"""Auditable native graph/JEPA/dynamics/imitation training and strict deployment."""

from __future__ import annotations

import hashlib
import itertools
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch_geometric.data import Data

from src.integrations.phase_rs.native_learning_data import (
    ACTION_DIM,
    FEATURE_VERSION,
    encode_action,
    observation,
    playable_indices,
    text_vector,
)
from src.knowledge.graph_embedder import CardGraphEmbedder
from src.world_model.controller import ControllerConfig
from src.world_model.dynamics_model import DynamicsModelConfig
from src.world_model.jepa_predictor import JEPAPredictorConfig
from src.world_model.state_encoder import StateEncoderConfig
from src.world_model.world_model import WorldModel, WorldModelConfig


def induce_graph(games: list[dict], output: Path) -> dict:
    """CO_VISIBLE exposure is observational evidence, never a causal combo."""
    evidence = []
    aggregates: dict[tuple[str, str], dict] = {}
    for game in games:
        if game["split"] != "train":
            raise ValueError("Induced graph accepts training games only")
        if not game["completed"]:
            continue
        pairs = set()
        for record in game["records"][:-1]:
            pairs.update(itertools.combinations(sorted(set(record["visible_cards"])), 2))
        reward = game["records"][-1]["reward"]
        for pair in sorted(pairs):
            evidence.append({
                "relation": "CO_VISIBLE", "cards": list(pair), "game_id": game["game_id"],
                "producer": game["producer"], "deck_sha256": game["deck_sha256"],
                "source_sha256": game["source_sha256"], "split": "train",
                "terminal_outcome": reward, "epistemic_status": "induced_observation",
            })
            entry = aggregates.setdefault(pair, {
                "cards": list(pair), "exposed_games": 0, "wins": 0, "losses": 0, "draws": 0,
            })
            entry["exposed_games"] += 1
            entry["wins" if reward > 0 else "losses" if reward < 0 else "draws"] += 1
    if not aggregates:
        raise ValueError("No observed training card-pair evidence; graph training cannot run")
    graph = {
        "schema_version": 1, "relation": "CO_VISIBLE",
        "interpretation": "observed co-visibility, not causal synergy or rules truth",
        "training_game_ids": [game["game_id"] for game in games if game["completed"]],
        "edges": list(aggregates.values()), "evidence": evidence,
    }
    output.write_text(json.dumps(graph, indent=2), encoding="utf-8")
    return graph


def train_graph(graph: dict, seed: int, epochs: int) -> tuple[dict, list[float]]:
    torch.manual_seed(seed)
    names = sorted({name for edge in graph["edges"] for name in edge["cards"]})
    indices = {name: i for i, name in enumerate(names)}
    pairs = sorted({tuple(sorted(indices[name] for name in edge["cards"]))
                    for edge in graph["edges"]})
    negatives = list(set(itertools.combinations(range(len(names)), 2)) - set(pairs))
    if not negatives:
        raise ValueError("Complete co-visibility graph has no negative pairs for link training")
    positive = torch.tensor(pairs, dtype=torch.long).T
    edges = torch.cat([positive, positive.flip(0)], dim=1)
    negative = torch.tensor(sorted(negatives), dtype=torch.long).T
    x = torch.tensor(np.stack([text_vector(name, 32) for name in names]))
    data = Data(x=x, edge_index=edges)
    model = CardGraphEmbedder(32, 64, 32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    history = []
    for _ in range(epochs):
        model.train()
        z = F.normalize(model(data.x, data.edge_index), dim=1)
        pos = (z[positive[0]] * z[positive[1]]).sum(1) * 4
        neg = (z[negative[0]] * z[negative[1]]).sum(1) * 4
        loss = F.binary_cross_entropy_with_logits(
            torch.cat([pos, neg]), torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
        )
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite graph loss")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    model.eval()
    with torch.no_grad():
        embeddings = F.normalize(model(data.x, data.edge_index), dim=1)
    return {
        "names": names, "embeddings": embeddings, "state_dict": model.state_dict(),
        "edge_index": edges, "feature_version": "hashed_name32",
        "training_game_ids": graph["training_game_ids"],
    }, history


def make_model(seed: int) -> WorldModel:
    torch.manual_seed(seed)
    return WorldModel(WorldModelConfig(
        encoder=StateEncoderConfig(
            hidden_dim=128, latent_dim=32, kg_embed_dim=32, dropout=0, decision_context_dim=128,
        ),
        dynamics=DynamicsModelConfig(
            latent_dim=32, action_dim=ACTION_DIM, hidden_dim=64,
            num_layers=1, num_gaussians=3, dropout=0,
        ),
        controller=ControllerConfig(latent_dim=32, hidden_state_dim=64, action_dim=ACTION_DIM),
        jepa=JEPAPredictorConfig(
            latent_dim=32, action_dim=ACTION_DIM, hidden_dim=64,
            use_transformer=False, dropout=0, jepa_beta=0.001,
        ), use_jepa=True,
    ))


def graph_context(names: list[str], graph: dict, enabled: bool = True) -> torch.Tensor:
    indices = {name: i for i, name in enumerate(graph["names"])}
    found = [indices[name] for name in names if name in indices]
    if not enabled or not found:
        return torch.zeros(1, 32)
    return graph["embeddings"][found].mean(0, keepdim=True)


def latent(model: WorldModel, record: dict, graph: dict, enabled: bool = True):
    features = {key: value.unsqueeze(0) for key, value in record["features"].items()}
    return model.encoder.encode(
        features, kg_embedding=graph_context(record["visible_cards"], graph, enabled),
    )[1]


def parameter_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def diagnostics(model: WorldModel, games: list[dict], graph: dict) -> dict:
    model.eval()
    prediction, imitation, accuracy, latents, sequence_metrics = [], [], [], [], []
    choice_imitation, choice_accuracy = [], []
    with torch.no_grad():
        for game in games:
            if not game["completed"]:
                continue
            records = game["records"]
            zs = torch.cat([latent(model, record, graph) for record in records])
            actions = torch.stack([
                record["actions"][record["chosen_index"]] for record in records
            ])
            sequence_metrics.append({
                key: float(value) for key, value in model.dynamics.sequence_loss(
                    zs.unsqueeze(0), actions.unsqueeze(0),
                    torch.tensor([[float(record["done"]) for record in records]]),
                    torch.tensor([[record["reward"] for record in records]]),
                ).items()
            })
            hidden = None
            for now, nxt in zip(game["records"][:-1], game["records"][1:]):
                z = latent(model, now, graph)
                target = latent(model, nxt, graph)
                latents.append(z.squeeze(0))
                action = now["actions"][now["chosen_index"]].unsqueeze(0)
                prediction.append(float(F.mse_loss(model.jepa_predictor(z, action), target)))
                h = model.dynamics.get_hidden_state_vector(hidden)
                scores = model.controller.score_actions(z, h, now["actions"].unsqueeze(0))
                imitation.append(float(-scores[0, now["chosen_index"]]))
                accuracy.append(int(scores.argmax(1).item() == now["chosen_index"]))
                eligible = playable_indices(now["legal_actions"])
                if len(eligible) > 1:
                    choice_scores = model.controller.score_actions(
                        z, h, now["actions"][eligible].unsqueeze(0),
                    )
                    selected = eligible.index(now["chosen_index"])
                    choice_imitation.append(float(-choice_scores[0, selected]))
                    choice_accuracy.append(int(choice_scores.argmax(1).item() == selected))
                _, hidden, _ = model.dynamics.step(z, action, hidden)
    if not prediction:
        raise ValueError("Held-out diagnostics require real terminal decision pairs")
    return {
        "pairs": len(prediction), "jepa_mse": float(np.mean(prediction)),
        "imitation_nll": float(np.mean(imitation)), "imitation_accuracy": float(np.mean(accuracy)),
        "nonforced_choice_count": len(choice_accuracy),
        "nonforced_imitation_nll": (
            float(np.mean(choice_imitation)) if choice_imitation else None
        ),
        "nonforced_imitation_accuracy": (
            float(np.mean(choice_accuracy)) if choice_accuracy else None
        ),
        "latent_variance_mean": float(torch.stack(latents).var(0, unbiased=False).mean()),
        "recurrent_metrics": {
            key: float(np.mean([metrics[key] for metrics in sequence_metrics]))
            for key in ("total", "z_pred", "reward", "done")
        },
    }


def train_components(model: WorldModel, games: list[dict], graph: dict, epochs: int) -> list[dict]:
    if any(game["split"] != "train" for game in games):
        raise ValueError("Optimizers accept training games only")
    pairs = [(now, nxt) for game in games if game["completed"]
             for now, nxt in zip(game["records"][:-1], game["records"][1:])]
    if not pairs:
        raise ValueError("No terminal native transitions for training")
    optimizer = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.jepa_predictor.parameters()), lr=0.001,
    )
    target_encoder = deepcopy(model.encoder).eval()
    target_encoder.requires_grad_(False)
    history = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for start in range(0, len(pairs), 64):
            batch = pairs[start:start + 64]
            features = {
                key: torch.stack([now["features"][key] for now, _ in batch])
                for key in batch[0][0]["features"]
            }
            following = {
                key: torch.stack([nxt["features"][key] for _, nxt in batch])
                for key in batch[0][1]["features"]
            }
            contexts = torch.cat([graph_context(now["visible_cards"], graph) for now, _ in batch])
            next_contexts = torch.cat([
                graph_context(nxt["visible_cards"], graph) for _, nxt in batch
            ])
            _, mu, logvar = model.encoder.encode(features, kg_embedding=contexts)
            with torch.no_grad():
                target = target_encoder.encode(following, kg_embedding=next_contexts)[1]
            actions = torch.stack([now["actions"][now["chosen_index"]] for now, _ in batch])
            predicted = model.jepa_predictor(mu, actions)
            variance_loss = F.relu(
                0.05 - torch.sqrt(mu.var(dim=0, unbiased=False) + 1e-6)
            ).square().mean()
            loss = F.mse_loss(predicted, target) + 10 * variance_loss + 0.001 * (
                -0.5 * (1 + logvar - mu.square() - logvar.exp()).mean()
            )
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite native JEPA loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            with torch.no_grad():
                for target_parameter, parameter in zip(
                    target_encoder.parameters(), model.encoder.parameters(), strict=True,
                ):
                    target_parameter.lerp_(parameter, 0.01)
            losses.append(float(loss.detach()))
        history.append({"epoch": epoch + 1, "jepa_loss": float(np.mean(losses))})

    model.eval()
    sequences = []
    with torch.no_grad():
        for game in games:
            if game["completed"]:
                records = game["records"]
                sequences.append((
                    torch.cat([latent(model, record, graph) for record in records]),
                    torch.stack([record["actions"][record["chosen_index"]] for record in records]),
                    torch.tensor([record["reward"] for record in records]),
                    torch.tensor([float(record["done"]) for record in records]),
                    records,
                ))
    optimizer = torch.optim.Adam(model.dynamics.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.dynamics.train()
        losses = []
        for zs, actions, rewards, dones, _ in sequences:
            # Overlap one observation to retain every consecutive target.
            for start in range(0, len(zs) - 1, 32):
                end = min(start + 33, len(zs))
                loss = model.dynamics.sequence_loss(
                    zs[start:end].unsqueeze(0), actions[start:end].unsqueeze(0),
                    dones[start:end].unsqueeze(0), rewards[start:end].unsqueeze(0),
                )["total"]
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite recurrent dynamics/reward loss")
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.dynamics.parameters(), 1)
                optimizer.step()
                losses.append(float(loss.detach()))
        history[epoch]["dynamics_loss"] = float(np.mean(losses))
    model.eval()
    policy_samples = []
    with torch.no_grad():
        for zs, actions, _, _, records in sequences:
            hidden = None
            for i, record in enumerate(records[:-1]):
                h = model.dynamics.get_hidden_state_vector(hidden).detach()
                policy_samples.append((zs[i:i+1], h, record["actions"], record["chosen_index"]))
                _, hidden, _ = model.dynamics.step(zs[i:i+1], actions[i:i+1], hidden)
    optimizer = torch.optim.Adam(model.controller.parameters(), lr=0.003)
    for epoch in range(epochs):
        losses = []
        for start in range(0, len(policy_samples), 64):
            batch = policy_samples[start:start + 64]
            loss = torch.stack([
                -model.controller.score_actions(z, h, actions.unsqueeze(0))[0, chosen]
                for z, h, actions, chosen in batch
            ]).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite native controller loss")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        history[epoch]["controller_imitation_loss"] = float(np.mean(losses))
    return history


def save_bundle(path: Path, model: WorldModel, graph: dict, metadata: dict) -> None:
    torch.save({
        "feature_version": FEATURE_VERSION, "config": model.config,
        "state_dict": model.state_dict(), "graph": graph, "metadata": metadata,
        "component_digests": {key: parameter_digest(getattr(model, key))
                              for key in ("encoder", "jepa_predictor", "dynamics", "controller")},
    }, path)


class NativeLearnedPicker:
    """Strict native-trained controller with a frozen graph intervention."""

    def __init__(
        self, checkpoint: Path, graph_enabled: bool = True, seed: int = 0,
        selection: str = "sample",
    ):
        if selection not in {"sample", "argmax"}:
            raise ValueError("Native selection must be sample or argmax")
        bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if bundle["feature_version"] != FEATURE_VERSION:
            raise ValueError("Native learning feature version mismatch")
        self.model = WorldModel(bundle["config"])
        self.model.load_state_dict(bundle["state_dict"], strict=True)
        if set(bundle["component_digests"]) != {
            "encoder", "jepa_predictor", "dynamics", "controller",
        }:
            raise ValueError("Native checkpoint requires all four component digests")
        for key, expected in bundle["component_digests"].items():
            if parameter_digest(getattr(self.model, key)) != expected:
                raise ValueError(f"Native component digest mismatch: {key}")
        self.model.eval()
        self.graph, self.graph_enabled = bundle["graph"], graph_enabled
        self.graph_digest = parameter_digest_from_graph(self.graph)
        self.name = "native_learned_graph" if graph_enabled else "native_learned_no_graph"
        self.hidden = None
        self.selection = selection
        self.generator = torch.Generator().manual_seed(seed)
        self.last_reasoning: dict = {}

    def pick(self, actions: list[dict], state: dict, seat: int) -> int:
        features, names = observation(state, seat)
        record = {"features": features, "visible_cards": names}
        eligible = playable_indices(actions)
        encodings = torch.stack([encode_action(actions[i], state, seat) for i in eligible])
        with torch.no_grad():
            z = latent(self.model, record, self.graph, self.graph_enabled)
            h = self.model.dynamics.get_hidden_state_vector(self.hidden)
            scores = self.model.controller.score_actions(z, h, encodings.unsqueeze(0))
            if not torch.isfinite(scores).all():
                raise ValueError("Nonfinite native controller action scores")
            selected = (
                torch.multinomial(scores.exp()[0], 1, generator=self.generator).item()
                if self.selection == "sample" else scores.argmax(1).item()
            )
            without = latent(self.model, record, self.graph, False)
            ablated = self.model.controller.score_actions(without, h, encodings.unsqueeze(0))
            delta = float((scores - ablated).abs().max())
            _, self.hidden, _ = self.model.dynamics.step(
                z, encodings[selected:selected+1], self.hidden,
            )
        found = sorted(set(names) & set(self.graph["names"]))
        self.last_reasoning = {
            "backend": "native_trained_controller", "feature_version": FEATURE_VERSION,
            "graph_enabled": self.graph_enabled, "graph_sha256": self.graph_digest,
            "retrieved_cards": found if self.graph_enabled else [],
            "graph_score_delta_max": delta, "chosen_index": eligible[selected],
            "controller_invoked": True, "dynamics_invoked": True,
            "component_loading": "strict_complete", "candidate_indices": eligible,
            "selection": self.selection, "candidate_probabilities": scores.exp()[0].tolist(),
        }
        return eligible[selected]


def parameter_digest_from_graph(graph: dict) -> str:
    return hashlib.sha256(
        json.dumps(graph["names"]).encode() + graph["embeddings"].numpy().tobytes()
    ).hexdigest()
