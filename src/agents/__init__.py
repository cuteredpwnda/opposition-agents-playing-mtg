"""Agent implementations and registry.

The registry lets callers build agents by short name (e.g. ``"random"``,
``"heuristic"``, ``"world_model"``).  This is what powers the
``--seat0 NAME`` flag promised in ``AGENTS.md`` and the overnight
training pipeline.

Heavy imports (torch, transformers, langchain, neo4j) are deferred to the
factory so importing this module stays cheap for the random/heuristic
fast path.
"""

from __future__ import annotations

from typing import Any, Callable

from src.agents.base_agent import MTGAgent
from src.agents.heuristic_agent import HeuristicAgent
from src.agents.random_agent import RandomAgent

# Factory signature: (player_id: str, **kwargs) -> MTGAgent.
_AgentFactory = Callable[..., MTGAgent]


def _make_random(player_id: str, **kw: Any) -> MTGAgent:
    return RandomAgent(player_id=player_id, name=kw.pop("name", f"Random({player_id})"))


def _make_heuristic(player_id: str, **kw: Any) -> MTGAgent:
    return HeuristicAgent(
        player_id=player_id,
        name=kw.pop("name", ""),
        seed=kw.pop("seed", None),
        prefer_aggressive=kw.pop("prefer_aggressive", True),
    )


def _make_kg_heuristic(player_id: str, **kw: Any) -> MTGAgent:
    from src.agents.kg_heuristic_agent import KGHeuristicAgent
    return KGHeuristicAgent(
        player_id=player_id,
        name=kw.pop("name", ""),
        seed=kw.pop("seed", None),
        kg=kw.pop("kg", None),
    )


def _make_human(player_id: str, **kw: Any) -> MTGAgent:
    from src.agents.human_agent import HumanAgent
    return HumanAgent(player_id=player_id, name=kw.pop("name", f"Human({player_id})"))


def _make_ollama(player_id: str, **kw: Any) -> MTGAgent:
    from src.agents.llm_agent import OllamaAgent
    return OllamaAgent(
        player_id=player_id,
        name=kw.pop("name", "Ollama"),
        model=kw.pop("model", "llama3"),
        base_url=kw.pop("base_url", "http://localhost:11434"),
    )


def _make_world_model(player_id: str, **kw: Any) -> MTGAgent:
    """Build a `WorldModelAgent`.

    Accepts either ``world_model=...`` + ``tokenizer=...`` directly, or a
    ``checkpoint=path/to/jepa_final.pt`` (the factory loads it).
    """
    from src.agents.world_model_agent import WorldModelAgent
    from src.world_model.world_model import WorldModel
    from src.world_model.game_tokenizer import GameTokenizer
    from src.world_model.card_embeddings import CardEmbeddingModel

    world_model = kw.pop("world_model", None)
    tokenizer = kw.pop("tokenizer", None)
    if world_model is None:
        ckpt = kw.pop("checkpoint", "checkpoints/jepa/jepa_final.pt")
        world_model = WorldModel.load(ckpt)
    if tokenizer is None:
        card_model = CardEmbeddingModel()
        try:
            from scripts.train_graph_embeddings import load_embedding_cache
            embeds, name_to_idx = load_embedding_cache()
            for name, idx in name_to_idx.items():
                card_model._embeddings[name] = embeds[idx].numpy()
        except Exception:
            pass
        tokenizer = GameTokenizer(card_embeddings=card_model.get_all_embeddings())

    return WorldModelAgent(
        player_id=player_id,
        world_model=world_model,
        tokenizer=tokenizer,
        kg_encoder=kw.pop("kg_encoder", None),
        name=kw.pop("name", "WorldModelAgent"),
        mode=kw.pop("mode", "direct"),
        device=kw.pop("device", "cpu"),
    )


def _make_active_inference(player_id: str, **kw: Any) -> MTGAgent:
    from src.agents.active_inference_agent import ActiveInferenceAgent
    return ActiveInferenceAgent(
        player_id=player_id,
        kg=kw.pop("kg", None),
        known_opponent_decklist=kw.pop("known_opponent_decklist", None),
        name=kw.pop("name", "ActiveInferenceAgent"),
    )


def _make_llm_fusion(player_id: str, **kw: Any) -> MTGAgent:
    """Build a `LLMFusionAgent` (LLM + world model + KG fusion).

    Optional kwargs: ``world_model`` / ``tokenizer`` / ``checkpoint``,
    ``knowledge_graph``, ``opponent_model``, ``llm_model``, plus any
    `FusionConfig` field overrides.
    """
    from src.agents.llm_fusion_agent import LLMFusionAgent, FusionConfig

    cfg_overrides: dict[str, Any] = {}
    for fname in (
        "llm_weight", "world_model_weight", "kg_weight", "heuristic_weight",
        "dream_rollouts", "dream_depth", "dream_temperature",
        "llm_provider", "llm_model", "llm_timeout",
        "skip_llm_if_obvious", "obvious_threshold",
    ):
        if fname in kw:
            cfg_overrides[fname] = kw.pop(fname)
    config = kw.pop("config", None) or FusionConfig(**cfg_overrides)

    world_model = kw.pop("world_model", None)
    tokenizer = kw.pop("tokenizer", None)
    if world_model is None and kw.get("checkpoint"):
        from src.world_model.world_model import WorldModel
        world_model = WorldModel.load(kw.pop("checkpoint"))
    if world_model is not None and tokenizer is None:
        from src.world_model.game_tokenizer import GameTokenizer
        from src.world_model.card_embeddings import CardEmbeddingModel
        card_model = CardEmbeddingModel()
        tokenizer = GameTokenizer(card_embeddings=card_model.get_all_embeddings())

    return LLMFusionAgent(
        player_id=player_id,
        name=kw.pop("name", "Fusion"),
        config=config,
        world_model=world_model,
        tokenizer=tokenizer,
        knowledge_graph=kw.pop("knowledge_graph", kw.pop("kg", None)),
        opponent_model=kw.pop("opponent_model", None),
    )


AGENT_REGISTRY: dict[str, _AgentFactory] = {
    "random": _make_random,
    "heuristic": _make_heuristic,
    "kg_heuristic": _make_kg_heuristic,
    "human": _make_human,
    "ollama": _make_ollama,
    "llm": _make_ollama,  # alias
    "world_model": _make_world_model,
    "active_inference": _make_active_inference,
    "llm_fusion": _make_llm_fusion,
    "fusion": _make_llm_fusion,  # alias
}


def make_agent(name: str, player_id: str, **kwargs: Any) -> MTGAgent:
    """Build an agent by short name. Raises ``KeyError`` on unknown names."""
    try:
        factory = AGENT_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(AGENT_REGISTRY))
        raise KeyError(
            f"Unknown agent '{name}'. Available: {available}"
        ) from exc
    return factory(player_id, **kwargs)


def list_agents() -> list[str]:
    """Return all registered agent short names."""
    return sorted(AGENT_REGISTRY)


__all__ = [
    "MTGAgent",
    "RandomAgent",
    "HeuristicAgent",
    "AGENT_REGISTRY",
    "make_agent",
    "list_agents",
]

