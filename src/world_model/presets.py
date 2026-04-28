"""Named ``WorldModelConfig`` presets used by ablation sweeps.

Every preset returns a fully-configured ``WorldModelConfig`` instance.
The presets vary the latent / hidden dimensions and LSTM depth so that we
can quantify how scale affects the world model's predictive accuracy and
the resulting agent's win rate.
"""

from __future__ import annotations

from dataclasses import dataclass

from .controller import ControllerConfig
from .dynamics_model import DynamicsModelConfig
from .jepa_predictor import JEPAPredictorConfig
from .state_encoder import StateEncoderConfig
from .world_model import WorldModelConfig


@dataclass(frozen=True)
class WorldModelPreset:
    """Hyper-parameters describing a single world-model size."""

    name: str
    latent_dim: int
    hidden_dim: int
    lstm_layers: int
    transformer_layers: int
    approx_params_million: float
    target_vram_gb: float
    notes: str = ""

    def build_config(self, *, use_jepa: bool = True,
                     action_dim: int = 136) -> WorldModelConfig:
        """Materialise this preset as a ``WorldModelConfig``."""
        encoder = StateEncoderConfig(
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
        )
        dynamics = DynamicsModelConfig(
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.lstm_layers,
            action_dim=action_dim,
        )
        controller = ControllerConfig(
            latent_dim=self.latent_dim,
            hidden_state_dim=self.hidden_dim,
            action_dim=action_dim,
        )
        jepa = JEPAPredictorConfig(
            latent_dim=self.latent_dim,
            action_dim=action_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.transformer_layers,
        )
        return WorldModelConfig(
            encoder=encoder,
            dynamics=dynamics,
            controller=controller,
            jepa=jepa,
            use_jepa=use_jepa,
        )


WORLD_MODEL_PRESETS: dict[str, WorldModelPreset] = {
    "tiny": WorldModelPreset(
        name="tiny",
        latent_dim=64,
        hidden_dim=128,
        lstm_layers=1,
        transformer_layers=1,
        approx_params_million=0.3,
        target_vram_gb=1.0,
        notes="CI / smoke tests.",
    ),
    "small": WorldModelPreset(
        name="small",
        latent_dim=128,
        hidden_dim=256,
        lstm_layers=2,
        transformer_layers=2,
        approx_params_million=2.0,
        target_vram_gb=2.0,
    ),
    "medium": WorldModelPreset(
        name="medium",
        latent_dim=256,
        hidden_dim=512,
        lstm_layers=2,
        transformer_layers=2,
        approx_params_million=12.0,
        target_vram_gb=4.0,
        notes="Default reference model used in the tech report.",
    ),
    "large": WorldModelPreset(
        name="large",
        latent_dim=384,
        hidden_dim=768,
        lstm_layers=3,
        transformer_layers=3,
        approx_params_million=40.0,
        target_vram_gb=10.0,
    ),
    "xl": WorldModelPreset(
        name="xl",
        latent_dim=512,
        hidden_dim=1024,
        lstm_layers=4,
        transformer_layers=4,
        approx_params_million=120.0,
        target_vram_gb=20.0,
        notes="Requires a 24GB GPU.",
    ),
}


def get_preset(name: str) -> WorldModelPreset:
    if name not in WORLD_MODEL_PRESETS:
        raise KeyError(
            f"Unknown world-model preset '{name}'. "
            f"Available: {sorted(WORLD_MODEL_PRESETS)}"
        )
    return WORLD_MODEL_PRESETS[name]


def list_presets(max_vram_gb: float | None = None) -> list[WorldModelPreset]:
    presets = list(WORLD_MODEL_PRESETS.values())
    if max_vram_gb is not None:
        presets = [p for p in presets if p.target_vram_gb <= max_vram_gb]
    return sorted(presets, key=lambda p: p.approx_params_million)
