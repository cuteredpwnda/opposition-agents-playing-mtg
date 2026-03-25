"""
World Model for Magic: The Gathering.

Applies Ha & Schmidhuber (2018) "World Models" framework to MTG:
- V (State Encoder): GameState → latent vector z
- M (Dynamics Model): Predicts P(z_{t+1} | z_t, a_t, h_t)
- C (Controller): Maps (z, h) → action distribution

Dual-input JEPA extension (LeWM-style):
- KGContextEncoder: visible cards → semantic context embedding
- JEPAPredictor:    (z_t, a_t) → ẑ_{t+1}  [prediction + KL loss, 1 hyperparameter]
- WorldModel.jepa_training_step() orchestrates the full dual-input training loop.

See docs/WORLD_MODEL_DESIGN.md and docs/KG_WORLD_MODEL_PRESENTATION.md for details.
"""

from src.world_model.card_embeddings import CardEmbeddingModel
from src.world_model.controller import Controller
from src.world_model.dynamics_model import DynamicsModel
from src.world_model.game_tokenizer import GameTokenizer
from src.world_model.jepa_predictor import JEPAPredictor, JEPAPredictorConfig
from src.world_model.kg_encoder import KGContextEncoder, KGContextEncoderConfig
from src.world_model.state_encoder import StateEncoder, StateEncoderConfig
from src.world_model.trajectory import Trajectory, TrajectoryStore, Transition
from src.world_model.world_model import WorldModel, WorldModelConfig

__all__ = [
    "WorldModel",
    "WorldModelConfig",
    "StateEncoder",
    "StateEncoderConfig",
    "DynamicsModel",
    "Controller",
    "GameTokenizer",
    "CardEmbeddingModel",
    "JEPAPredictor",
    "JEPAPredictorConfig",
    "KGContextEncoder",
    "KGContextEncoderConfig",
    "Trajectory",
    "Transition",
    "TrajectoryStore",
]
