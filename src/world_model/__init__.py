"""
World Model for Magic: The Gathering.

Applies Ha & Schmidhuber (2018) "World Models" framework to MTG:
- V (State Encoder): GameState → latent vector z
- M (Dynamics Model): Predicts P(z_{t+1} | z_t, a_t, h_t)
- C (Controller): Maps (z, h) → action distribution

See docs/WORLD_MODEL_DESIGN.md for full architecture details.
"""

from src.world_model.card_embeddings import CardEmbeddingModel
from src.world_model.controller import Controller
from src.world_model.dynamics_model import DynamicsModel
from src.world_model.game_tokenizer import GameTokenizer
from src.world_model.state_encoder import StateEncoder
from src.world_model.trajectory import Trajectory, TrajectoryStore, Transition
from src.world_model.world_model import WorldModel

__all__ = [
    "WorldModel",
    "StateEncoder",
    "DynamicsModel",
    "Controller",
    "GameTokenizer",
    "CardEmbeddingModel",
    "Trajectory",
    "Transition",
    "TrajectoryStore",
]
