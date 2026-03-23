"""
Training pipeline for the Controller (C model).

Supports two training methods:

1. CMA-ES (Covariance Matrix Adaptation Evolution Strategy):
   - Original World Models approach
   - Works well with tiny controllers (~1K params)
   - Population-based, gradient-free optimization
   - Evaluates controller fitness by running dream rollouts

2. Policy Gradient (REINFORCE / PPO):
   - More sample-efficient for larger controllers
   - Uses log-probabilities from score_actions()
   - Can train on dream rollouts or real game data
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

try:
    import torch
except ImportError:
    raise ImportError("Training requires PyTorch.")

import numpy as np

from ..controller import Controller, ControllerConfig
from ..world_model import WorldModel

logger = logging.getLogger(__name__)


@dataclass
class ControllerTrainingConfig:
    """Configuration for controller training."""

    method: str = "cma-es"          # "cma-es" or "policy_gradient"
    # CMA-ES settings
    population_size: int = 64
    num_generations: int = 100
    sigma_init: float = 0.1         # Initial mutation strength
    num_eval_rollouts: int = 16     # Dream rollouts per candidate
    rollout_max_steps: int = 50     # Max steps per dream rollout
    # Policy gradient settings
    pg_learning_rate: float = 1e-4
    pg_epochs: int = 50
    pg_batch_size: int = 32
    pg_gamma: float = 0.99          # Reward discount
    # General
    dream_temperature: float = 1.15
    checkpoint_dir: str = "checkpoints/controller"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train_controller_cmaes(
    world_model: WorldModel,
    config: ControllerTrainingConfig | None = None,
) -> Controller:
    """Train the controller using CMA-ES on dream rollouts.

    This is the original World Models approach:
    1. For each candidate parameter vector in the population
    2. Set controller params, run multiple dream rollouts
    3. Score = average cumulative reward across rollouts
    4. CMA-ES updates the distribution of parameter vectors

    Args:
        world_model: Full V+M+C world model (V and M should be pre-trained)
        config: Training configuration

    Returns:
        Trained Controller
    """
    config = config or ControllerTrainingConfig()
    controller = world_model.controller

    num_params = controller.num_parameters()
    logger.info(
        "Training controller with CMA-ES: %d parameters, pop=%d, gens=%d",
        num_params, config.population_size, config.num_generations,
    )

    try:
        import cma
    except ImportError:
        logger.error(
            "CMA-ES requires the 'cma' package. Install with: pip install cma"
        )
        raise

    # Initialize CMA-ES
    initial_params = controller.get_flat_params()
    es = cma.CMAEvolutionStrategy(
        initial_params.tolist(),
        config.sigma_init,
        {"popsize": config.population_size},
    )

    for generation in range(config.num_generations):
        # Get candidate solutions
        solutions = es.ask()
        fitnesses = []

        for params in solutions:
            # Set controller parameters
            controller.set_flat_params(np.array(params))

            # Evaluate via dream rollouts
            total_reward = _evaluate_in_dreams(
                world_model,
                num_rollouts=config.num_eval_rollouts,
                max_steps=config.rollout_max_steps,
                temperature=config.dream_temperature,
            )
            # CMA-ES minimizes, so negate reward
            fitnesses.append(-total_reward)

        es.tell(solutions, fitnesses)

        best_fitness = -min(fitnesses)
        mean_fitness = -np.mean(fitnesses)
        logger.info(
            "Generation %d: best=%.4f, mean=%.4f, sigma=%.4f",
            generation, best_fitness, mean_fitness, es.sigma,
        )

        if es.stop():
            logger.info("CMA-ES converged at generation %d", generation)
            break

    # Set best parameters
    best_params = es.result.xbest
    controller.set_flat_params(np.array(best_params))

    # Save
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(checkpoint_dir / "controller_cmaes_best.npy"), best_params)
    torch.save(controller.state_dict(), str(checkpoint_dir / "controller_final.pt"))

    return controller


def train_controller_pg(
    world_model: WorldModel,
    config: ControllerTrainingConfig | None = None,
) -> Controller:
    """Train the controller using REINFORCE policy gradient on dream rollouts.

    Args:
        world_model: Full V+M+C world model
        config: Training configuration

    Returns:
        Trained Controller
    """
    config = config or ControllerTrainingConfig()
    device = torch.device(config.device)
    controller = world_model.controller.to(device)

    optimizer = torch.optim.Adam(controller.parameters(), lr=config.pg_learning_rate)

    logger.info(
        "Training controller with policy gradient: %d params, %d epochs",
        controller.num_parameters(), config.pg_epochs,
    )

    for epoch in range(config.pg_epochs):
        # Collect dream rollouts
        trajectories = []
        for _ in range(config.pg_batch_size):
            traj = world_model.dream(
                z_start=torch.randn(1, world_model.config.encoder.latent_dim, device=device),
                temperature=config.dream_temperature,
            )
            trajectories.append(traj)

        # Compute policy gradient loss
        total_loss = torch.tensor(0.0, device=device)
        total_reward = 0.0

        for traj in trajectories:
            # Compute discounted returns
            returns = []
            G = 0.0
            for step in reversed(traj):
                G = step["reward"].item() + config.pg_gamma * G
                returns.insert(0, G)

            returns = torch.tensor(returns, device=device)
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            # REINFORCE loss: -sum(log_prob * return)
            for step, R in zip(traj, returns):
                total_loss -= step["log_prob"] * R

            total_reward += sum(s["reward"].item() for s in traj)

        total_loss /= max(len(trajectories), 1)
        avg_reward = total_reward / max(len(trajectories), 1)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            logger.info(
                "PG Epoch %d: loss=%.4f, avg_reward=%.4f",
                epoch, total_loss.item(), avg_reward,
            )

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(controller.state_dict(), str(checkpoint_dir / "controller_pg_final.pt"))

    return controller


def _evaluate_in_dreams(
    world_model: WorldModel,
    num_rollouts: int,
    max_steps: int,
    temperature: float,
) -> float:
    """Run dream rollouts and return average cumulative reward."""
    device = next(world_model.parameters()).device
    total_reward = 0.0

    with torch.no_grad():
        for _ in range(num_rollouts):
            z = torch.randn(1, world_model.config.encoder.latent_dim, device=device)
            traj = world_model.dream(z_start=z, temperature=temperature)
            episode_reward = sum(step["reward"].item() for step in traj)
            total_reward += episode_reward

    return total_reward / max(num_rollouts, 1)
