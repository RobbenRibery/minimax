"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from typing import Tuple, Optional
from functools import partial
from enum import IntEnum, Enum

import numpy as np

import chex
import jax
import jax.numpy as jnp

from .common import make_maze_map
from .maze import EnvState, EnvParams
from minimax.envs.registration import register_mutator


class Mutations(IntEnum):
    # No Operation, Flip the Wall, Move the Goal
    NO_OP = 0
    FLIP_WALL = 1
    MOVE_GOAL = 2

class MutationWeights(Enum): 
    # No Operation, Flip the Wall, Move the Goal
    NO_OP = 0.0
    FLIP_WALL = 0.0
    MOVE_GOAL = 0.0


def flip_wall(rng:chex.PRNGKey, state: EnvState) -> EnvState:
    """
    Flips a wall in the given state by randomly selecting a wall that is not the goal or agent position.

    Args:
        rng (ArrayImpl): The random number generator.
        state (EnvState): The current state of the environment.

    Returns:
        EnvState: The updated state with the flipped wall.

    """
    wall_map:chex.Array = state.wall_map
    h, w = wall_map.shape
    wall_mask = jnp.ones((h * w,), dtype=jnp.bool_)

    goal_idx = w * state.goal_pos[1] + state.goal_pos[0]
    agent_idx = w * state.agent_pos[1] + state.agent_pos[0]
    wall_mask = wall_mask.at[goal_idx].set(False)
    wall_mask = wall_mask.at[agent_idx].set(False)
    
    # sample a single wall position to flip
    flip_idx = jax.random.choice(rng, np.arange(h * w), p=wall_mask)
    flip_y = flip_idx // w
    flip_x = flip_idx % w

    flip_val = ~wall_map.at[flip_y, flip_x].get()
    next_wall_map = wall_map.at[flip_y, flip_x].set(flip_val)

    return state.replace(wall_map=next_wall_map)


def move_goal(rng:chex.PRNGKey, state:EnvState) -> EnvState:
    """
    Moves the goal position in the given state to a randomly chosen location that is not a wall or the agent's position.

    Args:
        rng (chex.PRNGKey): The random number generator.
        state (EnvState): The current state of the environment.

    Returns:
        EnvState: The updated state with the goal position moved.

    """
    
    wall_map:chex.Array = state.wall_map
    h, w = wall_map.shape
    wall_mask = wall_map.flatten()

    goal_idx = w * state.goal_pos[1] + state.goal_pos[0]
    agent_idx = w * state.agent_pos[1] + state.agent_pos[0]
    wall_mask = wall_mask.at[goal_idx].set(True)
    wall_mask = wall_mask.at[agent_idx].set(True)

    next_goal_idx = jax.random.choice(rng, np.arange(h * w), p=~wall_mask)
    next_goal_y = next_goal_idx // w
    next_goal_x = next_goal_idx % w

    next_wall_map = wall_map.at[next_goal_y, next_goal_x].set(False)
    next_goal_pos = jnp.array([next_goal_x, next_goal_y], dtype=jnp.uint32)

    return state.replace(wall_map=next_wall_map, goal_pos=next_goal_pos)


@partial(jax.jit, static_argnums=(1, 3))
def move_goal_flip_walls(
    rng, 
    params:EnvParams, 
    state:EnvState, 
    n:int=1, 
    mutators_prob:Optional[jax.Array]=None
    ) -> EnvState:
    """
    Mutates the given state based on the provided mutation step.

    Args:
        rng (jax.random.PRNGKey): The random number generator.
        params (EnvParams): The environment parameters.
        state (EnvState): The current state of the environment.
        n (int, optional): The number of mutations to perform. Defaults to 1.
        mutators_prob (jax.Array, optional): The probabilities of each mutation type. Defaults to None.

    Returns:
        EnvState: The mutated state after performing the mutations.

    Raises:
        None
    """
    
    if n == 0:
        return state

    def _mutate(carry:EnvState, step:Tuple):
        """
        Mutates the given state based on the provided mutation step.

        Args:
            carry (Any): The current state to be mutated.
            step (Tuple[jnp.ndarray, int]): A tuple containing the random number generator (rng) and the mutation step.

        Returns:
            Tuple[Any, None]: A tuple containing the mutated state and None.

        Raises:
            None
        """
        state = carry

        # mutation can only be 0, 1, or 2
        rng, mutation = step

        rng, arng, brng = jax.random.split(rng, 3)
        
        # mutation must be a scalar:int, for element-wise comparison 
        is_flip_wall = jnp.equal(mutation, Mutations.FLIP_WALL.value) 
        mutated_state = flip_wall(arng, state)
        # if going to, return the mutated state. Otherwise, return the original state 
        next_state = jax.tree_util.tree_map(
            lambda x, y: jax.lax.select(is_flip_wall, x, y), mutated_state, state
        )

        # mutation must be a scalar:int, for element-wise comparison 
        is_move_goal = jnp.equal(mutation, Mutations.MOVE_GOAL.value)
        mutated_state = move_goal(brng, state)
        # if going to, return the mutated state. Otherwise, return the original state
        next_state = jax.tree_util.tree_map(
            lambda x, y: jax.lax.select(is_move_goal, x, y), mutated_state, next_state
        )

        return next_state, None

    rng, nrng, *mrngs = jax.random.split(rng, n + 2)
    
    # chose types of mutations, indexed by 0, 1 and 2
    mutations = jax.random.choice(nrng, np.arange(len(Mutations)), (n,), p=mutators_prob)
    
    # get the final state after a carried over n-step mutations
    state, _ = jax.lax.scan(_mutate, state, (jnp.array(mrngs), mutations))

    # Update state maze_map
    next_maze_map = make_maze_map(
        params,
        state.wall_map,
        state.goal_pos,
        state.agent_pos,
        state.agent_dir_idx,
        pad_obs=True,
    )
    return state.replace(maze_map=next_maze_map)


# Register the mutators
if hasattr(__loader__, "name"):
    module_path = __loader__.name
elif hasattr(__loader__, "fullname"):
    module_path = __loader__.fullname

register_mutator(
    env_id="Maze", mutator_id=None, entry_point=module_path + ":move_goal_flip_walls"
)
