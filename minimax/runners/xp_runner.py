"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from typing import Tuple, Dict, Optional, Union

import copy
from functools import partial
from collections import defaultdict
import time

import numpy as np
import jax
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
import chex

from .eval_runner import EvalRunner
from .dr_runner import DRRunner
from .paired_runner import PAIREDRunner
from .plr_runner import PLRRunner
from minimax.util.loggers import Logger
from minimax.util.rl import VmapTrainState
import minimax.envs as envs
import minimax.models as models
import minimax.agents as agents


class RunnerInfo:
    def __init__(self, runner_cls, is_ued=False):
        self.runner_cls = runner_cls
        self.is_ued = is_ued


RUNNER_INFO = {
    "dr": RunnerInfo(
        runner_cls=DRRunner,
    ),
    "plr": RunnerInfo(
        runner_cls=PLRRunner,
    ),
    "paired": RunnerInfo(runner_cls=PAIREDRunner, is_ued=True),
}


class ExperimentRunner:
    def __init__(
        self,
        train_runner:str,
        env_name:str,
        agent_rl_algo,
        student_model_name,
        teacher_model_name=None,
        train_runner_kwargs:Dict={},
        env_kwargs:Dict={},
        ued_env_kwargs:Dict={},
        student_rl_kwargs:Dict={},
        teacher_rl_kwargs:Dict={},
        student_model_kwargs:Dict={},
        teacher_model_kwargs:Dict={},
        eval_kwargs:Dict={},
        eval_env_kwargs:Dict={},
        n_devices:int=1,
    ):
        self.env_name = env_name
        self.agent_rl_algo = agent_rl_algo
        self.is_ued = RUNNER_INFO[train_runner].is_ued

        dummy_env = envs.make(env_name, env_kwargs, ued_env_kwargs)[0]

        # ---- Make agent ----
        student_model_kwargs["output_dim"] = dummy_env.action_space().n
        student_model = models.make(
            env_name=env_name, model_name=student_model_name, **student_model_kwargs
        )

        student_agent = agents.PPOAgent(
            model=student_model, n_devices=n_devices, **student_rl_kwargs
        )

        # ---- Handle UED-related settings ----
        if self.is_ued:
            max_teacher_steps = dummy_env.ued_max_episode_steps()
            teacher_model_kwargs["n_scalar_embeddings"] = max_teacher_steps
            teacher_model_kwargs["max_scalar"] = max_teacher_steps
            teacher_model_kwargs["output_dim"] = dummy_env.ued_action_space().n

            teacher_model = models.make(
                env_name=env_name, model_name=teacher_model_name, **teacher_model_kwargs
            )

            teacher_agent = agents.PPOAgent(
                model=teacher_model, n_devices=n_devices, **teacher_rl_kwargs
            )

            train_runner_kwargs.update(dict(teacher_agents=[teacher_agent]))
            train_runner_kwargs.update(dict(ued_env_kwargs=ued_env_kwargs))

        # ---- Set up train runner ----
        runner_cls:Union[PLRRunner, DRRunner, PAIREDRunner] = RUNNER_INFO[train_runner].runner_cls

        # Set up learning rate annealing parameters
        lr_init = train_runner_kwargs.lr
        lr_final = train_runner_kwargs.lr_final
        lr_anneal_steps = train_runner_kwargs.lr_anneal_steps

        if lr_final is None:
            train_runner_kwargs.lr_final = lr_init
        if train_runner_kwargs.lr_final == train_runner_kwargs.lr:
            train_runner_kwargs.lr_anneal_steps = 0

        self.runner:Union[PLRRunner, DRRunner, PAIREDRunner] = runner_cls(
            env_name=env_name,
            env_kwargs=env_kwargs,
            student_agents=[student_agent],
            n_devices=n_devices,
            **train_runner_kwargs,
        )

        # ---- Make eval runner ----
        if eval_kwargs.get("env_names") is None:
            self.eval_runner = None
        else:
            self.eval_runner = EvalRunner(
                pop=self.runner.student_pop, env_kwargs=eval_env_kwargs, **eval_kwargs
            )

        self._start_tick = 0

        # ---- Set up device parallelism ----
        self.n_devices = n_devices
        if n_devices > 1:
            dummy_runner_state = self.runner.reset(jax.random.PRNGKey(0))
            self._shmap_run = self._make_shmap_run(dummy_runner_state)
        else:
            self._shmap_run = None

    @partial(jax.jit, static_argnums=(0,))
    def step(self, runner_state:Tuple, evaluate:bool=False):
        if self.n_devices > 1:
            run_fn = self._shmap_run
        else:
            run_fn = self.runner.run

        stats, *runner_state = run_fn(*runner_state)

        rng = runner_state[0]
        rng, subrng = jax.random.split(rng)

        if self.eval_runner is not None:
            params = runner_state[1].params
            eval_stats = jax.lax.cond(
                evaluate,
                self.eval_runner.run,
                self.eval_runner.fake_run,
                *(subrng, params),
            )
        else:
            eval_stats = {}

        return stats, eval_stats, rng, *runner_state[1:]

    def _make_shmap_run(self, runner_state):
        devices = mesh_utils.create_device_mesh((self.n_devices,))
        mesh = Mesh(devices, axis_names=("device"))

        in_specs, out_specs = self.runner.get_shmap_spec()

        return partial(
            shard_map,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_rep=False,
        )(self.runner.run)

    def train(
        self,
        rng:chex.PRNGKey,
        n_total_updates:int=1000,
        logger:Optional[Logger]=None,
        log_interval:int=1,
        test_interval:int=1,
        checkpoint_interval:int=0,
        archive_interval:int=0,
        archive_init_checkpoint:bool=False,
        from_last_checkpoint:bool=False,
    ):
        """
        Entry-point for training
        """
        # Load checkpoint if any
        runner_state = self.runner.reset(rng)

        if from_last_checkpoint:
            last_checkpoint_state = logger.load_last_checkpoint_state()
            if last_checkpoint_state is not None:
                runner_state = self.runner.load_checkpoint_state(
                    runner_state, last_checkpoint_state
                )
                self._start_tick = runner_state[1].n_iters[0]

        # Archive initialization weights if necessary
        if archive_init_checkpoint:
            logger.checkpoint(
                self.runner.get_checkpoint_state(runner_state),
                index=0,
                archive_interval=1,
            )

        # Train loop
        log_on = logger is not None and log_interval > 0
        checkpoint_on = checkpoint_interval > 0 or archive_interval > 0
        train_state:VmapTrainState = runner_state[1]

        tick = self._start_tick
        train_steps = (
            tick
            * self.runner.step_batch_size
            * self.runner.n_rollout_steps
            * self.n_devices
        )
        real_train_steps = train_steps // self.runner.n_students

        while (train_state.n_updates < n_total_updates).any():
            evaluate = test_interval > 0 and (tick + 1) % test_interval == 0

            start = time.time()
            stats, eval_stats, *runner_state = self.step(runner_state, evaluate)
            end = time.time()

            if evaluate:
                stats.update(eval_stats)
            else:
                stats.update({k: None for k in eval_stats.keys()})

            train_state = runner_state[1]

            dsteps = (
                self.runner.step_batch_size
                * self.runner.n_rollout_steps
                * self.n_devices
            )
            real_dsteps = dsteps // self.runner.n_students
            train_steps += dsteps
            real_train_steps += real_dsteps
            sps = int(dsteps / (end - start))
            real_sps = int(real_dsteps / (end - start))
            stats.update(
                dict(
                    steps=train_steps,
                    sps=sps,
                    real_steps=real_train_steps,
                    real_sps=real_sps,
                )
            )

            tick += 1

            if log_on and tick % log_interval == 0:
                logger.log(stats, tick, ignore_val=-np.inf)

            if checkpoint_on and tick > 0:
                if tick % checkpoint_interval == 0 or (
                    archive_interval > 0 and tick % archive_interval == 0
                ):
                    checkpoint_state = self.runner.get_checkpoint_state(runner_state)
                    logger.checkpoint(
                        checkpoint_state, index=tick, archive_interval=archive_interval
                    )