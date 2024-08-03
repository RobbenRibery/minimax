"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import sys
sys.path.append('/teamspace/studios/this_studio/minimax/src/minimax')

import os
import copy
from pprint import pprint 

# os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=2'
# os.environ['JAX_TRACEBACK_FILTERING'] = 'off'
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".70"

import jax
import wandb

from minimax.runners import ExperimentRunner
from minimax.util.loggers import Logger
from .arguments import parser


if __name__ == "__main__":
    with jax.disable_jit(False):
        args = parser.parse_args(preview=True)
        print(args.env_args)

        # === Setup the main runner ===
        _args = copy.deepcopy(args)  # Mutable record of args
        print(_args.env_args)
        xp_runner = ExperimentRunner(
            train_runner=_args.train_runner,
            env_name=_args.env_name,
            agent_rl_algo=_args.agent_rl_algo,
            student_model_name=_args.student_model_name,
            teacher_model_name=_args.teacher_model_name,
            train_runner_kwargs=_args.train_runner_args,
            env_kwargs=_args.env_args,
            ued_env_kwargs=_args.ued_env_args,
            student_rl_kwargs=_args.student_rl_args,
            teacher_rl_kwargs=_args.teacher_rl_args,
            student_model_kwargs=_args.student_model_args,
            teacher_model_kwargs=_args.teacher_model_args,
            eval_kwargs=_args.eval_args,
            eval_env_kwargs=_args.eval_env_args,
            n_devices=_args.n_devices,
        )

        # === Configure logging ===
        # Set up wandb
        wandb_args = args.wandb_args
        print(f"#### WandB args \n")
        pprint(wandb_args)
        # if wandb_args.base_url:
        #     #os.environ["WANDB_BASE_URL"] = wandb_args.base_url
        #     pass
        # if wandb_args.api_key:
        #     #os.environ["WANDB_API_KEY"] = wandb_args.api_key
        #     pass
        if wandb_args.project is not None:
            #if wandb_args.base_url or wandb_args.api_key:
            os.environ["WANDB_CACHE_DIR"] = "~/.cache/wandb"
            wandb.init(
                project=wandb_args.project,
                entity=None,
                config=args,
                name=args.xpid,
                group=wandb_args.group,
            )
            callback = wandb.log 
        else:
            callback = None

        logger = Logger(
            log_dir=args.log_dir,
            xpid=args.xpid,
            xp_args=args,
            callback=callback,
            from_last_checkpoint=args.from_last_checkpoint,
            verbose=args.verbose,
        )

        # === Start training ===
        rng = jax.random.PRNGKey(args.seed)
        xp_runner.train(
            rng=rng,
            n_total_updates=args.n_total_updates,
            logger=logger,
            log_interval=args.log_interval,
            test_interval=args.test_interval,
            checkpoint_interval=args.checkpoint_interval,
            archive_interval=args.archive_interval,
            archive_init_checkpoint=args.archive_init_checkpoint,
            from_last_checkpoint=args.from_last_checkpoint,
        )
