import argparse
import os,time
import sys

import gym
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces

import pfrl
from pfrl import experiments, explorers
from pfrl import nn as pnn
from pfrl import q_functions, replay_buffers, utils
from pfrl.agents.dqn import DQN
from pfrl.nn.mlp import MLP
from pfrl.utils.contexts import evaluating



def main(sys_args=None,log_file=None):
    import logging

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        default="results",
        help=(
            "Directory path to save output files."
            " If it does not exist, it will be created."
        ),
    )
    parser.add_argument("--env", type=str, default="CartPole-v0")
    parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 32)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--final-exploration-steps", type=int, default=10 ** 5)
    parser.add_argument("--start-epsilon", type=float, default=1.0)
    parser.add_argument("--end-epsilon", type=float, default=0.02)
    parser.add_argument("--noisy-net-sigma", type=float, default=None)
    parser.add_argument("--demo", action="store_true", default=False)
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--steps", type=int, default=10 ** 5)
    parser.add_argument("--prioritized-replay", action="store_true")
    parser.add_argument("--replay-start-size", type=int, default=1000)
    parser.add_argument("--target-update-interval", type=int, default=10 ** 2)
    parser.add_argument("--target-update-method", type=str, default="hard")
    parser.add_argument("--soft-update-tau", type=float, default=1e-2)
    parser.add_argument("--update-interval", type=int, default=1)
    parser.add_argument("--eval-n-runs", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=10 ** 5)
    parser.add_argument("--checkpoint-freq", type=int, default=10 ** 5)
    parser.add_argument("--n-hidden-channels", type=int, default=100)
    parser.add_argument("--n-hidden-layers", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--minibatch-size", type=int, default=None)
    parser.add_argument("--render-train", action="store_true")
    parser.add_argument("--render-eval", action="store_true")
    parser.add_argument("--monitor", action="store_true")
    parser.add_argument("--reward-scale-factor", type=float, default=1.0)
    parser.add_argument(
        "--actor-learner",
        action="store_true",
        help="Enable asynchronous sampling with asynchronous actor(s)",
    )  # NOQA
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help=(
            "The number of environments for sampling (only effective with"
            " --actor-learner enabled)"
        ),
    )  # NOQA
    parser.add_argument(
        "--batch-accumulator", type=str, default="mean", choices=["mean", "sum"]
    )
    parser.add_argument("--quantile-thresholds-N", type=int, default=64)
    parser.add_argument("--quantile-thresholds-N-prime", type=int, default=64)
    parser.add_argument("--quantile-thresholds-K", type=int, default=32)
    parser.add_argument("--cvar-alpha", type=float, default=1.0)
    parser.add_argument("--cvar-static", action="store_true", default=False)
    parser.add_argument("--observation-noise", action="store_true", default=False)
    args = parser.parse_args(sys_args)

    # Set a random seed used in PFRL
    utils.set_random_seed(args.seed)

    if not args.demo:
        args.outdir = experiments.prepare_output_dir(args, args.outdir, argv=sys.argv)
        print("Output files are saved in {}".format(args.outdir))
        
        if log_file is None:
            logging.basicConfig(level=logging.INFO)
        else:
            logging.basicConfig(level=logging.INFO,filename=args.outdir+'/'+log_file)
            

    # Set different random seeds for different subprocesses.
    # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
    # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
    process_seeds = np.arange(args.num_envs) + args.seed * args.num_envs
    assert process_seeds.max() < 2 ** 32

    def clip_action_filter(a):
        return np.clip(a, action_space.low, action_space.high)

    def make_env(idx=0, test=False):
        env = gym.make(args.env)
        # Use different random seeds for train and test envs
        process_seed = int(process_seeds[idx])
        env_seed = 2 ** 32 - 1 - process_seed if test else process_seed
        utils.set_random_seed(env_seed)
        # Cast observations to float32 because our model uses float32
        if args.observation_noise: ##lunar only
            env = pfrl.wrappers.CastObservationToFloat32(env,np.array([0.05,0.05,0,0,0,0,0,0]))
        else:
            env = pfrl.wrappers.CastObservationToFloat32(env)
        if args.monitor:
            env = pfrl.wrappers.Monitor(env, args.outdir)
        if isinstance(env.action_space, spaces.Box):
            utils.env_modifiers.make_action_filtered(env, clip_action_filter)
        if True: #not test:
            # Scale rewards (and thus returns) to a reasonable range so that
            # training is easier
            env = pfrl.wrappers.ScaleReward(env, args.reward_scale_factor)
        if (args.render_eval and test) or (args.render_train and not test):
            env = pfrl.wrappers.Render(env)
        return env

    env = make_env(test=False)
    timestep_limit = env.spec.max_episode_steps
    obs_space = env.observation_space
    obs_size = obs_space.low.size
    action_space = env.action_space

    if isinstance(action_space, spaces.Box):
        action_size = action_space.low.size
        # Use NAF to apply DQN to continuous action spaces
        q_func = q_functions.FCQuadraticStateQFunction(
            obs_size,
            action_size,
            n_hidden_channels=args.n_hidden_channels,
            n_hidden_layers=args.n_hidden_layers,
            action_space=action_space,
        )
        # Use the Ornstein-Uhlenbeck process for exploration
        ou_sigma = (action_space.high - action_space.low) * 0.2
        explorer = explorers.AdditiveOU(sigma=ou_sigma)
    else:
        n_actions = action_space.n
        q_func = pfrl.agents.iqn.ImplicitQuantileQFunction(
            psi=MLP(
                in_size=obs_size,
                out_size=args.n_hidden_channels,
                hidden_sizes=[args.n_hidden_channels] * args.n_hidden_layers,
                nonlinearity=F.relu,last_wscale=1.0
            ),
            phi=nn.Sequential(
                pfrl.agents.iqn.CosineBasisLinear(64, args.n_hidden_channels),
                nn.ReLU(),
            ),
            f=nn.Sequential(
                nn.Linear(args.n_hidden_channels, args.n_hidden_channels),
                nn.ReLU(),
                nn.Linear(args.n_hidden_channels, n_actions),
            ),
        )
        
        # Use epsilon-greedy for exploration
        explorer = explorers.LinearDecayEpsilonGreedy(
            args.start_epsilon,
            args.end_epsilon,
            args.final_exploration_steps,
            action_space.sample,
        )

    if args.noisy_net_sigma is not None:
        pnn.to_factorized_noisy(q_func, sigma_scale=args.noisy_net_sigma)
        # Turn off explorer
        explorer = explorers.Greedy()

    opt = optim.Adam(q_func.parameters(),lr=args.learning_rate)

    rbuf_capacity = 5 * 10 ** 5
    if args.minibatch_size is None:
        args.minibatch_size = 32
    if args.prioritized_replay:
        betasteps = (args.steps - args.replay_start_size) // args.update_interval
        rbuf = replay_buffers.PrioritizedReplayBuffer(
            rbuf_capacity, betasteps=betasteps
        )
    else:
        rbuf = replay_buffers.ReplayBuffer(rbuf_capacity)

    agent = pfrl.agents.IQN(
        q_func,
        opt,
        rbuf,
        gpu=args.gpu,
        gamma=args.gamma,
        explorer=explorer,
        replay_start_size=args.replay_start_size,
        target_update_interval=args.target_update_interval,
        update_interval=args.update_interval,
        minibatch_size=args.minibatch_size,
        target_update_method=args.target_update_method,
        soft_update_tau=args.soft_update_tau,
        batch_accumulator=args.batch_accumulator,
        quantile_thresholds_N=args.quantile_thresholds_N,
        quantile_thresholds_N_prime=args.quantile_thresholds_N_prime,
        quantile_thresholds_K=args.quantile_thresholds_K,
        cvar_alpha = args.cvar_alpha,
        cvar_static = args.cvar_static,
    )

    if args.load:
        agent.load(args.load)

    eval_env = make_env(test=True)

    if args.demo:
        eval_stats = experiments.eval_performance(
            env=eval_env,
            agent=agent,
            n_steps=None,
            n_episodes=args.eval_n_runs,
            max_episode_len=timestep_limit,
        )
        print(
            "n_runs: {} mean: {} median: {} stdev {}".format(
                args.eval_n_runs,
                eval_stats["mean"],
                eval_stats["median"],
                eval_stats["stdev"],
            )
        )
        return agent,eval_stats['scores'],eval_stats['scores2']

    elif not args.actor_learner:

        # print(
        #     "WARNING: Since https://github.com/pfnet/pfrl/pull/112 we have started"
        #     " setting `eval_during_episode=True` in this script, which affects the"
        #     " timings of evaluation phases."
        # )

        experiments.train_agent_with_evaluation(
            agent=agent,
            env=env,
            steps=args.steps,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs,
            eval_interval=args.eval_interval,
            outdir=args.outdir,
            checkpoint_freq=args.checkpoint_freq,
            eval_env=eval_env,
            train_max_episode_len=timestep_limit,
            eval_during_episode=False, #True,
        )
    else:
        # using impala mode when given num of envs

        # When we use multiple envs, it is critical to ensure each env
        # can occupy a CPU core to get the best performance.
        # Therefore, we need to prevent potential CPU over-provision caused by
        # multi-threading in Openmp and Numpy.
        # Disable the multi-threading on Openmp and Numpy.
        os.environ["OMP_NUM_THREADS"] = "1"  # NOQA

        (
            make_actor,
            learner,
            poller,
            exception_event,
        ) = agent.setup_actor_learner_training(args.num_envs)

        poller.start()
        learner.start()

        experiments.train_agent_async(
            processes=args.num_envs,
            make_agent=make_actor,
            make_env=make_env,
            steps=args.steps,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs,
            eval_interval=args.eval_interval,
            outdir=args.outdir,
            stop_event=learner.stop_event,
            exception_event=exception_event,
        )

        poller.stop()
        learner.stop()
        poller.join()
        learner.join()

    return args.outdir



if __name__ == "__main__":
    res_path='results'

    env_name=sys.argv[1]
    eval_seed=int(sys.argv[2])
    eval_alpha='{:.2f}'.format(float(sys.argv[3]))
    eval_static=bool(int(sys.argv[4])) #0 or 1

    n_eval='100'
    steps=1000000

    for path in os.listdir(res_path):
        outdir='{}/{}'.format(res_path,path)
        with open(outdir+'/command.txt','r') as fp:
            lines=fp.readlines()
            parts=lines[0].split(' ')
            seed=int(parts[[i for i,p in enumerate(parts) if p=='--seed'][-1]+1])
            env=parts[[i for i,p in enumerate(parts) if p=='--env'][-1]+1]
            cvar_alpha='{:.2f}'.format(float(parts[[i for i,p in enumerate(parts) if p=='--cvar-alpha'][-1]+1]))
            STATIC_IDX = [i for i,p in enumerate(parts) if p=='--cvar-static']
            cvar_static=len(STATIC_IDX)>0
            
            if env.startswith(env_name) and seed==eval_seed and cvar_alpha==eval_alpha and cvar_static==eval_static:
                dir_name='{}/{}_finish'.format(outdir,steps)
                args=['--seed','{}'.format(seed+steps),'--env',env,'--observation-noise',
                      '--cvar-alpha',cvar_alpha,
                      '--demo','--n-best-episodes',n_eval,'--load',dir_name
                      ]
                if cvar_static:
                    args.append('--cvar-static')
                agent,s,s2=main(args)
                s2=np.sort(s2)

                import matplotlib.pyplot as plt
                plt.ion()
                plt.figure()
                plt.clf()
                plt.hist(s2,20,edgecolor='k')
                break
