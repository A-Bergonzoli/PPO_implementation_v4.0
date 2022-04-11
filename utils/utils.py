import os
import argparse

import gym
import matplotlib.pyplot as plt
from env_wrappers import PendulumActionNormalizer

from distutils.util import strtobool


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment (default=1)")
    parser.add_argument("--mode", type=str, default='train', choices={"train", "test"},
                        help="run mode: train a model from scratch, training sess or test a model")
    parser.add_argument("--quantized", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        metavar='',
                        help="whether to quantize the network's hidden layers (default false)")
    parser.add_argument("--qkeras", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, metavar='',
                        help="whether to use the QKeras API for quantization (default false)")

    # Algorithm specific arguments
    parser.add_argument("--env_id", type=str, default="Myturtlebot-v0", metavar='',
                        help="the id of the environment")
    parser.add_argument("--num_iterations", type=int, default=2000, metavar='',
                        help="max number of iterations of rollout and training (default 2000)")
    parser.add_argument("--score_th", type=float, default=100, metavar='',
                        help="average score threshold above which we check if the model is sufficiently good")
    parser.add_argument("--plot_interval", type=int, default=10, metavar='',
                        help="interval with which to update the plot (default 10)")
    parser.add_argument("--actor_lr", type=float, default=1e-4, metavar='',
                        help="the learning rate of the optimizer (default 0.0001)")
    parser.add_argument("--critic_lr", type=float, default=5e-4, metavar='',
                        help="the learning rate of the optimizer (default 0.0005)")
    parser.add_argument("--actor_layers", type=list, default=[32, 32], metavar='',
                        help="list of neurons of the actor hidden layers (default = [32, 32])")
    parser.add_argument("--critic_layers", type=list, default=[64, 64], metavar='',
                        help="list of neurons of the critic hidden layers (default = [64, 64])")

    parser.add_argument("--epsilon", type=float, default=1e-7, metavar='',
                        help="Value of epsilon for the optimizer, used to avoid divide by 0 errors when gradient is small (default 1e-7)")
    parser.add_argument("--num_steps", type=int, default=500, metavar='',
                        help="the number of steps to run in each environment per policy rollout (default 500)")
    parser.add_argument("--gamma", type=float, default=0.99, metavar='',
                        help="the discount factor gamma (default 0.99)")
    parser.add_argument("--gae_lambda", type=float, default=0.95, metavar='',
                        help="the lambda for the general advantage estimation (default 0.95)")
    parser.add_argument("--minibatch_size", type=int, default=100, metavar='',
                        help="the size of mini-batches (default 100)")
    parser.add_argument("--train_epochs", type=int, default=10, metavar='',
                        help="the K epochs to update the policy (default 10)")
    parser.add_argument("--clip_coef", type=float, default=0.2, metavar='',
                        help="the surrogate clipping coefficient (default 0.2)")
    parser.add_argument("--vf_coef", type=float, default=0.5, metavar='',
                        help="coefficient of the value function (default 0.5)")
    parser.add_argument("--entropy_coef", type=float, default=0.5, metavar='',
                        help="coefficient of the entropy portion of the loss (default 0.5)")

    args = parser.parse_args()

    return args


def make_env(env_id, config):
    env = gym.make(env_id)

    env = PendulumActionNormalizer(env)
    # env = gym.wrappers.ClipAction(env)  # clip the continuous action within the valid bounds
    # env = gym.wrappers.NormalizeObservation(env)
    # env = gym.wrappers.NormalizeReward(env)

    # for repeatability
    env.seed(config.seed)

    return env
