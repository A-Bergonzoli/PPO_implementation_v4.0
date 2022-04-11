import numpy as np
import tensorflow as tf
import random
from agent import Agent
from utils.utils import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Ignore general tf warnings


class GlobalConfig:
    def __init__(self):
        self.seed = 555


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


if __name__ == "__main__":
    config = GlobalConfig()
    seed_everything(config.seed)
    args = parse_args()

    # environment setup
    env = make_env(args.env_id, config)

    agent_params = {
        'ENV_ID': args.env_id,
        'EXP_NAME': args.exp_name,

        'NUM_ITERATIONS': args.num_iterations,
        'EPISODE_STEPS': args.num_steps,
        'NUM_TRAIN_EPOCHS': args.train_epochs,
        'MINIBATCH_SIZE': args.minibatch_size,

        'GAMMA': args.gamma,
        'CLIP': args.clip_coef,
        'LAMBDA': args.gae_lambda,
        'VALUE_LOSS_COEF': args.vf_coef,
        'ENTROPY_LOSS_COEF': args.entropy_coef,

        'ACTOR_LR': args.actor_lr,
        'CRITIC_LR': args.critic_lr,
        'ACTOR_HU': args.actor_layers,
        'CRITIC_HU': args.critic_layers,
    }

    #
    agent = Agent(
        params=agent_params,
        env=env,
        obs_dim=env.observation_space.shape[0],
        act_dim=env.action_space.shape[0],
        score_thresh=-10,
        is_evaluate=False,
        plot_interval=args.plot_interval,
        train_history_path='unused'
    )

    if args.mode == "train":
        agent.train()
    elif args.mode == "test":
        agent.test()
