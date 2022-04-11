import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from IPython.display import clear_output
from networks import Actor, Critic
from ppo_memory import Memory


class Agent(object):

    def __init__(self,
                 params: dict,
                 env,
                 obs_dim,
                 act_dim,
                 score_thresh: float,
                 is_evaluate: bool,
                 plot_interval: int,
                 train_history_path: str):
        self.params = params
        self.env = env
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # agent neural networks
        self.actor = Actor(
            input_dim=self.obs_dim,
            output_dim=self.act_dim,
            hidden_units=self.params["ACTOR_HU"]
        )
        self.critic = Critic(
            input_dim=self.obs_dim,
            hidden_units=self.params["CRITIC_HU"]
        )

        # neural networks optimizers
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.params["ACTOR_LR"], epsilon=1e-7)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.params["CRITIC_LR"], epsilon=1e-7)

        # memory buffer
        self.agent_mem = Memory()

        # logging stuff
        self.actor_loss_history = []
        self.critic_loss_history = []
        self.score_history = []
        self.plot_interval = plot_interval
        self.train_history_path = train_history_path

        self.is_evaluate = is_evaluate
        self.score_threshold = score_thresh

    def _choose_action(self, state: np.ndarray) -> float:
        """Choose action based on observation. If currently not evaluating, store step info in buffer
        :param state: observation returned from the environment
        :return: action
        """
        state = tf.convert_to_tensor(state, tf.float32)
        action, distribution = self.actor(state)

        if not self.is_evaluate:
            value = self.critic(state)

            # store data of current time step of the episode
            self.agent_mem.states.append(state)
            self.agent_mem.actions.append(action)
            self.agent_mem.log_probs.append(distribution.log_prob(action))
            self.agent_mem.values.append(value)

        return list(action.numpy()).pop()

    def _step(self, action: float):
        nxt_state, reward, done, info = self.env.step(action)

        # add fake dim to match dimensions with batch size
        nxt_state = np.reshape(nxt_state, (1, -1)).astype(np.float64)
        reward = np.reshape(reward, (1, -1)).astype(np.float64)
        done = np.reshape(done, (1, -1))

        if not self.is_evaluate:
            # convert ndarray returned by step to tf.Tensors and
            # store data of current time step of the episode
            self.agent_mem.rewards.append(tf.convert_to_tensor(reward, dtype=tf.float32))
            self.agent_mem.dones.append(tf.convert_to_tensor((1 - done), dtype=tf.float32))

        return nxt_state, reward, done, info

    def train(self):

        score = 0
        state = self.env.reset()
        state = np.reshape(state, (1, -1))

        for update in range(self.params["NUM_ITERATIONS"]):

            start_time = time.time()

            # perform rollout
            for _ in range(self.params["EPISODE_STEPS"]):
                action = self._choose_action(state)
                next_state, reward, done, info = self._step(action)

                state = next_state
                score += reward[0][0]

                if done[0][0]:
                    self.score_history.append(score)
                    score = 0
                    state = self.env.reset()
                    state = np.reshape(state, (1, -1))

            # update plot
            if update % self.plot_interval == 0:
                self._plot_train_history()

            # check if model is good enough to stop training
            if np.mean(self.score_history[-self.plot_interval:]) > self.score_threshold:
                print(f"Solved before {self.params['NUM_ITERATIONS']}!")
                break

            # actually train the agent
            value = self.critic(tf.convert_to_tensor(next_state, dtype=tf.float32))
            self.agent_mem.values.append(value)
            self._update_agent()

            end_time = time.time()

            # print some useful infos
            print(f"Update {update}, Total timesteps completed (iteration*T) {self.params['EPISODE_STEPS']}\n \
                  Last episode score: {self.score_history[update]:.3f}\n \
                  Last {self.plot_interval} score average: {np.mean(self.score_history[-self.plot_interval:]):.3f}\n \
                  elapsed update time {end_time - start_time:.2f} seconds\n")

        self._save_train_history()
        self.env.close()

    def _update_agent(self):

        actor_losses, critic_losses = [], []
        returns = self._get_gae(
            self.agent_mem.rewards,
            self.agent_mem.values,
            self.agent_mem.dones
        )

        # flatten a list of tf.tensors into vectors
        states = tf.reshape(tf.concat(self.agent_mem.states, axis=0), shape=(-1, self.obs_dim)).numpy()
        actions = tf.concat(self.agent_mem.actions, axis=0).numpy()
        returns = tf.concat(returns, axis=0).numpy()
        log_probs = tf.concat(self.agent_mem.log_probs, axis=0).numpy()
        values = tf.concat(self.agent_mem.values, axis=0).numpy()
        advantages = returns - values[:-1]

        # loop through a minibatch and perform gradient computations
        for state, action, return_, old_log_prob, old_value, advantage in self._batch_generator(
                states,
                actions,
                returns,
                log_probs,
                values,
                advantages
        ):
            with tf.GradientTape() as tape1:
                # compute policy ratio
                _, distribution = self.actor(state)
                current_log_prob = distribution.log_prob(action)
                ratio = tf.exp(current_log_prob - old_log_prob)
                clipped_ratio = tf.clip_by_value(ratio, 1 - self.params["CLIP"], 1 + self.params["CLIP"])

                # entropy
                entropy = tf.reduce_mean(distribution.entropy())

                # compute actor loss
                surrogate = ratio * advantage
                surrogate_ = clipped_ratio * advantage
                actor_loss = - tf.reduce_mean(tf.minimum(surrogate, surrogate_)) \
                             - self.params["ENTROPY_LOSS_COEF"] * entropy

                actor_losses.append(actor_loss.numpy())    # numpy() because actor_loss is a tf.Tensor

            actor_gradients = tape1.gradient(actor_loss, self.actor.trainable_weights)
            self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_weights))

            with tf.GradientTape() as tape2:
                # compute critic loss
                current_value = self.critic(state)
                critic_loss = tf.reduce_mean(tf.pow(return_ - current_value, 2))

                critic_losses.append(critic_loss.numpy())  # numpy() because tf.critic_loss is a tf.Tensor

            critic_gradients = tape2.gradient(critic_loss, self.critic.trainable_weights)
            self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_weights))

        # clean memory of the episode
        self.agent_mem.clear_mem()

        # update logs
        self.actor_loss_history.append(sum(actor_losses) / len(actor_losses))
        self.critic_loss_history.append(sum(critic_losses) / len(critic_losses))

    def test(self):
        self.is_evaluate = True

        # load model weights
        self.actor.load_weights(self.params["ENV_ID"] + '/' + self.params["EXP_NAME"] + "/actor")
        self.critic.load_weights(self.params["ENV_ID"] + '/' + self.params["EXP_NAME"] + "/critic")

        ### ROLLOUT ###
        for _ in range(5):  # repeat rollout for 5 times
            state = self.env.reset()
            state = np.reshape(state, (1, -1))
            done = False
            while not done:
                self.env.render()
                action = self._choose_action(state)
                next_state, _, done, info = self.env.step(action)

                state = next_state
                state = np.reshape(state, (1, -1))

        self.env.close()

    def _plot_train_history(self):
        data = [
            self.score_history,
            self.actor_loss_history,
            self.critic_loss_history]
        labels = [
            f"score: {np.mean(self.score_history[-self.plot_interval:]):.3f}",
            f"actor loss: {np.mean(self.actor_loss_history[-self.plot_interval:]):.4f}",
            f"critic loss: {np.mean(self.critic_loss_history[-self.plot_interval:]):.4f}"
        ]

        clear_output(True)
        with plt.style.context("seaborn-dark-palette"):
            fig, axes = plt.subplots(3, 1, figsize=(6, 8))
            for i, ax in enumerate(axes):
                ax.plot(data[i], c="crimson")
                ax.set_title(labels[i])

            plt.tight_layout()
            plt.show()

    def _save_train_history(self):
        # save actor and critic weights
        self.actor.save_weights(self.params["ENV_ID"] + '/' + self.params["EXP_NAME"] + "/actor")
        self.critic.save_weights(self.params["ENV_ID"] + '/' + self.params["EXP_NAME"] + "/critic")

        # save losses and score histories as csv files
        pd.DataFrame(
            {"actor loss": self.actor_loss_history,
             "critic_loss": self.critic_loss_history}
        ).to_csv("loss_logs.csv")
        pd.DataFrame(
            {"scores": self.score_history}
        ).to_csv("score_logs.csv")

        # save run configuration into a json file
        run_config = {
            "environment id": self.params["ENV_ID"],
            "experiment name": self.params["EXP_NAME"],
            "actor_hidden_layers": self.params["ACTOR_HU"],
            "critic hidden layers": self.params["CRITIC_HU"],
            "actor learning rate": self.params["ACTOR_LR"],
            "critic learning rate": self.params["CRITIC_LR"],
            "num iterations": self.params["NUM_ITERATIONS"],
            "duration of an episode": self.params["EPISODE_STEPS"],
            "num train epochs": self.params["NUM_TRAIN_EPOCHS"],
            "minibatch size": self.params["MINIBATCH_SIZE"],
            "discount factor": self.params["GAMMA"],
            "clip parameter": self.params["CLIP"],
            "value loss coefficient": self.params["VALUE_LOSS_COEF"],
            "entropy loss coefficient": self.params["ENTROPY_LOSS_COEF"],
            "gae lambda": self.params["LAMBDA"],
        }
        j = json.dumps(run_config)
        with open("run_config.json", 'w') as f:
            f.write(j)

    def _get_gae(self, rewards, values, dones):
        """Computes a list of estimators of the advantage function at each timestep of the episode
        :param rewards: list of rewards
        :param values: list of value function values
        :param dones: list of terminal state flags
        :return: discounted sum of rewards
        """
        gae = 0
        returns = []
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.params["GAMMA"] * values[i + 1] * dones[i] - values[i]
            gae = delta + self.params["GAMMA"] * self.params["LAMBDA"] * dones[i] * gae
            returns.insert(0, gae + values[i])  # we are traversing in reversed order (we keep overwriting)

        return returns

    def _batch_generator(self,
                         states: tf.Tensor,
                         actions: tf.Tensor,
                         returns: tf.Tensor,
                         log_probs: tf.Tensor,
                         values: tf.Tensor,
                         advantages: tf.Tensor):
        """Generates batch for the training of the agent. Note that the memory must be sampled randomly
        """
        ##data_len = states.get_shape().as_list()[0]
        # data_len = tf.shape(states)[0]
        data_len = 4000
        for _ in range(self.params["NUM_TRAIN_EPOCHS"]):
            for _ in range(data_len // self.params["MINIBATCH_SIZE"]):
                idxs = np.random.choice(data_len, self.params["MINIBATCH_SIZE"])
                yield states[idxs, :], actions[idxs], returns[idxs], log_probs[idxs], values[idxs], advantages[idxs]

    # TODO implement _get_succes_rate
