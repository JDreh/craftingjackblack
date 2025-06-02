# sarsaagent.py

from agent import BlackJackAgent
import numpy as np


class SARSABlackJackAgent(BlackJackAgent):
    def __init__(
        self,
        env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):

        super().__init__(
            env=env,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
        )

    def get_action(self, env, obs: tuple[int, int, bool]) -> int:
        # with probability epsilon return a random action to explore the environment
        if np.random.rand() < self.epsilon:
            return env.action_space.sample()
        # with probability (1 - epsilon) act greedily (exploit)
        return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
        next_action: int,
    ):

        current_q = self.q_values[obs][action]
        if terminated:
            target = reward
        else:
            target = reward + self.discount_factor * self.q_values[next_obs][next_action]

        td_error = target - current_q
        self.q_values[obs][action] += self.lr * td_error
        self.training_error.append(abs(td_error))
