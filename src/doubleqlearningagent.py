import numpy as np

from agent import BlackJackAgent

class DoubleQLearningBlackJackAgent(BlackJackAgent):
    def get_action(self, env, obs: tuple[int, int, bool]) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return env.action_space.sample()

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        if np.random.random() < 0.5:
            future_q_value = (not terminated) * np.max(self.q_values[next_obs])
            temporal_difference = (reward + self.discount_factor * future_q_value - self.q_values[obs][action])

            self.q_values[obs][action] = (self.q_values[obs][action] + self.lr * temporal_difference)
            self.training_error.append(temporal_difference)
        else:
            future_q_value = (not terminated) * np.min(self.q_values[next_obs])
            temporal_difference = (reward + self.discount_factor * future_q_value - self.q_values[obs][action])

            self.q_values[obs][action] = (self.q_values[obs][action] + self.lr * temporal_difference)
            self.training_error.append(temporal_difference)

