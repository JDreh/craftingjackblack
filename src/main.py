from enum import Enum
import sys
import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from collections import defaultdict
import numpy as np
import seaborn as sns

from qlearningagent import QLearningBlackJackAgent
from deepqlearningagent import DeepQLearningBlackJackAgent
from ppoagent import PPOBlackJackAgent
from doubleqlearningagent import DoubleQLearningBlackJackAgent
from sarsaagent import SARSAAgent

class Model(Enum):
    PPO = "ppo"
    DQN = "dqn"
    SQL = "sql"
    DQL = "dql"
    SARSA = "sarsa"

def main():
    env = gym.make("Blackjack-v1", sab=True)
    done = False

    # info dictionary is useless
    obs, _ = env.reset()
    print(obs)

    learning_rate = 0.01
    n_episodes = 10000
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (n_episodes / 2)
    final_epsilon = 0.1

    model_used = Model.PPO if len(sys.argv) < 2 else Model(sys.argv[1])

    match model_used:
        case Model.PPO:
            learning_rate = 0.0003  # Lower learning rate for PPO
            n_episodes = 100000  # Increase episodes for PPO
            agent = PPOBlackJackAgent(
                env=env,
                learning_rate=learning_rate,
                discount_factor=0.99,
                initial_epsilon=start_epsilon,
                epsilon_decay=epsilon_decay,
                final_epsilon=final_epsilon,
                batch_size=8
            )
        case Model.DQN:
            agent = DeepQLearningBlackJackAgent(
                env=env,
                learning_rate=learning_rate,
                discount_factor=0.99,
                initial_epsilon=start_epsilon,
                epsilon_decay=epsilon_decay,
                final_epsilon=final_epsilon
            )
        case Model.DQL:
                agent = DoubleQLearningBlackJackAgent(
                env=env,
                learning_rate=learning_rate,
                initial_epsilon=start_epsilon,
                epsilon_decay=epsilon_decay,
                final_epsilon=final_epsilon,
            )
        case Model.SQL:
                agent = QLearningBlackJackAgent(
                env=env,
                learning_rate=learning_rate,
                initial_epsilon=start_epsilon,
                epsilon_decay=epsilon_decay,
                final_epsilon=final_epsilon,
            )
        case Model.SARSA:
            agent = SARSAAgent(
                env=env,
                learning_rate=learning_rate,
                initial_epsilon=start_epsilon,
                epsilon_decay=epsilon_decay,
                final_epsilon=final_epsilon,
                discount_factor=0.95,
            )

    print(f"training with agent {model_used.value}")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    episode_rewards = []
    for episode in tqdm(range(n_episodes)):
        obs, info = env.reset()
        done = False
        total_reward = 0

        # pick the first action from state before entering the loop
        if isinstance(agent, SARSAAgent):
            action = agent.get_action(env, obs)

            while not done:
                next_obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward

                done = terminated or truncated

                # choose next_action from next_obs (even if it turns out to be terminal)
                next_action = agent.get_action(env, next_obs) if not done else None

                # update agent (pass next_action; if done, next_action=None)
                agent.update(obs, action, reward, terminated, next_obs, next_action)

                # move to next state‐action pair
                obs = next_obs
                action = next_action

            agent.decay_epsilon()
            episode_rewards.append(total_reward)
        else:

            # play one episode
            while not done:
                action = agent.get_action(env, obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)

                # update the agent
                agent.update(obs, action, reward, terminated, next_obs)

                done = terminated or truncated
                obs = next_obs
                total_reward += reward

            agent.decay_epsilon()
            episode_rewards.append(total_reward)

    rolling_length = 500
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
    axs[0].set_title("Episode rewards")
    # compute and assign a rolling average of the data to provide a smoother graph
    reward_moving_average = (
        np.convolve(
            np.array(episode_rewards), np.ones(rolling_length), mode="valid"
        )
        / rolling_length
    )
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
    axs[1].set_title("Episode lengths")
    length_moving_average = (
        np.convolve(
            np.array(env.length_queue).flatten(), np.ones(rolling_length), mode="same"
        )
        / rolling_length
    )
    axs[1].plot(range(len(length_moving_average)), length_moving_average)
    axs[2].set_title("Training Error")
    training_error_moving_average = (
        np.convolve(np.array(agent.training_error), np.ones(rolling_length), mode="same")
        / rolling_length
    )
    axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
    plt.tight_layout()
    plt.show()
    # state values & policy with usable ace (ace counts as 11)
    value_grid, policy_grid = create_grids(agent, usable_ace=True)
    fig1 = create_plots(value_grid, policy_grid, title="With usable ace")
    plt.show()

    env.close()



def create_grids(agent, usable_ace=False):
    """Create value and policy grid given an agent."""
    # convert our state-action values to state values
    # and build a policy dictionary that maps observations to actions
    state_value = defaultdict(float)
    policy = defaultdict(int)
    for obs, action_values in agent.q_values.items():
        state_value[obs] = float(np.max(action_values))
        policy[obs] = int(np.argmax(action_values))

    player_count, dealer_count = np.meshgrid(
        # players count, dealers face-up card
        np.arange(12, 22),
        np.arange(1, 11),
    )

    # create the value grid for plotting
    value = np.apply_along_axis(
        lambda obs: state_value[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    value_grid = player_count, dealer_count, value

    # create the policy grid for plotting
    policy_grid = np.apply_along_axis(
        lambda obs: policy[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    return value_grid, policy_grid


def create_plots(value_grid, policy_grid, title: str):
    """Creates a plot using a value and policy grid."""
    # create a new figure with 2 subplots (left: state values, right: policy)
    player_count, dealer_count, value = value_grid
    fig = plt.figure(figsize=plt.figaspect(0.4))
    fig.suptitle(title, fontsize=16)

    # plot the state values
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(
        player_count,
        dealer_count,
        value,
        rstride=1,
        cstride=1,
        cmap="viridis",
        edgecolor="none",
    )
    plt.xticks(range(12, 22), range(12, 22))
    plt.yticks(range(1, 11), ["A"] + list(range(2, 11)))
    ax1.set_title(f"State values: {title}")
    ax1.set_xlabel("Player sum")
    ax1.set_ylabel("Dealer showing")
    ax1.zaxis.set_rotate_label(False)
    ax1.set_zlabel("Value", fontsize=14, rotation=90)
    ax1.view_init(20, 220)

    # plot the policy
    fig.add_subplot(1, 2, 2)
    ax2 = sns.heatmap(policy_grid, linewidth=0, annot=True, cmap="Accent_r", cbar=False)
    ax2.set_title(f"Policy: {title}")
    ax2.set_xlabel("Player sum")
    ax2.set_ylabel("Dealer showing")
    ax2.set_xticklabels(range(12, 22))
    ax2.set_yticklabels(["A"] + list(range(2, 11)), fontsize=12)

    # add a legend
    legend_elements = [
        Patch(facecolor="lightgreen", edgecolor="black", label="Hit"),
        Patch(facecolor="grey", edgecolor="black", label="Stick"),
    ]
    ax2.legend(handles=legend_elements, bbox_to_anchor=(1.3, 1))
    return fig




if __name__ == "__main__":
    main()
