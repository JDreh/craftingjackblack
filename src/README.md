# Crafting Jack Black

## Running

This project uses [uv](https://github.com/astral-sh/uv) to run. To begin, go into the `src` directory.

To train specific agents, run `uv run main.py <agent1> [agent2]...`, where the agent name is one of the following:

- sql: Single Q-Learning

- dql: Double Q-Learning

- dqn: Deep Q-Learning

- ppo: Proximal Policy Optimization

- sarsa: SARSA

To train all of the agents, just run `uv run main.py` (with no extra arguments)

## Outputs

After running the program, there will be a directory called `output`. A directory will be created for each agent that ran, and each one will contain the following items:

- fig1.png shows the moving average episode rewards, episode lengths, and training errors
- fig2.png shows the state and policy values *with* a usable ace
- fig3.png shows the state and policy values *without* a usable ace
- errors.csv
- errors_avg.csv
- lengths.csv
- lengths_avg.csv
- rewards.csv
- rewards_avg.csv

## Commit History

Files and work have been directly committed to the main branch.