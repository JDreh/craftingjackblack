import matplotlib.pyplot as plt
import pandas as pd
import os

AGENTS = [
    "QLearning",
    "DeepQLearning",
    "DoubleQLearning",
    "SARSA",
    "PPO"
]

COLORS = ["blue", "red", "green", "orange", "purple"]
LINE_STYLES = ["-", "--", "-.", ":", "-"]

# Plotting rewards
plt.figure(figsize=(12, 8))

for agent, color, line_style in zip(AGENTS, COLORS, LINE_STYLES):
    file_path = os.path.join("output", agent, "rewards.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        # Optional smoothing using rolling average
        df['Smoothed Reward'] = df['Episode Reward'].rolling(window=10000).mean()
        df['Cumulative Smoothed Reward'] = df['Smoothed Reward'].cumsum()
        plt.plot(df['Episode'], df['Smoothed Reward'], label=agent, color=color, linestyle=line_style)
    else:
        print(f"Warning: {file_path} not found.")

plt.title("Episode Rewards by Agent", fontsize=16)
plt.xlabel("Episode", fontsize=14)
plt.ylabel("Reward", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=12, frameon=True)
plt.tight_layout()
plt.savefig("./output/Analysis/rewards_plot.png")
plt.show()

# Plotting rewards: rolling average
plt.figure(figsize=(12, 8))
for agent, color, line_style in zip(AGENTS, COLORS, LINE_STYLES):
    file_path = os.path.join("output", agent, "rewards.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df['Smoothed Reward'] = df['Episode Reward'].rolling(window=10000).mean()
        plt.plot(df['Episode'], df['Smoothed Reward'], label=agent, color=color, linestyle=line_style)
    else:
        print(f"Warning: {file_path} not found.")
plt.title("Episode Rewards (Rolling Average) by Agent", fontsize=16)
plt.xlabel("Episode", fontsize=14)
plt.ylabel("Reward (Rolling Avg)", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=12, frameon=True)
plt.tight_layout()
plt.savefig("./output/Analysis/rewards_rolling_plot.png")
plt.show()

# Plotting rewards: cumulative
plt.figure(figsize=(12, 8))
for agent, color, line_style in zip(AGENTS, COLORS, LINE_STYLES):
    file_path = os.path.join("output", agent, "rewards.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df['Smoothed Reward'] = df['Episode Reward'].rolling(window=10000).mean()
        df['Cumulative Smoothed Reward'] = df['Smoothed Reward'].cumsum()
        plt.plot(df['Episode'], df['Cumulative Smoothed Reward'], label=agent, color=color, linestyle=line_style)
    else:
        print(f"Warning: {file_path} not found.")
plt.title("Cumulative Episode Rewards by Agent", fontsize=16)
plt.xlabel("Episode", fontsize=14)
plt.ylabel("Cumulative Reward", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=12, frameon=True)
plt.tight_layout()
plt.savefig("./output/Analysis/rewards_cumulative_plot.png")
plt.show()

# Plotting training errors
plt.figure(figsize=(12, 8))

for agent, color, line_style in zip(AGENTS, COLORS, LINE_STYLES):
    file_path = os.path.join("output", agent, "errors.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        # Optional smoothing using rolling average
        df['Smoothed Error'] = df['Training Error'].rolling(window=10000).mean()
        df['Cumulative Smoothed Error'] = df['Smoothed Error'].cumsum()
        plt.plot(df['Episode'], df['Smoothed Error'], label=agent, color=color, linestyle=line_style)
    else:
        print(f"Warning: {file_path} not found.")

plt.title("Training Errors by Agent", fontsize=16)
plt.xlabel("Episode", fontsize=14)
plt.ylabel("Training Error", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=12, frameon=True)
plt.tight_layout()
plt.savefig("./output/Analysis/training_errors_plot.png")
plt.show()

# Plotting training errors: rolling average
plt.figure(figsize=(12, 8))
for agent, color, line_style in zip(AGENTS, COLORS, LINE_STYLES):
    file_path = os.path.join("output", agent, "errors.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df['Smoothed Error'] = df['Training Error'].rolling(window=10000).mean()
        plt.plot(df['Episode'], df['Smoothed Error'], label=agent, color=color, linestyle=line_style)
    else:
        print(f"Warning: {file_path} not found.")
plt.title("Training Errors (Rolling Average) by Agent", fontsize=16)
plt.xlabel("Episode", fontsize=14)
plt.ylabel("Training Error (Rolling Avg)", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=12, frameon=True)
plt.tight_layout()
plt.savefig("./output/Analysis/training_errors_rolling_plot.png")
plt.show()

# Plotting training errors: cumulative
plt.figure(figsize=(12, 8))
for agent, color, line_style in zip(AGENTS, COLORS, LINE_STYLES):
    file_path = os.path.join("output", agent, "errors.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df['Smoothed Error'] = df['Training Error'].rolling(window=10000).mean()
        df['Cumulative Smoothed Error'] = df['Smoothed Error'].cumsum()
        plt.plot(df['Episode'], df['Cumulative Smoothed Error'], label=agent, color=color, linestyle=line_style)
    else:
        print(f"Warning: {file_path} not found.")
plt.title("Cumulative Training Errors by Agent", fontsize=16)
plt.xlabel("Episode", fontsize=14)
plt.ylabel("Cumulative Training Error", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=12, frameon=True)
plt.tight_layout()
plt.savefig("./output/Analysis/training_errors_cumulative_plot.png")
plt.show()

# Plotting episode lengths
plt.figure(figsize=(12, 8))

for agent, color, line_style in zip(AGENTS, COLORS, LINE_STYLES):
    file_path = os.path.join("output", agent, "lengths.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        # Optional smoothing using rolling average
        df['Smoothed Length'] = df['Episode Lengths'].rolling(window=10000).mean()
        df['Cumulative Smoothed Length'] = df['Smoothed Length'].cumsum()
        plt.plot(df.index, df['Smoothed Length'], label=agent, color=color, linestyle=line_style)
    else:
        print(f"Warning: {file_path} not found.")

plt.title("Episode Lengths by Agent", fontsize=16)
plt.xlabel("Episode", fontsize=14)
plt.ylabel("Episode Length", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=12, frameon=True)
plt.tight_layout()
plt.savefig("./output/Analysis/episode_lengths_plot.png")
plt.show()

# Plotting episode lengths: rolling average
plt.figure(figsize=(12, 8))
for agent, color, line_style in zip(AGENTS, COLORS, LINE_STYLES):
    file_path = os.path.join("output", agent, "lengths.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df['Smoothed Length'] = df['Episode Lengths'].rolling(window=10000).mean()
        plt.plot(df.index, df['Smoothed Length'], label=agent, color=color, linestyle=line_style)
    else:
        print(f"Warning: {file_path} not found.")
plt.title("Episode Lengths (Rolling Average) by Agent", fontsize=16)
plt.xlabel("Episode", fontsize=14)
plt.ylabel("Episode Length (Rolling Avg)", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=12, frameon=True)
plt.tight_layout()
plt.savefig("./output/Analysis/episode_lengths_rolling_plot.png")
plt.show()

# Plotting episode lengths: cumulative
plt.figure(figsize=(12, 8))
for agent, color, line_style in zip(AGENTS, COLORS, LINE_STYLES):
    file_path = os.path.join("output", agent, "lengths.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df['Smoothed Length'] = df['Episode Lengths'].rolling(window=10000).mean()
        df['Cumulative Smoothed Length'] = df['Smoothed Length'].cumsum()
        plt.plot(df.index, df['Cumulative Smoothed Length'], label=agent, color=color, linestyle=line_style)
    else:
        print(f"Warning: {file_path} not found.")
plt.title("Cumulative Episode Lengths by Agent", fontsize=16)
plt.xlabel("Episode", fontsize=14)
plt.ylabel("Cumulative Episode Length", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=12, frameon=True)
plt.tight_layout()
plt.savefig("./output/Analysis/episode_lengths_cumulative_plot.png")
plt.show()
