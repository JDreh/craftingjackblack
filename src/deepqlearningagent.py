import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from agent import BlackJackAgent

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DeepQLearningBlackJackAgent(BlackJackAgent):
    def __init__(self, state_space_dim, action_space_dim, lr=1e-3, gamma=0.99, epsilon=0.1, device=None):
        # Do not call super().__init__() since BlackJackAgent expects Q-learning args
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = QNetwork(state_space_dim, action_space_dim).to(self.device)
        self.target_network = QNetwork(state_space_dim, action_space_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.discount_factor = gamma
        self.epsilon = epsilon
        self.training_error = []
        self.memory = []
        self.batch_size = 32
        self.update_target_steps = 1000
        self.learn_step_counter = 0

    def obs_to_tensor(self, obs):
        # obs: (player_sum, dealer_card, usable_ace)
        return torch.tensor([obs[0], obs[1], int(obs[2])], dtype=torch.float32, device=self.device)

    def get_action(self, env, obs: tuple[int, int, bool]) -> int:
        if np.random.random() < self.epsilon:
            return env.action_space.sample()
        with torch.no_grad():
            state = self.obs_to_tensor(obs)
            q_values = self.q_network(state)
            return int(torch.argmax(q_values).item())

    def update(self, obs, action, reward, terminated, next_obs):
        # Store transition in memory
        self.memory.append((obs, action, reward, terminated, next_obs))
        if len(self.memory) < self.batch_size:
            return
        # Sample random minibatch
        indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        obs_batch, action_batch, reward_batch, terminated_batch, next_obs_batch = zip(*batch)
        obs_batch = torch.stack([self.obs_to_tensor(o) for o in obs_batch])
        next_obs_batch = torch.stack([self.obs_to_tensor(o) for o in next_obs_batch])
        action_batch = torch.tensor(action_batch, dtype=torch.long, device=self.device)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32, device=self.device)
        terminated_batch = torch.tensor(terminated_batch, dtype=torch.float32, device=self.device)
        # Q(s,a)
        q_values = self.q_network(obs_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        # max_a' Q_target(s',a')
        with torch.no_grad():
            next_q_values = self.target_network(next_obs_batch).max(1)[0]
            target = reward_batch + self.discount_factor * next_q_values * (1 - terminated_batch)
        loss = nn.MSELoss()(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.training_error.append(loss.item())
        # Update target network
        self.learn_step_counter += 1
        if self.learn_step_counter % self.update_target_steps == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        # Simple linear decay for epsilon
        self.epsilon = max(0.1, self.epsilon * 0.995)

    @property
    def q_values(self):
        # Return a dict-like object that computes Q-values on demand for plotting
        class QValueDict:
            def __init__(self, agent):
                self.agent = agent
            def __getitem__(self, obs):
                obs_tensor = self.agent.obs_to_tensor(obs)
                with torch.no_grad():
                    qvals = self.agent.q_network(obs_tensor)
                return qvals.cpu().numpy()
            def items(self):
                # For plotting, generate all possible states
                for player_sum in range(12, 22):
                    for dealer_card in range(1, 11):
                        for usable_ace in [False, True]:
                            obs = (player_sum, dealer_card, usable_ace)
                            yield obs, self[obs]
        return QValueDict(self)
