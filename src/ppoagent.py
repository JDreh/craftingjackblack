import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from agent import BlackJackAgent

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.action_head = nn.Linear(64, output_dim)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        action_logits = self.action_head(x)
        state_value = self.value_head(x)
        return action_logits, state_value

class PPOBlackJackAgent(BlackJackAgent):
    def __init__(
        self,
        env,
        learning_rate: float,
        initial_epsilon: float, # For exploration only, not PPO's clip
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.99,
        device=None,
        clip_epsilon=0.2,
        update_epochs=4,
        batch_size=32,
    ):
        super().__init__(env, learning_rate, initial_epsilon, epsilon_decay, final_epsilon, discount_factor)
        state_space_dim = len(env.observation_space)
        action_space_dim = env.action_space.n
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PolicyNetwork(state_space_dim, action_space_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.training_error = []
        self.memory = []
        self.action_space_dim = action_space_dim

    def obs_to_tensor(self, obs):
        return torch.tensor([obs[0], obs[1], int(obs[2])], dtype=torch.float32, device=self.device)

    def get_action(self, env, obs: tuple[int, int, bool]) -> int:
        state = self.obs_to_tensor(obs)
        with torch.no_grad():
            logits, _ = self.policy(state)
            probs = torch.softmax(logits, dim=-1)
            if np.random.rand() < self.epsilon:
                return env.action_space.sample()
            action = torch.multinomial(probs, 1).item()
        return action

    def store_transition(self, obs, action, reward, done, log_prob, value):
        self.memory.append((obs, action, reward, done, log_prob, value))

    def update(self, obs, action, reward, terminated, next_obs):
        state = self.obs_to_tensor(obs)
        logits, value = self.policy(state)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_prob = dist.log_prob(torch.tensor(action, device=self.device))
        self.store_transition(obs, action, reward, terminated, log_prob.item(), value.item())
        # Learn when enough transitions are collected
        if len(self.memory) >= self.batch_size:
            self._learn()
            self.memory = []
        # Always learn at the end of an episode if any transitions remain
        elif terminated and len(self.memory) > 0:
            self._learn()
            self.memory = []

    def _learn(self):
        # Convert memory to tensors
        obs_list, action_list, reward_list, done_list, old_log_probs, values = zip(*self.memory)
        obs_batch = torch.stack([self.obs_to_tensor(o) for o in obs_list])
        action_batch = torch.tensor(action_list, dtype=torch.long, device=self.device)
        reward_batch = torch.tensor(reward_list, dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(done_list, dtype=torch.float32, device=self.device)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)
        values = torch.tensor(values, dtype=torch.float32, device=self.device)

        # Compute returns and advantages
        returns = []
        G = 0
        for r, d in zip(reversed(reward_list), reversed(done_list)):
            G = r + self.discount_factor * G * (1 - d)
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages = returns - values
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # Normalize advantages
        if torch.isnan(advantages).any():
            advantages = torch.zeros_like(advantages)



        for _ in range(self.update_epochs):
            logits, value_preds = self.policy(obs_batch)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(action_batch)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(value_preds.view(-1), returns)
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.training_error.append(loss.item())

    def decay_epsilon(self):
        self.epsilon = max(0.01, self.epsilon * 0.995)

    @property
    def q_values(self):
        # For plotting, return action probabilities
        class QValueDict:
            def __init__(self, agent):
                self.agent = agent
            def __getitem__(self, obs):
                obs_tensor = self.agent.obs_to_tensor(obs)
                if obs_tensor.dim() == 1:
                    obs_tensor = obs_tensor.unsqueeze(0)
                with torch.no_grad():
                    logits, _ = self.agent.policy(obs_tensor)
                    probs = torch.softmax(logits, dim=-1).cpu().numpy().squeeze()
                    return probs # Return action probabilities
            def items(self):
                for player_sum in range(12, 22):
                    for dealer_card in range(1, 11):
                        for usable_ace in [False, True]:
                            obs = (player_sum, dealer_card, usable_ace)
                            yield obs, self[obs]
        return QValueDict(self)

    @q_values.setter
    def q_values(self, _):
        pass
