import random
import numpy as np
from sql_util.run import get_result_set


class ReinforcementLearningAgent:
    def __init__(self, db_path, action_space, alpha=0.1, gamma=0.9):
        self.db_path = db_path
        self.action_space = action_space
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.q_table = {}  # Q-Table: state -> action -> Q-value

    def get_state(self, db_path):
        """
        Get the current state (e.g., database size).
        """
        from sql_util.dbinfo import get_total_size_from_path
        return get_total_size_from_path(db_path)

    def select_action(self, state, epsilon=0.1):
        """
        Select an action based on epsilon-greedy policy.
        """
        if random.random() < epsilon:
            # Exploration: Random action index
            return random.randint(0, len(self.action_space) - 1)
        else:
            # Exploitation: Action with the highest Q-value
            if state in self.q_table:
                action_q_values = self.q_table[state]
                return max(range(len(self.action_space)), key=lambda i: action_q_values.get(i, 0))
            else:
                # If the state is unknown, default to a random action
                return random.randint(0, len(self.action_space) - 1)

    def update_q_value(self, state, action, reward, next_state):
        """
        Update the Q-value for a given state-action pair.
        """
        if state not in self.q_table:
            self.q_table[state] = {i: 0 for i in range(len(self.action_space))}

        if next_state not in self.q_table:
            self.q_table[next_state] = {i: 0 for i in range(len(self.action_space))}

        max_next_q = max(self.q_table[next_state].values())
        self.q_table[state][action] += self.alpha * (
            reward + self.gamma * max_next_q - self.q_table[state][action]
        )

    def train(self, action_space, reward_calculator, episodes=10):
        """
        Train the agent using the given action space and reward calculator.
        """
        for episode in range(episodes):
            print(f"Episode {episode + 1}/{episodes}")
            state = self.get_state(self.db_path)

            for _ in range(len(action_space)):
                action_index = self.select_action(state, epsilon=0.1)
                query = action_space[action_index]

                # Calculate reward
                reward = reward_calculator.calculate_reward(query, self.db_path)

                # Simulate a new state after taking the action
                next_state = self.get_state(self.db_path)

                # Update Q-value
                self.update_q_value(state, action_index, reward, next_state)

                # Update the state
                state = next_state
