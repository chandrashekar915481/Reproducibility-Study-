import os
import pickle
from reinforcement_learning_agent import ReinforcementLearningAgent
from sql_util.run import get_result_set
from sql_util.dbinfo import get_total_size_from_path
from sql_util.writedb import delete_random_fraction

# Define the database path
db_path = "dog_kennels.sqlite"

# Define the action space (list of queries)
action_space = [
    "SELECT MAX(FIRST_NAME) FROM professionals;",
    "SELECT MIN(FIRST_NAME) FROM professionals;",
    "SELECT COUNT(*) FROM professionals;",
    "SELECT AVG(CHARGE_AMOUNT) FROM Charges;",
    # Add more queries if needed
]

# Define a simple reward calculator
class RewardCalculator:
    def calculate_reward(self, query, db_path):
        result = get_result_set([query], db_path)
        if result:
            # Reward for a successful query result
            return 10
        else:
            # Penalty for query failure
            return -5

reward_calculator = RewardCalculator()

# Initialize the RL agent
agent = ReinforcementLearningAgent(db_path, action_space)

# Training Phase
print("Training the agent...")
agent.train(action_space, reward_calculator, episodes=10)
print("Training completed!")

# Save the trained Q-table for reuse (optional)
with open("q_table.pkl", "wb") as f:
    pickle.dump(agent.q_table, f)

# Testing Phase
print("\nTesting the agent...")
# Copy the database to a test database to ensure the original remains unchanged
test_db_path = "dog_kennels_test.sqlite"
if os.path.exists(test_db_path):
    os.remove(test_db_path)
os.system(f"cp {db_path} {test_db_path}")

# Define unseen queries for testing
test_queries = [
    "SELECT MAX(LAST_NAME) FROM professionals;",
    "SELECT MIN(LAST_NAME) FROM professionals;",
    "SELECT COUNT(*) FROM Charges WHERE CHARGE_AMOUNT > 100;",
    "SELECT AVG(AGE) FROM dogs;",
    # Add more test queries if needed
]

# Run the test queries using the trained agent
for query in test_queries:
    print(f"Testing query: {query}")
    # Use the agent to select an action (query index in action_space)
    current_size = get_total_size_from_path(test_db_path)
    action_index = agent.select_action(current_size, epsilon=0.1)  # Use epsilon-greedy policy

    # Validate the action index and retrieve the query
    if isinstance(action_index, int) and 0 <= action_index < len(action_space):
        selected_query = action_space[action_index]
    else:
        raise ValueError(f"Invalid action returned: {action_index}")

    # Apply the action (e.g., process query and modify database)
    print(f"Executing selected query: {selected_query}")
    result = get_result_set([selected_query], test_db_path)
    print(f"Query result: {result}")

    # Simulate database modification for testing (e.g., drop random entries)
    delete_random_fraction(test_db_path, test_db_path, "dogs", 0.1)  # Modify database slightly

    # Observe the new state (database size)
    new_size = get_total_size_from_path(test_db_path)
    print(f"Database size after modification: {new_size}")
    print("")

print("Testing completed!")
