class TrainingEnvironment:
    def __init__(self, db_path, queries):
        self.db_path = db_path
        self.queries = queries

    def get_possible_actions(self):
        """Return a list of possible SQL queries (actions)."""
        return self.queries

    def calculate_reward(self, result):
        """Calculate a reward based on query results."""
        if result is None:
            return -1  # Negative reward for invalid queries
        return len(result)  # Reward proportional to the number of rows returned
