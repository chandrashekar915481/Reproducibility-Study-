class RewardCalculator:
    def __init__(self):
        pass

    def calculate_reward(self, result):
        """Define a custom reward calculation logic."""
        if result is None:
            return -1
        return len(result)
