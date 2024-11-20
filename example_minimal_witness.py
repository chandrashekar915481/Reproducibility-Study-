from sql_util.minimal_witness import minimize_distinguishing_for_set, get_expected_remaining_entropy
from sql_util.dbinfo import database_pprint
from natural_fuzz import NaturalFuzzGenerator
import os

# Paths to the database files
database_path = 'dog_kennels.sqlite'  # Original database
random_db_path = 'random_db.sqlite'  # Randomized database
tmp_random_db_path = 'tmp_random_db.sqlite'  # Temporary database

# Initialize the fuzzer
fuzzer = NaturalFuzzGenerator(database_path, None, overwrite=True)

# Define SQL queries based on the provided schema
# a sql distribution
sqls = [
    'SELECT MAX(first_name) FROM Professionals;',
    'SELECT MIN(first_name) FROM Professionals;',
]
probabilities = [0.5, 0.5]

# using the fuzzer to generate a large number of large random databases
# with high information gain and keep the one with the smallest expected
# remaining entropy
smallest_remaining_entropy = float('inf')
for _ in range(1000):
    fuzzer.generate_one_db(sqls, tmp_random_db_path)
    remaining_entropy = get_expected_remaining_entropy(tmp_random_db_path, sqls, probabilities=probabilities)
    if smallest_remaining_entropy > remaining_entropy:
        smallest_remaining_entropy = remaining_entropy
        os.system('mv {} {}'.format(tmp_random_db_path, random_db_path))
    if remaining_entropy > 1e-10:
        break

if os.path.exists(tmp_random_db_path):
    os.remove(tmp_random_db_path)


# Minimize the distinguishing database
minimization_result = minimize_distinguishing_for_set(
    sqls, distinguishing_db_path=database_path, num_minimization_restart=1, verbose=True,
    max_total_row=float('inf'), distinguish_criteria='list', probabilities=probabilities
)
new_db_path = minimization_result['best_db_path']

# Print database information before and after minimization
print('Original database:')
database_pprint(database_path, print_empty=False)

print('\nRandomized database before minimization:')
database_pprint(random_db_path, print_empty=False)

print('\nMinimized database:')
database_pprint(new_db_path, print_empty=False)

# Clean up temporary files
if os.path.exists(random_db_path):
    os.remove(random_db_path)
if os.path.exists(new_db_path):
    os.remove(new_db_path)



db_path = "dog_kennels.sqlite"
queries = [
    "SELECT MAX(FIRST_NAME) FROM professionals;",
    "SELECT MIN(FIRST_NAME) FROM professionals;",
    "SELECT COUNT(*) FROM breeds;",
]

print("Minimizing database with reinforcement learning...")
q_table = minimize_distinguishing_for_set(queries, db_path)

print("Reinforcement Learning Q-Table:")
print(q_table)
