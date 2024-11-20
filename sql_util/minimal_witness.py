import os
from sql_util.dbinfo import get_indexing_from_db, get_total_size_from_path, database_pprint, get_table2size_from_path, get_total_printed_size_from_path
from sql_util.run import get_result_set, listify_result, setify_result, exec_db_path_
from typing import List
import numpy as np
from sql_util.writedb import delete_entries_by_mod, delete_random_fraction
from shutil import copyfile
import time
import random


drop_block_count = 20
min_result_size = 2


def compute_entropy(probabilities: List[float]) -> float:
    probabilities = np.array(probabilities)
    # smoothing
    probabilities += 1e-10
    probabilities /= np.sum(probabilities)
    probabilities = np.array(probabilities)
    entropy = -np.sum(np.log2(probabilities) * probabilities)
    return entropy


def add_s_before_format_suffix(path, s):
    return path.replace('.sqlite', s + '.sqlite')


def compute_expected_entropy(probabilities_groups: List[List[float]]) -> float:
    return np.sum([sum(probabilities) * compute_entropy(probabilities) for probabilities in probabilities_groups])


# remove the databases that are not on the pareto frontier
def update_db2size_entropy_frontier(db2size_entropy_frontier, exempted_db_path):
    deleted_dbs = set()
    for db1 in db2size_entropy_frontier:
        for db2 in db2size_entropy_frontier:
            if db1 != db2:
                s1, e1 = db2size_entropy_frontier[db1]
                s2, e2 = db2size_entropy_frontier[db2]
                if s1 >= s2 and e1 >= e2:
                    if s1 == s2 and e1 == e2:
                        if db1 < db2:
                            deleted_dbs.add(db1)
                    else:
                        deleted_dbs.add(db1)

    for db in deleted_dbs:
        if db != exempted_db_path:
            del db2size_entropy_frontier[db]
            if os.path.exists(db):
                os.unlink(db)

                
def get_expected_remaining_entropy(db_path, queries, probabilities=None, distinguish_criteria='list'):
    if probabilities is None:
        n = len(queries)
        probabilities = np.ones(n) / n
    else:
        probabilities = probabilities / np.sum(probabilities)
    assert len(probabilities) == len(queries)
    results = get_result_set(queries, db_path, distinguish_criteria)
    if results is None:
        print("Error: get_result_set returned None.")
        print(f"Queries: {queries}")
        print(f"Database Path: {db_path}")
        return float('inf')  # Return a high entropy value to indicate failure
    expected_entropy = compute_expected_entropy([[probabilities[idx] for idx in results[key]] for key in results])
    return expected_entropy


def get_expected_remaining_entropy_all_info(db_path, q2prob, distinguish_criteria='list', gold=None):
    queries = sorted(q2prob, key=lambda q: q2prob[q], reverse=True)
    probabilities = [q2prob[q] for q in queries]
    
    f = listify_result if distinguish_criteria == 'list' else setify_result
    
    if probabilities is None:
        n = len(queries)
        probabilities = np.ones(n) / n
    else:
        probabilities = probabilities / np.sum(probabilities)
    assert len(probabilities) == len(queries)
    results = get_result_set(queries, db_path, distinguish_criteria)
    if results is None:
        e = compute_entropy(probabilities)
        return {
            'remaining_entropy': e,
            'information_gain': 0.,
            'q2pobs': [
                q2prob
            ],
            'q2prob_left': q2prob,
            'actual_entropy': e
        }
    expected_entropy = compute_expected_entropy([[probabilities[idx] for idx in results[key]] for key in results])
    
    returned_dict = {
        'remaining_entropy': expected_entropy,
        'information_gain': compute_entropy(probabilities) - expected_entropy,
        'q2pobs': [
            normalize_q2prob({queries[idx]: probabilities[idx] for idx in results[key]})
            for key in results
        ]
    }
    
    if gold is not None:
        gold_result = f(exec_db_path_(db_path, gold)[1])
        if gold_result in results:
            q2probs_left = normalize_q2prob({queries[idx]: probabilities[idx] for idx in results[gold_result]})
        else:
            q2probs_left = None
        returned_dict['q2prob_left'] = q2probs_left
        if q2probs_left is None:
            returned_dict['actual_entropy'] = 0
        else:
            returned_dict['actual_entropy'] = compute_entropy([v for v in q2probs_left.values()])
    return returned_dict
    


def normalize_q2prob(q2prob):
    norm = sum(q2prob.values())
    return {q: q2prob[q] / norm for q in q2prob}


from sql_util.reinforcement_learning_agent import ReinforcementLearningAgent
from sql_util.reward_calculator import RewardCalculator

def minimize_distinguishing_for_set(queries: List[str], distinguishing_db_path: str,
                                    num_minimization_restart=1, verbose=False, 
                                    max_total_row=float('inf'), 
                                    distinguish_criteria='list', probabilities=None):
    assert distinguishing_db_path.endswith('.sqlite')
    if probabilities is None:
        probabilities = np.ones(len(queries)) / len(queries)
    else:
        probabilities = probabilities / np.sum(probabilities)
    step_count = 0
    db2size_entropy_frontier = {}
    rl_agent = ReinforcementLearningAgent(distinguishing_db_path)  # RL agent initialization
    reward_calculator = RewardCalculator()  # Reward calculator for entropy reduction

    if verbose:
        print('original size of the database at %s is %d' % (distinguishing_db_path, get_total_size_from_path(distinguishing_db_path)))
        print('queries: ')
        for a in queries[:20]:
            print(a)
        if len(queries) > 20:
            print('...')

    # Reinitialize the database to the original one
    for restart_idx in range(num_minimization_restart):
        db_target_path = add_s_before_format_suffix(distinguishing_db_path, '_generated_restart%d' % restart_idx)
        db_target_lookahead_path = add_s_before_format_suffix(db_target_path, 'lookahead')

        # Make a copy of the original database
        os.system('touch %s' % (db_target_path))
        copyfile(distinguishing_db_path, db_target_path)
        cur_size = get_total_size_from_path(db_target_path)

        epoch_idx = 0
        if verbose:
            print('computing splits based on the starting database')

        # RL-driven loop for database minimization
        while cur_size > min_result_size:
            # RL: Determine action (which table/entry to drop)
            table2size = get_table2size_from_path(db_target_path)
            actions = list(table2size.keys())
            state = rl_agent.get_state(db_target_path)
            action = rl_agent.choose_action(state, actions)

            # Perform the action and calculate reward
            num_deleted_rows = delete_random_fraction(db_target_path, db_target_lookahead_path, action, 1.0 / len(actions))
            result = get_result_set(queries, db_target_lookahead_path)
            reward = reward_calculator.calculate_reward(result)

            if num_deleted_rows > 0:
                # Update the RL agent with the observed reward
                next_state = rl_agent.get_state(db_target_lookahead_path)
                rl_agent.update_q_value(state, action, reward, next_state)

                # Apply the changes if reward is positive
                os.rename(db_target_lookahead_path, db_target_path)
                cur_size = get_total_size_from_path(db_target_path)
                step_count += 1

                if verbose:
                    print(f'Successfully dropped entries from {action}. Current size: {cur_size}. Reward: {reward}')
            else:
                os.unlink(db_target_lookahead_path)
                if verbose:
                    print(f'Failed to drop entries from {action}. Skipping.')

            if cur_size <= min_result_size:
                break

            # Update Pareto frontier
            db2size_entropy_frontier[db_target_path] = (cur_size, compute_expected_entropy(probabilities))

            epoch_idx += 1

        if verbose:
            print('Minimization complete. Final size:', cur_size)

        # Keep track of the best database on the Pareto frontier
        update_db2size_entropy_frontier(db2size_entropy_frontier, exempted_db_path='')

    # Select the best database from the Pareto frontier
    best_db_path = max([db for db, (s, e) in db2size_entropy_frontier.items() if s <= max_total_row], 
                       key=lambda k: db2size_entropy_frontier[k][0])

    # Clean up other databases
    for db in db2size_entropy_frontier:
        if db != best_db_path:
            os.unlink(db)
            
    return {
        'best_db_path': best_db_path,
        'best_size': get_total_printed_size_from_path(best_db_path, queries),
        'best_expected_entropy': db2size_entropy_frontier[best_db_path][1],
        'frontier': db2size_entropy_frontier,
        "step_count": step_count
    }



def drop_random_to_target_size_approximate(db_path, target_size, target_path):
    cur_path = target_path
    tmp_path = cur_path + '_'
    os.system('cp %s %s' % (db_path, cur_path))
    os.system('chmod 777 %s' % cur_path)

    cur_size = get_total_size_from_path(cur_path)
    continue_dropping = True
    num_attempt = 20
    
    while num_attempt > 0:
        
        table2size = get_table2size_from_path(cur_path)
        table = random.choice(list(table2size))
        if table2size[table] == 0:
            continue
        
        fraction = max(0.2, 1. / table2size[table])
        num_r_dropped = delete_random_fraction(cur_path, tmp_path, table, fraction)
        new_size = get_total_size_from_path(tmp_path)

        if new_size <= target_size * 0.8:
            num_attempt -= 1
            continue
        elif new_size <= target_size:
            break
        else:
            os.system('mv %s %s' % (tmp_path, cur_path))
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)
        
    return cur_path


def drop_random_to_target_size(db_path, target_size):
    cur_path = 'tmp/' + str(time.time())
    tmp_path = cur_path + '_'
    os.system('cp %s %s' % (db_path, cur_path))
    os.system('chmod 777 %s' % cur_path)

    cur_size = get_total_size_from_path(cur_path)
    continue_dropping = True
    num_attempt = 10
    
    while num_attempt > 0:
        
        table2size = get_table2size_from_path(cur_path)
        table = random.choice(list(table2size))
        if table2size[table] == 0:
            continue
        
        fraction = max(0.2, 1. / table2size[table])
        num_r_dropped = delete_random_fraction(cur_path, tmp_path, table, fraction)
        new_size = get_total_size_from_path(tmp_path)

        if new_size < target_size:
            os.unlink(tmp_path)
            num_attempt -= 1
        else:
            os.system('mv %s %s' % (tmp_path, cur_path))
        if new_size == target_size:
            return cur_path
    return cur_path

def refine_query_with_feedback(sql_query: str, feedback: dict) -> str:
    """
    Refine a SQL query based on semantic feedback.
    Args:
        sql_query: The SQL query to refine.
        feedback: A dictionary where keys are old table/column names and values are corrections.
    """
    for original, replacement in feedback.items():
        sql_query = sql_query.replace(original, replacement)
    return sql_query


