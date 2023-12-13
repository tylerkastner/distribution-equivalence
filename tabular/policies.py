from typing import Literal, Tuple

import numpy as np
import value_and_policy_iteration as vpi


def make_random_deterministic_policy(
        num_states: int,
        num_actions: int
) -> np.ndarray:
    actions = np.random.randint(0, num_actions, size=[num_states])
    one_hot_actions = np.zeros([num_states, num_actions])
    one_hot_actions[np.arange(num_states), actions] = 1
    return one_hot_actions


def make_random_stochastic_policy(
        num_states: int,
        num_actions: int
) -> np.ndarray:
    pi = np.random.uniform(0, 1, size=[num_states, num_actions])
    return pi / np.sum(pi, axis=1)[:, None]


def collect_random_policies(
        num_policies: int,
        ve_policy_mode: Literal['det', 'stoch', 'det_and_stoch'],
        num_ve_steps: int,
        num_states: int,
        num_actions: int,
        true_r: np.ndarray,
        true_p: np.ndarray,
        gamma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    policies = np.zeros([num_policies, num_states, num_actions], dtype=np.float32)
    values = np.zeros([num_policies, num_states], dtype=np.float32)
    for i in range(num_policies):
        if ve_policy_mode == 'det':
            pi = make_random_deterministic_policy(num_states, num_actions)
        elif ve_policy_mode == 'stoch':
            pi = make_random_stochastic_policy(num_states, num_actions)
        else:
            use_stoch = np.random.uniform(0, 1) < 0.5
            if use_stoch:
                pi = make_random_stochastic_policy(num_states, num_actions)
            else:
                pi = make_random_deterministic_policy(num_states, num_actions)
        policies[i] = pi
    for i, pi in enumerate(policies):
        if i % 100 == 0:
            print(f'done {i}')
        if num_ve_steps < np.inf:
            v_pi = np.random.uniform(-1, 1, size=[num_states])
        else:
            v_pi = vpi.exact_policy_evaluation(gamma, pi, true_r, true_p)
        values[i] = v_pi
    return policies, values


def collect_random_moment_vectors(
        num_policies: int,
        ve_policy_mode: Literal['det', 'stoch', 'det_and_stoch'],
        num_ve_steps: int,
        num_moments: int,
        num_states: int,
        num_actions: int,
        true_r: np.ndarray,
        true_p: np.ndarray,
        gamma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    policies = np.zeros([num_policies, num_states, num_actions], dtype=np.float32)
    values = np.zeros([num_policies, num_moments, num_states], dtype=np.float32)
    for i in range(num_policies):
        if ve_policy_mode == 'det':
            pi = make_random_deterministic_policy(num_states, num_actions)
        elif ve_policy_mode == 'stoch':
            pi = make_random_stochastic_policy(num_states, num_actions)
        else:
            use_stoch = np.random.uniform(0, 1) < 0.5
            if use_stoch:
                pi = make_random_stochastic_policy(num_states, num_actions)
            else:
                pi = make_random_deterministic_policy(num_states, num_actions)
        policies[i] = pi
    for i, pi in enumerate(policies):
        if i % 50 == 0:
            print(f'done {i} value functions')
        if num_ve_steps < np.inf:
            v_pi = np.random.uniform(-1, 1, size=(num_moments, num_states))
        else:
            v_pi = vpi.m_moment_policy_evaluation(gamma, pi, true_r, true_p)
        values[i] = v_pi
    return policies, values


def collect_iteration_policies(
        num_base_policies: int,
        scale: int,
        ve_policy_mode: Literal['det', 'stoch', 'det_and_stoch'],
        num_ve_steps: int,
        num_states: int,
        num_actions: int,
        true_r: np.ndarray,
        true_p: np.ndarray,
        gamma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    base_policies = np.zeros([num_base_policies, num_states, num_actions], dtype=np.float32)
    policies = np.zeros([num_base_policies * scale, num_states, num_actions], dtype=np.float32)
    values = np.zeros([num_base_policies * scale, num_states], dtype=np.float32)
    vstar = np.mean(vpi.run_value_iteration(gamma, true_r, true_p, np.zeros([num_states])))

    i = 0
    while i < num_base_policies:
        pi = make_random_deterministic_policy(num_states, num_actions)
        v_pi = vpi.exact_policy_evaluation(gamma, pi, true_r, true_p)
        base_policies[i] = pi
        i += 1
        while np.mean(v_pi) < 0.9 * vstar and i < num_base_policies:
            v_pi, update_pi, update = vpi.run_value_iteration(
                    gamma, true_r, true_p, v_pi, num_iters=1, prob_update=0.1,
                    return_policy=True)
            pi = update[:, None] * update_pi + (1 - update[:, None]) * pi
            base_policies[i] = pi
            i += 1

    if ve_policy_mode == 'det':
        fuzz_pi = lambda: make_random_deterministic_policy(num_states, num_actions)
    else:
        fuzz_pi = lambda: make_random_stochastic_policy(num_states, num_actions)

    i = 0
    for pi in base_policies:
        for _ in range(scale):
            rand_pi = fuzz_pi()
            chance_swap = (np.random.uniform(0, 1, size=[num_states]) < 0.1).astype(np.float32)
            new_pi = chance_swap[:, None] * rand_pi + (1 - chance_swap[:, None]) * pi
            policies[i] = new_pi

            if num_ve_steps < np.inf:
                values[i] = np.random.uniform(-1, 1, size=[num_states])
            else:
                values[i] = vpi.exact_policy_evaluation(gamma, new_pi, true_r, true_p)
            i += 1
    return policies, values
