import numpy as np
from typing import Tuple, Union


def exact_policy_evaluation(
    gamma: float,
    policy: np.ndarray,
    rewards: np.ndarray,
    transitions: np.ndarray
) -> np.ndarray:
    num_states, num_actions = rewards.shape
    r_pi = np.sum(rewards * policy, axis=1)  # [s]
    p_pi = np.sum(policy[:, :, None] * transitions, axis=1)  # [s, s]
    v = np.linalg.inv((np.eye(num_states) - gamma*p_pi)) @ r_pi[:, None]
    return v[:, 0]


def run_value_iteration(
        gamma: float,
        rewards: np.ndarray,  # [s, a]
        transitions: np.ndarray,  # [s, a, s]
        initial_v: np.ndarray,  # [s]
        threshold: float = 1e-6,
        num_iters: int = np.inf,
        prob_update: float = 1.0,
        return_policy: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if len(rewards.shape) == 3:
        rewards = np.sum(rewards * transitions, axis=2)

    # print(rewards.shape)

    num_states, num_actions = rewards.shape
    v = initial_v[:, None]  # [s, 1]
    p = np.transpose(transitions, [1, 0, 2])  # [a, s, s]
    r = np.transpose(rewards, [1, 0])[:, :, None]  # [a, s, 1]
    i = 0
    old_v = initial_v
    while i < num_iters:
        pv = (p @ v[None, :, :])  # [a, s, 1]
        va = r + gamma * pv
        pi = np.argmax(va, axis=0)[:, 0]  # [s]
        new_v = np.max(va, axis=0)  # [s, 1]
        err = np.max(np.abs(new_v[:, 0] - v[:, 0]), axis=0)
        if err <= threshold:
            break
        v = new_v
        i += 1
    additional_output = ()
    if prob_update < 1:
        update = (np.random.uniform(0, 1, size=[num_states]) < prob_update).astype(np.float32)
        new_v = new_v * update + old_v[:, None] * (1 - update)
        additional_output = (update,)
    if return_policy:
        new_onehot = np.zeros(shape=rewards.shape)
        for s in range(rewards.shape[0]):
            new_onehot[s, pi[s]] = 1
        return (new_v[:, 0], new_onehot) + additional_output
    else:
        return (new_v[:, 0],) + additional_output

def run_m_moment_value_iteration(
        gamma: float,
        rewards: np.ndarray,  # [s, a, s]
        transitions: np.ndarray,  # [s, a, s]
        threshold: float = 1e-6,
        num_iters: int = np.inf,
        prob_update: float = 1.0,
        return_policy: bool = False,
        variance_weight: float = 0.5,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:

    if len(rewards.shape) == 2:
        rewards = np.repeat(rewards[:, :, None], rewards.shape[0], axis=2)


    num_states, num_actions, _ = rewards.shape

    v = np.zeros((2, num_states))[:, :, None]  # [2, s, 1]

    p = np.transpose(transitions, [1, 0, 2])  # [a, s, s]
    # r = np.transpose(rewards[], [1, 0])[:, :, None]  # [a, s, 1]
    r = rewards
    i = 0
    old_v = v

    while i < num_iters:
        if i % 1000 == 0:
            print(i)
        t_pi_M = []

        pval = p @ v[0, :, :] # [a, s, 1]
        pvar = p @ v[1, :, :] # [a, s, 1]

        r_expec = np.sum(r * transitions, axis=2)
        r_squared_expec = np.sum(np.power(r, 2) * transitions, axis=2)

        r_expec = np.transpose(r_expec, [1, 0])[:, :, None] # [a, s, 1]
        r_squared_expec = np.transpose(r_squared_expec, [1, 0])[:, :, None] # [a, s, 1]

        va = r_expec + gamma * pval # [a, s, 1]
        va2 = r_squared_expec + 2 * gamma * r_expec * pval + gamma ** 2 * pvar # [a, s, 1]


        mean_variance_vector = va + variance_weight * np.power(va, 2) - variance_weight * va2

        pi = np.argmax(mean_variance_vector, axis=0)[:, 0]  # [s]

        argmax_indices = np.argmax(mean_variance_vector, axis=0)

        new_v = va[argmax_indices, np.arange(va.shape[1])[:,None], np.arange(va.shape[2])] # [s, 1]
        new_second_moment = va2[argmax_indices, np.arange(va.shape[1])[:,None], np.arange(va.shape[2])] # [s, 1]

        err = np.max(np.abs(new_v[:, 0] - v[0, :, 0]), axis=0)
        if err <= threshold:
            break

        v = np.stack([new_v, new_second_moment])
        i += 1


    if return_policy:
        new_onehot = np.zeros(shape=(num_states, num_actions))
        for s in range(rewards.shape[0]):
            new_onehot[s, pi[s]] = 1

        return (v, new_onehot)

    else:
        return v


def m_moment_policy_evaluation(
    gamma: float,
    policy: np.ndarray, # [s, a]
    rewards: np.ndarray, # [s, a, s]
    transitions: np.ndarray, # [s, a, s]
    num_iters: int = np.inf,
    threshold: float = 1e-6,
) -> np.ndarray:
    num_states, num_actions, _ = rewards.shape

    v = np.zeros((2, num_states)) # [2, s]
    p = np.transpose(transitions, [1, 0, 2])  # [a, s, s]

    r = rewards
    i = 0
    old_v = v

    while i < num_iters:
        t_pi_M = []

        first_moment = v[0] # [s]
        second_moment = v[1] # [s]

        first_moment_p_expec = p @ first_moment # [a, s]
        first_moment_p_expec = np.transpose(first_moment_p_expec, [1, 0]) # [s, a]
        first_moment_pi_expec = np.sum(policy * first_moment_p_expec, axis=1) # [s]

        second_moment_p_expec = p @ second_moment # [a, s]
        second_moment_p_expec = np.transpose(second_moment_p_expec, [1, 0]) # [s, a]
        second_moment_pi_expec = np.sum(policy * second_moment_p_expec, axis=1) # [s]

        r_expec = np.sum(r * transitions, axis=2) # [s, a]
        r_pi_expec = np.sum(policy * r_expec, axis=1) # [s]

        r_squared_expec = np.sum(np.power(r, 2) * transitions, axis=2) # [s, a]
        r_squared_pi_expec = np.sum(policy * r_squared_expec, axis=1) # [s]


        bellman_op_first_moment = r_pi_expec + gamma * first_moment_pi_expec # [s]
        bellman_op_second_moment = r_squared_pi_expec + 2 * gamma * r_pi_expec * first_moment_pi_expec + gamma ** 2 * second_moment_pi_expec # [s]

        err = np.max(np.abs(bellman_op_first_moment - first_moment), axis=0)
        if err <= threshold:
            break

        v = np.stack([bellman_op_first_moment, bellman_op_second_moment])
        i += 1

    return v

