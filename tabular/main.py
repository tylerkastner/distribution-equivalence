import jax
import numpy as np
import jax.numpy as jnp
import jax.random as jrng
import optax
from ray import tune
import random
import numpy as np
import pickle
import os
import sys
import gridworld
import windy_cliff
import frozen_lake

import policies as policies_module
import value_and_policy_iteration as vpi
from math import comb


def init_model(key, num_states, num_actions, config):
    key, r_key, d_key, k_key = jrng.split(key, 4)
    uni_init = config['uni_init']
    if config['sas']:
        r = jrng.uniform(r_key, [num_states, num_actions, num_states], minval=-1, maxval=1)
    else:
        r = jrng.uniform(r_key, [num_states, num_actions], minval=-1, maxval=1)
    if config['restrict_capacity']:
        d = jrng.uniform(d_key, [num_actions, num_states, config['model_rank']], minval=-uni_init, maxval=uni_init)
        k = jrng.uniform(k_key, [num_actions, config['model_rank'], num_states], minval=-uni_init, maxval=uni_init)
        return key, (r, d, k)
    else:
        p = jrng.uniform(d_key, [num_actions, num_states, num_states], minval=-uni_init, maxval=uni_init)
        return key, (r, p)


def params_to_model(params, config):
    if config['restrict_capacity']:
        r, d, k = params
        pd = jax.nn.softmax(d, axis=2)
        pk = jax.nn.softmax(k, axis=2)
        p = pd @ pk
    else:
        r, p = params
        p = jax.nn.softmax(p, axis=2)
    return r, jnp.transpose(p, [1, 0, 2])


def bellman_update(pi, v, r, p, gamma):
    # pi : [num_states, num_actions]
    # v : [num_states]
    # r : [num_states, num_actions]
    # p : [num_states, num_actions, num_states]
    vv = jnp.transpose(p, [1, 0, 2]) @ v[None, :, None]  # [num_actions, num_states, 1]
    rvv = r + gamma * jnp.transpose(vv[:, :, 0], [1, 0])  # [num_states, num_actions]
    return jnp.sum(pi * rvv, axis=1)  # [num_states]


def m_moment_bellman_update(pi, M, r, p, m, gamma):
    # pi : [num_states, num_actions]
    # M : [num_moments, num_states]
    # m : num_moments
    # r : [num_states, num_actions, num_states]
    # p : [num_states, num_actions, num_states]

    t_pi_M = []
    num_states = r.shape[0]

    r_powers = []


    for i in range(m):
        isum = np.zeros(num_states)
        for j in range(i):
            moment_next_state_expec = jnp.transpose(p, [1, 0, 2]) @ M[i-j, :, None] # [num_actions, num_states, 1]
            moment_next_state_expec = jnp.transpose(moment_next_state_expec[:, :, 0], [1, 0]) # [num_states, num_actions]
            moment_pi_expec = jnp.sum(pi * moment_next_state_expec, axis=1) # [num_states]

            jth_power_reward_pi_expec = jnp.sum(pi * np.sum(jnp.power(r, j) * p, axis=2), axis=1)
            isum = isum + gamma ** (i - j) * comb(i, j) * jth_power_reward_pi_expec * moment_pi_expec

        t_pi_M.append(isum)

    t_pi_M = jnp.stack(t_pi_M, axis= 0)

    return t_pi_M # [num_moments, num_states]

def two_moment_bellman_update(pi, M, r, p, m, gamma):
#     gamma: float,
#     policy: np.ndarray, # [s, a]
#     rewards: np.ndarray, # [s, a, s]
#     transitions: np.ndarray, # [s, a, s]
#     num_iters: int = np.inf,
#     threshold: float = 1e-6,
# ) -> np.ndarray:
    num_states, num_actions, _ = r.shape
    # print('==== here ====')

    # p = np.transpose(transitions, [1, 0, 2])  # [a, s, s]

    # r = rewards

    first_moment = M[0] # [s]
    second_moment = M[1] # [s]

    first_moment_p_expec = jnp.transpose(p, [1, 0, 2]) @ first_moment # [a, s]
    first_moment_p_expec = jnp.transpose(first_moment_p_expec, [1, 0]) # [s, a]
    first_moment_pi_expec = jnp.sum(pi * first_moment_p_expec, axis=1) # [s]

    second_moment_p_expec = jnp.transpose(p, [1, 0, 2]) @ second_moment # [a, s]
    second_moment_p_expec = jnp.transpose(second_moment_p_expec, [1, 0]) # [s, a]
    second_moment_pi_expec = jnp.sum(pi * second_moment_p_expec, axis=1) # [s]

    r_expec = jnp.sum(r * p, axis=2) # [s, a]
    r_pi_expec = jnp.sum(pi * r_expec, axis=1) # [s]

    r_squared_expec = jnp.sum(jnp.power(r, 2) * p, axis=2) # [s, a]
    r_squared_pi_expec = jnp.sum(pi * r_squared_expec, axis=1) # [s]


    bellman_op_first_moment = r_pi_expec + gamma * first_moment_pi_expec # [s]
    bellman_op_second_moment = r_squared_pi_expec + 2 * gamma * r_pi_expec * first_moment_pi_expec + gamma ** 2 * second_moment_pi_expec # [s]

    v = jnp.stack([bellman_op_first_moment, bellman_op_second_moment])

    return v


def n_step_bellman_update(pi, v, r, p, n, gamma):
    for i in range(n):
        v = bellman_update(pi, v, r, p, gamma)
    return v


def n_step_two_moment_bellman_update(pi, M, r, p, n, m, gamma):
    for i in range(n):
        M  = two_moment_bellman_update(pi, M, r, p, m, gamma)
    return M

def ve_loss(params, pi_batch, v_batch, true_r, true_p, config):
    r, p = params_to_model(params, config)
    T = jax.vmap(n_step_bellman_update, (0, 0, None, None, None, None), 0)
    tv_model = T(pi_batch, v_batch, r, p, config['ve_mode'][0], config['gamma'])
    tv = T(pi_batch, v_batch, true_r, true_p, config['ve_mode'][0], config['gamma'])
    return jnp.mean(jnp.sum(jnp.square(tv - tv_model), axis=1), axis=0)


def mmve_loss(params, pi_batch, M_batch, true_r, true_p, config):
    r, p = params_to_model(params, config)
    T = jax.vmap(n_step_two_moment_bellman_update, (0, 0, None, None, None, None, None), 0)
    tv_model = T(pi_batch, M_batch, r, p, config['ve_mode'][0], config['m'], config['gamma'])
    tv = T(pi_batch, M_batch, true_r, true_p, config['ve_mode'][0], config['m'], config['gamma'])

    return jnp.mean(jnp.sum(jnp.square(tv - tv_model), axis=(1,2)), axis=0)


def fpve_loss(params, pi_batch, true_v_pi_batch, config):
    r, p = params_to_model(params, config)
    T = jax.vmap(bellman_update, (0, 0, None, None, None), 0)
    tv_model = T(pi_batch, true_v_pi_batch, r, p, config['gamma'])
    return jnp.mean(jnp.sum(jnp.square(tv_model - true_v_pi_batch), axis=1), axis=0)


def pmmve_loss(params, pi_batch, true_v_pi_batch, config):
    r, p = params_to_model(params, config)
    # T = jax.vmap(m_moment_bellman_update, (0, 0, None, None, None, None), 0)
    T = jax.vmap(two_moment_bellman_update, (0, 0, None, None, None, None), 0)
    tv_model = T(pi_batch, true_v_pi_batch, r, p, config['m'], config['gamma'])
    return jnp.mean(jnp.sum(jnp.square(tv_model - true_v_pi_batch), axis=(1, 2)), axis=0)


def update_ve(params, state, opt, pi, v, true_r, true_p, config):
    loss, grads = jax.value_and_grad(ve_loss)(params, pi, v, true_r, true_p, config)
    updates, state = opt.update(grads, state, params)
    params = optax.apply_updates(params, updates)
    return params, state, loss


def update_mmve(params, state, opt, pi, v, true_r, true_p, config):
    loss, grads = jax.value_and_grad(mmve_loss)(params, pi, v, true_r, true_p, config)
    updates, state = opt.update(grads, state, params)
    params = optax.apply_updates(params, updates)
    return params, state, loss


def update_fpve(params, state, opt, pi, true_v_pi, config):
    loss, grads = jax.value_and_grad(fpve_loss)(params, pi, true_v_pi, config)
    updates, state = opt.update(grads, state, params)
    params = optax.apply_updates(params, updates)
    return params, state, loss


def update_mmpve(params, state, opt, pi, true_v_pi, config):
    loss, grads = jax.value_and_grad(pmmve_loss)(params, pi, true_v_pi, config)
    updates, state = opt.update(grads, state, params)
    params = optax.apply_updates(params, updates)
    return params, state, loss


def run_experiment(config):
    # set up seed
    # np.random.seed(config['seed'])
    # random.seed(config['seed'])
    # construct env and get env params
    # env = gridworld.FourRooms(p_intended=0.8)

    # env = windy_cliff.CliffWalk()
    env = frozen_lake.EightByEight()

    if config['sas']:
        true_r = env.get_sas_reward_matrix()
    else:
        true_r = env.get_reward_matrix()

    true_p = env.get_transition_tensor()

    if config['sas']:
        num_states, num_actions, _ = np.shape(true_r)
    else:
        num_states, num_actions = np.shape(true_r)

    # initialize jax stuff
    # key = jrng.PRNGKey(config['seed'])

    key = jrng.PRNGKey(1234)
    key, model_params = init_model(key, num_states, num_actions, config)
    opt = optax.adam(config['learning_rate'])
    state = opt.init(model_params)

    num_ve_steps, ve_policy_mode = config['ve_mode']

    use_m_moment = config['m_moment']
    if use_m_moment:
        num_moments = config['m']

    def _update_ve(params, state, pi, v):
        return update_ve(params, state, opt, pi, v, true_r, true_p, config)
    _update_ve = jax.jit(_update_ve)

    def _update_mmve(params, state, pi, v):
        return update_mmve(params, state, opt, pi, v, true_r, true_p, config)
    _update_mmve = jax.jit(_update_mmve)

    def _update_fpve(params, state, pi, true_v_pi):
        return update_fpve(params, state, opt, pi, true_v_pi, config)
    _update_fpve = jax.jit(_update_fpve)

    def _update_mmpve(params, state, pi, true_v_pi):
        return update_mmpve(params, state, opt, pi, true_v_pi, config)
    _update_mmpve = jax.jit(_update_mmpve)

    # collect policies and values
    if not use_m_moment:
        if not config['use_vip']:
            policies, values = policies_module.collect_random_policies(
                10_000, ve_policy_mode, num_ve_steps, num_states, num_actions, true_r, true_p, config['gamma'])
        else:
            policies, values = policies_module.collect_iteration_policies(
                1000, 100, ve_policy_mode, num_ve_steps, num_states, num_actions, true_r, true_p, config['gamma'])
    else:
        if not config['use_vip']:
            policies, values = policies_module.collect_random_moment_vectors(
                10_000, ve_policy_mode, num_ve_steps, num_moments, num_states, num_actions, true_r, true_p, config['gamma'])
            # changed num from 100_000 to 1000
        else:
            # TODO: make this m moment
            policies, values = policies_module.collect_iteration_policies(
                1000, 100, ve_policy_mode, num_ve_steps, num_states, num_actions, true_r, true_p, config['gamma'])

    stored_models = []
    # stored_models_path = os.path.join(tune.get_trial_dir(), 'models.pickle')
    stored_models_path = os.path.join(config['dir_to_save'], 'models.pickle')

    for ts in range(1, config['num_iters']+1):
        idx = np.random.randint(0, len(policies), size=[config['batch_size']])

        if not use_m_moment:
            pi_batch = policies[idx, :, :]
            v_batch = values[idx, :]
        else:
            pi_batch = policies[idx, :, :]
            v_batch = values[idx, :, :]

        if not use_m_moment:
            if num_ve_steps == np.inf:
                model_params, state, loss = _update_fpve(model_params, state, pi_batch, v_batch)
            else:
                model_params, state, loss = _update_ve(model_params, state, pi_batch, v_batch)
        else:
            if num_ve_steps == np.inf:
                model_params, state, loss = _update_mmpve(model_params, state, pi_batch, v_batch)
            else:
                model_params, state, loss = _update_mmve(model_params, state, pi_batch, v_batch)

        to_report = {}

        if ts % config['store_model_every'] == 0:
            r, p = params_to_model(model_params, config)
            stored_models.append((ts, r, p))
            to_report['ts'] = ts

        if ts % config['store_loss_every'] == 0:
            to_report['loss'] = loss
            to_report['ts'] = ts

        if ts % config['eval_model_every'] == 0:
            r, p = params_to_model(model_params, config)
            r, p = np.array(r), np.array(p)
            _, pi = vpi.run_value_iteration(
                config['gamma'], r, p, np.zeros([num_states]), threshold=1e-4, return_policy=True)
            value_pi = vpi.exact_policy_evaluation(config['gamma'], pi, true_r, true_p)
            to_report['mean_value'] = np.mean(value_pi)
            to_report['ts'] = ts
        
        if len(stored_models) > 0 and ts == (config['num_iters'] - 1):
            with open(stored_models_path, 'wb+') as f:
                pickle.dump(stored_models, f)
            to_report['model_path'] = stored_models_path
        if len(to_report) > 0:
            # tune.report(**to_report)
            print(to_report)


if __name__ == '__main__':

    model_capacity_search_space = {
        'seed': tune.randint(0, 500_000),
        'gamma': 0.99,
        'batch_size': 50,
        've_mode': tune.grid_search([(np.inf, 'stoch'), (np.inf, 'det')]),
        'model_rank': tune.grid_search([20, 30, 40, 50, 60, 70, 80, 90, 100, 104]),
        'learning_rate': 5e-4,
        'eval_model_every': 10_000,
        'store_loss_every': 10_000,
        'num_iters': 1000,
        'restrict_capacity': True,
        'store_model_every': np.inf,
        'uni_init': 5,
        'use_vip': True,
        'm_moment': False,
        'sas': False,
    }
   
    # n_step_search = [1, 30, 40, 50, 60, np.inf]
    # n_step_search = [1, 50, np.inf]
    n_step_search = [1, 5, 10, 20]
    m_search = [2, 4, 6, 8]

    diameter_search_space = {
        # 'seed': tune.randint(0, 500_000),
        'gamma': 0.99,
        'batch_size': 50,
        # 've_mode': tune.grid_search([(i, 'det_and_stoch') for i in n_step_search]),
        # 've_mode': (np.inf, 'det_and_stoch'),
        've_mode': (np.inf, 'det_and_stoch'),
        'model_rank': 104,
        'learning_rate': 1e-3,
        'eval_model_every': np.inf,
        'store_loss_every': 100,
        'num_iters': 1_000_000,
        'restrict_capacity': False,
        'store_model_every': 1_000,
        'uni_init': 5,
        'use_vip': False,
        'm_moment': False,
        'sas': False,
    }

    moment_diameter_search_space = {
        # 'seed': tune.randint(0, 500_000),
        'gamma': 0.99,
        'batch_size': 50,
        # 'm': tune.grid_search([m for m in m_search]),
        'm': 2,
        # 've_mode': tune.grid_search([(i, 'det_and_stoch') for i in n_step_search]),
        # 've_mode': (np.inf, 'det_and_stoch'),
        've_mode': (10, 'det_and_stoch'),
        'model_rank': 104,
        'learning_rate': 1e-3,
        'eval_model_every': np.inf,
        'store_loss_every': 100,
        'num_iters': 1_000_000,
        'restrict_capacity': False,
        'store_model_every': 1_000,
        'uni_init': 5,
        'use_vip': False,
        'm_moment': True,
        'sas': True,
    }

    mode = sys.argv[1]
    local_dir = sys.argv[2]


    assert mode in ['diameter', 'capacity', 'm_moment_diameter']

    # k = moment_diameter_search_space['ve_mode'][0]
    k = diameter_search_space['ve_mode'][0]

    dir_to_save = os.path.join(local_dir, f'{mode}_k={k}')

    moment_diameter_search_space['dir_to_save'] = dir_to_save

    diameter_search_space['dir_to_save'] = dir_to_save

    run_experiment(diameter_search_space)




