import sys
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import rcParams
from matplotlib.collections import LineCollection

import value_and_policy_iteration as vpi
import read_experiment_data as dd
import gridworld
import frozen_lake
import windy_cliff

import ray
import tqdm
import pickle
import random
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable


env = gridworld.FourRooms(p_intended=0.8)
# env = frozen_lake.EightByEight()
# env = windy_cliff.CliffWalk()



true_r = env.get_reward_matrix()
true_r_sas = env.get_sas_reward_matrix()

true_p = env.get_transition_tensor()
num_states, num_actions = np.shape(true_r)
gamma = 0.99
r_max = 2


def set_tickless(ax):
    ax.tick_params(
        axis='x',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False)
    ax.tick_params(
        axis='y',
        which='both',
        left=False,
        right=False,
        labelleft=False)


def get_model_from_vector(vector):
    flat_r_len = np.product(np.shape(true_r))
    flat_r_sas_len = np.product(np.shape(true_r_sas))

    if len(vector) - flat_r_sas_len == flat_r_len:
        flat_r, flat_p = vector[:flat_r_len], vector[flat_r_len:]
        r = np.reshape(flat_r, np.shape(true_r))
    else:
        flat_r, flat_p = vector[:flat_r_sas_len], vector[flat_r_sas_len:]
        r = np.reshape(flat_r, np.shape(true_r_sas))

    p = np.reshape(flat_p, np.shape(true_p))
    return r, p


def get_model_vec_from_file_path(path):
    model_vecs = []
    with open(path, 'rb') as f:
        model_data = pickle.load(f)

    for ts, r, p in model_data:
        model_vecs.append(np.concatenate([np.reshape(r, [-1]), np.reshape(p, [-1])], axis=0))
    model_vecs = np.array(model_vecs)

    return model_vecs


mmve_model_vecs = get_model_vec_from_file_path('filepath')
pve_model_vecs = get_model_vec_from_file_path('filepath')
pve_model_vecs = get_model_vec_from_file_path('filepath')

last_mmve_model_vec = mmve_model_vecs[-1]
last_pve_model_vec = pve_model_vecs[-1]
# last_one_step_mm_model_vec = one_step_mm_model_vecs[-1]

# last_windy_pve_model_vec = windy_pve_model_vecs[-1]
# last_windy_mmve_model_vec = windy_mmve_model_vecs[-1]


# last_frozen_lake_mmve_model_vec = frozen_lake_mmve_model_vecs[-1]
# last_frozen_lake_pve_model_vec = frozen_lake_pve_model_vecs[-1]

mmve_model = get_model_from_vector(last_mmve_model_vec)
pve_model = get_model_from_vector(last_pve_model_vec)
# last_one_step_mm_model = get_model_from_vector(last_one_step_mm_model_vec)

# windy_pve_model = get_model_from_vector(last_windy_pve_model_vec)
# windy_mmve_model = get_model_from_vector(last_windy_mmve_model_vec)

# frozen_lake_mmve_model = get_model_from_vector(last_frozen_lake_mmve_model_vec)
# frozen_lake_pve_model = get_model_from_vector(last_frozen_lake_pve_model_vec)

def visualize_trajs(ax, start_pos, model, env, pi, num_trajs=100, traj_len=100):
    r, p = model
    if len(r.shape) == 2:
        num_states, num_actions = r.shape
    else:
        num_states, num_actions, _ = r.shape
    ppi = np.sum(pi[:, :, None] * p, axis=1) # [s, s]
    ppi = ppi / np.sum(ppi, axis=1)[:, None]


    environment_reward_matrix = env.get_sas_reward_matrix()

    all_traj = []
    all_traj_returns = []
    for _ in tqdm.tqdm(range(num_trajs)):
        traj = []
        traj_return = 0.
        pos = start_pos
        s = env._free_tile_mapping[pos]
        traj.append(pos)
        for _ in range(traj_len-1):
            prev_s = s

            action = np.argmax(pi[s])

            s = np.random.choice(num_states, p=p[s, action])
            pos = env._s_to_xy[s]

            reward = environment_reward_matrix[prev_s, action, s]
            traj_return += reward
            traj.append(pos)


            if s in env._final_states:
                break

        all_traj.append(traj)
        all_traj_returns.append(traj_return)

    # cmap = colors.ListedColormap(['white', 'gray', 'green', 'paleturquoise'])
    # bounds = [0,1,2,3, 4]
    # norm = colors.BoundaryNorm(bounds, cmap.N)
    # im = ax.imshow(env.visualize(), cmap=cmap, norm=norm)

    # for traj in tqdm.tqdm(all_traj):
    #     traj = np.array(traj)
    #     xs, ys = zip(*traj)
    #     # perc = (np.arange(len(xs), dtype=np.float32) + 1) / len(xs)
    #     # lc = LineCollection(zip(traj[:-1], traj[1:]), array=perc, cmap=plt.cm.plasma, alpha=0.02)

    #     lc = LineCollection(zip(traj[:-1], traj[1:]), cmap=plt.cm.plasma, alpha=0.02)

    #     # lc = LineCollection(zip(traj[:-1], traj[1:]), cmap=plt.cm.plasma)#, alpha=0.02)

    #     ax.add_collection(lc)
    #     ax.margins(0.1)
    #     #ax.plot(xs, ys, c=perc, cmap=traj_cmap)
    # set_tickless(ax)

    # ax.scatter([start_pos[0]], [start_pos[1]], c='r', s=150)

    print(all_traj_returns)


# model = fpve_model
# model = ve_model
# _, pi = vpi.run_value_iteration(
    # gamma, env.get_reward_matrix(), env.get_transition_tensor(), np.zeros([num_states]), return_policy=True)
# scale = 6
# #pi = 0.2 * (np.ones_like(pi) / num_actions) + 0.8 * pi
# f, ax = plt.subplots(1, 1, figsize=(scale, scale))
# true_model = (env.get_reward_matrix(), env.get_transition_tensor())
# visualize_trajs(ax, (1, 11), true_model, env, pi, num_trajs=5000, traj_len=30)
# ax.set_title(r'{\huge Environment}')
# plt.tight_layout()
# f.savefig('./visuals/opt_traj_environment.png')
# f.savefig('./visuals/opt_traj_environment.pdf')

scale = 6
f, ax = plt.subplots(1, 1, figsize=(scale, scale))
true_model = (env.get_reward_matrix(), env.get_transition_tensor())


# _, pi = vpi.run_value_iteration(
#     gamma, env.get_reward_matrix(), env.get_transition_tensor(), np.zeros([num_states]), return_policy=True)

# _, pi = vpi.run_value_iteration(
#     gamma, mmve_model[0], mmve_model[1], np.zeros([num_states]), return_policy=True)

# _, pi = vpi.run_value_iteration(
#     gamma, pve_model[0], pve_model[1], np.zeros([num_states]), return_policy=True)


# _, pi = vpi.run_value_iteration(
#     gamma, windy_pve_model[0], windy_pve_model[1], np.zeros([num_states]), return_policy=True)


_, pi = vpi.run_m_moment_value_iteration(gamma, env.get_sas_reward_matrix(), env.get_transition_tensor(), return_policy=True)

# _, pi = vpi.run_m_moment_value_iteration(gamma, mmve_model[0], mmve_model[1], return_policy=True)
# _, pi = vpi.run_m_moment_value_iteration(gamma, pve_model[0], pve_model[1], return_policy=True)


# _, pi = vpi.run_m_moment_value_iteration(gamma, frozen_lake_pve_model[0], frozen_lake_pve_model[1], return_policy=True)

# _, pi = vpi.run_m_moment_value_iteration(gamma, frozen_lake_mmve_model[0], frozen_lake_mmve_model[1], return_policy=True)
# _, pi = vpi.run_m_moment_value_iteration(gamma, pve_model[0], pve_model[1], return_policy=True)

# _, pi = vpi.run_m_moment_value_iteration( gamma, windy_pve_model[0], windy_pve_model[1], return_policy=True)
# _, pi = vpi.run_m_moment_value_iteration(gamma, windy_mmve_model[0], windy_mmve_model[1], return_policy=True)


# _, pi = vpi.run_m_moment_value_iteration(gamma, last_one_step_mm_model[0], last_one_step_mm_model[1], return_policy=True)
# last_one_step_mm_model

visualize_trajs(ax, (1, 11), mmve_model, env, pi, num_trajs=20000, traj_len=1000)
# visualize_trajs(ax, (1, 11), true_model, env, pi, num_trajs=5000, traj_len=1000)


# FrozenLake
# visualize_trajs(ax, (0, 0), true_model, env, pi, num_trajs=20000, traj_len=1000)



# Windy
# visualize_trajs(ax, (0, 0), true_model, env, pi, num_trajs=20000, traj_len=1000)


# ax.set_title(r'{\huge PVE Model}')
# plt.tight_layout()
plt.plot()
plt.show()
# f.savefig('./visuals/opt_traj_pve.png')
# f.savefig('./visuals/opt_traj_pve.pdf')
