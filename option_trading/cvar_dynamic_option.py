import pickle,time,sys,os
import numpy as np
import torch
import gym
from replay_buffer import ReplayBuffer
import hashlib
import matplotlib.pyplot as plt
import dqr_policy
from option_env import OptionEnv
from dqr_policy import DQR_Policy


def run_train():
    if not os.path.exists('./res'):
        os.mkdir('./res')
    for seed in [1]:#,2,3]: #[int(sys.argv[1])]:
        for n_cvar in [10]: #,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]: #[int(sys.argv[2])]:
            for stock_i in range(10):
                tr_env=OptionEnv(tr_all_mean[stock_i],tr_all_std[stock_i],force_start=0)
                val_env=OptionEnv(tr_all_mean[stock_i],tr_all_std[stock_i],force_start=0)
                tst_env=OptionEnv(tr_all_mean[stock_i],tr_all_std[stock_i],force_start=0)

                out_prefix='res/dqr_option_res{}_{:g}_{}_{}_{}_{}'.format(stock_i,alpha,max_iter,n_quantiles,n_cvar,seed)
                pol=DQR_Policy(alpha,gamma,seed,tr_env.observation_space.shape[0],tr_env.action_space.n,n_quantiles,n_cvar)
                res=pol.train(epsilon,buffer_size,batch_size,seed,tr_env,val_env,tst_env,max_iter,explore_iter,eval_freq,
                              max_episode_len,out_prefix=out_prefix)

                pol.model.cpu()
                dqr_policy.DEVICE='cpu'
                mres=np.load('{}.npy'.format(out_prefix))
                m_idx=np.argsort((mres[:,5]+mres[:,6])/2)[-10:]
                model_file='{}.pth'.format(out_prefix)
                models=torch.load(model_file,map_location='cpu')
                res=[]
                for idx in list(m_idx)+[-1]:
                    model=models[idx]
                    pol.model.load_state_dict(model)
                    tst_env.seed(123)
                    r,_=pol.eval_policy(tst_env,n_rep,max_episode_len,gamma=gamma,s_step=0)
                    res.append( sorted(r) )
                np.save('{}_eval.npy'.format(out_prefix), np.array(res))

                pol.model.cuda()
                dqr_policy.DEVICE='cuda'
                

    

np.set_printoptions(5,suppress=True,linewidth=150)

### stocks=['AAPL','UNH','HD','GS','V','MCD','MSFT','BA','MMM','JNJ']
data=np.load('dow_top10_2005_2019.npy')
val_st_idx=1962 ### train / test split on Jan 1, 2016
z=np.log(data[:,1:val_st_idx,3]/data[:,:val_st_idx-1,3])
z2=np.log(data[:,val_st_idx+1:,3]/data[:,val_st_idx:-1,3])
tr_all_mean=np.mean(z,axis=1)
tr_all_std=np.std(z,axis=1)
tst_all_mean=np.mean(z2,axis=1)
tst_all_std=np.std(z2,axis=1)

max_episode_len=999

alpha=0.0001
gamma=0.999 #discount
epsilon=0.02 #exploration
buffer_size=20000
batch_size=32
n_quantiles=100

max_iter=200000
explore_iter=20000
eval_freq=1000
n_rep=1000

##### To train and save outputs
run_train()

