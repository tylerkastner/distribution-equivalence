import os,sys,pickle
import torch
import numpy as np
import gym
import matplotlib.pyplot as plt
import seaborn as sea
import pandas as pd
import dqr_policy
import dqr4d_policy
from dqr_policy import DQR_Policy
from dqr4d_policy import DQR4D_Policy

dqr_policy.DEVICE='cpu'
dqr4d_policy.DEVICE='cpu'

class RealOptionEnv:
    def __init__(self,data):
        self.action_space=gym.spaces.discrete.Discrete(2)
        self.observation_space=gym.spaces.box.Box(-np.inf,np.inf,shape=(2,),dtype='float64')
        self.T=100
        self.K=1
        self.s0=1
        self.data=data/data[0]

    def seed(self,s):
        pass

    def get_obs(self,s,t):
        return np.array([s,(self.T-t)/self.T])

    def reset(self,s_step=None):
        self.state_t=0
        self.state_s=self.s0
        return self.get_obs(self.state_s,self.state_t)

    def step(self,act):
        act=act%2
        if self.state_t==self.T-2:
            act=1
        if self.state_t<self.T-1:
            self.state_t+=1
        if act>0:
            r=np.maximum(0,self.K-self.state_s)
            done=True
        else:
            r=0
            done=False
        self.state_s=self.data[self.state_t]
        return self.get_obs(self.state_s,self.state_t),r,done,None

    
def load_pol(seed,n_cvar,env,Policy):
    q_alpha = 0.0001
    gamma = 0.999
    n_obs = env.observation_space.shape[0]
    pol=Policy(q_alpha,gamma,seed,n_obs,env.action_space.n,100,n_cvar)
    return pol


def eval_all(alg,n_cvar_rng):
    Policy=DQR_Policy if alg=='dqr' else DQR4D_Policy
    all_res={}
    for n_cvar in n_cvar_rng:
        this_res=[]
        for stock_i in range(10):
            tr_env=RealOptionEnv(data[stock_i,val_st_idx:val_st_idx+100,3])
            env_name='option_res{}_0.0001_200000_100'.format(stock_i)
            res_path='res/{}_{}_{}'.format(alg,env_name,n_cvar)
            res=np.array([np.load('{}_{}_eval.npy'.format(res_path,s)) for s in [1,2,3]])
            if res.shape[0]!=3 or res.shape[2]!=1000:
                raise
            best_idx= np.argmax( np.mean(res[:,:,:round(n_cvar*1000/100)],axis=2), axis=1 )
            sres=[]
            for seed in [1,2,3]:
                pol=load_pol(seed,n_cvar,tr_env,Policy)

                mres=np.load('{}_{}.npy'.format(res_path,seed))
                m_idx=list(np.argsort((mres[:,5]+mres[:,6])/2)[-10:])+[-1]
                model_file='{}_{}.pth'.format(res_path,seed)
                models=torch.load(model_file,map_location='cpu')
                pol.model.load_state_dict( models[ m_idx[best_idx[seed-1]] ] )
                res=[]
                print('Evaluating {} stock {} cvar {} seed {} ...'.format(alg,stock_i,n_cvar,seed),flush=True)
                for offset in np.arange(0,1000,100):
                    tst_env=RealOptionEnv(data[stock_i,val_st_idx+offset:val_st_idx+offset+100,3])
                    r,_=pol.eval_policy(tst_env,1,max_episode_len,gamma=gamma,s_step=0)
                    res.append(r)
                res=np.sort(np.array(res).flatten())
                sres.append(res)
            this_res.append(np.array(sres))
        all_res[n_cvar]=np.array(this_res)
    return all_res

def extract_res(li_res,cvar_rng):
    res=[]
    for n_cvar in cvar_rng:
        m= np.mean(li_res[n_cvar][:,:,:round(n_cvar*n_rep/100)],axis=2)
        res.append( np.mean( m,axis=1 ) )
    return np.array(res)

def extract_res_exp(li_res,cvar_rng):
    res_exp=li_res[100]
    res=[]
    for n_cvar in cvar_rng:
        m= np.mean(res_exp[:,:,:round(n_cvar*n_rep/100)],axis=2)
        res.append( np.mean( m,axis=1) )
    return np.array(res)


if __name__=='__main__':
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
    gamma=0.999
    n_quantiles=100

    plt.ion()

    dqr_cvar_rng=np.arange(10,110,10)
    dqr4d_cvar_rng=np.arange(10,100,10)
    
    res_dqr=eval_all('dqr',dqr_cvar_rng)
    res_dqr4d=eval_all('dqr4d',dqr4d_cvar_rng)
    # with open('res_option_eval_real3.pickle','wb') as fp:
    #     pickle.dump(res_dqr,fp)
    #     pickle.dump(res_dqr4d,fp)

    # with open('res_option_eval_real3.pickle','rb') as fp:
    #     res_dqr=pickle.load(fp)
    #     res_dqr4d=pickle.load(fp)

    n_rep=10
    colors=plt.rcParams['axes.prop_cycle'].by_key()['color'] #'rgbmyck'
    specs=[(c,'*-') for c in colors]*3
    specs2=[(c,'*--') for c in colors]*3

    all_res_exp=extract_res_exp(res_dqr,dqr_cvar_rng)
    all_res_dqr=extract_res(res_dqr,dqr_cvar_rng)
    all_res_dqr4d=extract_res(res_dqr4d,dqr4d_cvar_rng)
    
    
    fig=plt.figure(2)
    fig.set_tight_layout(True)
    plt.clf()
    pd_exp=pd.melt(pd.DataFrame(all_res_exp.T,columns=dqr_cvar_rng/100),var_name='alpha level',value_name='CVaR')
    pd_dqr=pd.melt(pd.DataFrame(all_res_dqr.T,columns=dqr_cvar_rng/100),var_name='alpha level',value_name='CVaR')
    pd_dqr4d=pd.melt(pd.DataFrame(all_res_dqr4d.T,columns=dqr4d_cvar_rng/100),var_name='alpha level',value_name='CVaR')
    pd_all=pd.concat([pd_exp,pd_dqr,pd_dqr4d],keys=['Risk-neutral','Markov action-selection','Proposed algorithm'],names=['Algorithm'])
    sea.set_theme(style="darkgrid")
    sea.set(font_scale=2)
    h=sea.lineplot(data=pd_all,x='alpha level',y='CVaR',hue='Algorithm',style='Algorithm',markers=True,seed=1)
    h.legend_.set_title(None)
