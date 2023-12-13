import pickle,time,sys,os
import numpy as np
import gym
from dqr_policy import DQR_Policy
from dqr4d_policy import DQR4D_Policy
from dqr_policy import LinNet
import matplotlib.pyplot as plt

class FiniteEnv:
    def __init__(self,P,terminal=None,start_state=None): #P shape: states x actions x types x outcomes
        n_states=len(P)
        self.action_space=gym.spaces.discrete.Discrete(len(P[0]))
        self.observation_space=gym.spaces.box.Box(0,1.0,shape=(n_states,),dtype='float64')
        self.randgen=np.random.RandomState()
        self.n_states=n_states
        self.P=P
        self.terminal=-1 if terminal is None else terminal
        self.start_state=start_state #force start state

    def seed(self,s):
        self.randgen.seed(s)

    def reset(self,s_state=None):
        if self.start_state is not None:
            self.state=self.start_state
        elif s_state is not None:
            self.state=s_state%self.n_states
        else:
            self.state=0
        obs=np.zeros(self.n_states)
        obs[self.state]=1.0
        return obs

    def step(self,act):
        act=act%len(self.P[0])
        outcomes=self.P[self.state][act]
        p=outcomes[0]
        idx=self.randgen.choice(len(p),1,p=p)[0]
        r=outcomes[2][idx]
        self.state=outcomes[1][idx]
        if self.state==self.terminal:
            done=True
        else:
            done=False
        obs=np.zeros(self.n_states)
        obs[self.state]=1.0
        return obs,r,done,None

def rollout(P,terminal,N,pol,gamma,seed=None,start_state=None):
    env=FiniteEnv(P,terminal)
    if seed is not None:
        env.seed(seed)
    returns=[]
    for _ in range(N):
        obs=env.reset(start_state)
        ret=0
        i=0
        while True:
            a=round(pol@obs)
            obs,r,done,_=env.step(a)
            ret+=(gamma**i) * r
            i+=1
            if done:
                break
        returns.append(ret)
    return returns


def train_policy(alg,n_cvar,Penv,n_round,round_iter):
    alpha=0.001
    gamma=0.999
    seed=1
    n_quantiles=100
    
    epsilon=0.02
    buffer_size=20000
    batch_size=32
    explore_iter=20000
    eval_freq=1000
    max_episode_len=999
    out_prefix=None
    
    tr_env=FiniteEnv(Penv,terminal=0,start_state=1)
    val_env=FiniteEnv(Penv,terminal=0,start_state=1)
    tst_env=FiniteEnv(Penv,terminal=0,start_state=1)
    env=FiniteEnv(Penv,terminal=0,start_state=1)
    

    if alg=='dqr':
        pol=DQR_Policy(alpha,gamma,seed,tr_env.observation_space.shape[0],tr_env.action_space.n,n_quantiles,n_cvar,Net=LinNet)
    elif alg=='dqr4d':
        pol=DQR4D_Policy(alpha,gamma,seed,tr_env.observation_space.shape[0],tr_env.action_space.n,n_quantiles,n_cvar,Net=LinNet)
    else:
        raise

    all_z=[]
    all_res=[]
    all_eval=[]
    max_iter=round_iter
    env.seed(123)
    for j in range(n_round):
        res=pol.train(epsilon,buffer_size,batch_size,seed,tr_env,val_env,tst_env,max_iter,explore_iter,eval_freq,
                      max_episode_len,out_prefix=out_prefix,n_rep=1)

        all_z.append([ pol.model.fc1.weight[:,1].view(-1,2,100).squeeze(0).detach().cpu().numpy(),
                       pol.model.fc1.weight[:,2].view(-1,2,100).squeeze(0).detach().cpu().numpy() ])
        all_res.append(res)

        r,d=pol.eval_policy(env,1000,999,gamma=gamma)
        all_eval.append( np.mean(sorted(r)[:round(n_cvar*1000/n_quantiles)]) )
            
        seed=None
        max_iter+=round_iter
    return all_z,all_res,all_eval


def plot_z(fnum,all_z,round_iter):
    fig=plt.figure(fnum)
    fig.set_figwidth(12)
    fig.set_figheight(19)
    fig.set_tight_layout(True)
    plt.clf()

    n_quantiles=len(all_z[0][0][0])
    q_levels=np.linspace(0.5/n_quantiles,1-0.5/n_quantiles,n_quantiles)
    
    n=len(all_z)
    for j in range(n):
        plt.subplot(n,2,j*2+1)
        z=all_z[j][0]
        plt.plot(q_levels,z[0],'*')
        plt.plot(q_levels,z[1],'*')
        plt.grid()
        plt.title('State X1, # steps={}'.format((j+1)*round_iter))
        if j==0 or j==n-1:
            plt.xlabel('Quantile level')
        plt.subplot(n,2,j*2+2)
        z=all_z[j][1]
        plt.plot(q_levels,z[0],'*')
        plt.plot(q_levels,z[1],'*')
        plt.grid()
        plt.title('State X2, # steps={}'.format((j+1)*round_iter))
        if j==0 or j==n-1:
            plt.xlabel('Quantile level')

            
def get_Penv(mdp,p,epsilon=None):
    if mdp=='1':
        x11=[ [p,1-p], [2,3], [0,0] ]
        x12=[ [1], [0], [epsilon] ]
        x21=x22=[ [1-p,p], [0,0], [1,0] ]
        x31=x32=[ [1], [0], [1] ]
        x01=x02=[ [1], [0], [0] ]
        P=[ [x01,x02], [x11,x12], [x21,x22], [x31,x32] ]
        # pol=[0, 0, 0, 0]
        # #ret=rollout(P1,0,1000,pol,1,seed=1,start_state=1)
    elif mdp=='1c':
        x11=x12=[ [p,1-p], [2,0], [0,1] ]
        x21=[ [1-p,p], [0,0], [1,0] ]
        x22=[ [1], [0], [epsilon] ]
        x01=x02=[ [1], [0], [0] ]
        P=[ [x01,x02], [x11,x12], [x21,x22] ]
        # pol=[0,0,1]
        # #ret=rollout(P2,0,1000,pol,1,seed=np.random.randint(1000),start_state=1)
    elif mdp=='2':
        x11=x12=[ [1-p,p], [2,2], [1,0] ]
        x21=[ [1-p,p], [0,0], [2,0] ]
        x22=[ [1], [0], [1] ]
        x01=x02=[ [1], [0], [0] ]
        P=[ [x01,x02], [x11,x12], [x21,x22] ]
    else:
        raise
    return P
    
if __name__=='__main__':
    np.set_printoptions(5,suppress=True,linewidth=150)

    plt.ion()
    plt.rc('font',size=16)


    ######### For Figure 5 #############################
    epsilon=0.01
    p_rng=np.arange(0.1,0.9,0.1)
    cvar_rng=[round(90*p) for p in p_rng]

    dqr_z1={}
    dqr_eval1={}
    dqr4d_z1={}
    dqr4d_eval1={}
    for n_cvar,p in zip(cvar_rng,p_rng):
        Penv=get_Penv('1c',p,epsilon)

        z,res,ev=train_policy('dqr',n_cvar,Penv,10,5000)
        dqr_z1[n_cvar]=z
        dqr_eval1[n_cvar]=ev
        z,res,ev=train_policy('dqr4d',n_cvar,Penv,10,5000)
        dqr4d_z1[n_cvar]=z
        dqr4d_eval1[n_cvar]=ev

    fig=plt.figure(60)
    fig.set_tight_layout(True)
    plt.clf()
    opt=0.999*(0.9*p_rng-p_rng**2)/(0.9*p_rng)
    for i,n_cvar in enumerate(cvar_rng):
        plt.subplot(len(cvar_rng),1,i+1)
        plt.plot(np.arange(5000,55000,5000),dqr_eval1[n_cvar],'*-',label='Markov action-selection')
        plt.plot(np.arange(5000,55000,5000),dqr4d_eval1[n_cvar],'*-',label='Proposed')
        plt.plot([5000,50000],[opt[i],opt[i]],'--',label='Optimal')
        plt.xlabel('# training steps')
        plt.ylabel('p-CVaR')
        plt.title('p = {:.1f}'.format(p_rng[i]))
        if i==0:
            plt.legend(loc='center left')
    ##########################################################
    

    ######### For Figures 6,7,8 #############################
    p_rng=[0.1,0.2,0.3,0.4]
    cvar_rng=[10,20,30,40]

    dqr_z={}
    dqr_eval={}
    dqr4d_z={}
    dqr4d_eval={}
    for n_cvar,p in zip(cvar_rng,p_rng):
        Penv=get_Penv('2',p)
        
        z,res,ev=train_policy('dqr',n_cvar,Penv,10,5000)
        dqr_z[n_cvar]=z
        dqr_eval[n_cvar]=ev
        z,res,ev=train_policy('dqr4d',n_cvar,Penv,10,5000)
        dqr4d_z[n_cvar]=z
        dqr4d_eval[n_cvar]=ev

    for n_cvar,p in zip([20],[0.2]): #pick one alpha for further inspection
        Penv=get_Penv('2',p)
        
        z,res,ev=train_policy('dqr',n_cvar,Penv,10,500)
        dqr_z20=z
        z,res,ev=train_policy('dqr4d',n_cvar,Penv,10,500)
        dqr4d_z20=z
        

    plot_z(51,dqr_z[20],5000)
    plot_z(52,dqr4d_z[20],5000)
    plot_z(53,dqr_z20,500)
    plot_z(54,dqr4d_z20,500)

    fig=plt.figure(50)
    fig.set_tight_layout(True)
    plt.clf()
    opt=2*0.999*(1-np.array(p_rng))
    for i,n_cvar in enumerate(cvar_rng):
        plt.subplot(len(cvar_rng),1,i+1)
        plt.plot(np.arange(5000,55000,5000),dqr_eval[n_cvar],'*-',label='Markov action-selection')
        plt.plot(np.arange(5000,55000,5000),dqr4d_eval[n_cvar],'*-',label='Proposed')
        plt.plot([5000,50000],[opt[i],opt[i]],'--',label='Optimal')
        plt.xlabel('# training steps')
        plt.ylabel('p-CVaR')
        plt.title('p = {:.1f}'.format(p_rng[i]))
        if i==0:
            plt.legend(loc='center left')
    ##########################################################
    
        

        
