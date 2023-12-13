import os,sys,time
import gym
import numpy as np
import dqr_policy
from dqr_policy import DQR_Policy
import dqr4d_policy
from dqr4d_policy import DQR4D_Policy
import torch
import matplotlib.pyplot as plt
from matplotlib import patches

def ptSegDist(x1,y1,x2,y2,px,py):
    pd2 = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
    if pd2 == 0:
        x = x1
        y = y2
    else:
        u = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / pd2;
        if u < 0:
            x = x1
            y = y1
        elif u > 1.0:
            x = x2
            y = y2
        else:
            x = x1 + u * (x2 - x1)
            y = y1 + u * (y2 - y1)
    return np.sqrt( (x - px) * (x - px) + (y - py) * (y - py) )

 
class PuddleEnv:
    def __init__(self):
        self.action_space=gym.spaces.discrete.Discrete(4)
        self.observation_space=gym.spaces.box.Box(0,1,shape=(2,),dtype='float64')
        self.randgen=np.random.RandomState()
        self.puddle=np.array([ [0.2, 1, 0.2, 0.75, 0.1], [0.5, 1, 0.5, 0.5, 0.1], [0.8, 1, 0.8, 0.25, 0.1] ])
        self.init_s=np.array([0.1, 0.95])

    def seed(self,s):
        self.randgen.seed(s)

    def reset(self,ix=None):
        if ix is not None and ix==0:
            self.state=np.array(self.init_s)
        else:
            self.state=self.randgen.rand(2)
            while self.state[0]>=0.9 and self.state[1]>=0.9:
                self.state=self.randgen.rand(2)
        return self.state.copy()

    def step(self,act):
        act=act%4
        if act==0:
            self.state[0]+=self.randgen.randn()*0.01
            self.state[1]+=0.05+self.randgen.randn()*0.01
        elif act==1:
            self.state[0]+=self.randgen.randn()*0.01
            self.state[1]-=0.05+self.randgen.randn()*0.01
        elif act==2:
            self.state[1]+=self.randgen.randn()*0.01
            self.state[0]+=0.05+self.randgen.randn()*0.01
        else:
            self.state[1]+=self.randgen.randn()*0.01
            self.state[0]-=0.05+self.randgen.randn()*0.01
        self.state=np.clip(self.state,0,1)
        if self.state[0]>=0.9 and self.state[1]>=0.9:
            r=0
            done=True
        else:
            r=1.0
            done=False
        for p in self.puddle:
            d=ptSegDist(p[0],p[1],p[2],p[3],self.state[0],self.state[1])
            if d<p[4]:
                if self.randgen.rand()<0.1:
                    r+=300*(p[4]-d)
        r=-r
        return self.state.copy(),r,done,None

    def query(self,s):
        x,y=s
        if x>0.9:
            return 0,None
        if y<0.4:
            return 2,None
        if x>0.8:
            return 2,None
        return 1,None

    
def draw(env,li_traj,fnum=999,title=None):
    for p in env.puddle:
        plt.gca().add_patch(patches.Rectangle([p[0]-p[4],p[3]],2*p[4],p[1]-p[3],color='#aaaaaa'))
        plt.gca().add_patch(patches.Circle([p[2],p[3]],p[4],color='#aaaaaa'))
        plt.plot(p[[0,2]],p[[1,3]],'-',color='#888888')
    plt.gca().add_patch(patches.Rectangle([0.9,0.9],0.1,0.1,color='#ffff55'))
    plt.plot(.1,.95,'x',markersize=20,markeredgewidth=5,color='g')        
    if not isinstance(li_traj,list):
        li_traj=[li_traj]
    for traj in li_traj:
        x=[t[0][0] for t in traj]+[traj[-1][3][0]]
        y=[t[0][1] for t in traj]+[traj[-1][3][1]]
        plt.plot(x,y,'*-')
    plt.axis('square')
    plt.axis([0,1,0,1])
    plt.grid(True)
    if title is not None:
        plt.title(title)


def dqr_rollout(pol,env,max_episode_len=100,gamma=0.99,s_step=0):
    traj=[]
    st=time.time()
    s=env.reset(s_step)
    phi=s.astype('float32')
    ep_r=0
    steps=0
    while steps<max_episode_len:
        a,_=pol.query(phi)
        next_s,r,done,info=env.step(a)
        ep_r+=(gamma**steps)*r
        traj.append((phi,a,r,next_s.astype('float32'),done))
        phi=next_s.astype('float32')
        steps+=1
        if done:
            break
    et=time.time()
    #print('.. done eval_policy: {} steps, {:.1f} secs, ({:.1f} steps/sec)'.format(steps,et-st,steps/(et-st+1e-8)))
    return ep_r,traj

def dqr4d_rollout(pol,env,max_episode_len=100,gamma=0.99,s_step=0):
    traj=[]
    st=time.time()
    s=env.reset(s_step)
    phi=s.astype('float32')
    ep_r=0
    steps=0
    a,cutoff,_=pol.query_init(phi)
    while steps<max_episode_len:
        next_s,r,done,info=env.step(a)
        ep_r+=(gamma**steps)*r
        traj.append((phi,a,r,next_s.astype('float32'),done))
        phi=next_s.astype('float32')
        steps+=1
        if done:
            break
        cutoff=(cutoff-r)/pol.gamma
        a,_=pol.query(phi,cutoff)
    et=time.time()
    #print('.. done eval_policy: {} steps, {:.1f} secs, ({:.1f} steps/sec)'.format(steps,et-st,steps/(et-st+1e-8)))
    return ep_r,traj

if __name__=='__main__':
    np.set_printoptions(5,suppress=True,linewidth=150)

    tr_env=PuddleEnv()
    val_env=PuddleEnv()
    tst_env=PuddleEnv()

    alpha=0.001
    gamma=0.99
    n_quantiles=100
    n_hidden=100

    alg=sys.argv[1]
    seed=int(sys.argv[2])
    n_cvar=int(sys.argv[3])
    
    epsilon=float(sys.argv[4])
    buffer_size=int(sys.argv[5])
    eval_only=int(sys.argv[6])
    batch_size=32
    max_iter=1000000
    explore_iter=500000
    eval_freq=1000
    max_episode_len=50
    out_prefix=None

    if alg=='dqr':
        policy=dqr_policy
        policy.DEVICE='cuda'
        pol=DQR_Policy(alpha,gamma,seed,tr_env.observation_space.shape[0],tr_env.action_space.n,
                       n_quantiles,n_cvar,n_hidden=n_hidden)
    elif alg=='dqr4d':
        policy=dqr4d_policy
        policy.DEVICE='cuda'
        pol=DQR4D_Policy(alpha,gamma,seed,tr_env.observation_space.shape[0],tr_env.action_space.n,
                         n_quantiles,n_cvar,n_hidden=n_hidden)

    if not os.path.exists('./res_puddle2'):
        os.mkdir('./res_puddle2')
    out_prefix='res_puddle2/{}_{:.2f}_{}_{}_{}'.format(alg,epsilon,buffer_size,seed,n_cvar)
    if eval_only==0:
        pol.train(epsilon,buffer_size,batch_size,seed,tr_env,val_env,tst_env,max_iter,explore_iter,eval_freq,
                  max_episode_len,out_prefix=out_prefix,n_rep=1)

    if eval_only==1:
        max_episode_len=100
        
        pol.model.cpu()
        policy.DEVICE='cpu'
        model_file='{}.pth'.format(out_prefix)
        models=torch.load(model_file,map_location='cpu')
        if len(models)<1000:
            raise
        n_rep=1000
        res=[]
        for model in models:
            pol.model.load_state_dict(model)
            tst_env.seed(123)
            r,_=pol.eval_policy(tst_env,n_rep,max_episode_len,gamma=gamma,s_step=0)
            res.append( sorted(r) )
        np.save('{}_eval2b.npy'.format(out_prefix), np.array(res))
        
    if eval_only==2:
        policy.DEVICE='cpu'
        pol.model.cpu()

        t_cvar=20
        max_episode_len=100
        plt.ion()
        plt.rc('font',size=16)
        if seed==1:
            fig=plt.figure((1000+n_cvar) if alg=='dqr' else 1010)
            fig.set_figwidth(18)
            fig.set_figheight(6)
            fig.set_tight_layout(True)
            plt.clf()
        
        res_eval2b=np.load('{}_eval2b.npy'.format(out_prefix))
        m_idx=np.argmax(np.mean(res_eval2b[:,:n_cvar*10],axis=1))
        model_file='{}.pth'.format(out_prefix)
        models=torch.load(model_file,map_location='cpu')
        pol.model.load_state_dict(models[m_idx])
        n_rep=1000
        tst_env.seed(123)
        if alg=='dqr':
            res=[dqr_rollout(pol,tst_env,max_episode_len,s_step=0) for _ in range(n_rep)]
        elif alg=='dqr4d':
            res=[dqr4d_rollout(pol,tst_env,max_episode_len,s_step=0) for _ in range(n_rep)]
        else:
            raise
        r=sorted([tup[0] for tup in res])
        print(np.mean(r[:round(t_cvar*n_rep/100)]),np.mean(r))
        idx=np.argsort([tup[0] for tup in res])
        plt.subplot(1,3,seed)
        draw(tst_env,[tup[1] for tup in res[:100]],
             title='0.2-CVaR: {:.1f}, Avg: {:.1f}'.format(np.mean(r[:round(t_cvar*n_rep/100)]),np.mean(r)))
        print(np.array(r[:10]))

