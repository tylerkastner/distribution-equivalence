import pickle,time,sys,os
import numpy as np
import gym


class LadderEnv:
    def __init__(self,n_states):
        self.action_space=gym.spaces.discrete.Discrete(2)
        self.observation_space=gym.spaces.box.Box(0,1.0,shape=(n_states,),dtype='float64')
        self.randgen=np.random.RandomState()
        self.n_states=n_states
        self.mean=[1.0, 0.8]
        self.std=[1.0, 0.4]

    def seed(self,s):
        self.randgen.seed(s)

    def reset(self,s_state=None):
        if s_state is not None:
            self.state=s_state%self.n_states
        else:
            self.state=0
        obs=np.zeros(self.n_states)
        obs[self.state]=1.0
        return obs

    def step(self,act):
        act=act%2
        if self.state<self.n_states-1:
            r=self.randgen.randn()*self.std[act]+self.mean[act]
            self.state+=1
        else:
            r=0
        if self.state>=self.n_states-1:
            done=True
        else:
            done=False
        obs=np.zeros(self.n_states)
        obs[self.state]=1.0
        return obs,r,done,None

def rollout(start_state,n_states,N,pol,gamma,seed=None):
    env=LadderEnv(n_states)
    if seed is not None:
        env.seed(seed)
    returns=[]
    for _ in range(N):
        obs=env.reset(start_state)
        ret=0
        i=0
        while True:
            a=int(pol@obs)
            obs,r,done,_=env.step(a)
            ret+=(gamma**i) * r
            i+=1
            if done:
                break
        returns.append(ret)
    return returns
