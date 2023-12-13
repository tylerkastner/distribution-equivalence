import pickle,time,sys,os
import numpy as np
import gym

class OptionEnv:
    def __init__(self,m,s,force_start=None):
        self.action_space=gym.spaces.discrete.Discrete(2)
        self.observation_space=gym.spaces.box.Box(-np.inf,np.inf,shape=(2,),dtype='float64')
        self.randgen=np.random.RandomState()
        self.mean=m
        self.std=s
        self.T=100
        self.K=1
        self.s0=1
        self.force_start=force_start

    def seed(self,s):
        self.randgen.seed(s)

    def get_obs(self,s,t):
        return np.array([s,(self.T-t)/self.T])

    def reset(self,s_step=None):
        if self.force_start is not None:
            self.state_t=self.force_start
        elif s_step is not None:
            self.state_t=s_step
        else:
            self.state_t=self.randgen.randint(self.T-2)
        self.state_s=self.s0
        self.state_s=self.state_s*np.exp(self.mean*self.state_t+self.std*np.sqrt(self.state_t)*self.randgen.randn())
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
        self.state_s=self.state_s*np.exp(self.mean+self.std*self.randgen.randn())
        return self.get_obs(self.state_s,self.state_t),r,done,None

    
