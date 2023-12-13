import pickle,time,sys,os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gym
from replay_buffer import ReplayBufferCutoff
import hashlib
import matplotlib.pyplot as plt
from ladder_env import LadderEnv

TARGET_DTYPE = torch.float
DEVICE = 'cuda'
    
def one_hot(n,a):
    x=np.zeros(n,dtype='float32')
    x[a]=1
    return x


class DenseNet(nn.Module):
    def __init__(self,n_features,n_actions,n_quantiles):
        super(DenseNet,self).__init__()

        self.fc1=nn.Linear(n_features,n_actions*n_quantiles,bias=False)
        self.to(device=DEVICE,dtype=TARGET_DTYPE)

        for p in self.parameters():
            if p.dim()>1:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.zeros_(p)

    def forward(self,x):
        return self.fc1(x)

    
class QPolicy:
    def __init__(self,q_alpha,gamma,seed,n_obs,n_actions,n_quantiles,n_cvar):
        torch.manual_seed(seed)
        self.gamma=gamma
        self.model=DenseNet(n_obs,n_actions,n_quantiles)
        self.target_model=DenseNet(n_obs,n_actions,n_quantiles)
        for p in self.target_model.parameters():
            p.requires_grad=False
        self.update_target()
        self.n_actions=n_actions
        self.n_quantiles=n_quantiles
        self.n_cvar=n_cvar
        self.q_levels=torch.linspace(0.5/n_quantiles,1-0.5/n_quantiles,n_quantiles,dtype=TARGET_DTYPE,device=DEVICE)
        self.kappa=torch.tensor(1.0,dtype=TARGET_DTYPE,device=DEVICE)
        self.q_optimizer=optim.Adam(self.model.parameters(),lr=q_alpha) #AdamW  ,weight_decay=0.01)
        self.clip_norm=10.0 #tf.constant(10.0)
        self.target_update_freq=500
        self.model.train(False)

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())
        # for var,var_target in zip(self.model.parameters(),self.target_model.parameters()):
        #     var_target.data.copy_(var.data)
            
    def extract_state_feature(self,obs):
        return obs.astype('float32')

    def eval_model(self,phi,cutoff):
        with torch.no_grad():
            quantiles=self.model(torch.tensor(phi,device=DEVICE,dtype=TARGET_DTYPE)).view(self.n_actions,self.n_quantiles)
            q=torch.mean(torch.clamp(quantiles-cutoff,max=0),axis=1)
        return torch.argmax(q),q,quantiles

    
    def query(self,phi,cutoff):
        a,q,_=self.eval_model(phi,torch.tensor(cutoff,dtype=TARGET_DTYPE,device=DEVICE))
        return a.cpu().numpy(),q.cpu().numpy()
    
    # def query_all(self,phi,cutoff):
    #     a,q,quantiles=self.eval_model(phi,torch.tensor(cutoff,dtype=TARGET_DTYPE,device=DEVICE))
    #     return a.cpu().numpy(),q.cpu().numpy(),quantiles.cpu().numpy()

    
    def eval_model_init(self,phi):
        with torch.no_grad():
            quantiles=self.model(torch.tensor(phi,device=DEVICE,dtype=TARGET_DTYPE)).view(self.n_actions,self.n_quantiles)
            q=torch.mean( quantiles[:,:self.n_cvar], axis=1 )
            a=torch.argmax(q)
            cutoff=0.5*(quantiles[a,self.n_cvar-1]+quantiles[a,self.n_cvar])
        return a,cutoff,q
    
    def query_init(self,phi):
        a,cutoff,q=self.eval_model_init(phi)
        return a.cpu().numpy(),cutoff.cpu().numpy(),q.cpu().numpy()

    def eval_policy(self,env,n_rep,max_episode_len,gamma=1.0,render=False):
        n_actions=env.action_space.n
        ep_rewards=[]
        decs=[]
        for rep_i in range(n_rep):
            s=env.reset()
            if render:
                env.render()
            phi=self.extract_state_feature(s)
            ep_r=0
            steps=0
            a,cutoff,_=self.query_init(phi)
            while steps<max_episode_len:
                decs.append(one_hot(n_actions,a))
                next_s,r,done,_=env.step(a)
                if render:
                    env.render()
                ep_r+=(gamma**steps)*r
                phi=self.extract_state_feature(next_s)
                steps+=1
                if done:
                    break
                cutoff=(cutoff-r)/self.gamma
                a,_=self.query(phi,cutoff)
            ep_rewards.append(ep_r)
        return ep_rewards,np.array(decs)
                      
    def update_q(self,cutoffs,phis,acts,rewards,next_phis,dones):
        self.q_optimizer.zero_grad()
        
        quantiles=self.model(phis).view(-1,self.n_actions,self.n_quantiles)
        q=quantiles.gather(1,acts.view(-1,1,1).expand(-1,-1,self.n_quantiles)).squeeze(1)
        
        #cutoffs=(q[:,self.n_cvar-1]-rewards)/self.gamma
        qn=self.target_model(next_phis).view(-1,self.n_actions,self.n_quantiles)
        qn_acts=torch.argmax( torch.mean( torch.clamp(qn-cutoffs.view(-1,1,1),max=0),axis=2), axis=1)
        next_quantiles=qn.gather(1,qn_acts.view(-1,1,1).expand(-1,-1,self.n_quantiles)).squeeze(1)
        
        target = rewards.unsqueeze(1)+self.gamma*(1-dones.unsqueeze(1))*next_quantiles
        tau=self.q_levels.view(1,1,-1).expand(len(q),self.n_quantiles,self.n_quantiles).reshape(-1)
        target=target.unsqueeze(2).expand(len(q),self.n_quantiles,self.n_quantiles).reshape(-1)
        q=q.unsqueeze(1).expand(len(q),self.n_quantiles,self.n_quantiles).reshape(-1)
        u=target-q
        q_loss=torch.mean( (tau-(1-torch.sign(u))*0.5) * u )
        
        q_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(),self.clip_norm)
        self.q_optimizer.step()
        
        # with tf.GradientTape() as tape:
        #     quantiles=tf.reshape( self.model(phis), [-1,self.n_actions,self.n_quantiles] )
        #     q=tf.math.reduce_sum(acts[:,:,tf.newaxis]*quantiles, axis=1)
        #     cutoff=(q[:,self.n_cvar-1]-rewards)/self.gamma
        #     qn=tf.reshape( self.target_model(next_phis), [-1,self.n_actions,self.n_quantiles] )
        #     qn_acts=tf.math.argmax( tf.math.reduce_mean( tf.math.minimum(qn-cutoff[:,tf.newaxis,tf.newaxis],0), axis=2 ), axis=1 )
        #     next_quantiles=tf.gather_nd(qn, qn_acts[:,tf.newaxis], batch_dims=1)
        #     target=tf.stop_gradient( rewards[:,tf.newaxis]+self.gamma*(1-dones[:,tf.newaxis])*next_quantiles )
        #     tau=tf.reshape(tf.broadcast_to(self.q_levels,[len(q),self.n_quantiles,self.n_quantiles]), [-1])
        #     target=tf.reshape(tf.broadcast_to(target[:,:,tf.newaxis],[len(q),self.n_quantiles,self.n_quantiles]), [-1])
        #     q=tf.reshape(tf.broadcast_to(q[:,tf.newaxis,:],[len(q),self.n_quantiles,self.n_quantiles]), [-1])
        #     u=target-q
        #     q_loss=tf.reduce_mean( (tau-(1-tf.math.sign(u))*0.5) * u )
            
        # grad=tape.gradient(q_loss,self.model.trainable_variables)
        # grad,g_norm=tf.clip_by_global_norm(grad,self.clip_norm)
        # self.q_optimizer.apply_gradients(zip(grad,self.model.trainable_variables))
        return q_loss
        
    def train(self,epsilon,buffer_size,batch_size,seed,env,val_env,tst_env,max_iter,explore_iter,eval_freq,max_episode_len):
        q_losses=[]
        q_vals=[]
        grad_norms=[]
        n_rep=1
        res=[]
        if seed is not None:
            np.random.seed(seed)
            env.seed(seed)
            val_env.seed(seed+1)
            tst_env.seed(seed+2)
            self.replay_buf=ReplayBufferCutoff(buffer_size)
            
            self.iter=0
            self.steps=0
            self.ep_done=0
            self.ep_rewards=[0.0]
            
            r_val,d_val=self.eval_policy(val_env,n_rep,max_episode_len)
            r_tst,d_tst=self.eval_policy(tst_env,n_rep,max_episode_len)
            res.append(np.concatenate(([self.iter,0,0,0,np.mean(self.ep_rewards[-100:]),
                                        np.mean(r_val),np.mean(r_tst)],np.mean(d_val,axis=0),np.mean(d_tst,axis=0))))
            print(res[-1])
            
            self.phi=self.extract_state_feature(env.reset())
            self.a_next,self.cutoff,self.last_q=self.query_init(self.phi)
        else:
            self.ep_rewards=self.ep_rewards[-100:].copy()
            
        st=time.time()
        while True:
            ##query policy
            a=self.a_next
            q_vals.append(self.last_q)
            p_explore=1-min(1,self.iter/explore_iter)*(1-epsilon)
            if np.random.rand()<p_explore:
                a=np.random.randint(env.action_space.n)
            obs,r,done,_=env.step(a)
            next_phi=self.extract_state_feature(obs)
            self.ep_rewards[-1]+=r
            self.steps+=1
            self.cutoff=(self.cutoff-r)/self.gamma
            self.a_next,self.last_q=self.query(next_phi,self.cutoff)
            self.replay_buf.add(self.cutoff,self.phi,a,np.array(r,dtype='float32'),next_phi,np.array(done,dtype='float32'))
            if done or self.steps>=max_episode_len:
                self.phi=self.extract_state_feature(env.reset())
                self.steps=0
                self.ep_done+=1
                self.ep_rewards.append(0.0)
                self.a_next,self.cutoff,self.last_q=self.query_init(self.phi)
            else:
                self.phi=next_phi

            #q update
            cutoffs,phis,acts,rewards,next_phis,dones=self.replay_buf.sample(batch_size)
            self.model.train(True)
            q_loss=self.update_q(torch.tensor(cutoffs,dtype=TARGET_DTYPE,device=DEVICE),
                                 torch.tensor(phis,dtype=TARGET_DTYPE,device=DEVICE),
                                 torch.tensor(acts,device=DEVICE,dtype=torch.int64),
                                 torch.tensor(rewards,dtype=TARGET_DTYPE,device=DEVICE),
                                 torch.tensor(next_phis,dtype=TARGET_DTYPE,device=DEVICE),
                                 torch.tensor(dones,dtype=TARGET_DTYPE,device=DEVICE))
            q_losses.append(q_loss.item())
            self.model.train(False)
            
            norm=0
            for p in self.model.parameters():
                if p.grad is not None:
                    norm+=torch.norm(p.grad).item()**2
            grad_norms.append(  np.sqrt(norm) )
            
            self.iter+=1
            if self.iter%self.target_update_freq==0:
                self.update_target()
                
            if self.iter%eval_freq==0:
                r_val,d_val=self.eval_policy(val_env,n_rep,max_episode_len)
                r_tst,d_tst=self.eval_policy(tst_env,n_rep,max_episode_len)
                res.append(np.concatenate(([self.iter,np.median(grad_norms),np.max(grad_norms),
                                            np.mean(q_losses),np.mean(self.ep_rewards[-100:]),
                                            np.mean(r_val),np.mean(r_tst)],np.mean(d_val,axis=0),np.mean(d_tst,axis=0))))
                q_losses=[]
                grad_norms=[]
                
                print(self.ep_done,'({:.3f} secs)'.format(time.time()-st),res[-1],np.max(np.array(q_vals),axis=0),
                      np.min(np.array(q_vals),axis=0),
                      np.sum([ex[4] for ex in self.replay_buf._storage]),flush=True)
                q_vals=[]
                st=time.time()
            if self.iter>=max_iter:
                break
        return np.array(res)

    

def run_train():
    if not os.path.exists('./res'):
        os.mkdir('./res')
    for seed in [1,2,3]: #[int(sys.argv[1])]:
        for n_cvar in [50,55,60,65,70,75,80,85,90,95]: #[int(sys.argv[2])]:
            pol=QPolicy(alpha,gamma,seed,tr_env.observation_space.shape[0],tr_env.action_space.n,n_quantiles,n_cvar)
            res=pol.train(epsilon,buffer_size,batch_size,seed,tr_env,val_env,tst_env,max_iter,explore_iter,eval_freq,max_episode_len)
            np.save('res/dqr4d_ladder_res_{:g}_{}_{}_{}_{}.npy'.format(alpha,max_iter,n_quantiles,n_cvar,seed),res)
            torch.save(pol.model.state_dict(),'res/dqr4d_ladder_res_model_{:g}_{}_{}_{}_{}.pth'.format(alpha,
                                                                                                      max_iter,n_quantiles,n_cvar,seed))
            tst_env.seed(123)
            st=time.time()
            r,_=pol.eval_policy(tst_env,n_rep,max_episode_len)
            print('eval {}: {:.1f} secs'.format(n_rep,time.time()-st))
            tst_env.seed(123)
            st=time.time()
            r2,_=pol.eval_policy(tst_env,n_rep,max_episode_len,gamma=gamma)
            print('eval2 {}: {:.1f} secs'.format(n_rep,time.time()-st))
            np.save('res/dqr4d_ladder_res_eval_{:g}_{}_{}_{}_{}_{}.npy'.format(alpha,max_iter,n_quantiles,n_cvar,n_rep,seed), np.array([r,r2]))


    
np.set_printoptions(5,suppress=True,linewidth=150)

max_episode_len=200
n_states=4
tr_env=LadderEnv(n_states)
val_env=LadderEnv(n_states)
tst_env=LadderEnv(n_states)

alpha=0.0001
gamma=0.9 #discount
epsilon=0.02 #exploration
buffer_size=50000
batch_size=32
n_quantiles=100

max_iter=200000
explore_iter=5000
eval_freq=1000
n_rep=1000


##### To train and save outputs
run_train()
