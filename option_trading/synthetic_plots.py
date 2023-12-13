import matplotlib.pyplot as plt
import numpy as np
from ladder_env import rollout

def gtruth(from_state,fig_num):
    n_states=4
    gamma=0.9
    n_quantiles=100
    n_rep=1000
    
    z=np.array([np.array(list('{{:0{}b}}'.format(n_states-1).format(i)),dtype='int32') for i in range(2**(n_states-1-from_state))])
    ret=[]
    N=n_rep
    for pol in z:
        ret.append( np.sort( rollout(from_state,n_states,N,np.append(pol,0),gamma,seed=100) ) )
    ret=np.array(ret)
    tau=np.linspace(0.5/n_quantiles,1-0.5/n_quantiles,n_quantiles)
    q_levels=np.ceil(tau*N).astype('int32')-1
    fig=plt.figure(fig_num)
    fig.set_tight_layout(True)
    plt.clf()
    for r in ret:
        plt.plot(np.arange(1,n_quantiles+1)/n_quantiles,np.cumsum(r[q_levels])/np.arange(1,n_quantiles+1),'.-')
    plt.grid()
    plt.xlabel('alpha level')
    plt.ylabel('CVaR')
    plt.legend([str(pol) for pol in z])
    res=[]
    pols=[]
    for n_cvar in np.arange(500,1000,50):
        r= np.mean(ret[:,:n_cvar],axis=1)
        i=np.argmax(r)
        res.append([n_cvar/1000,r[i]])
        pols.append(z[i])
    return np.array(res),np.array(pols)

    
def load_res(res_id):
    alpha=0.0001
    n_quantiles=100
    max_iter=200000
    n_rep=1000
    res=[]
    for n_cvar in np.arange(50,100,5):
        r2=[]
        for seed in [1,2,3]:
            r=np.sort(np.load('res/{}_ladder_res_eval_{:g}_{}_{}_{}_{}_{}.npy'.format(res_id,alpha,max_iter,n_quantiles,n_cvar,n_rep,seed)))
            r2.append( np.mean(r[1,:(n_cvar*n_rep//100)]) )
        res.append([n_cvar/100,np.mean(r2)])
    return np.array(res)


    
plt.ion()

plt.rc('font',size=16)
res0,pols=gtruth(0,1)
#gtruth(1,10)
gtruth(2,2)
res1=load_res('dqr')
res2=load_res('dqr4d')
fig=plt.figure(3)
fig.set_tight_layout(True)
plt.clf()
plt.plot(res0[:,0],res0[:,1],'*-')
plt.plot(res1[:,0],res1[:,1],'*-')
plt.plot(res2[:,0],res2[:,1],'*-')
plt.grid()
plt.legend(['Optimal stationary','Markov action-selection','Proposed algorithm'])
plt.xlabel('alpha level')
plt.ylabel('CVaR')

