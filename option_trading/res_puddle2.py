import numpy as np
import matplotlib.pyplot as plt

def do_plot(i,j,all_res,r_idx,spec='*-',title=None):
    plt.subplot(1,i,j)
    for res in all_res:
        plt.plot(res[:,0],np.mean(res[:,r_idx],1),spec)
    plt.grid(True)
    if title is not None:
        plt.title('{} {}'.format(r_idx,title))

def do_plots(f,all_res,title=None,fig_title=None):
    fig=plt.figure(f)
    fig.set_figwidth(25)
    fig.set_figheight(6)
    fig.set_tight_layout(True)
    if fig_title is not None:
        fig.canvas.set_window_title(fig_title)
    plt.clf()
    do_plot(4,1,all_res,[1],title=title)
    do_plot(4,2,all_res,[3],title=title)
    do_plot(4,3,all_res,[4],title=title)
    do_plot(4,4,all_res,[5,6],spec='*',title=title)
    
    
np.set_printoptions(5,suppress=True,linewidth=150)
plt.ion()


s_rng=np.arange(1,6)
epsilon=0.02
buf_size=50000

for fnum,t_cvar in enumerate([20]):
    fig=plt.figure(fnum+1)
    fig.set_figwidth(15)
    fig.set_figheight(18)
    fig.set_tight_layout(True)
    plt.clf()
    for j,(alg,n_cvar,leg) in enumerate(zip(['dqr','dqr','dqr4d'],[100,t_cvar,t_cvar],['Risk-neutral','Markov','Proposed'])):
        env='{:.2f}_{}'.format(epsilon,buf_size)
        res_eval2b=[np.load('res_puddle2/{}_{}_{}_{}_eval2b.npy'.format(alg,env,s,n_cvar)) for s in s_rng]
        res=np.array([res[ np.argmax(np.mean(res[:,:n_cvar*10],axis=1)) ] for res in res_eval2b])
        for i,r in enumerate(res):
            plt.subplot(len(s_rng),3,i*3+j+1)
            plt.hist(r,30,edgecolor='k')
            plt.title('{:.1f}-CVaR: {:.1f}, Avg: {:.1f}'.format(t_cvar/100,np.mean(r[:t_cvar*10]),np.mean(r)) )
            plt.gca().set(xlim=(np.min(res),np.max(res)))
            plt.xlabel('Return')
            plt.legend(['{} seed {}'.format(leg,i+1)],loc='upper left')
            
