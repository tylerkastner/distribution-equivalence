import os,sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea
import pandas as pd

def load_all(alg,n_cvar_rng):
    all_res={}
    for n_cvar in n_cvar_rng:
        this_res=[]
        for stock_i in range(1):# for stock_i in range(10):
            env_name='option_res{}_0.0001_4000_100'.format(stock_i)
            res_path='res/{}_{}_{}'.format(alg,env_name,n_cvar)
            save_path='{}_eval.npy'.format(res_path)
            if os.path.exists(save_path):
                res=np.load(save_path)
            else: 
                res=np.array([np.load('{}_{}_eval.npy'.format(res_path,s)) for s in [1,2, 3, 4,5 ]]) #,2,3,4,5]])#, 2, 3, 4, 5]])#[1,2,3]])
            # if res.shape[0]!=3 or res.shape[2]!=1000:
            #     print(res.shape)
            #     raise
            this_res.append(res)
        all_res[n_cvar]=np.array(this_res)
    return all_res

def extract_res(li_res,cvar_rng):
    res=[]
    for n_cvar in cvar_rng:
        m = np.mean(li_res[n_cvar][:,:,:,:round(n_cvar*n_rep/100)],axis=3)
        # print(f'm.shape: {m.shape}')
        # print(f'np.max(m[:,:,:],axis=2).shape: {np.max(m[:,:,:],axis=2).shape}')

        # res.append( np.mean( np.max(m[:,:,:],axis=2), axis=1 ) )
        res.append(np.max(m[:,:,:],axis=2))
    return np.array(res)

def extract_res_exp(li_res,cvar_rng):
    res_exp=li_res[100]
    res=[]
    for n_cvar in cvar_rng:
        m= np.mean(res_exp[:,:,:,:round(n_cvar*n_rep/100)],axis=3)
        res.append( np.mean( np.max(m[:,:,:],axis=2),axis=1 ) )
    return np.array(res)

if __name__=='__main__':
    # np.set_printoptions(5,suppress=True,linewidth=150)
    # plt.ion()

    dqr_cvar_rng=np.arange(10,105,5)
    # dqr4_cvar_rng=np.arange(10,100,5)
    dqr4_cvar_rng= np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 99])#, 50])
    max_cvar=105
    max_cvar4=100
    
    # res_dqr=load_all('dqr',dqr_cvar_rng)
    mmve = load_all('dqr4dmmve',dqr4_cvar_rng)
    dqr = load_all('dqr4d',dqr4_cvar_rng)
    ve = load_all('dqr4dve',dqr4_cvar_rng)

    n_rep=1000
    colors=plt.rcParams['axes.prop_cycle'].by_key()['color'] #'rgbmyck'
    specs=[(c,'*-') for c in colors]*3
    specs2=[(c,'*--') for c in colors]*3

    # print(f'mmve: {[(a,v.shape) for a, v in mmve.items()]}')

    all_res_ve=extract_res(ve,dqr4_cvar_rng).squeeze()
    all_res_mmve=extract_res(mmve,dqr4_cvar_rng).squeeze()
    all_res_dqr=extract_res(dqr,dqr4_cvar_rng).squeeze()


    # print(all_res_ve.shape)

    # print(pd.DataFrame(all_res_ve.T,columns=dqr4_cvar_rng/100))

    # fig=plt.figure(1)
    # fig.set_tight_layout(True)
    # plt.clf()
    pd_dqr=pd.melt(pd.DataFrame(all_res_dqr.T,columns=dqr4_cvar_rng/100),var_name=r'CVaR level ($\tau$)',value_name='CVaR of Return')
    pd_ve=pd.melt(
        pd.DataFrame(all_res_ve.T,columns=dqr4_cvar_rng/100),
        var_name=r'CVaR level ($\tau$)',value_name='CVaR of Return')

    pd_mmve=pd.melt(pd.DataFrame(all_res_mmve.T,columns=dqr4_cvar_rng/100),var_name=r'CVaR level ($\tau$)',value_name='CVaR of Return')
    # pd_dqr=pd.melt(pd.DataFrame(all_res_dqr.T,columns=dqr4_cvar_rng/100),var_name='Tau level',value_name='CVaR')

    # print(f'pd_ve.shape: {pd_ve.shape}')
    # print(f'pd_mmve.shape: {pd_mmve.shape}')
    # print(f'pd_dqr.shape: {pd_dqr.shape}')

    pd_all=pd.concat([pd_mmve,pd_ve,pd_dqr],keys=[r'$\psi$-equivalent model', 'VE model','Original'],names=['Algorithm'])

    # print(pd_dqr4d)

    # print(pd_all)

    # sea.set_theme(style="darkgrid")
    # sea.set(font_scale=2)

    # sea.set_palette('colorblind')

    rc = {'figure.figsize':(7.8,5),
      'axes.facecolor':'white',
      # 'axes.grid' : True,
      # 'grid.color': '.8',
      # 'font.family':'Times New Roman',
      'mathtext.fontset': 'cm',
      'font.size' : 15}


    plt.rcParams.update(rc)

    # sea.despine(bottom = True, left = True)

    sea.despine(fig=None, ax=None, top=True, right=True, left=False, bottom=False, offset=None, trim=False)

    palette = sea.color_palette('colorblind')[:3][::-1]

    sea.lineplot(data=pd_all,x=r'CVaR level ($\tau$)',y='CVaR of Return' ,hue='Algorithm', style='Algorithm', errorbar=('ci', 75), palette=palette)
    # h.legend_.set_title(None)

    sea.despine()


    plt.savefig('test.png')
    # plt.show()
    
