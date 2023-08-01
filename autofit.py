import matplotlib.pyplot as plt
import numpy as np

from lmfit import Minimizer,Parameters
#define class for components

def gaussian(x,h,mu,sig):
    return h*np.exp(-np.power(x-mu,2.)/(2*np.power(sig,2.)))

class component:
    def __init__(self,type,id):
        self.type = type
        self.id = id

    def updateIParams(self,params):
        #Update the parameters of the particular component
        id = self.id
        self.c = params[f'c{id}']
        self.w = params[f'w{id}']
        if self.type != "HINSA":
            self.h = params[f'h{id}']
        if self.type != "WNM":
            self.t =params[f't{id}']

    def I_emit(self,x,b=0):
        #The radiation emitted by a cloud
        if self.type == 'HINSA':
            return 0
        if self.type == 'WNM':
            return gaussian(x,self.h,self.c+b,self.w)
        if self.type == 'CNM':
            tau = gaussian(x,self.t,self.c+b,self.w)
            return self.h*(1-np.exp(-tau))
        
    def I_passthrough(self,I,x,b=0):
        #The optical depth provided by a component when radioation passes through
        if self.type == "WNM":
            return I
        tau = gaussian(x,self.t,self.c+b,self.w)
        return I*np.exp(-tau)   

    def updateVParams(self,params):
        #Update the stokes V parameters
        id = self.id
        self.b = params[f'b{id}']
        self.berr = params[f'b{id}'].stderr
    
    def V_emit(self,x):
        #Similar to I_emit but separates into LCP and RCP
        RCP = 0.5*self.I_emit(x,self.b)
        LCP = 0.5*self.I_emit(x,-self.b)
        return (RCP,LCP)

    def V_passthrough(self,V,x):
        #Similar to I_passthrough but separates into LCP and RCP
        RCP = self.I_passthrough(V[0],x,self.b)
        LCP = self.I_passthrough(V[1],x,-self.b)
        return (RCP,LCP)
    
    def calcB(self):
        #Calculate the magnetic field strength
        zf = 2.8
        dvdf = 2.9979e5/1420.40575e6
        B = self.b/dvdf*2/zf
        Berr = self.berr/dvdf*2/zf
        return (B,Berr)


#Class for Stokes I and Stokes V
class ZeemanStokes:
    def __init__(self,x,stk_i,stk_v):
        self.Iparams = Parameters()
        self.Vparams = Parameters()
        self.cnmCount = 0
        self.wnmCount = 0
        self.hinsaCount = 0
        self.x = x
        self.stk_i = stk_i
        self.stk_v = stk_v
        self.components = {}
    
    def addCNM(self,id,props,vary=[True,True,True,True,True],
               lbound=[None,None,None,None,None],
               ubound=[None,None,None,None,None]):
        '''
        Takes the properties and bounds of the CNM component as input.
        Adds that CNM to the model
        '''
        self.cnmCount += 1
        self.Iparams.add_many((f'h{id}',props[0],vary[0],lbound[0],ubound[0],None,None),
                             (f't{id}',props[1],vary[1],lbound[1],ubound[1],None,None),
                             (f'c{id}',props[2],vary[2],lbound[2],ubound[2],None,None),
                             (f'w{id}',props[3],vary[3],lbound[3],ubound[3],None,None))
        self.Vparams.add(f'b{id}',props[-1],vary[-1],lbound[-1],ubound[-1],None,None)
        self.components[id] = component('CNM',id)


    def addWNM(self,id,props,vary=[True,True,True],
               lbound=[None,None,None],
               ubound=[None,None,None]):
        '''
        Takes the properties and bounds of the CNM component as input.
        Adds that WNM to the model
        '''
        self.wnmCount += 1
        self.Iparams.add_many((f'h{id}',props[0],vary[0],lbound[0],ubound[0],None,None),
                             (f'c{id}',props[1],vary[1],lbound[1],ubound[1],None,None),
                             (f'w{id}',props[2],vary[2],lbound[2],ubound[2],None,None))
        
        self.Vparams.add(f'b{id}',props[-1],vary[-1],lbound[-1],ubound[-1],None,None)
        self.components[id] = component('WNM',id)
        

    def addHINSA(self,id,props,vary=[True,True,True],
                 lbound=[None,None,None],
                 ubound=[None,None,None]):
        '''
        Takes the properties and bounds of the CNM component as input.
        Adds HINSA to the model
        '''
        self.Iparams.add_many((f't{id}',props[0],vary[0],lbound[0],ubound[0],None,None),
                             (f'c{id}',props[1],vary[1],lbound[1],ubound[1],None,None),
                             (f'w{id}',props[2],vary[2],lbound[2],ubound[2],None,None))
        self.Vparams.add(f'b{id}',props[-1],vary[-1],lbound[-1],ubound[-1],None,None)
        self.hinsaCount += 1
        self.components[id] = component('HINSA',id)

    def performIFit(self,order):
        self.order = order
        fitter = Minimizer(self.fitfuncI,self.Iparams)
        self.Iresult = fitter.minimize(method='leastsq')
        for id in order:
            self.components[id].updateIParams(self.Iresult.params)

    def fitfuncI(self,params):
        model = self.stokesI(params)
        return abs(self.stk_i-model)

    def stokesI(self,params):
        #Returns the total Stokes I based on the passed values of the parameters
        order = self.order
        x = self.x
        tot = 0
        for n,i in enumerate(order):
            comp = self.components[i]
            comp.updateIParams(params)
            I = comp.I_emit(x)
            for j in order[:n]:
                comp_ = self.components[j]
                comp_.updateIParams
                I = comp_.I_passthrough(I,x)
            tot+=I
        return tot

    def plotI(self):
        #Plot the Stokes I, Stokes I fit and the several components
        I_tot = 0
        I_noHINSA_tot = 0
        fig,ax =plt.subplots(1)
        ax.plot(self.x,self.stk_i)
        ax.set_ylabel("Stokes I(K)")
        ax.set_xlabel(r"$V_{LSR}$ (Kms$^{-1}$)")
        for n,id in enumerate(self.order):
            comp = self.components[id]
            if comp.type == 'HINSA':
                continue
            I = comp.I_emit(self.x)
            I_noHINSA = I
            for j in self.order[:n]:
                comp_ = self.components[j]
                if comp_.type != 'HINSA':
                    I_noHINSA = comp_.I_passthrough(I_noHINSA,self.x)
                I = comp_.I_passthrough(I,self.x)
            ax.plot(self.x,I_noHINSA,label=id)
            I_tot += I
            I_noHINSA_tot += I_noHINSA
        ax.plot(self.x,I_tot,label='fit')
        ax.plot(self.x,I_noHINSA_tot-I_tot,label = 'HINSA')
        ax.legend()

    def performVFit(self,e=[None,None,None,None]):
        #Perform Stokes V fit
        self.Vparams.add('e',e[0],e[1],e[2],e[3],None,None)
        fitter = Minimizer(self.fitfuncV,self.Vparams)
        self.Vresult = fitter.minimize(method='leastsq')
        for id in self.order:
            self.components[id].updateVParams(self.Vresult.params)

    def fitfuncV(self,params):
        model = self.stokesV(params)
        return abs(self.stk_v-model)

    def stokesV(self,params):
        #Take the parameters and return the Stokes V model based on the previously fitted Stokes I
        order = self.order
        x = self.x
        tot_RCP = 0
        tot_LCP = 0
        for n,i in enumerate(order):
            comp = self.components[i]
            comp.updateVParams(params)
            V_RCP,V_LCP = comp.V_emit(x)
            for j in order[:n]:
                comp_ = self.components[j]
                comp_.updateVParams(params)
                V = (V_RCP,V_LCP)
                V_RCP,V_LCP = comp_.V_passthrough(V,x)
            tot_RCP += V_RCP
            tot_LCP += V_LCP
        return tot_RCP-tot_LCP + params['e']*self.stk_i

    def plotV(self,plotComponents=False,pc=False):
        """Plots the Stokes V with the fit and the componetns in oone plot. 
        If plotComponents is set to true, each component is plotted separately
        If pc is set t False, then the optical depth effects undergone by the radiation 
        due to a cloud in between that and the observer are ignored"""
        V_tot_RCP = 0
        V_tot_LCP = 0
        V_noHINSA_tot_RCP = 0
        V_noHINSA_tot_LCP = 0
        fig1,ax = plt.subplots(1)
        ax.set_ylabel("Stokes V(K)")
        ax.set_xlabel(r"$V_{LSR}$ (Kms$^{-1}$)")
        ax.step(self.x,self.stk_v-self.Vresult.params['e']*self.stk_i)

        for n,id in enumerate(self.order):
            comp = self.components[id]
            if comp.type == 'HINSA':
                continue
            V = comp.V_emit(self.x)
            V_noHINSA = V
            for j in self.order[:n]:
                comp_ = self.components[j]
                if comp_.type != 'HINSA':
                    V_noHINSA = comp_.V_passthrough(V_noHINSA,self.x)
                V = comp_.V_passthrough(V,self.x)
            ax.plot(self.x,V[0]-V[1],linestyle='--',alpha=0.5,label=id)
            #ax.plot(self.x,V_noHINSA,label=id)
            V_tot_RCP += V[0]
            V_tot_LCP += V[1]
            V_noHINSA_tot_RCP += V_noHINSA[0]
            V_noHINSA_tot_LCP += V_noHINSA[1]
        V_tot = V_tot_RCP - V_tot_LCP
        V_noHINSA_tot = V_noHINSA_tot_RCP - V_noHINSA_tot_LCP
        fit = V_tot #+ self.Vresult.params['e']*self.stk_i
        resV = self.stk_v - fit - self.Vresult.params['e']*self.stk_i
        HINSAV = V_tot - V_noHINSA_tot
        ax.plot(self.x,fit,label='fit')
        ax.plot(self.x,HINSAV,label = 'HINSA')
        ax.legend()
        if plotComponents == True:
            self.plotComponents(pc,resV,HINSAV)

    def plotComponents(self,pc,res,HINSAV):
        #Function to plot the components separately
        fig,ax = plt.subplots(len(self.order),figsize=(6,2*len(self.order)),sharex=True,gridspec_kw={'hspace':0})
        fig.supylabel("Stokes V(K)")
        fig.supxlabel(r"$V_{LSR}$ (Kms$^{-1}$)")
        for n,id in enumerate(self.order):
            comp = self.components[id]
            if comp.type !='HINSA':
                V = comp.V_emit(self.x)
                if pc:
                    for j in self.order[:n]: 
                        comp_ = self.components[j]
                        if comp_.type != 'HINSA':
                            V = comp_.V_passthrough(V,self.x)
                V = V[0]-V[1] 
            else:
                V = HINSAV
            B = comp.calcB()
            #ymin = 1.1*np.max(abs(res+v))
            #ymax = -1.1*np.min(abs(res+v))
            #ax[n].set_xlim(-10,20)
            ax[n].plot(self.x, V,'r-')
            ax[n].step(self.x, res+V,'k-', label=r"data",linewidth=0.5)
            ax[n].text(0.08,1-0.15*8/6, self.order[n],horizontalalignment='center',verticalalignment='center',
                    transform=ax[n].transAxes, fontsize=10)
            ax[n].text(0.25,1-0.15*8/6,"{:2.1f}\u00B1{:2.1f} ".format(B[0],B[1])+r'$\mu G$' ,horizontalalignment='center',verticalalignment='center',
                    transform=ax[n].transAxes, fontsize=10)