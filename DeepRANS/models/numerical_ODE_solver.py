import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
from scipy.integrate import solve_bvp
import warnings
import os 
from utils_preprocess import channel_parameters, make_dataframe
from utils_train import ModelTrainer
import pandas as pd

def solve_ODE(turb_model,Retau,points,tol,A_plus,karman,method):
    
    Re_tau, u_tau, nu = channel_parameters(Retau)
    # Fixed Channel Parameters
    delta = 1 # meters
    y_min = 0 # meters
    k = karman # Van Karmen Constant
    A_plus = A_plus # van/Driest Damping Constant
    y_max = delta # meters

    # Define mesh
    if method == 'logspace':
        y_norm_bvp = np.logspace(-6, 0, points)

    else:
        y_norm_bvp = np.linspace(y_min, y_max, points)

    # Define ordinary differential equation models to solve
    def fun_vanDriest(y_norm, u_norm):
        C = delta*u_tau/(A_plus*nu)
        return np.vstack((u_norm[1], (-1-(2*C*k**2*y_norm**2*np.exp(-C*y_norm)*(1-np.exp(-C*y_norm))*u_norm[1]**2)-(2*k**2*y_norm*(1-np.exp(-C*y_norm))**2*u_norm[1]**2))/((1/Re_tau)+(2*k**2*y_norm**2*(1-np.exp(-C*y_norm))**2*u_norm[1]) )))

    def fun_Prandtl(y_norm, u_norm):
        return np.vstack((u_norm[1], (-1-(2*k**2*y_norm*u_norm[1]**2))/((1/Re_tau)+(2*k**2*y_norm**2*u_norm[1]))  ))

    def fun_Rui(y_norm, u_norm):
        
        #Import Model for Anisotropic Tensor
        trainer = ModelTrainer()
        trainer.load_checkpoint('checkpoint_MLP_Re_BC.pt')
        rui_model = trainer.model
        rui_model.to(torch.float64) 
        rui_model = rui_model.to(torch.float64)
        factor = 1e3 # the anisotropic stress tensor is multiplied by this factor

        # Create differentiation using autograd
        def diff(u, t, order=1):
            ones = torch.ones_like(u)
            der, = autograd.grad(u, t, create_graph=True, grad_outputs=ones, allow_unused=True)
            if der is None:
                return torch.zeros_like(t, requires_grad=True)
            else:
                der.requires_grad_()
            for i in range(1, order):
                ones = torch.ones_like(der)
                der, = autograd.grad(der, t, create_graph=True, grad_outputs=ones, allow_unused=True)
                if der is None:
                    return torch.zeros_like(t, requires_grad=True)
                else:
                    der.requires_grad_()
            return der

        # Convert inputs to tensors to allow diff
        y_norm_tensor = torch.tensor(y_norm,requires_grad=True).view(-1, 1)
        y_plus_tensor = y_norm_tensor*Re_tau
        du_dy_norm_tensor = torch.tensor(u_norm[1],requires_grad=True).view(-1, 1)
        du_dy_tensor = du_dy_norm_tensor*u_tau
        Re_tau_tensor = Re_tau*torch.ones_like(y_norm_tensor,requires_grad=True)

        return np.vstack((u_norm[1],Re_tau**2*diff(rui_model(du_dy_tensor,Re_tau_tensor,y_plus_tensor)/(factor*u_tau**2),y_plus_tensor, order = 1).detach().cpu().numpy().reshape(torch.tensor(u_norm[1]).shape) ))       


    def fun_Ocariz(y_norm,u_norm):

        ocariz_model = torch.load('short_convbcre_fourth')
        ocariz_model.to(torch.float64) 
        ocariz_model = ocariz_model.to(torch.float64)
        factor = 1e1

        # Interpolating k
        df_Ocariz = pd.concat([make_dataframe(Retau, nondim='k-eps')])
        k_turb = df_Ocariz['k']
        y_norm_DNS = df_Ocariz['y']
        from scipy.interpolate import interp1d
        f = interp1d(y_norm_DNS, k_turb,kind='nearest', fill_value='extrapolate')
        xnew = y_norm
        k_turb_new = f(xnew)

        eps = df_Ocariz['eps']
        from scipy.interpolate import interp1d
        f2 = interp1d(y_norm_DNS, eps,kind='nearest', fill_value='extrapolate')
        xnew = y_norm
        eps_new = f2(xnew)   

        # Import model
        ocariz_model = torch.load('short_convbcre_fourth')
        ocariz_model.to(torch.float64) 
        ocariz_model = ocariz_model.to(torch.float64)
        factor = 1e1

        # Channel Parameters
        Re_tau, u_tau, nu = channel_parameters(Retau)

        # Format validation x 
        y_norm_tensor = torch.tensor(y_norm,requires_grad=True)
        u_norm_tensor = torch.tensor(u_norm[0],requires_grad=True)
        du_dy_norm_tensor = torch.tensor(u_norm[1],requires_grad=True)
        du_dy_norm_k_tensor = du_dy_norm_tensor*torch.tensor(k_turb_new/(2.*eps_new*Re_tau),requires_grad=True)
        du_dy_in = du_dy_norm_k_tensor.view(1,1,len(y_norm)).float()

        # Add Reynolds number
        Re_tensor = Re_tau*1e-8*torch.ones_like(du_dy_norm_tensor,requires_grad=True)
        Re_in = Re_tensor.view(1,1,len(Re_tensor)).float()

        # Input vector including gradient and Reynolds
        x_in = torch.cat((du_dy_in,Re_in),dim=1)

        # Format validation y+ (for boundary condition enforcement)
        yplus = y_norm*u_tau/nu
        yplus_torch = torch.tensor(yplus,requires_grad=True)
        yplus_in = yplus_torch.view(1,1,len(yplus)).float()

        # Get dimensioning correct
        maxlength = y_norm.shape[0]
        zerosval = torch.zeros(1,1,maxlength-y_norm.shape[0])
        zerosval_double = torch.zeros(1,2,maxlength-y_norm.shape[0])

        x_in=torch.cat((x_in,zerosval_double), 2)
        yplus_in=torch.cat((yplus_in,zerosval), 2)

        shapexval=y_norm.shape[0]
        x_in=x_in[:,:,0:shapexval]
        yplus_in=yplus_in[:,:,0:shapexval]

        # Create differentiation using autograd 
        def diff(u, t, order=1):
            ones = torch.ones_like(u)
            der, = autograd.grad(u, t, create_graph=True, grad_outputs=ones, allow_unused=True)
            if der is None:
                return torch.zeros_like(t, requires_grad=True)
            else:
                der.requires_grad_()
            for i in range(1, order):
                ones = torch.ones_like(der)
                der, = autograd.grad(der, t, create_graph=True, grad_outputs=ones, allow_unused=True)
                if der is None:
                    return torch.zeros_like(t, requires_grad=True)
                else:
                    der.requires_grad_()
            return der

        k_turb_new = k_turb_new.reshape(u_norm[1].shape)
        eps_new = eps_new.reshape(u_norm[1].shape)

        return np.vstack((u_norm[1],Re_tau**2*diff(torch.mean(ocariz_model(x_in,yplus_in), dim=0).reshape(torch.tensor(k_turb_new).shape)*2*torch.tensor(k_turb_new)/factor,yplus_in.reshape(torch.tensor(u_norm[1]).shape)).detach().cpu().numpy()-Re_tau ))

    # Define boundary conditions for channel
    def bc(ya, yb):
        return np.array([ya[0], yb[1]])

    if turb_model == 'Prandtl':
        u_norm_bvp_init = np.ones((2, points))
        res = solve_bvp(fun_Prandtl, bc, y_norm_bvp, u_norm_bvp_init,tol=tol,max_nodes=100000)
    elif turb_model == 'vanDriest':
        u_norm_bvp_init = np.ones((2, points))
        res = solve_bvp(fun_vanDriest, bc, y_norm_bvp, u_norm_bvp_init,tol=tol,max_nodes=100000)
    elif turb_model == 'Rui':
        u_norm_bvp_init = np.ones((2, points))
        res = solve_bvp(fun_Rui, bc, y_norm_bvp, u_norm_bvp_init,tol=tol,max_nodes=100000)
    elif turb_model == 'Ocariz':
        u_norm_bvp_init = np.ones((2, points))
        res = solve_bvp(fun_Ocariz, bc, y_norm_bvp, u_norm_bvp_init,tol=tol,max_nodes=100000)
    else:
        print('Error')

    return [y_norm_bvp, res.sol(y_norm_bvp)[0], res.sol(y_norm_bvp)[1]]



