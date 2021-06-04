import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import pandas as pd
import warnings
import os 
from utils_preprocess import channel_parameters, make_dataframe
from utils_train import ModelTrainer
from scipy.interpolate import interp1d

'''Defining differentiation operator'''

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


'''Calculating anisotropic Stress Tensors'''

def a_uv_from_Prandtl(du_dy_norm,y_norm,Retau,output,output_format):
    
    k = 0.384 # Van Karmen Constant
    
    # Convert inputs to tensors to allow diff
    y_norm_tensor = torch.tensor(y_norm,requires_grad=True).view(-1, 1)
    du_dy_norm_tensor = torch.tensor(du_dy_norm,requires_grad=True).view(-1, 1)

    a_uv_norm = -k**2*y_norm_tensor**2*du_dy_norm_tensor**2
    da_uv_dy_norm = diff(a_uv_norm,y_norm_tensor)

    if output_format == 'np':
        a_uv_norm = a_uv_norm.detach().cpu().numpy()
        a_uv_norm = a_uv_norm.reshape(y_norm_tensor.shape)
        da_uv_dy_norm = da_uv_dy_norm.detach().cpu().numpy().reshape(y_norm_tensor.shape)

    elif output_format == 'torch':
        a_uv_norm = a_uv_norm
        da_uv_dy_norm = da_uv_dy_norm
    else:
        print('Error') 
    return a_uv_norm if output == 'a_uv' else da_uv_dy_norm


def a_uv_from_vanDriest(du_dy_norm,y_norm,Retau,output,output_format):
    
    # Channel Parameters
    Re_tau, u_tau, nu = channel_parameters(Retau)

    # Fixed Channel Parameters
    delta = 1 # meters
    k = 0.384 # Van Karmen Constant
    A_plus = 26 # van/Driest Damping Constant
    C = delta*u_tau/(A_plus*nu)

    # Convert inputs to tensors to allow diff
    y_norm_tensor = torch.tensor(y_norm,requires_grad=True).view(-1, 1)
    du_dy_norm_tensor = torch.tensor(du_dy_norm,requires_grad=True).view(-1, 1)

    a_uv_norm = -k**2*y_norm_tensor**2*du_dy_norm_tensor**2*(1-torch.exp(-C*y_norm_tensor))**2
    da_uv_dy_norm = diff(a_uv_norm,y_norm_tensor)

    if output_format == 'np':
        a_uv_norm = a_uv_norm.detach().cpu().numpy()
        a_uv_norm = a_uv_norm.reshape(y_norm_tensor.shape)
        da_uv_dy_norm = da_uv_dy_norm.detach().cpu().numpy().reshape(y_norm_tensor.shape)

    elif output_format == 'torch':
        a_uv_norm = a_uv_norm
        da_uv_dy_norm = da_uv_dy_norm
    else:
        print('Error') 
    return a_uv_norm if output == 'a_uv' else da_uv_dy_norm


def a_uv_from_Rui(du_dy_norm,y_norm,Retau,output,output_format):
    # Channel Parameters
    Re_tau, u_tau, nu = channel_parameters(Retau)

    # Convert inputs to tensors to allow diff
    y_norm_tensor = torch.tensor(y_norm,requires_grad=True).view(-1, 1)
    du_dy_norm_tensor = torch.tensor(du_dy_norm,requires_grad=True).view(-1, 1)

    #Import Model for Anisotropic Tensor
    rui_model = torch.load('model_Rui_CNN.pt')
    rui_model = rui_model.to(torch.float64)
    rui_model = rui_model.cuda()
    factor = 1e3 # the anisotropic stress tensor is multiplied by this factor

    a_uv_norm = rui_model(du_dy_norm_tensor*u_tau,Re_tau*torch.ones_like(torch.tensor(y_norm,requires_grad=True).view(-1, 1)),y_norm_tensor*Re_tau)/(factor*(u_tau**2))
    da_uv_dy_norm = diff(rui_model(du_dy_norm_tensor*u_tau,Re_tau*torch.ones_like(torch.tensor(y_norm,requires_grad=True).view(-1, 1)),y_norm_tensor*Re_tau)/(factor*(u_tau**2)),y_norm_tensor)

    if output_format == 'np':
        a_uv_norm = a_uv_norm.detach().cpu().numpy()
        a_uv_norm = a_uv_norm.reshape(y_norm_tensor.shape)
        da_uv_dy_norm = da_uv_dy_norm.detach().cpu().numpy().reshape(y_norm_tensor.shape)

    elif output_format == 'torch':
        a_uv_norm = a_uv_norm
        da_uv_dy_norm = da_uv_dy_norm
    else:
        print('Error') 
    return a_uv_norm if output == 'a_uv' else da_uv_dy_norm


def a_uv_from_Ocariz(du_dy_norm,y_norm,Retau,output,output_format):

    df_Ocariz = pd.concat([make_dataframe(Retau, nondim='k-eps')])
    y_norm_DNS = df_Ocariz['y']
    k_turb = df_Ocariz['k']
    eps = df_Ocariz['eps']
    f = interp1d(y_norm_DNS, k_turb,kind='nearest', fill_value='extrapolate')
    f2 = interp1d(y_norm_DNS, eps,kind='nearest', fill_value='extrapolate')

    # Channel Parameters
    Re_tau, u_tau, nu = channel_parameters(Retau)

    # Inputs
    y_norm_tensor = torch.tensor(y_norm,requires_grad=True).view(-1, 1)
    xnew = y_norm_tensor.detach().cpu().numpy()
    k_turb_new = f(xnew)
    eps_new = f2(xnew)   
    k_turb_new = torch.tensor(k_turb_new.reshape(y_norm_tensor.shape))
    eps_new = torch.tensor(eps_new.reshape(y_norm_tensor.shape))
    factor = 1e1
    
    #Ocariz's Model
    ocariz_model = torch.load('model_Ocariz_CNN')
    ocariz_model = ocariz_model.to(torch.float64)
    ocariz_model = ocariz_model.cpu()

    # Format validation x 
    du_dy_norm_tensor = torch.tensor(du_dy_norm,requires_grad=True).view(-1, 1)
    du_dy_norm_k_tensor = du_dy_norm_tensor*torch.tensor(k_turb_new/(2.*eps_new*Re_tau),requires_grad=True)
    du_dy_in = du_dy_norm_k_tensor.view(1,1,len(y_norm)).float()

    # Add Reynolds number
    Re_tensor = Re_tau*1e-8*torch.ones_like(du_dy_norm_tensor,requires_grad=True)
    Re_in = Re_tensor.view(1,1,len(Re_tensor)).float()

    # Get dimensioning correct
    maxlength = y_norm_tensor.shape[0]
    zerosval = torch.zeros(1,1,maxlength-y_norm_tensor.shape[0])
    zerosval_double = torch.zeros(1,2,maxlength-y_norm_tensor.shape[0])
    shapexval=y_norm_tensor.shape[0]
    
    a_uv_norm = torch.mean(ocariz_model(torch.cat((torch.cat((du_dy_in,Re_in),dim=1),zerosval_double), 2)[:,:,0:shapexval].cpu(),torch.cat(((u_tau/nu)*y_norm_tensor.view(1,1,len(y_norm_tensor)).float(),zerosval), 2)[:,:,0:shapexval].cpu()), dim=0).reshape(torch.tensor(k_turb_new.cpu()).shape)*2*torch.tensor(k_turb_new.cpu())/factor
    da_uv_dy_norm = diff(torch.mean(ocariz_model(torch.cat((torch.cat((du_dy_in,Re_in),dim=1),zerosval_double), 2)[:,:,0:shapexval].cpu(),torch.cat(((u_tau/nu)*y_norm_tensor.view(1,1,len(y_norm_tensor)).float(),zerosval), 2)[:,:,0:shapexval].cpu()), dim=0).reshape(torch.tensor(k_turb_new.cpu()).shape)*2*torch.tensor(k_turb_new.cpu())/factor,y_norm_tensor)

    if output_format == 'np':
        a_uv_norm = a_uv_norm.detach().cpu().numpy()
        a_uv_norm = a_uv_norm.reshape(du_dy_norm_tensor.shape)
        da_uv_dy_norm = da_uv_dy_norm.detach().cpu().numpy().reshape(du_dy_norm_tensor.shape)

    elif output_format == 'torch':
        a_uv_norm = a_uv_norm
        da_uv_dy_norm = da_uv_dy_norm
    else:
        print('Error') 
    return a_uv_norm if output == 'a_uv' else da_uv_dy_norm


