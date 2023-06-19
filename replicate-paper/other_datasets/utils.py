import numpy as np
import torch
import torch.nn.functional as F
from ds import *
from networks import *
import os, sys
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from tqdm import *
import matplotlib.pyplot as plt

def train_classifier_loop(
    train_loader,
    val_loader,
    Net,
    n_epochs=100,
    init_lr=1e-4,
    eval_every = 10,
    dtype = torch.cuda.FloatTensor,
    device='cuda',
    ckpt_path = '../ckpt/Net'
):
    Net.to(device)
    Net.type(dtype)
    optimNet = torch.optim.Adam(list(Net.parameters()), lr=init_lr)
    optimNet_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimNet, n_epochs)
    
    last_best_acc = 0
    for eph in range(n_epochs):
        Net.train()
        with tqdm(train_loader, unit='batch') as tepoch:
            for xin, yout in tepoch:
                tepoch.set_description(f"Epoch {eph}")
            
                xin, yout = xin.to(device), yout.to(device)
                optimNet.zero_grad()
                output = F.log_softmax(Net(xin))
                predictions = output.argmax(dim=1, keepdim=True).squeeze()
                loss = F.nll_loss(output, yout)
                correct = (predictions == yout).sum().item()
                accuracy = correct / xin.shape[0]
                
                loss.backward()
                optimNet.step()

                tepoch.set_postfix(loss=loss.item(), accuracy=100.*accuracy)
        torch.save(Net.state_dict(), ckpt_path+'_last.pth')
        
        if eph%eval_every == 0 or eph == n_epochs-1:
            acc = eval_classifier_loop(
                val_loader,
                Net,
                dtype, device
            )
            if acc > last_best_acc:
                last_best_acc = acc
                torch.save(Net.state_dict(), ckpt_path+'_best.pth')
                
def eval_classifier_loop(
    eval_loader,
    Net,
    dtype = torch.cuda.FloatTensor,
    device='cuda',
):
    Net.to(device)
    Net.type(dtype)
    Net.eval()
    tot_corr = 0
    tot_samples = 0
    with tqdm(eval_loader, unit='batch') as tepoch:
        for xin, yout in tepoch:
            tepoch.set_description("Evaluating ...")

            xin, yout = xin.to(device), yout.to(device)
            output = F.log_softmax(Net(xin))
            predictions = output.argmax(dim=1, keepdim=True).squeeze()
            correct = (predictions == yout).sum().item()
            accuracy = correct / xin.shape[0]
            tot_corr += correct
            tot_samples += xin.shape[0]
            tepoch.set_postfix(accuracy=100.*accuracy)
    print('overall accuracy: {}'.format(100*tot_corr/tot_samples))
    return tot_corr/tot_samples

##################### AutoEncoder

def train_AE_loop(
    train_loader,
    val_loader,
    Net,
    n_epochs=100,
    init_lr=1e-4,
    eval_every = 10,
    dtype = torch.cuda.FloatTensor,
    device='cuda',
    ckpt_path = '../ckpt/Net'
):
    Net.to(device)
    Net.type(dtype)
    optimNet = torch.optim.Adam(list(Net.parameters()), lr=init_lr)
    optimNet_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimNet, n_epochs)
    
    last_best_score = 10000
    for eph in range(n_epochs):
        Net.train()
        with tqdm(train_loader, unit='batch') as tepoch:
            for xin, yout in tepoch:
                tepoch.set_description(f"Epoch {eph}")
            
                xin, yout = xin.to(device), yout.to(device)
                optimNet.zero_grad()
                output, _ = Net(xin)
                loss = F.l1_loss(output, xin)
                
#                 plt.imshow(output[0].to('cpu').data.numpy()[0])
#                 plt.show()
#                 plt.imshow(xin[0].to('cpu').data.numpy()[0])
#                 plt.show()
                
                loss.backward()
                optimNet.step()

                tepoch.set_postfix(loss=loss.item())
        torch.save(Net.state_dict(), ckpt_path+'_last.pth')
        
        if eph%eval_every == 0 or eph == n_epochs-1:
            acc = eval_AE_loop(
                val_loader,
                Net,
                dtype, device
            )
            if acc < last_best_score:
                last_best_score = acc
                torch.save(Net.state_dict(), ckpt_path+'_best.pth')
                
def eval_AE_loop(
    eval_loader,
    Net,
    dtype = torch.cuda.FloatTensor,
    device='cuda',
):
    Net.to(device)
    Net.type(dtype)
    Net.eval()
    tot_err = 0
    tot_samples = 0
    with tqdm(eval_loader, unit='batch') as tepoch:
        for xin, yout in tepoch:
            tepoch.set_description("Evaluating ...")

            xin, yout = xin.to(device), yout.to(device)
            output, _ = Net(xin)
            err = F.l1_loss(output, xin).item()
            tot_err += err 
            tot_samples += 1 #since l1 loss avgs over batch
            tepoch.set_postfix(Error=err)
    print('overall accuracy: {}'.format(tot_err/tot_samples))
    return tot_err/tot_samples


def show_AE(
    eval_loader,
    Net,
    dtype = torch.cuda.FloatTensor,
    device='cuda',
    n_show = 5,
):
    Net.to(device)
    Net.type(dtype)
    Net.eval()
    tot_err = 0
    tot_samples = 0
    for idx, (xin, yout) in enumerate(eval_loader):
        if idx>= n_show:
            break
        xin, yout = xin.to(device), yout.to(device)
        output, _ = Net(xin)
        err = F.l1_loss(output, xin).item()
        tot_err += err 
        tot_samples += 1 #since l1 loss avgs over batch
        n_batch = xin.shape[0]
        for j in range(n_batch):
            i1 = xin[j].data.cpu().transpose(0,2).transpose(0,1).numpy()
            i1_rec = output[j].data.cpu().transpose(0,2).transpose(0,1).numpy()
            
            plt.figure()
            plt.title(yout[j].data.cpu().numpy())
            plt.subplot(1,2,1)
            plt.imshow(i1)
            plt.subplot(1,2,2)
            plt.imshow(i1_rec)
            plt.show()
            
def train_classifier_w_AE_loop(
    train_loader,
    val_loader,
    NetAE,
    Net,
    n_epochs=100,
    init_lr=1e-4,
    eval_every = 10,
    dtype = torch.cuda.FloatTensor,
    device='cuda',
    ckpt_path = '../ckpt/Net'
):
    NetAE.to(device)
    NetAE.type(dtype)
    # Freeze training for all layers
    for param in NetAE.parameters():
        param.require_grad = False
    NetAE.eval()
    
    Net.to(device)
    Net.type(dtype)
    optimNet = torch.optim.Adam(list(Net.parameters()), lr=init_lr)
    optimNet_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimNet, n_epochs)
    
    last_best_acc = 0
    for eph in range(n_epochs):
        Net.train()
        with tqdm(train_loader, unit='batch') as tepoch:
            for xin, yout in tepoch:
                tepoch.set_description(f"Epoch {eph}")
            
                xin, yout = xin.to(device), yout.to(device)
                xin_rec, _ = NetAE(xin)
                optimNet.zero_grad()
                output = F.log_softmax(Net(xin_rec))
                predictions = output.argmax(dim=1, keepdim=True).squeeze()
                loss = F.nll_loss(output, yout)
                correct = (predictions == yout).sum().item()
                accuracy = correct / xin.shape[0]
                
                loss.backward()
                optimNet.step()

                tepoch.set_postfix(loss=loss.item(), accuracy=100.*accuracy)
        torch.save(Net.state_dict(), ckpt_path+'_last.pth')
        
        if eph%eval_every == 0 or eph == n_epochs-1:
            acc = eval_classifier_w_AE_loop(
                val_loader,
                NetAE,
                Net,
                dtype, device
            )
            if acc > last_best_acc:
                last_best_acc = acc
                torch.save(Net.state_dict(), ckpt_path+'_best.pth')
                
def eval_classifier_w_AE_loop(
    eval_loader,
    NetAE,
    Net,
    dtype = torch.cuda.FloatTensor,
    device='cuda',
):
    NetAE.to(device)
    NetAE.type(dtype)
    # Freeze training for all layers
    for param in NetAE.parameters():
        param.require_grad = False
    NetAE.eval()
    
    Net.to(device)
    Net.type(dtype)
    Net.eval()
    tot_corr = 0
    tot_samples = 0
    with tqdm(eval_loader, unit='batch') as tepoch:
        for xin, yout in tepoch:
            tepoch.set_description("Evaluating ...")

            xin, yout = xin.to(device), yout.to(device)
            xin_rec, _ = NetAE(xin)
            output = F.log_softmax(Net(xin_rec))
            predictions = output.argmax(dim=1, keepdim=True).squeeze()
            correct = (predictions == yout).sum().item()
            accuracy = correct / xin.shape[0]
            tot_corr += correct
            tot_samples += xin.shape[0]
            tepoch.set_postfix(accuracy=100.*accuracy)
    print('overall accuracy: {}'.format(100*tot_corr/tot_samples))
    return tot_corr/tot_samples