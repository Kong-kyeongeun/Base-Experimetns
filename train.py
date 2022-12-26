import os, sys
import argparse
import shutil
import time
import numpy as np
import pandas as pd

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from collections import OrderedDict
from termcolor import colored

import nni
from nni.compression.pytorch.utils import count_flops_params

import models

parser = argparse.ArgumentParser(description="train")
parser.add_argument('--arch', type=str, default = 'resnet')
parser.add_argument('--dataset', type=str, default = 'cifar10')
parser.add_argument("--save", type=str, default="")
parser.add_argument("--evaluate",  action='store_true', default=False)
parser.add_argument("--extract",  action='store_true', default=False)
parser.add_argument("--weight_path", type=str, default="")
parser.add_argument("--datapath", type=str, default="/disk")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--ngpu", type=str, default="cuda:0")
parser.add_argument("--cuda", type=bool, default=True)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--test_batch_size", type=int, default=256)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--wd", type=float, default=5e-4)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument('--milestones', type=int, nargs='+', default=[30, 60])
parser.add_argument("--epochs", type=int, default=200)

args = parser.parse_args()
################################ Check #######################################
print("=> Parameter : {}".format(args))

if os.path.isdir(args.save) and not(args.evaluate) and not(args.extract):
    print("weight already exists")
    exit()

if args.arch == "vgg":
    model_name = "vgg16"
else:
    model_name = args.arch
################################ Check #######################################


################################# train method #########################################
torch.manual_seed(args.seed)
args.cuda = args.cuda and torch.cuda.is_available()
if args.cuda:
    torch.cuda.manual_seed(args.seed)
device = args.ngpu if args.cuda else "cpu"
datapath = os.path.join(args.datapath , args.dataset)
g_epoch = 0
def train(model, optimizer, criterion):
    global g_epoch
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        #data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                  g_epoch, batch_idx * len(data), len(train_loader.dataset),
                  100. * batch_idx / len(train_loader), loss.data.item()))
    g_epoch += 1
    return loss.data.item()

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        tot_correct = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            tot_correct.append(correct_k)
        return tot_correct
    
def test(model, ece=False):
    model.eval()
    test_loss = 0
    corr1 = 0
    corr5 = 0
    criterion = nn.CrossEntropyLoss().to(device)
    
    if ece:
        logit_set = np.zeros([10000, 10])
        counter = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            #data, target = Variable(data, volatile=True), Variable(target, volatile=True)
            output = model(data)
            if ece:
                for i, per_sample in enumerate(output):
                    logit_set[counter+i] = per_sample.detach().cpu().numpy()
                counter += output.size(0)

            test_loss += criterion(output, target).item()

            corr1_, corr5_ = accuracy(output, target, topk=(1, 5))
            corr1 += corr1_
            corr5 += corr5_

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}\nTop-1 Accuracy: {}/{} ({:.2f}%), Top-5 Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, corr1.item(), len(test_loader.dataset),
            100. * float(corr1.item() / len(test_loader.dataset)),
            corr5.item(), len(test_loader.dataset),
            100. * float(corr5.item() / len(test_loader.dataset))))
    if ece:
        return float(corr1.item()/len(test_loader.dataset)), float(corr5.item()/len(test_loader.dataset)), test_loss, logit_set
    else:
        return float(corr1.item()/len(test_loader.dataset)), float(corr5.item()/len(test_loader.dataset)), test_loss
def save_checkpoint(state, is_best, filepath):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))

################################# load Dataset #########################################
if args.dataset.startswith('cifar'):
    CIFAR = datasets.CIFAR10 if args.dataset=="cifar10" else datasets.CIFAR100
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.1994, 0.2010))
    kwargs = {'num_workers': 4, 'pin_memory': True} 
    train_loader = torch.utils.data.DataLoader(
        CIFAR(datapath, train=True, download=False, transform=transforms.Compose([
                            transforms.Pad(4),
                            transforms.RandomCrop(32),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize
                            ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        CIFAR(datapath, train=False, download=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            normalize,
            ])),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)


elif args.dataset == "imagenet":
    kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}
    traindir = os.path.join(datapath, 'train')
    testdir = os.path.join(datapath, 'val3')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(traindir, transforms.Compose([
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize,
                                    ]))
    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, sampler=None, **kwargs)

    test_dataset = datasets.ImageFolder(testdir, transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    normalize,
                                    ]))
    test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, shuffle=False, sampler=None, **kwargs)

################################# load Dataset #########################################

################################# load model ##########################################
# init
if args.dataset == 'imagenet':
    if args.arch.startswith('resnet'):
        model = models.__dict__[args.arch]()
    else:
        model = models.__dict__[args.arch]()
    
    flops, params, results = count_flops_params(model, torch.randn([128, 3, 224, 224]))

elif args.dataset == 'cifar10' or args.dataset == 'cifar100':
    kwargs = {'dataset': args.dataset}
    model = models.__dict__[args.arch](**kwargs)
    flops, params, results = count_flops_params(model, torch.randn([128, 3, 32, 32]))

if args.cuda:
    model = model.to(device)
################################# load model ##########################################

################################# main ###############################################
if __name__ == "__main__":
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=args.milestones, gamma=0.1)
    criterion = nn.CrossEntropyLoss().to(device)
    ################################# test model #########################################
    if args.evaluate:
        print("Evaluate")
        print("Batch Size : ", args.test_batch_size)
        reptitions = 300
        time = [0] * reptitions
        dict_ = torch.load(args.weight_path,map_location=args.ngpu)
        if args.dataset == 'imagenet':
            if args.arch.startswith('resnet'):
                model = models.__dict__[args.arch](cfg = dict_["cfg"])
            else:
                model = models.__dict__[args.arch](cfg = dict_["cfg"])
        
        elif args.dataset == 'cifar10' or args.dataset == 'cifar100':
            kwargs = {'dataset': args.dataset}
            model = models.__dict__[args.arch](**kwargs , cfg = dict_["cfg"])
            flops, params, results = count_flops_params(model, torch.randn([128, 3, 32, 32]))
        
        model.load_state_dict(dict_["state_dict"])

        if args.cuda:
            model = model.to(device)

        model.eval()
        with torch.no_grad():
            for i in range(reptitions):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = model(torch.randn([64, 3, 32, 32]).to(device))
        #        test(model)
                end.record()
                torch.cuda.synchronize()
                time[i] = start.elapsed_time(end)
            print("Inference Time : " , sum(time) / len(time))
        ################################# test model #########################################
    elif args.extract:
        print("filter extract")
        savepath = "./filter_weights/{}_{}/baseline".format(model_name,args.dataset)
        if os.path.exists(savepath)==False:
            os.makedirs(savepath)
        dict_ = torch.load(args.weight_path)
        model.load_state_dict(dict_["state_dict"])
        test(model)

        conv_layer = 0
        for layer in model.modules():
            print(layer)
            if isinstance(layer, nn.Conv2d):
                print("weights shape ", layer.weight.shape)
                layer_weight = layer.weight.data.cpu().numpy()
                np.save(os.path.join(savepath,"{}".format(conv_layer)),layer_weight)
                conv_layer +=1
            print("--")
    ################################# train model #########################################
    else:
        print("train")
        best_prec1 = 0.
        if os.path.exists(args.save)==False:
            os.makedirs(args.save)
        for i in range(args.epochs):
            train_loss = train(model, optimizer, criterion)
            prec1, prec5, test_loss = test(model)

            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': g_epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best, filepath=args.save)
            print("Best accuracy: "+str(best_prec1))
            scheduler.step()
        print("Finished saving training history")
    ################################# train model #########################################

