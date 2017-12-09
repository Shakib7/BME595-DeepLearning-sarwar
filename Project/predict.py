import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-d','--data', metavar='DIR',  default='../imagenet2012_resized/',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet34',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-c', '--classes',default=1000, type=int, metavar='N',
                    help='number of total classes')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1000, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-pf', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--predict', dest='predict', default='', type=str, metavar='PATH',
                    help='predict on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()
    #num_classes=200
    #print(str(args.classes) + 'prediction')
    if args.predict:
        model = models.__dict__[args.arch]()
        model.fc = nn.Linear(512, args.classes)
        model = torch.nn.DataParallel(model).cuda()
        print(model)
        #out_prob=torch.FloatTensor(10000,num_classes)
        if os.path.isfile(args.predict):
            #print("=> loading model ".format(args.predict))
            checkpoint = torch.load(args.predict)
            #args.start_epoch = checkpoint['epoch']
            #best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            #optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded model")
        else:
            print("=> no model found at '{}'".format(args.resume))
        valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        predict(val_loader, model)
        return


def predict(val_loader, model):
    model.eval()
    out_prob=torch.FloatTensor(1,args.classes)
    #tr = torch.FloatTensor(1)
    #print('trg',tr)
    for i, (input, target) in enumerate(val_loader):
        print('batch',i)
        #target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        #target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        tout=output.data
        #print ('tout',tout)
        out_prob=torch.cat((out_prob,tout.cpu()),0)
        #print('tr',(target.type(torch.FloatTensor)).cpu())
        #tr = torch.cat((tr, (target.type(torch.FloatTensor).cpu())), 0)
        #print ('prob',tout)
        tout, index = torch.max(tout, 1)
        #print('idx b4 sqz', index)
        #index=torch.squeeze(index,1)
        #print('idx',index)
        #print('t',target)
        #if i is 2:
        #    print(tr)
        #    break
    torch.save(out_prob[1:],(str(args.classes) + 'L4prediction500Org'))
    ts=torch.load((str(args.classes) + 'L4prediction500Org'))
    print ('loaded tensor',ts)
    #return top1.avg

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
