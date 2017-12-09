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


'''parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-d','--data', metavar='DIR',  default='models/',
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

best_prec1 = 0'''


def main():
    #global args, best_prec1
    #args = parser.parse_args()
    tr = torch.load('1000targetOrg')
    p5=torch.load('500prediction')

    p3=torch.load('300prediction500Org')
    p2 =torch.load('200prediction500Org')
    p=torch.cat((p5,p3,p2),1)

    pp3 = torch.load('300prediction')
    pp2 = torch.load('200prediction')
    pp = torch.cat((p5, pp3, pp2), 1)

    ppp3 = torch.load('300L4prediction500Org')
    ppp2 = torch.load('200L4prediction500Org')
    ppp = torch.cat((p5, ppp3, ppp2), 1)


    prec1, prec5 = accuracy(p, tr, topk=(1, 5))
    print('Accuracy Incremental w/o share 500 start','p1',prec1,'p5',prec5)

    prec1, prec5 = accuracy(pp, tr, topk=(1, 5))
    print('Accuracy Incremental w/ share L34 500 start','p1',prec1,'p5',prec5)

    prec1, prec5 = accuracy(ppp, tr, topk=(1, 5))
    print('Accuracy Incremental w/ share L4 500 start','p1',prec1,'p5',prec5)



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    #print('op', output, 'tr', target)
    target=target.type(torch.LongTensor)
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
