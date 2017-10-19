import argparse
import os
import shutil
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import cv2
import random

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
#parser.add_argument('--data', metavar='DIR',help='path to dataset')
parser.add_argument('--model', metavar='DIR',
                    help='path to get trained model')
parser.add_argument('--arch', '-a', metavar='ARCH', default='alexnet',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: alexnet)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=100, type=int,
                    metavar='N', help='mini-batch size (default: 100)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')


best_prec1 = 0


def test():
    global args, best_prec1
    args = parser.parse_args()

    # load model
    model=AlexNet(200)
    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()
    # define loss function (criterion) and optimizer
    #criterion = nn.CrossEntropyLoss().cuda()
    if args.model:
        mod_file=os.path.join(args.model,'model_best.pth.tar')
        if os.path.isfile(mod_file):
            print("=> loading model ")
            checkpoint = torch.load(mod_file)
            model.load_state_dict(checkpoint['state_dict'])
            #optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded model 'alexnet'")
        else:
            print("=> no model found at '{}'".format(args.model))

    #print(model)
    cudnn.benchmark = True
    '''
    # Data loading code
    valdir = os.path.join(args.data, 'val')

    #~~~~~~~Validation Parse~~~~~~~#
    annote_source=os.path.join(valdir,"val_annotations.txt")
    val_annote = open(annote_source, "r")
    for x in val_annote.readlines():
        source = os.path.join(valdir,'images', x.split()[0])
        destination = os.path.join(valdir,x.split()[1], x.split()[0])
        if os.path.exists(source):
            if not os.path.exists((os.path.join(valdir,x.split()[1]))):
                os.makedirs((os.path.join(valdir,x.split()[1])))
            os.rename(source, destination)
    val_annote.close()
    if os.path.exists(os.path.join(valdir,'images')):
        os.rmdir(os.path.join(valdir,'images'))
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
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

    if 0:
        validate(val_loader, model, criterion)
        return
    '''
    cam(model,0)

def cam (model,idx=0):
    cap = cv2.VideoCapture(0)
    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        #~~~~~~~~~~~~~use random image as input if webcam doesn't work~~~~~~~~~~~~~~~~~~~#
        '''serial=('val_1880.JPG','val_3903.JPG','val_4924.JPG','val_611.JPG','val_691.JPG');  ##random image
        frame = cv2.imread(serial[random.randrange(0,5)], cv2.IMREAD_COLOR) '''
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

        show=frame

        # Our operations on the frame come here
        if len(frame) is 480:
            frame = frame[:, 80:559]
            frame = cv2.resize(frame, None, fx=2.143, fy=2.143, interpolation=cv2.INTER_AREA)
        else:
            frame=cv2.resize(frame,(224,224),interpolation=cv2.INTER_LINEAR)
        frame=torch.from_numpy(frame);        frame=frame.type(torch.FloatTensor);        frame=torch.transpose(torch.FloatTensor(frame),0,2);        frame=frame.float().div(255)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        frame=normalize(frame);        frame=torch.unsqueeze(frame,0)
        model.eval()
        x= model(torch.autograd.Variable(frame))

        ### Create Caption
        tout = x.data
        tout, index = torch.max(tout, 1)
        index = torch.squeeze(index.cpu(), 1)       #class
        #capList = ('0','1','2','3','4')            #couldn't get class caption
        #caption = capList[index]

        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        show = cv2.resize(show, (32 * 8, 32 * 8), interpolation=cv2.INTER_CUBIC)
        cv2.putText(show, str(index.numpy()), (10, 120), font, 1.5, (0, 0, 255))
        print('Press "q" or "Ctrl c" to close window')
        cv2.imshow('Image', show)
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break
        cv2.destroyAllWindows()
    # When everything done, release the capture
    cap.release(); cv2.destroyAllWindows()

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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

class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    test()
