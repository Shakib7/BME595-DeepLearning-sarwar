from __future__ import print_function
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
import cv2

class Img2Obj(nn.Module):
    def __init__(self):
        super(Img2Obj,self).__init__()
        self.conv1=nn.Conv2d(3,6,5)
        self.conv2 =nn.Conv2d(6, 16, 5)
        self.fc1 =nn.Linear(16*5*5, 120)
        self.fc2 =nn.Linear(120, 84)
        self.fc3 =nn.Linear(84, 100)
        self.criterion=nn.MSELoss(size_average=True)

    def __call__(self, img):
        return self.forward(img)

    def forward(self, img):
        img = (img.type(torch.FloatTensor))/ 255
        if (img.numpy()).ndim is 3:
            x=Variable(torch.unsqueeze(img,0))
        else:
            x=Variable(img.type(torch.FloatTensor))
        x=F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        ###convert to feature vector
        size=x.size()[1:]
        num_features=1
        for s in size:
            num_features*=s
        x = x.view(-1, num_features)
        ### Fully Connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        ### Create Caption
        tout=x.data
        tout, index = torch.max(tout, 1)
        index=torch.squeeze(index,1)
        capList=('apple','aquarium_fish','baby','bear','beaver','bed','bee','beetle','bicycle','bottle','bowl','boy','bridge','bus','butterfly','camel','can','castle','caterpillar','cattle','chair','chimpanzee','clock','cloud','cockroach','couch','crab','crocodile','cup','dinosaur','dolphin','elephant','flatfish','forest','fox','girl','hamster','house','kangaroo','keyboard','lamp','lawn_mower','leopard','lion','lizard','lobster','man','maple_tree','motorcycle','mountain','mouse','mushroom','oak_tree','orange','orchid','otter','palm_tree','pear','pickup_truck','pine_tree','plain','plate','poppy','porcupine','possum','rabbit','raccoon','ray','road','rocket','rose','sea','seal','shark','shrew','skunk','skyscraper','snail','snake','spider','squirrel','streetcar','sunflower','sweet_pepper','table','tank','telephone','television','tiger','tractor','train','trout','tulip','turtle','wardrobe','whale','willow_tree','wolf','woman','worm')
        caption=capList[index[0]]
        return caption

    def view(self,img):
        caption=self.forward(img)
        img = (torch.transpose(img, 0, 2))/255
        show=img.numpy()
        font=cv2.FONT_HERSHEY_COMPLEX_SMALL
        show=cv2.resize(show,(32*8,32*8),interpolation=cv2.INTER_LINEAR)
        cv2.putText(show,caption,(10,120),font,1.5,(0,0,255))
        cv2.imshow('Image',show)
        print ('Press anykey to close window')
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def cam (self,idx=0):
        cap = cv2.VideoCapture(0)
        while (True):
            # Capture frame-by-frame
            ret, frame = cap.read()
            #serial=('1.jpg','2.jpg','3.jpg');   frame = cv2.imread(serial[random.randrange(0,3)], cv2.IMREAD_COLOR)
            # Our operations on the frame come here
            if len(frame) is 480:
                frame = frame[:, 80:559]
                frame = cv2.resize(frame, None, fx=.0667, fy=.0667, interpolation=cv2.INTER_AREA)
            else:
                frame=cv2.resize(frame,(32,32),interpolation=cv2.INTER_LINEAR)

            frame=torch.transpose(torch.ByteTensor(frame),0,2)
            caption = self.forward(frame)
            frame = (torch.transpose(frame, 0, 2))
            show = frame.numpy()
            font = cv2.FONT_HERSHEY_COMPLEX_SMALL
            show = cv2.resize(show, (32 * 8, 32 * 8), interpolation=cv2.INTER_CUBIC)
            cv2.putText(show, caption, (10, 120), font, 1.5, (0, 0, 255))
            print('Press "q" to close window')
            cv2.imshow('Image', show)
            if cv2.waitKey(250) & 0xFF == ord('q'):
                break
            cv2.destroyAllWindows()
        # When everything done, release the capture
        cap.release(); cv2.destroyAllWindows()

    def train(self):
        learning_rate = 0.75
        optimizer = optim.SGD(self.parameters(),lr=learning_rate,momentum=0.0,weight_decay=0.00)

        # ~~~~~~~~~~~~~~~~~Train Network~~~~~~~~~~~~~~~~~~~#
        # Prep Dataset
        train_loader = torchvision.datasets.CIFAR100('../data_CIFAR100', train=True, download=True, transform=True,target_transform=True)

        epoch = 251;  batch_size = 100;   numCls = 100;    iteration = len(train_loader) / batch_size
        input = (torch.from_numpy(train_loader.train_data)).type(torch.FloatTensor)
        input = (torch.transpose(input.view(iteration, batch_size, 32, 32, 3), 2, 4))/255
        target = torch.zeros((len(train_loader.train_labels), numCls))

        ##one hot encode
        for i in range(len(train_loader.train_labels)):
            target[i][train_loader.train_labels[i]] = 1
        #print(train_loader.train_labels[i])
        #print('target', target)
        target = Variable(target.view(iteration, batch_size, numCls))

        #### epochs
        ###train
        for i in range(epoch):
            index = random.sample(range(iteration), iteration)
            e = 0
        #    with open("outLoss"+str(i)+".csv", "w") as out_file:
            for j in range(iteration):
                optimizer.zero_grad()
                ###Forward
                x=Variable (input[index[j]])
                x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
                x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
                size = x.size()[1:]
                num_features = 1
                for s in size:
                    num_features *= s
                x = x.view(-1, num_features)
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)

                ###back-propagation
                loss=self.criterion(x, target[index[j]])
                e +=loss.data[0]
                #print ('loss(ij)',i,j,':',loss.data[0] )
                loss.backward()
                optimizer.step()
            print ('loss(i)',i,':',e)
                #out_file.write(str(e))
            if (i % 10):
                pass
            else:
                self.test()

    def test(self):
        #~~~~~~~~~~~~~~~~~Test Network~~~~~~~~~~~~~~~~~~~#
        test_loader = torchvision.datasets.CIFAR100('../data_CIFAR100', train=False, download=True, transform=True,target_transform=True)
        batch_size = 1000;	numCls = 100;	iterationT = len(test_loader) / batch_size
        inputT = (torch.from_numpy(test_loader.test_data)).type(torch.FloatTensor)
        inputT = (torch.transpose(inputT.view(iterationT, batch_size, 32, 32, 3), 2, 4))/255
        targetT = torch.zeros((len(test_loader.test_labels), numCls))

        ##one hot encode
        for i in range(len(test_loader.test_labels)):
            targetT[i][test_loader.test_labels[i]] = 1
        targetT = Variable(targetT.view(iterationT, batch_size, numCls))
        err=0
        for i in range(iterationT):   #iterationT
            ###Forward
            x = Variable(inputT[i])
            x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
            x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
            size = x.size()[1:]
            num_features = 1
            for s in size:
                num_features *= s
            x = x.view(-1, num_features)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            outputT = self.fc3(x)
            #outputT=self.forward(Variable (inputT[i]))
            #print ('O',outputT)
            tout, index = torch.max(outputT.data, 1)
            predict = torch.zeros(len(index), 100)
            ##one hot encode
            for j in range(len(index)):
                predict[j][index[j]] = 1
            outputT =predict
            #print('O', outputT)
            #print ('T',targetT[i].data)
            err += len(torch.nonzero(outputT - targetT[i].data)) / 2
        print ('acc',100-((err*100)/len(test_loader.test_labels)))
