from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import time
import pandas as pd
from PIL import ImageFile
from collections.abc import Iterable

# set the device to run the training
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64

# # set the transform on the training set, including resize, HorizontalFlip, VerticalFlip, RandomRotation, RandomCrop
# transform_train = transforms.Compose([
#     transforms.Resize((150,150)),
#     transforms.RandomHorizontalFlip(0.5),
#     transforms.RandomVerticalFlip(0.5),
#     transforms.RandomRotation(180),
#     transforms.RandomCrop(128),
#     transforms.ToTensor(),
#     # transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
# ])

# set the transform on the test set, only including resize
transform_test = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
])

class MyDataset(Dataset):
    def __init__(self,root_path,target_transform=None):
        self.root_path = root_path
        self.data_files = os.listdir(self.root_path)
    def __getitem__(self,item):
        data_file = self.data_files[item]
#         path = data_file.split('/')[0]
        file = os.path.join(self.root_path,data_file)
        name = data_file.split('.')[0]
        label = int(name.split('_')[-1])
        img = Image.open(file).convert('RGB')
        img = transform_test(img)
        label = torch.tensor(label)
        return img, label,name
    def __len__(self):
        return len(self.data_files)


train_datasets = MyDataset(r'./NIMS_image/Total_checked_Train/')
test_datasets = MyDataset(r'./NIMS_image/Total_checked_Test/')

trainloader = torch.utils.data.DataLoader(train_datasets,batch_size=BATCH_SIZE,shuffle=True)
testloader = torch.utils.data.DataLoader(test_datasets,batch_size=BATCH_SIZE,shuffle=True)

for ite in range(20):
    class VGGNet(nn.Module):
        def __init__(self,num_class = 2):
            super(VGGNet,self).__init__()
            #net = models.vgg16(pretrained=False)
            net = models.vgg16(pretrained=True)
            net.classifier = nn.Sequential()
            self.features = net
            self.classifier = nn.Sequential(
                nn.Linear(512*7*7,1024),
                nn.ReLU(True),
                nn.Dropout(0.3),
                nn.Linear(1024,100),
                nn.ReLU(True),
                nn.Dropout(0.3),
                nn.Linear(100,10),
                nn.ReLU(True),
                nn.Dropout(0.3),
                nn.Linear(10,num_class),
            )
            self.softmax = nn.Softmax(dim=1)
        def forward(self,x):
            x = self.features(x)
            x = x.view(x.size(0),-1)
            x = self.classifier(x)
            prob = self.softmax(x)
            return x,prob

    class DenseNet(nn.Module):
        def __init__(self,num_class = 2):
            super(DenseNet,self).__init__()
            net = models.vgg16(pretrained=True)
            net.classifier = nn.Sequential()
            self.features = net
            self.classifier = nn.Sequential(
                nn.Linear(512*7*7,100),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(100,num_class),
            )
            self.softmax = nn.Softmax(dim=1)
        def forward(self,x):
            x = self.features(x)
            x = x.view(x.size(0),-1)
            x = self.classifier(x)
            prob = self.softmax(x)
            return x,prob

    class ResNet(nn.Module):
        def __init__(self, num_class=2):
            super(ResNet, self).__init__()
            net = models.resnet18(pretrained=True)
            net.classifier = nn.Sequential()
            self.features = net
            self.classifier = nn.Sequential(
                nn.Linear(1000,100),
                nn.ReLU(True),
                nn.Dropout(0.3),
                nn.Linear(100,10),
                nn.ReLU(True),
                nn.Dropout(0.3),
                nn.Linear(10,num_class),
            )
            self.softmax = nn.Softmax(dim=1)
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            prob = self.softmax(x)
            return x,prob

    # net = DenseNet().to(device)

    def set_freeze_by_names(model,layer_names,freeze=True):
        if not isinstance(layer_names, Iterable):
            layer_names = [layer_names]
        for name, child in model.named_children():
            if name not in layer_names:
                continue
            for  param in child.parameters():
                param.require_grad = not freeze
    def freeze_by_names(model,layer_names):
        set_freeze_by_names(model,layer_names,True)
        
    def unfreeze_by_names(model,layer_names):
        set_freeze_by_names(model,layer_names,False)

    net = VGGNet().to(device)
        
    optimizer = optim.Adam(net.parameters(),lr=0.00005)
    criterion = nn.CrossEntropyLoss()
    criterion.to(device=device)
    y_pre = []
    y_true = []
    y_path = []
    EPOCH = 100
    pre_epoch = 0
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    Train_loss = []
    Train_accu = []
    Test_accu = []

    #training
    freeze_by_names(net,('features'))

    print("Start VGG Training!")
    best_accu = 80
    f_path=open(f'prevgg_result_{ite}.txt','a',encoding='utf8')

    for epoch in range(pre_epoch, EPOCH):
        print( '\nEpoch: %d'%(epoch+ 1))
        net.train()
        sum_loss=0.0
        correct=0.0
        total=0.0
        for i, data in enumerate(trainloader, 0):
            length=len(trainloader)
            inputs,labels, path =data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            # forward+ backward
            outputs,prob = net(inputs)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            
            sum_loss += loss.item()*length
            total += labels.size(0)
            train_pred = torch.argmax(prob,dim=1) 
            train_criterion = (train_pred==labels).float()
            train_correct=torch.sum(train_criterion)
            correct = correct + train_correct

        #print loss and accuracy on training set for every epoch
        print('[Train] Loss:%.03f | Acc:%.3f%%'%(sum_loss/total,100.*correct/total))
        Train_loss.append(sum_loss/total)
        Train_accu.append(100.*correct.item()/total)
        #print accuracy on test set for every epoch
        with torch.no_grad():
            correct=0
            total=0
            for data in testloader:
                net.eval()
                images,labels,path=data
                images, labels = images.to(device), labels.to(device)
                outputs,prob=net(images)
                
                total += labels.size(0)
                test_pred = torch.argmax(prob,dim=1) 
                test_criterion = (test_pred==labels).float()
                test_correct=torch.sum(test_criterion)
                correct += test_correct
                
                #y_pre.append(test_pred)
                #y_true.append(labels)
                #y_path.append(path)

                #record the true label and prediction for each image in test set
                f_path.write(str(path))
                f_path.write('\n')
                f_path.write(str(test_pred))
                f_path.write('\n')
                f_path.write(str(labels))
                f_path.write('\n')
            print('[Test] Accuracy:%.3f%%'%(100*correct/total))
        accu = 100.*correct.item()/total
        if accu>80:
            # best_accu = accu
            torch.save(net,'prevgg_%d_1208_%.3f.pth'%(ite,accu))
        Test_accu.append(100.*correct.item()/total)
    # print('Train_loss:',list(Train_loss))
    # print('Train_accu:',list(Train_accu))
    # print('Test_accu:',list(Test_accu))
    print(max(list(Test_accu)))
    f_path.write(str(list(Train_loss)))
    f_path.write('\n')
    f_path.write(str(list(Train_accu)))
    f_path.write('\n')
    f_path.write(str(list(Test_accu)))
    f_path.write('\n')
    f_path.close()

    net = DenseNet().to(device)
        
    optimizer = optim.Adam(net.parameters(),lr=0.00005)
    criterion = nn.CrossEntropyLoss()
    criterion.to(device=device)
    y_pre = []
    y_true = []
    y_path = []
    EPOCH = 100
    pre_epoch = 0
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    Train_loss = []
    Train_accu = []
    Test_accu = []

    #training
    freeze_by_names(net,('features'))

    print("Start Dense Training!")
    best_accu = 80
    f_path=open(f'predense_result_{ite}.txt','a',encoding='utf8')

    for epoch in range(pre_epoch, EPOCH):
        print( '\nEpoch: %d'%(epoch+ 1))
        net.train()
        sum_loss=0.0
        correct=0.0
        total=0.0
        for i, data in enumerate(trainloader, 0):
            length=len(trainloader)
            inputs,labels, path =data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            # forward+ backward
            outputs,prob = net(inputs)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            
            sum_loss += loss.item()*length
            total += labels.size(0)
            train_pred = torch.argmax(prob,dim=1) 
            train_criterion = (train_pred==labels).float()
            train_correct=torch.sum(train_criterion)
            correct = correct + train_correct

        #print loss and accuracy on training set for every epoch
        print('[Train] Loss:%.03f | Acc:%.3f%%'%(sum_loss/total,100.*correct/total))
        Train_loss.append(sum_loss/total)
        Train_accu.append(100.*correct.item()/total)
        #print accuracy on test set for every epoch
        with torch.no_grad():
            correct=0
            total=0
            for data in testloader:
                net.eval()
                images,labels,path=data
                images, labels = images.to(device), labels.to(device)
                outputs,prob=net(images)
                
                total += labels.size(0)
                test_pred = torch.argmax(prob,dim=1) 
                test_criterion = (test_pred==labels).float()
                test_correct=torch.sum(test_criterion)
                correct += test_correct
                
                #y_pre.append(test_pred)
                #y_true.append(labels)
                #y_path.append(path)

                #record the true label and prediction for each image in test set
                f_path.write(str(path))
                f_path.write('\n')
                f_path.write(str(test_pred))
                f_path.write('\n')
                f_path.write(str(labels))
                f_path.write('\n')
            print('[Test] Accuracy:%.3f%%'%(100*correct/total))
        accu = 100.*correct.item()/total
        if accu>80:
            # best_accu = accu
            torch.save(net,'predense_%d_1208_%.3f.pth'%(ite,accu))
        Test_accu.append(100.*correct.item()/total)
    # print('Train_loss:',list(Train_loss))
    # print('Train_accu:',list(Train_accu))
    # print('Test_accu:',list(Test_accu))
    print(max(list(Test_accu)))
    f_path.write(str(list(Train_loss)))
    f_path.write('\n')
    f_path.write(str(list(Train_accu)))
    f_path.write('\n')
    f_path.write(str(list(Test_accu)))
    f_path.write('\n')
    f_path.close()

    net = ResNet().to(device)
        
    optimizer = optim.Adam(net.parameters(),lr=0.00005)
    criterion = nn.CrossEntropyLoss()
    criterion.to(device=device)
    y_pre = []
    y_true = []
    y_path = []
    EPOCH = 100
    pre_epoch = 0
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    Train_loss = []
    Train_accu = []
    Test_accu = []

    #training
    freeze_by_names(net,('features'))

    print("Start Res Training!")
    best_accu = 80
    f_path=open(f'preres_result_{ite}.txt','a',encoding='utf8')

    for epoch in range(pre_epoch, EPOCH):
        print( '\nEpoch: %d'%(epoch+ 1))
        net.train()
        sum_loss=0.0
        correct=0.0
        total=0.0
        for i, data in enumerate(trainloader, 0):
            length=len(trainloader)
            inputs,labels, path =data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            # forward+ backward
            outputs,prob = net(inputs)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            
            sum_loss += loss.item()*length
            total += labels.size(0)
            train_pred = torch.argmax(prob,dim=1) 
            train_criterion = (train_pred==labels).float()
            train_correct=torch.sum(train_criterion)
            correct = correct + train_correct

        #print loss and accuracy on training set for every epoch
        print('[Train] Loss:%.03f | Acc:%.3f%%'%(sum_loss/total,100.*correct/total))
        Train_loss.append(sum_loss/total)
        Train_accu.append(100.*correct.item()/total)
        #print accuracy on test set for every epoch
        with torch.no_grad():
            correct=0
            total=0
            for data in testloader:
                net.eval()
                images,labels,path=data
                images, labels = images.to(device), labels.to(device)
                outputs,prob=net(images)
                
                total += labels.size(0)
                test_pred = torch.argmax(prob,dim=1) 
                test_criterion = (test_pred==labels).float()
                test_correct=torch.sum(test_criterion)
                correct += test_correct
                
                #y_pre.append(test_pred)
                #y_true.append(labels)
                #y_path.append(path)

                #record the true label and prediction for each image in test set
                f_path.write(str(path))
                f_path.write('\n')
                f_path.write(str(test_pred))
                f_path.write('\n')
                f_path.write(str(labels))
                f_path.write('\n')
            print('[Test] Accuracy:%.3f%%'%(100*correct/total))
        accu = 100.*correct.item()/total
        if accu>80:
            # best_accu = accu
            torch.save(net,'preres_%d_1208_%.3f.pth'%(ite,accu))
        Test_accu.append(100.*correct.item()/total)
    # print('Train_loss:',list(Train_loss))
    # print('Train_accu:',list(Train_accu))
    # print('Test_accu:',list(Test_accu))
    print(max(list(Test_accu)))
    f_path.write(str(list(Train_loss)))
    f_path.write('\n')
    f_path.write(str(list(Train_accu)))
    f_path.write('\n')
    f_path.write(str(list(Test_accu)))
    f_path.write('\n')
    f_path.close()