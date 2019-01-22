 #######
# Developed by Pablo Tostado Marcos
# Last modified: Feb 15th 2018
#######

import os
os.environ['QT_QPA_PLATFORM']='offscreen' #Needed when ssh to avoid display on screen

from __future__ import print_function
# from data_loader import *
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle
from  PIL import Image
import numpy as np
import random
import sys
sys.dont_write_bytecode = True
import IPython as IP
import os
from data_loader_mnist import *
#Main to run neural network
# from stochastic_weights import *
import pickle
from mpl_toolkits.mplot3d import Axes3D
from my_helper_pytorch import *


################################## DEFAULT LOADING ######################################
###### Initialize variables:

model = 4

# mnist_dir = '/home/pablotostado/Desktop/PT/ML_Datasets/MNIST/'
mnist_dir = 'home/Users/pablo_tostado/Pablo_Tostado/ML_Datasets'
batch_size = 20

#### Default loading
transform = transforms.Compose([
                transforms.ToTensor(), # first, convert image to PyTorch tensor
                transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
            ])

trainset = torchvision.datasets.MNIST(root=mnist_dir, train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=False, num_workers=2)
#
testset = torchvision.datasets.MNIST(root=mnist_dir, train=False,
                                       transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=False, num_workers=2)

validation = True
if (validation == True):

    valid_dataset = datasets.MNIST(root=mnist_dir, train=True,
                download=True, transform=transform)
    validationloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size,
                                              shuffle=False, num_workers=2)

    validationloader.dataset.train_data = validationloader.dataset.train_data[0:6000]
    validationloader.dataset.train_labels = validationloader.dataset.train_labels[0:6000]

    trainloader.dataset.train_data = trainloader.dataset.train_data[5999:-1]
    trainloader.dataset.train_labels = trainloader.dataset.train_labels[5999:-1]

classes = ('0', '1', '2', '3',
           '4', '5', '6', '7', '8', '9')


############################ My loading ###############################################

# mnist_dir = '/home/pablotostado/Desktop/PT/ML_Datasets/MNIST/'
# global batch_size
# batch_size = 20
#
# # Load Training + Validation
# trainloader, validationloader = get_train_valid_loader(data_dir=mnist_dir,
#                                                        batch_size=batch_size,
#                                                        augment=False,
#                                                        random_seed=2,
#                                                        show_sample=False,
#                                                        shuffle=False)
# # Load Testing
# testloader = get_test_loader(data_dir=mnist_dir,
#                              batch_size=batch_size,
#                              pin_memory=True)
#
# classes = ('0', '1', '2', '3',
#            '4', '5', '6', '7', '8', '9')

######################## Data Path ########################

# Static class to add the Stochastic Rounding module to my Network:
class Binarize_and_StochRound(torch.autograd.Function):
    @staticmethod
    def forward(x):
        x = binarize_and_stochRound(x)
        return x
    @staticmethod
    def backward(grad_output):
        grad_output = binarize_and_stochRound(grad_output)
        return grad_input

class Net_mnist(nn.Module):
    def __init__(self):
        super(Net_mnist, self).__init__()
        #convolutional layers
        #the input is 3 RB

        self.binarize_and_round = Binarize_and_StochRound()

        self.fc = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 10)


        # self.fc1 = nn.Linear(1200, 1200)
        #
        # self.fc2 = nn.Linear(1200, 1200)
        # self.fc3 = nn.Linear(1200, 10)

        self.batch200 = nn.BatchNorm1d(200)
        self.batch200 = nn.BatchNorm1d(10)

        self.dropOut = nn.Dropout(p=0.5)

    def forward(self, x):

        # IP.embed()
        x = x.view(x.size(0), -1)
        x = self.binarize_and_round(x) # ! BINARIZE INPUTS

        x = F.relu(self.fc(x))
        # x = self.dropOut(x)

        x = self.binarize_and_round(x) # ! BINARIZE ACTIVATIONS

        # x = F.relu(self.fc1(x))
        # x = self.dropOut(x)
        #
        # x = F.relu(self.fc2(x))
        # x = self.dropOut(x)

        x = F.relu(self.fc2(x))

        # x = F.relu(self.fc1(x))
        # x = self.fc4(x)
        # x = self.fc2(x)
        return F.log_softmax(x) #softmax classifier


#####################################################################

use_cuda = True
init_weights = False

net = Net_mnist()
if use_cuda:
    net.cuda()

#Initialize weights from normal dist.
if init_weights:
    std = 0.01
    torch.nn.init.normal(net.fc.weight, mean=0, std=std)
    torch.nn.init.normal(net.fc1.weight, mean=0, std=std)
    torch.nn.init.normal(net.fc2.weight, mean=0, std=std)
    torch.nn.init.normal(net.fc3.weight, mean=0, std=std)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.00003)
# optimizer = optim.SGD(net.parameters(), lr=0.0001)#, momentum=0.9)
print('Defined Everything')


train_accuracy = []
test_accuracy = []
# validation_accuracy = []

train_class_accuracy = []
test_class_accuracy = []
# validation_class_accuracy = []

epochs = 150
for epoch in range(epochs):
    # print (epoch)

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        if use_cuda:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)               #FORWARD pass
        loss = criterion(outputs, labels)


        # IP.embed()

        loss.backward()  #Compute dloss/dx for each weight
        optimizer.step() #Update weights using the gradients

        # print statistics
        running_loss += loss.data[0]
        running_loss = 0.0

    # IP.embed()

    # print('Completed an Epoch %d'%(epoch + 1))
    train_accuracy.append(get_accuracy(trainloader, net, classes, use_cuda))
    test_accuracy.append(get_accuracy(testloader, net, classes, use_cuda))
    print('Epoch {} | Accuracy {}'.format(epoch+1, test_accuracy[-1]))
    # print('Epoch accuracy {}'.format(test_accuracy[-1]))
    # validation_accuracy.append(get_accuracy(validationloader, net, classes))

    # train_class_accuracy.append(get_class_accuracy(trainloader, net, classes))
    # test_class_accuracy.append(get_class_accuracy(testloader, net, classes))
    # validation_class_accuracy.append(get_class_accuracy(validationloader, net, classes))


print('test accuracy:\n')
print(get_accuracy(testloader, net, classes))

model_name = 'networksv2/networks_NOdropout/mnist_model_NOdropout_0'+str(model+1)+'.pt'
save_dir = os.getcwd() + '/' + model_name

#SAVE
torch.save(net.state_dict(), save_dir)

model+=1

# print('validation accuracy:\n')
# print(get_accuracy(validationloader, net, classes))

#########################################################################################################
### PLOTS
#
# '''
# Plotting
# '''
#
# plt.style.use('ggplot')
#
# '''
# Total accuracy
# '''
# plt.figure()
# plt.plot(range(epochs), train_accuracy, label='Train accuracy')
# plt.plot(range(epochs), test_accuracy, label='Test accuracy')
# # plt.plot(range(epochs), validation_accuracy, label='Validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Percent Accuracy')
# plt.title('Training accuracy over: \n{} Iterations'.format(epochs), fontsize=16)
# plt.legend(loc='lower right')
# plt.show(block=False)
#
# '''
# Accuracy by class.
# '''
#
# f, axarr = plt.subplots(2, 5, figsize=(18,9))
# for i in range(len(classes)):
#     if int((i) / 5) > 0:
#         row = 1
#         col = i % 5
#     else:
#         row = 0
#         col = i
#
#     print(row, col)
#     axarr[row, col].plot(range(len(train_class_accuracy)), list(np.array(train_class_accuracy)[:, i]), label='Train accuracy')
#     axarr[row, col].plot(range(len(test_class_accuracy)), list(np.array(test_class_accuracy)[:, i]), label='Test accuracy')
#     # axarr[row, col].plot(range(len(validation_class_accuracy)), list(np.array(validation_class_accuracy)[:, i]), label='Validation accuracy')
#     axarr[row, col].set_title('Accuracy for\nclass: {}'.format(classes[i]))
#
# # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
# plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
# plt.suptitle('Accuracy By Class over {} Epochs'.format(len(train_accuracy)), fontsize=16)
# plt.figlegend(loc = 'lower center', ncol=5, labelspacing=0. )
# plt.show()

#####################################33
### Save models

#PATH
model_name = 'networks/networks_dropout_SGD/mnist_model_dropout_0'+str(model+1)+'.pt'
save_dir = os.getcwd() + '/' + model_name

#SAVE
torch.save(net.state_dict(), save_dir)
#LOAD
net.load_state_dict(torch.load(save_dir))
