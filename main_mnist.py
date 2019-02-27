 #######
# Developed by Pablo Tostado Marcos
# Last modified:
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
from model import *


####################################################################
###### Initialize variables:
####################################################################

net = Net_mnist()
if torch.cuda.is_available():
    net.cuda()

model = 1
batch_size = 20
init_weights = False
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0000005)

################## DATA LOADING #####################################
# mnist_dir = '/home/pablotostado/Desktop/PT/ML_Datasets/MNIST/'
mnist_dir = os.getcwd() + '/ML_Datasets'

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

validation = False
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

#####################################################################

#Initialize weights from normal / Xavier dist.
if init_weights:
#     mean, std = 0., 0.01
#     torch.nn.init.normal(net.fc.weight, mean=mean, std=std)
#     torch.nn.init.normal(net.fc1.weight, mean=mean, std=std)
#     torch.nn.init.normal(net.fc2.weight, mean=mean, std=std)

    torch.nn.init.xavier_uniform(net.fc.weight)
    torch.nn.init.xavier_uniform(net.fc1.weight)
    torch.nn.init.xavier_uniform(net.fc2.weight)

# optimizer = optim.SGD(net.parameters(), lr=0.0001)#, momentum=0.9)
print('Net Initialized')


def train(ep):
    train_loss = []
    train_accuracy = []
    validation_accuracy = []
    test_accuracy = []

    # train_class_accuracy = []
    # test_class_accuracy = []
    # validation_class_accuracy = []
    net.train()

    for e in range(ep):
        # print (epoch)

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # Wrap nputs in variables
            inputs, labels = data
            if torch.cuda.is_available():
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad() # zero the parameter gradients, otherwise they accumulate

            # net.fc.weight.data = binarize_and_stochRound(net.fc.weight.data)
            # net.fc1.weight.data = binarize_and_stochRound(net.fc1.weight.data)

            outputs = net(inputs)               #FORWARD pass
            loss = criterion(outputs, labels)

            loss.backward()  #Compute dloss/dx for each weight
            optimizer.step() #Update weights using the gradients

            # print statistics
            train_loss.append(loss.data.cpu().numpy())
            running_loss = 0.0


        # Get performance after epoch:
        train_accuracy.append(get_accuracy(trainloader, net, classes, torch.cuda.is_available()))
        test_accuracy.append(get_accuracy(testloader, net, classes, torch.cuda.is_available()))
        if (validation):
            validation_accuracy.append(get_accuracy(validationloader, net, classes, torch.cuda.is_available()))
            print('Epoch {} | Validation Accuracy {} | Train Accuracy {} | Loss {}'.format(e+1, validation_accuracy[-1], train_accuracy[-1], train_loss[-1]))
        else:
            print('Epoch {} | Test Accuracy {} | Train Accuracy {} | Loss {}'.format(e+1, test_accuracy[-1], train_accuracy[-1], train_loss[-1]))

        # train_class_accuracy.append(get_class_accuracy(trainloader, net, classes))
        # test_class_accuracy.append(get_class_accuracy(testloader, net, classes))
        # validation_class_accuracy.append(get_class_accuracy(validationloader, net, classes))

    print('DONE TRAINING. Test accuracy:\n')
    print(get_accuracy(testloader, net, classes, torch.cuda.is_available()))

    return train_loss, train_accuracy, validation_accuracy, test_accuracy


def test():
    net.eval()
    test_performance = get_accuracy(testloader, net, classes, torch.cuda.is_available())
    print ('Test accuracy is {}'.format(test_performance))
    return test_performance



#####################################33
### Load / Save STATE DIR

def save_model():
    model_name = 'networks/networks_dropout_SGD/mnist_model_dropout_0'+str(model+1)+'.pt'
    model_name = 'networks/mnist_model_dropout50_0'+str(model+1)+'.pt'
    save_dir = os.getcwd() + '/' + model_name
    torch.save(net.state_dict(), save_dir)

def load_model():
    model_name = 'networks/mnist_model_dropout50_0'+str(model+1)+'.pt'
    save_dir = os.getcwd() + '/' + model_name
    net.load_state_dict(torch.load(save_dir))



#########################################################################################################
### PLOTS
# '''
# Total accuracy
def plot_results():
    # '''
    plt.figure()
    plt.plot(range(epochs), train_accuracy, label='Train accuracy')
    plt.plot(range(epochs), test_accuracy, label='Test accuracy')
    # plt.plot(range(epochs), validation_accuracy, label='Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Percent Accuracy')
    plt.title('Training accuracy over: \n{} Iterations'.format(epochs), fontsize=16)
    plt.legend(loc='lower right')
    plt.show(block=False)
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
