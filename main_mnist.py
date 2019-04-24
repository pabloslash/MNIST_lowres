 #######
# Developed by Pablo Tostado Marcos
# Last modified:
#######

import os
# os.environ['QT_QPA_PLATFORM']='offscreen' #Needed when ssh to avoid display on screen

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
import datetime
import time
from utils import *


####################################################################
###### Initialize Net & variables:
####################################################################

net = Net_mnist()
if torch.cuda.is_available():
    net.cuda()

model = 1
init_weights = False
batch_size = 3
lr = 0.0005
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(net.parameters(), lr=lr)
optimizer = optim.SGD(net.parameters(), lr=lr)

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
    mean, std = 0., 1.
    torch.nn.init.normal(net.fc.weight, mean=mean, std=std)
    torch.nn.init.normal(net.fc1.weight, mean=mean, std=std)
    print('Weights initialized from Normal Distribution')
    # torch.nn.init.normal(net.fc2.weight, mean=mean, std=std)
    # torch.nn.init.xavier_uniform(net.fc.weight)
    # torch.nn.init.xavier_uniform(net.fc1.weight)

print('Net Initialized')


def train(ep):
    train_loss = []
    train_accuracy = []
    validation_accuracy = []
    test_accuracy = []

    # train_class_accuracy = []
    # test_class_accuracy = []
    # validation_class_accuracy = []

    for e in range(ep):

        net.train()

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # Wrap nputs in variables
            inputs, labels = data
            if torch.cuda.is_available():
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad() # zero the parameter gradients, otherwise they accumulate
            '''
            # Stochastic binarization of weights:
            net.fc.weight.data = binarize_and_stochRound(net.fc.weight.data)
            net.fc1.weight.data = binarize_and_stochRound(net.fc1.weight.data)

            # DITHER.
            # Save original weights to un-do dithered matrices before backprop.
            l1 = net.fc.weight.data
            l2 = net.fc1.weight.data

            # print('calculate mask')
            # IP.embed()
            # Choose percentage prob. to dither. A higher perc. means most of the weights will dither.
            dith_mask1, net.fc.weight.data = weight_dithering(net.fc.weight.data, 50, dith_levels=1)
            dith_mask2, net.fc1.weight.data = weight_dithering(net.fc1.weight.data, 50, dith_levels=1)

            # net.fc.weight.data = fullPrec_grid_dithering(net.fc.weight.data)
            # net.fc1.weight.data = fullPrec_grid_dithering(net.fc1.weight.data)
            # IP.embed()
            outputs = net(inputs)               #FORWARD pass
            loss = criterion(outputs, labels)
            # loss.backward()  #Compute dloss/dx for each weight
            # plot_grad_flow(net.named_parameters())

            error1 = loss.data

            net.fc.weight.data = l1
            net.fc1.weight.data = l2

            #Calculate complimentary mask (what was x0.5->x2, x1->x1, x2->x0.5)
            comp_dith_mask1 = dith_mask1/(dith_mask1**2)
            comp_dith_mask2 = dith_mask2/(dith_mask2**2)

            net.fc.weight.data = net.fc.weight.data * comp_dith_mask1
            net.fc1.weight.data = net.fc1.weight.data * comp_dith_mask2

            outputs = net(inputs)               #FORWARD pass
            loss = criterion(outputs, labels)
            error2 = loss.data

            net.fc.weight.data = l1
            net.fc1.weight.data = l2

            grad_fc = -((error1 - error2) / (( dith_mask1*l1 - comp_dith_mask1*l1 ).norm()) ) * (dith_mask1 - comp_dith_mask1)
            grad_fc1 = -((error1 - error2) / (dith_mask2 - comp_dith_mask2).norm()) * (dith_mask2 - comp_dith_mask2)

            # print('hola')
            # IP.embed()
            net.fc.weight.grad =  Variable(grad_fc)
            net.fc1.weight.grad =  Variable(grad_fc1)

            # # Undo DITHER and apply gradients
            # net.fc.weight.data = l1
            # net.fc1.weight.data = l2
            optimizer.step() #Update weights using the gradients
            '''
            IP.embed()
            l1 = net.fc.weight.data
            l2 = net.fc1.weight.data

            net.fc.weight.data = fullPrec_grid_dithering(net.fc.weight.data)
            net.fc1.weight.data = fullPrec_grid_dithering(net.fc1.weight.data)

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            net.fc.weight.data = l1
            net.fc1.weight.data = l2

            optimizer.step()


            # print statistics
            train_loss.append(loss.data.cpu().numpy())
            running_loss = 0.0
            # print('Test accuracy = {}'.format(get_accuracy(testloader, net, classes, torch.cuda.is_available())))

            #
            # net.fc.weight.data = binarize_and_stochRound(net.fc.weight.data)
            # net.fc1.weight.data = binarize_and_stochRound(net.fc1.weight.data)
            # if i %100 == 0:
            #     print(get_accuracy(testloader, net, classes, torch.cuda.is_available()))
        # Get performance after epoch:
        # if e % 10 == 0: lr /= 3
        net.eval()
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

    # save_model()
    # print('Max. test accuracy of last 15 epochs: {}'.format(np.max(test_accuracy[-15:])))

    return train_loss, train_accuracy, validation_accuracy, test_accuracy


def test():
    net.eval()
    test_performance = get_accuracy(testloader, net, classes, torch.cuda.is_available())
    print ('Test accuracy is {}'.format(test_performance))
    return test_performance


#####################################
### Load / Save STATE DIR

def save_model():
    save_dir = os.getcwd() + "/results/NIPS/"
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    date = datetime.datetime.now()
    date_dir = save_dir + str(date.year) + str(date.month) + str(date.day) + '_' + str(date.hour) + str(date.minute)+ '/'  # Save in todays date.
    os.mkdir(date_dir)
    model_name = date_dir + 'mnist_model_fullPrec' + '_lr' + str(lr) + 'bs' + str(batch_size) + 'dithering_50_3levels.pt'
    torch.save(net.state_dict(), model_name)

def load_model():
    model_name = os.getcwd() + "/results/NIPS/201945_1553/mnist_model_fullPrec_lr0.00025_noDropOut.pt"
    net.load_state_dict(torch.load(save_dir))



#########################################################################################################
### PLOTS
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
