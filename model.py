import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import IPython as IP
from my_helper_pytorch import *

class Net_mnist(nn.Module):
    def __init__(self):
        super(Net_mnist, self).__init__()

        # We need as many instantiations as rounding rounds to do the backward pass
        # self.binarize_and_round1 = Binarize_and_StochRound()

        self.fc = nn.Linear(784, 200)
        self.fc1 = nn.Linear(200, 10)

        self.batch200 = nn.BatchNorm1d(200)
        self.batch200 = nn.BatchNorm1d(10)

        self.dropOut = nn.Dropout(p=0.5)

        # self.truncate_and_round_fc = Truncate_and_StochRound()
        # self.truncate_and_round_fc1 = Truncate_and_StochRound()

        # self.binary_table = torch.from_numpy(build_binary_table_v2(0, 8, -6))
        # if torch.cuda.is_available(): self.binary_table = Variable((self.binary_table).cuda())
        # else:                         self.binary_table = Variable(self.binary_table)


    def forward(self, x):
        x = x.view(x.size(0), -1)

        # x = self.binarize_and_round1(x) # ! BINARIZE INPUTS

        # IP.embed()
        #
        # self.fc.weight.data = truncate_and_stoch_round(self.fc.weight.data, self.binary_table)
        # self.fc1.weight.data = truncate_and_stoch_round(self.fc1.weight.data, self.binary_table)

        x = F.relu(self.fc(x))

        # x = self.dropOut(x)
        x = F.relu(self.fc1(x))
        # IP.embed()
        return F.log_softmax(x, dim=1) #softmax classifier

# Static class to add the Stochastic Rounding module to my Network:
class Binarize_and_StochRound(torch.autograd.Function):
    def forward(self, x):

        x = binarize_and_stochRound(x)
        return x
    def backward(self, grad_output):
        grad_output = binarize_and_stochRound(grad_output)
        return grad_output

# class Truncate_and_StochRound(torch.autograd.Function):
#     def forward(self, x, binary_table):
#         print('truncate class')
#         IP.embed()
#         x_truncated = truncate_and_stoch_round(x, binary_table)
#         if torch.cuda.is_available():
#             x_truncated = torch.from_numpy(x_truncated).float().cuda() #Back to cuda tensor
#         else:
#             x_truncated = torch.from_numpy(x_truncated).float() #Back to tensor
#
#         print('truncate class')
#         IP.embed()
#         return x_truncated.data
     # def backward(self, grad_output):
     #      grad_output = binarize_and_stochRound(grad_output)
     #      return grad_output
