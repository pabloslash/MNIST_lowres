# Helper Functions


import numpy as np
from torch.autograd import Variable
import torch
import IPython as IP

###### Get Accuracy

def get_accuracy(dataloader, net, classes, use_cuda, cuda=0):
    correct = 0
    total = 0

    for data in dataloader:
        inputs, labels = data
        if use_cuda:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum().cpu().numpy()
    return 100.0 * correct / total

def get_class_accuracy(dataloader, net, classes, use_cuda, cuda=0):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    for data in dataloader:
        inputs, labels = data
        if use_cuda:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels).squeeze()
        # IP.embed()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1

    class_perc = []

    for i in range(10):
        class_perc.append(100.0 * class_correct[i] / class_total[i])
    return class_perc


############################################################
'''TRUNCATION FOR INFERENCE'''
############################################################

#Build table
'''This function builds a symmetryc (+/-) table of possible binary values given a precission for mantisa and exponent.
    m_bits = mantissa bits, e_bits = exp bits,
    max_exp = Maximum exponent you want to allow, if you want to bias the weights towards smaller values (i.e. have more negative than positive exponents).'''
def build_binary_table(m_bits, e_bits, max_exp=None):
    m_bits = m_bits-1 #Save 1 bit for sign

    #Get BINARY possible mantissas
    bin_man = []
    for b in xrange(2**m_bits):
        b_man = bin(b)[2:]
        bin_man.append('0'*(m_bits-len(b_man)) + b_man)

    #Get DECIMAL possible exponents, biased by one towards negative exponents by default.
    exp_bias = 2**(e_bits-1)
    dec_exp = []
    for e in xrange(2**e_bits):
        dec_exp.append(e-exp_bias)

    #Bias fuurther if a MAXIMUM EXP is inputed by the user
    if (max_exp!=None and np.max(dec_exp)>max_exp):
        dec_exp = dec_exp - (np.max(dec_exp)-max_exp)

    #Get DECIMAL possible mantissas
    dec_man = []
    for m in xrange(len(bin_man)):
        dec_m = '1.'+(bin_man[m])
        dec_man.append(bin_2_dec(dec_m))

    #Build TABLE
    binary_table = []
    for e in xrange(len(dec_exp)):
        for m in xrange(len(dec_man)):
            binary_table.append( dec_man[m]*(2**int(dec_exp[e])) )
    binary_table = np.array(binary_table)
    neg_table = -binary_table
    # Make table symmetric according to the bit we took for the sign:
    binary_table = np.concatenate((neg_table, binary_table))

    #Take the largest representation for the 0.0 value
    idx = np.where(binary_table == np.max(binary_table))[0][0]
    binary_table[idx] = 0.0
    return np.sort(binary_table)

#Build table
'''Version 2 of build binary table. Fixed-point at low dynamic range, binary-floating point at higher exponents.
    exp_offset = Exponent offset, (e.g. have more negative than positive exponents).'''
def build_binary_table_v2(m_bits, e_bits, exp_offset=None):
    m_bits = m_bits-1 #Save 1 bit for sign

    # IP.embed()

    #Get BINARY possible mantissas
    bin_man = []
    for b in xrange(2**m_bits):
        b_man = bin(b)[2:]
        bin_man.append('0'*(m_bits-len(b_man)) + b_man)

    #Get DECIMAL possible exponents, biased by one towards negative exponents by default.
    # exp_bias = 2**(e_bits-1)
    dec_exp = []
    for e in xrange(2**e_bits):
        dec_exp.append(e)

    #Get DECIMAL possible mantissas
    dec_man = []
    for m in xrange(len(bin_man)):
        dec_m = '1.'+(bin_man[m])
        dec_man.append(bin_2_dec(dec_m))

    #Build TABLE
    binary_table = []
    for e in xrange(len(dec_exp)):
        for m in xrange(len(dec_man)):
            if (dec_exp[e]==0):
                dec_man_corrected = [x-1 for x in dec_man]
                binary_table.append(dec_man_corrected[m]*2)
            else:
                binary_table.append( dec_man[m]*(2**int(dec_exp[e])) )
    binary_table = np.array(binary_table)
    neg_table = -binary_table
    # Make table symmetric according to the bit we took for the sign:
    binary_table = np.concatenate((neg_table, binary_table))

    # Offset exonent
    if (exp_offset!=None):
        binary_table = [x * (2**exp_offset) for x in binary_table]

    return np.sort(binary_table)


'''This function will convert a binary fraction number to decimal: e.g. 0.11 -> 0.75)
   INPUT: string binary fractional mantissa: "0.xxxx" '''
def bin_2_dec(num):
    frac_part = num[2:] #Get rid of "0.""
    dec_num = 1+ np.sum(([ float(frac_part[pos])*(2**(-pos-1)) for pos in xrange(len(frac_part))]))
    return dec_num

# Truncate weights
'''This function rounds an inputted WEIGHT MATRIX to either of the closest possible
floating binary vaulues in a lookup table. If it's larger/smaller than the maximum/minimum values allowed,
it will round it deterministically to the extreme.'''
def stochastic_rounding(w, lookup_table):
    #Add new values in the edges of the lookup table in case we have weights at the extremes
    #
    # print('stoch rounding')
    # IP.embed()

    lookup_table = expand_lookup_table(lookup_table, w)

    #Get positions on which weights would be inserted into this lookuptable
    pos = np.searchsorted(lookup_table, w)

    #Round them to one of the adjacent allowed binary values probabilistically
    interval_len = lookup_table[pos] - lookup_table[pos-1] #Distance between possible values of my weight
    round_prob = (w - lookup_table[pos-1]) / interval_len #This prob indicates how far my weight is from the previous binary representation
    flip_coin = np.random.random(round_prob.shape) > round_prob #Flip coin to decide what value to update to
    w = flip_coin*lookup_table[pos-1] + np.invert(flip_coin)*lookup_table[pos] #Update to value that came up probabilistically.

    #Correct those weights that were rounded to the made-up extreme values (above/below the allowed values in the original lookup table)
    w[w<lookup_table[1]] = lookup_table[1]
    w[w>lookup_table[-2]] = lookup_table[-2]

    return w


'''This function rounds an inputted WEIGHT MATRIX to either the closest possible.
If it's larger/smaller than the maximum/minimum values allowed,
it will round it to the extreme value.'''
def deterministic_rounding(w, lookup_table):
    #Add new values in the edges of the lookup table in case we have weights at the extremes
    lookup_table = expand_lookup_table(lookup_table, w)

    #Get positions on which weights would be inserted into this lookuptable
    pos = np.searchsorted(lookup_table, w)

    #Round them to one of the adjacent allowed binary values probabilistically
    interval_len = lookup_table[pos] - lookup_table[pos-1] #Distance between possible values of my weight
    round_prob = (w - lookup_table[pos-1]) / interval_len #This prob indicates how far my weight is from the previous binary representation

    deterministic_prob = round_prob > 0.5
    w = deterministic_prob*lookup_table[pos] + np.invert(deterministic_prob)*lookup_table[pos-1]

    #Correct those weights that were rounded to the made-up extreme values (above/below the allowed values in the original lookup table)
    w[w<lookup_table[1]] = lookup_table[1]
    w[w>lookup_table[-2]] = lookup_table[-2]

    return w

'''Add new values in the edges of the lookup table in case we have weights at the extremes'''
def expand_lookup_table(lookup_table, w):
    min_value = np.min([np.min(w), np.min(lookup_table)]) - 1
    max_value = np.max([np.max(w), np.max(lookup_table)]) + 1
    lookup_table = np.insert(lookup_table, 0, min_value)
    lookup_table = np.append(lookup_table, max_value)
    return lookup_table

############################################################
'''TRUNCATION DURING TRAINING'''
############################################################

'''This function receives a matrix (torch tensor) of floats, binarizes it and stochastically rounds to closest binary exponent'''
def binarize_and_stochRound(x):

    x += 1e-8                                       # Avoid problems calculating log(0)
    exp = torch.floor(torch.log2(torch.abs(x)))              # Get binary exponent (absolute value)
    man = (x / 2**exp)                              # Get mantisa (man \in [1,2])

    flip_coin = (torch.rand(man.shape).cuda() if torch.cuda.is_available() else torch.rand(man.shape)) < (torch.abs(man) - 1.0)              #Flip_coin to stochstically round full_prec binary number
    exp += flip_coin.float() * 1.0

    stoch_rounded = torch.sign(x)* 2.0**exp
    return stoch_rounded

'''Same as above function, deterministic rounding'''
def binarize_and_detRound(x):
    x += 1e-8
    exp = torch.round(torch.log2(torch.abs(x))) #deterministically round full_prec binary number
    det_rounded = torch.sign(x)* 2.0**exp

    return det_rounded

'''This function receives a matrix (torch tensor) of floats, and stochastically rounds them according to the given binary table.'''
def truncate_and_stoch_round(x, binary_table):

    w_array = x.cpu().numpy() # Convert to numpy array
    lookup_table = (binary_table).numpy()

    w_array_truncated = stochastic_rounding(w_array, lookup_table)     # Truncate
# w_array_truncated = deterministic_rounding(w_array, bin_table)

    if torch.cuda.is_available():
        w_tensor_truncated = torch.from_numpy(w_array_truncated).float().cuda() #Back to cuda tensor
    else:
        w_tensor_truncated = torch.from_numpy(w_array_truncated).float() #Back to tensor
    return w_tensor_truncated

############################################################
'''DITHERING'''
############################################################

# Returns dithered matrix: x0.5, x1, x2: to half, maintain or double weight value according to the dithering percentage.
# Maintains expected value of the weights. Used with full-precision input matrix.
'''Hard to explain function.'''
def weight_dithering(x, dith_percentage, dith_levels=3):

    maintain_val = (100-float(dith_percentage)) / 100.0

    k = [2**(level+1) for level in xrange(dith_levels)] # x2 - x1/2, x4 - x 1/4, x8 - x1/8 etc.
    k_prob = [level/(level+1.0) for level in k] # For each level, get the probability to go to the lower step.
    # The higher step will be complimentary, which will ensure maintaining the expected value of the weights

    # Get probability of each step and complimentary:
    k_step_prob = [f(p) for p in k_prob for f in(lambda p: p, lambda p: 1-p)]

    # print('inside weight dith funct')
    # IP.embed()

    dith_prob = [maintain_val] + [(1-maintain_val)/(dith_levels) * prob for prob in k_step_prob] #Divide prob equally to go to each step
    new_k = [1.0] + [f(level) for level in k for f in(lambda l: 1/float(l), lambda l: float(l))]

    flip_coin = np.random.random(x.shape) # Create matrix of random nums of size of input Tensor
    cum_prob_levels =  np.cumsum([0.0] + dith_prob)

    dith_mask = np.zeros(x.shape)
    for idx in xrange(np.size(new_k)):
        dith_mask = dith_mask + ((flip_coin > cum_prob_levels[idx]) & (flip_coin < cum_prob_levels[idx+1])) * new_k[idx]

    return x * torch.from_numpy(dith_mask).float().cuda()

'''This function gets a full-precision input matrix and returns a dithered weight matrix where all
the dithering is to quantized possible values: 2^(x-1), 2^(x), 2^(x+1), 2^(x+2). If (w) lies in one
concrete quantiation value then there are only 3 potential dithering positions. (GERT CAUWENBERGHS' idea.)'''
def fullPrec_grid_dithering(x):

    wsgn = np.sign(x)
    w = np.abs(x)
    wexp = np.floor(np.log2(w))
    w_i_left_1 = 2**np.floor(np.log2(w)-0)

    w_i_left_2 = 2**np.floor(np.log2(w)-1)
    w_i_right_1 = 2**np.floor(np.log2(w)+1)
    w_i_right_2 = 2**np.floor(np.log2(w)+2)

    p_i_minus_2 = 0.5 * (w-w_i_right_1)/(w_i_left_2-w_i_right_1);
    p_i_minus_1 = p_i_minus_2 + 0.5*(w-w_i_right_2)/(w_i_left_1-w_i_right_2);
    p_i_plus_1 = p_i_minus_1 + 0.5*(w-w_i_left_2)/(w_i_right_1-w_i_left_2);
    p_i_plus_2 = p_i_plus_1 + 0.5*(w-w_i_left_1)/(w_i_right_2-w_i_left_1);

    prob = np.random.random(x.shape)

    x = wsgn * ( 2**( wexp - 1*(prob<p_i_minus_2).float() + 1*((prob >= p_i_minus_1)&(prob<p_i_plus_1)).float() + 2*(prob>=p_i_plus_1).float() ) )

    return x.cuda()


'''Dropconnect at the weight level.'''
def custom_dropconnect(x, dc_percentage):
    dc = dc_percentage / 100.0
    dc_matrix = np.random.choice([0, 1], size=(x.shape), p=[dc, 1-dc]) # Binary matrix with dc percentage of zeros.
    x = x * torch.from_numpy((dc_matrix / (1-dc))).float().cuda()

    return x
