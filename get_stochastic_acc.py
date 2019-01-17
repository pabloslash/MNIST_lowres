# Script to test deterministically trucated weights during Inference
from my_helper_pytorch import *
import pickle

import IPython as IP


############################# Dropout #############################

validation = True

if validation:
    set = 'validation'
else:
    set = 'training'


model_names = ['mnist_model_dropout_01.pt', 'mnist_model_dropout_02.pt', 'mnist_model_dropout_03.pt',
                'mnist_model_dropout_04.pt','mnist_model_dropout_05.pt' ]

sweet_spot = ['sweetSpot_stoch_'+set+'_dropout_01.pkl', 'sweetSpot_stoch_'+set+'_dropout_02.pkl', 'sweetSpot_stoch_'+set+'_dropout_03.pkl',
                'sweetSpot_stoch_'+set+'_dropout_04.pkl','sweetSpot_stoch_'+set+'_dropout_05.pkl' ]


m_round = [1,2,3,4,5]
e_round = [1,2,3,4,5]
e_max = [-4, -3, -2, -1, 0, 1, 2, 3, 4]


truncation_error = np.zeros((len(m_round),len(e_round)))
truncation_std = np.zeros((len(m_round),len(e_round)))


for e in xrange(len(e_round)):
    for m in xrange(len(m_round)):

        stoch_error = []

        for model in xrange(0, len(model_names)):

                load_dir = os.getcwd() + '/networksv2/networks_dropout/' + model_names[model]
                sweet_spot_dir = os.getcwd() + '/networksv2/networks_dropout/SweetSpot/' + sweet_spot[model]

                with open(sweet_spot_dir) as f:  # Python 3: open(..., 'rb')
                    ss, _, _, _, _ = pickle.load(f)

                for r in xrange (0,50):
                    #LOAD
                    net = Net_mnist()
                    net.cuda()
                    net.load_state_dict(torch.load(load_dir))

                    bin_table = build_binary_table_v2(m_round[m], e_round[e], e_max[int(ss[e][m])])

                    #Truncate weights

                    w_tensor = net.fc.weight.data
                    w_array = w_tensor.cpu().numpy()                                #Convert torch.cuda.tensor to array
                    w_array_truncated = stochastic_rounding(w_array, bin_table)     # Truncate
                    w_tensor_truncated = torch.from_numpy(w_array_truncated).float().cuda() #Back to cuda tensor
                    net.fc.weight.data = w_tensor_truncated

                    #fc2
                    w_tensor = net.fc2.weight.data
                    w_array = w_tensor.cpu().numpy()                                #Convert torch.cuda.tensor to array
                    w_array_truncated = stochastic_rounding(w_array, bin_table)     # Truncate
                    w_tensor_truncated = torch.from_numpy(w_array_truncated).float().cuda() #Back to cuda tensor
                    net.fc2.weight.data = w_tensor_truncated

                    stoch_error.append((get_accuracy(testloader, net, classes)))
                    # print (r)


        print(np.mean(stoch_error))
        truncation_error[e][m] = np.mean(stoch_error)
        truncation_std[e][m] = np.std(stoch_error)

## Save parameters
## Save parameters
with open('networksv2/networks_dropout/results/stoch_'+set+'_error_dropout.pkl', 'w') as f:  # Python 3: open(..., 'wb')
    pickle.dump([truncation_error, truncation_std], f)


with open('networks/networks_dropout/SweetSpot/sweetSpot_stoch_validation_dropout_02.pkl') as f:  # Python 3: open(..., 'wb')
    # stoch_error_dropout, stoch_std_dropout = pickle.load(f)
    sweet_spot, truncation_error, truncation_std, rounds_error_save, params = pickle.load(f)

############################# NO Dropout #############################


mode = 'stoch'
validation = False

if validation:
    set = 'validation'
else:
    set = 'train'



model_names = ['mnist_model_NOdropout_01.pt', 'mnist_model_NOdropout_02.pt', 'mnist_model_NOdropout_03.pt', 'mnist_model_NOdropout_04.pt','mnist_model_NOdropout_05.pt']

sweet_spot = ['sweetSpot_'+mode+'_'+set+'_NOdropout_01.pkl','sweetSpot_'+mode+'_'+set+'_NOdropout_02.pkl', 'sweetSpot_'+mode+'_'+set+'_NOdropout_03.pkl',
                'sweetSpot_'+mode+'_'+set+'_NOdropout_04.pkl', 'sweetSpot_'+mode+'_'+set+'_NOdropout_05.pkl']


m_round = [1,2,3,4,5]
e_round = [1,2,3,4,5]
e_max = [-4, -3, -2, -1, 0, 1, 2, 3, 4]

truncation_error = np.zeros((len(m_round),len(e_round)))
truncation_std = np.zeros((len(m_round),len(e_round)))


for e in xrange(len(e_round)):
    for m in xrange(len(m_round)):

        stoch_error = []

        for model in xrange(0, len(model_names)-3):

                load_dir = os.getcwd() + '/networksv2/networks_NOdropout/' + model_names[model]
                sweet_spot_dir = os.getcwd() + '/networksv2/networks_NOdropout/SweetSpot/' + sweet_spot[model]

                with open(sweet_spot_dir) as f:  # Python 3: open(..., 'rb')
                    ss, _, _, _, _ = pickle.load(f)


                rounds = 50
                for r in xrange (0,rounds):
                    #LOAD
                    net = Net_mnist()
                    net.cuda()
                    net.load_state_dict(torch.load(load_dir))

                    bin_table = build_binary_table_v2(m_round[m], e_round[e], e_max[int(ss[e][m])])


                    #Truncate weights

                    w_tensor = net.fc.weight.data
                    w_array = w_tensor.cpu().numpy()                                #Convert torch.cuda.tensor to array
                    # w_array_truncated = deterministic_rounding(w_array, bin_table)     # Truncate
                    w_array_truncated = stochastic_rounding(w_array, bin_table)     # Truncate
                    w_tensor_truncated = torch.from_numpy(w_array_truncated).float().cuda() #Back to cuda tensor
                    net.fc.weight.data = w_tensor_truncated

                    #fc1
                    w_tensor = net.fc2.weight.data
                    w_array = w_tensor.cpu().numpy()                                #Convert torch.cuda.tensor to array
                    # w_array_truncated = deterministic_rounding(w_array, bin_table)     # Truncate
                    w_array_truncated = stochastic_rounding(w_array, bin_table)     # Truncate
                    w_tensor_truncated = torch.from_numpy(w_array_truncated).float().cuda() #Back to cuda tensor
                    net.fc2.weight.data = w_tensor_truncated

                    stoch_error.append((get_accuracy(testloader, net, classes)))

        print(np.mean(stoch_error))
        truncation_error[e][m] = np.mean(stoch_error)
        truncation_std[e][m] = np.std(stoch_error)



## Save parameters
with open('networksv2/networks_NOdropout/results/stoch_'+set+'_error_NOdropout.pkl', 'w') as f:  # Python 3: open(..., 'wb')
    pickle.dump([truncation_error, truncation_std], f)
