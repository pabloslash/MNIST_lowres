# Script to test deterministically trucated weights during Inference
from my_helper_pytorch import *
import pickle

import IPython as IP


############################# Dropout #############################

validation = False

if validation:
    set = 'validation'
else:
    set = 'training'


model_names = ['mnist_model_dropout_01.pt', 'mnist_model_dropout_02.pt', 'mnist_model_dropout_03.pt',
                'mnist_model_dropout_04.pt','mnist_model_dropout_05.pt' ]

sweet_spot = ['sweetSpot_det_'+set+'_dropout_01.pkl', 'sweetSpot_det_'+set+'_dropout_02.pkl', 'sweetSpot_det_'+set+'_dropout_03.pkl',
                'sweetSpot_det_'+set+'_dropout_04.pkl','sweetSpot_det_'+set+'_dropout_05.pkl' ]


m_round = [1,2,3,4,5]
e_round = [1,2,3,4,5]
e_max = [-4, -3, -2, -1, 0, 1, 2, 3, 4]


truncation_error = np.zeros((len(m_round),len(e_round)))
truncation_std = np.zeros((len(m_round),len(e_round)))


for e in xrange(len(e_round)):
    for m in xrange(len(m_round)):

        det_error = []

        for model in xrange(0, len(model_names)):

                # IP.embed()

                load_dir = os.getcwd() + '/networksv2/networks_dropout/' + model_names[model]
                sweet_spot_dir = os.getcwd() + '/networksv2/networks_dropout/SweetSpot/' + sweet_spot[model]

                with open(sweet_spot_dir) as f:  # Python 3: open(..., 'rb')
                    ss, _, _, _, _ = pickle.load(f)

                #LOAD
                net = Net_mnist()
                net.cuda()
                net.load_state_dict(torch.load(load_dir))

                bin_table = build_binary_table(m_round[m], e_round[e], e_max[int(ss[e][m])])


                #Truncate weights

                w_tensor = net.fc.weight.data
                w_array = w_tensor.cpu().numpy()                                #Convert torch.cuda.tensor to array
                w_array_truncated = deterministic_rounding(w_array, bin_table)     # Truncate
                w_tensor_truncated = torch.from_numpy(w_array_truncated).float().cuda() #Back to cuda tensor
                net.fc.weight.data = w_tensor_truncated

                #fc1
                w_tensor = net.fc2.weight.data
                w_array = w_tensor.cpu().numpy()                                #Convert torch.cuda.tensor to array
                w_array_truncated = deterministic_rounding(w_array, bin_table)     # Truncate
                w_tensor_truncated = torch.from_numpy(w_array_truncated).float().cuda() #Back to cuda tensor
                net.fc2.weight.data = w_tensor_truncated

                det_error.append((get_accuracy(validationloader, net, classes)))
                # print(det_error)

        print(np.mean(det_error))
        truncation_error[e][m] = np.mean(det_error)
        truncation_std[e][m] = np.std(det_error)




## Save parameters
with open('networksv2/networks_dropout/results/det_'+set+'_error_dropout.pkl', 'w') as f:  # Python 3: open(..., 'wb')
    pickle.dump([truncation_error, truncation_std], f)


############################# NO Dropout #############################
mode = 'det'
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

        det_error = []

        for model in xrange(0, len(model_names)):

            load_dir = os.getcwd() + '/networksv2/networks_NOdropout/' + model_names[model]
            sweet_spot_dir = os.getcwd() + '/networksv2/networks_NOdropout/SweetSpot/' + sweet_spot[model]

            with open(sweet_spot_dir) as f:  # Python 3: open(..., 'rb')
                ss, _, _, _, _ = pickle.load(f)

            # print(ss)
            #LOAD
            net = Net_mnist()
            net.cuda()
            net.load_state_dict(torch.load(load_dir))

            bin_table = build_binary_table(m_round[m], e_round[e], e_max[int(ss[e][m])])
            # print(bin_table)

            #Truncate weights

            w_tensor = net.fc.weight.data
            w_array = w_tensor.cpu().numpy()                                #Convert torch.cuda.tensor to array
            w_array_truncated = deterministic_rounding(w_array, bin_table)     # Truncate
            w_tensor_truncated = torch.from_numpy(w_array_truncated).float().cuda() #Back to cuda tensor
            net.fc.weight.data = w_tensor_truncated

            #fc1
            w_tensor = net.fc2.weight.data
            w_array = w_tensor.cpu().numpy()                                #Convert torch.cuda.tensor to array
            w_array_truncated = deterministic_rounding(w_array, bin_table)     # Truncate
            w_tensor_truncated = torch.from_numpy(w_array_truncated).float().cuda() #Back to cuda tensor
            net.fc2.weight.data = w_tensor_truncated

            det_error.append((get_accuracy(validationloader, net, classes)))

        # IP.embed()

        print(np.mean(det_error))
        truncation_error[e][m] = np.mean(det_error)
        truncation_std[e][m] = np.std(det_error)


## Save parameters
with open('networksv2/networks_NOdropout/results/deterministic_error_'+set+'NOdropout.pkl', 'w') as f:  # Python 3: open(..., 'wb')
    pickle.dump([truncation_error, truncation_std], f)


with open('results/NOdropout/deterministic_error_SSvalidation_NOdropout.pkl') as f:  # Python 3: open(..., 'rb')
    truncation_error, _ = pickle.load(f)
