# Script to test trucated weights during Inference
from my_helper_pytorch import *
import pickle

### Load model
#PATH
model_names = ['mnist_model_dropout_01.pt', 'mnist_model_dropout_02.pt', 'mnist_model_dropout_03.pt', 'mnist_model_dropout_04.pt',
                'mnist_model_dropout_05.pt' ]

for model in xrange(0, len(model_names)):


    load_dir = os.getcwd() + '/networksv2/networks_dropout/' + model_names[model]


    #LOAD
    net = Net_mnist()
    net.cuda()
    net.load_state_dict(torch.load(load_dir))


    # get_accuracy(testloader, net, classes)

    ########## Truncate weights and evaluate on testing set:
    m_round = [1,2,3,4,5]
    e_round = [1,2,3,4,5]
    # m_round = [4]
    # e_round = [4]
    e_max = [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5] #Do not pass e_max arg if no limit desired


    truncation_error = np.zeros((len(e_round),len(m_round)))
    truncation_std = np.zeros((len(e_round),len(m_round)))
    sweet_spot = np.zeros((len(e_round),len(m_round)))

    params = []
    rounds_error_save = []

    for e in xrange(len(e_round)):
        for m in xrange(len(m_round)):

            em_error = []
            em_std = []

            for em in xrange(len(e_max)):
                #Build binary table with current precision
                bin_table = build_binary_table_v2(m_round[m], e_round[e], e_max[em])

                rounds_error = []
                rounds = 1
                for r in xrange(rounds):

                    # Load weights & truncate
                    net.load_state_dict(torch.load(load_dir))

                    #FC
                    w_tensor = net.fc.weight.data
                    w_array = w_tensor.cpu().numpy()                                #Convert torch.cuda.tensor to array
                    # w_array_truncated = stochastic_rounding(w_array, bin_table)     # Truncate
                    w_array_truncated = deterministic_rounding(w_array, bin_table)     # Truncate
                    w_tensor_truncated = torch.from_numpy(w_array_truncated).float().cuda() #Back to cuda tensor
                    net.fc.weight.data = w_tensor_truncated

                    #fc1
                    w_tensor = net.fc2.weight.data
                    w_array = w_tensor.cpu().numpy()                                #Convert torch.cuda.tensor to array
                    # w_array_truncated = stochastic_rounding(w_array, bin_table)     # Truncate
                    w_array_truncated = deterministic_rounding(w_array, bin_table)     # Truncate
                    w_tensor_truncated = torch.from_numpy(w_array_truncated).float().cuda() #Back to cuda tensor
                    net.fc2.weight.data = w_tensor_truncated

                    rounds_error.append((get_accuracy(validationloader, net, classes)))

                print('Round {}'.format(em))
                rounds_error_save.append(rounds_error)
                params.append([m_round[m],e_round[e],e_max[em]])

                em_error.append(np.mean(rounds_error))
                em_std.append(np.std(rounds_error))

                # if (em>0) & (em_error[em] < em_error[em-1]):
                #     break

                # print(em_error)

            # IP.embed()
            sweet_spot[e][m] = np.argmax(em_error)
            truncation_error[e][m] = em_error[int(sweet_spot[e][m])]
            truncation_std[e][m] = em_std[int(sweet_spot[e][m])]

    ## Save parameters
    with open('networksv2/networks_dropout/SweetSpot/sweetSpot_det_validation_dropout_0'+str(model+1)+'.pkl', 'w') as f:  # Python 3: open(..., 'wb')
        pickle.dump([sweet_spot, truncation_error, truncation_std, rounds_error_save, params], f)





## Load parameters
with open('Dropout_deterministic_trunc.pkl') as f:  # Python 3: open(..., 'rb')
    sweet_spot, truncation_error, truncation_std, rounds_error_save, params = pickle.load(f)








##################################### NO DROPOUT ################################################33
# Script to test trucated weights during Inference
from my_helper_pytorch import *
import pickle


model_names = ['mnist_model_NOdropout_01.pt', 'mnist_model_NOdropout_02.pt', 'mnist_model_NOdropout_03.pt',
                'mnist_model_NOdropout_04.pt', 'mnist_model_NOdropout_05.pt']

for model in xrange(0, len(model_names)):


    load_dir = os.getcwd() + '/networksv2/networks_NOdropout/' + model_names[model]


    #LOAD
    net = Net_mnist()
    net.cuda()
    net.load_state_dict(torch.load(load_dir))


    # get_accuracy(testloader, net, classes)


    ########## Truncate weights and evaluate on testing set:
    m_round = [1,2,3,4,5]
    e_round = [1,2,3,4,5]
    # m_round = [4]
    # e_round = [4]
    e_max = [-4, -3, -2, -1, 0, 1, 2, 3, 4] #Do not pass e_max arg if no limit desired



    truncation_error = np.zeros((len(m_round),len(e_round)))
    truncation_std = np.zeros((len(m_round),len(e_round)))
    sweet_spot = np.zeros((len(m_round),len(e_round)))

    params = []
    rounds_error_save = []

    for e in xrange(len(e_round)):
        for m in xrange(len(m_round)):

            em_error = []
            em_std = []

            for em in xrange(len(e_max)):
                #Build binary table with current precision
                bin_table = build_binary_table_v2(m_round[m], e_round[e], e_max[em])

                rounds_error = []
                rounds = 10
                for r in xrange(rounds):

                    # Load weights & truncate
                    net.load_state_dict(torch.load(load_dir))

                    #FC
                    w_tensor = net.fc.weight.data
                    w_array = w_tensor.cpu().numpy()                                #Convert torch.cuda.tensor to array
                    w_array_truncated = stochastic_rounding(w_array, bin_table)     # Truncate
                    # w_array_truncated = deterministic_rounding(w_array, bin_table)     # Truncate
                    w_tensor_truncated = torch.from_numpy(w_array_truncated).float().cuda() #Back to cuda tensor
                    net.fc.weight.data = w_tensor_truncated

                    #fc1
                    w_tensor = net.fc2.weight.data
                    w_array = w_tensor.cpu().numpy()                                #Convert torch.cuda.tensor to array
                    w_array_truncated = stochastic_rounding(w_array, bin_table)     # Truncate
                    # w_array_truncated = deterministic_rounding(w_array, bin_table)     # Truncate
                    w_tensor_truncated = torch.from_numpy(w_array_truncated).float().cuda() #Back to cuda tensor
                    net.fc2.weight.data = w_tensor_truncated

                    rounds_error.append((get_accuracy(validationloader, net, classes)))

                print('Round {}'.format(em))
                rounds_error_save.append(rounds_error)
                params.append([m_round[m],e_round[e],e_max[em]])

                em_error.append(np.mean(rounds_error))
                em_std.append(np.std(rounds_error))

                # print(em_error)

            # IP.embed()
            sweet_spot[e][m] = np.argmax(em_error)
            truncation_error[e][m] = em_error[int(sweet_spot[e][m])]
            truncation_std[e][m] = em_std[int(sweet_spot[e][m])]

    ## Save parameters
    with open('networksv2/networks_NOdropout/SweetSpot/sweetSpot_stoch_validation_NOdropout_0'+str(model+1)+'.pkl', 'w') as f:  # Python 3: open(..., 'wb')
        pickle.dump([sweet_spot, truncation_error, truncation_std, rounds_error_save, params], f)
