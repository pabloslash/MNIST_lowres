# Load stuff
import pickle

### Load FINAL error:
with open('networkv2s/networks_dropout/results/final_error_dropout.pkl') as f:  # Python 3: open(..., 'rb')
    final_error_dropout, final_std_dropout = pickle.load(f)

with open('networksv2/networks_NOdropout/results/final_error_NOdropout.pkl') as f:  # Python 3: open(..., 'rb')
    final_error_NOdropout, final_std_NOdropout = pickle.load(f)


### Load DETERMINISTIC error:
with open('networksv2/networks_dropout/results/det_training_error_dropout.pkl') as f:  # Python 3: open(..., 'wb')
    det_error_dropout, det_std_dropout = pickle.load(f)

with open('networksv2/networks_NOdropout/results/deterministic_error_trainNOdropout.pkl') as f:  # Python 3: open(..., 'wb')
    det_error_NOdropout, det_std_NOdropout = pickle.load(f)


### Load STOCHASTIC error:
with open('networksv2/networks_dropout/results/stoch_training_error_dropout.pkl') as f:  # Python 3: open(..., 'wb')
    stoch_error_dropout, stoch_std_dropout = pickle.load(f)

with open('networksv2/networks_NOdropout/results/stoch_training_error_NOdropout.pkl') as f:  # Python 3: open(..., 'wb')
    stoch_error_NOdropout, stoch_std_NOdropout = pickle.load(f)






### Test

model_error = []
model_names = ['mnist_model.2.pt', 'mnist_model.pt', 'mnist_model_dropout.pt',
                'mnist_model_Dropout.pt','mnist_model_noDropout.2.pt', 'mnist_model_noDropout.2.pt',
                'mnist_model_NOdropout.pt']

for model in xrange(2, len(model_names)):

    load_dir = os.getcwd() + '/Restored_models/' + model_names[model]
    net = Net_mnist()
    net.cuda()
    net.load_state_dict(torch.load(load_dir))


    model_error.append((get_accuracy(testloader, net, classes)))
    print (get_accuracy(testloader, net, classes))



e_round = [1,3]
m_round = [1,3]
e_max = [-4, -3, -2, -1, 0, 1, 2, 3, 4]


# truncation_error = np.zeros((len(m_round),len(e_round)))
# truncation_std = np.zeros((len(m_round),len(e_round)))

# truncation_error = np.zeros((len(m_round),len(e_round)))
# truncation_std = []

model = 6
load_dir = os.getcwd() + '/Restored_models/' + model_names[model]

truncation_error = np.zeros((len(m_round),len(e_round)))

for e in xrange(len(e_round)):
    for m in xrange(len(m_round)):

        em_error = []
        for em in xrange(len(e_max)):
            # print(em)

            # with open(sweet_spot_dir) as f:  # Python 3: open(..., 'rb')
            #     ss, _, _, _, _ = pickle.load(f)
            # print(ss)
            stoch_error = []
            for r in xrange (0,5):
                #LOAD
                net = Net_mnist()
                net.cuda()
                net.load_state_dict(torch.load(load_dir))

                bin_table = build_binary_table(m_round[m], e_round[e], e_max[em])


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
                # print (r)
            # IP.embed()
            # sweet_spot[e][m] = np.argmax(stoch_error)
            em_error.append(np.mean(stoch_error))
            print(em_error)
# IP.embed()
        truncation_error[e][m] = np.max(em_error)
# truncation_std.append(np.std(em_error))
