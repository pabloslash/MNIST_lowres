# Script to get paper plot of cross-linked networks

from my_helper_pytorch import *

mode = 'stoch'
validation = True

if validation:
    set = 'validation'
else:
    set = 'training'



model_names = ['mnist_model_NOdropout_01.pt', 'mnist_model_NOdropout_02.pt', 'mnist_model_NOdropout_03.pt', 'mnist_model_NOdropout_04.pt','mnist_model_NOdropout_05.pt']

sweet_spot = ['sweetSpot_'+mode+'_'+set+'_NOdropout_01.pkl','sweetSpot_'+mode+'_'+set+'_NOdropout_02.pkl', 'sweetSpot_'+mode+'_'+set+'_NOdropout_03.pkl',
                'sweetSpot_'+mode+'_'+set+'_NOdropout_04.pkl', 'sweetSpot_'+mode+'_'+set+'_NOdropout_05.pkl']


m_round = 1
e_round = 1
e_max = [-4, -3, -2, -1, 0, 1, 2, 3, 4]


validationloader_error = []
testloader_error = []



for model in xrange(0, len(model_names)):


    load_dir = os.getcwd() + '/networks/networks_NOdropout/' + model_names[model]
    sweet_spot_dir = os.getcwd() + '/networks/networks_NOdropout/SweetSpot/' + sweet_spot[model]

    with open(sweet_spot_dir) as f:  # Python 3: open(..., 'rb')
        ss, _, _, _, _ = pickle.load(f)


    rounds = 50
    for r in xrange (0,rounds):
        print(r)
        #LOAD

        net = Net_mnist()
        net.load_state_dict(torch.load(load_dir))

        bin_table = build_binary_table(m_round, e_round, e_max[int(ss[0][0])])


        #Truncate weights

        w_tensor = net.fc.weight.data
        w_array = w_tensor.numpy()                                #Convert torch.cuda.tensor to array
        # w_array_truncated = deterministic_rounding(w_array, bin_table)     # Truncate
        w_array_truncated = stochastic_rounding(w_array, bin_table)     # Truncate
        w_tensor_truncated = torch.from_numpy(w_array_truncated).float() #Back to cuda tensor
        net.fc.weight.data = w_tensor_truncated

        #fc1
        w_tensor = net.fc2.weight.data
        w_array = w_tensor.numpy()                                #Convert torch.cuda.tensor to array
        # w_array_truncated = deterministic_rounding(w_array, bin_table)     # Truncate
        w_array_truncated = stochastic_rounding(w_array, bin_table)     # Truncate
        w_tensor_truncated = torch.from_numpy(w_array_truncated).float() #Back to cuda tensor
        net.fc2.weight.data = w_tensor_truncated

        validationloader_error.append((get_accuracy(validationloader, net, classes)))
        testloader_error.append((get_accuracy(testloader, net, classes)))


## Save parameters
with open('networks/networks_NOdropout/results/crosslinked_nets.pkl', 'w') as f:  # Python 3: open(..., 'wb')
    pickle.dump([validationloader_error, testloader_error], f)


## Figure cross-crosslinked_nets

plt.figure()
ax = plt.subplot(111)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.scatter(validationloader_error[0:50], testloader_error[0:50], c='r')
plt.scatter(validationloader_error[50:100], testloader_error[50:100], c='orange')
plt.scatter(validationloader_error[100:150], testloader_error[100:150], c='gold')
plt.scatter(validationloader_error[150:200], testloader_error[150:200], c='green')
plt.scatter(validationloader_error[200:250], testloader_error[200:250], c='b')
plt.ylabel('Test Error (%)')
plt.xlabel('Validation Error (%)')
ax.set_xticks(np.arange(75,100,5))
ax.set_yticks(np.arange(75,100,5))
plt.gca().set_aspect('equal', adjustable='box')
plt.title('Validation vs. Test Error comparison')
plt.legend(['DNN 1','DNN 2', 'DNN 3', 'DNN 4', 'DNN 5'], loc='lower right')
plt.show(block=False)
