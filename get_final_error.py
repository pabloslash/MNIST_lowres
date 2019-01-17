# Get deterministic accuracy
import os

############################## DropOut ##############################

model_names = ['mnist_model_dropout_01.pt', 'mnist_model_dropout_02.pt', 'mnist_model_dropout_03.pt',
                'mnist_model_dropout_05.pt' ]


final_error = []

for model in xrange(0, len(model_names)):

    load_dir = os.getcwd() + '/networks/networks_dropout/' + model_names[model]

    #LOAD
    net = Net_mnist()
    net.cuda()
    net.load_state_dict(torch.load(load_dir))

    final_error.append(get_accuracy(testloader, net, classes))

fin_dropout_error = np.mean(final_error)
fin_dropout_std = np.std(final_error)

with open('networks/networks_dropout/results/final_error_dropout.pkl', 'w') as f:  # Python 3: open(..., 'wb')
    pickle.dump([fin_dropout_error, fin_dropout_std], f)



############################## NO dropOut ##############################

model_names = ['mnist_model_NOdropout_01.pt', 'mnist_model_NOdropout_02.pt', 'mnist_model_NOdropout_03.pt',
                'mnist_model_NOdropout_05.pt' ]


final_error = []

for model in xrange(0, len(model_names)):

    load_dir = os.getcwd() + '/networks/networks_NOdropout/' + model_names[model]

    #LOAD
    net = Net_mnist()
    net.cuda()
    net.load_state_dict(torch.load(load_dir))

    final_error.append(get_accuracy(testloader, net, classes))

fin_NOdropout_error = np.mean(final_error)
fin_NOdropout_std = np.std(final_error)

with open('networks/networks_NOdropout/results/final_error_NOdropout.pkl', 'w') as f:  # Python 3: open(..., 'wb')
    pickle.dump([fin_NOdropout_error, fin_NOdropout_std], f)
