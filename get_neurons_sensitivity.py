# Script to test trucated weights during Inference
from my_helper_pytorch import *
import pickle

### Load model
#PATH
model_names = ['mnist_model_dropout_01.pt']
sweet_spot = ['sweetSpot_det_validation_dropout_01.pkl']
model = 0


load_dir = os.getcwd() + '/networksv2/networks_dropout/' + model_names[model]


#LOAD
net = Net_mnist()
net.cuda(0)
net.load_state_dict(torch.load(load_dir))

# bin_table = build_binary_table_v2(m_round[m], e_round[e], e_max[em])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0003)
# optimizer = optim.SGD(net.parameters(), lr=0)
optimizer.zero_grad()


fc1_grad = np.zeros(np.shape(net.fc.weight))
fc2_grad = np.zeros(np.shape(net.fc2.weight))


for i, data in enumerate(testloader, 0):

    # get the inputs
    inputs, labels = data


    # wrap them in Variable
    inputs, labels = Variable(inputs.cuda(0)), Variable(labels.cuda(0))
    # print(labels)

    # optimizer.zero_grad() Do Not zero - gradients, just accumulate and divide by number of batches
    outputs = net(inputs)
    loss = criterion(outputs,labels)

    optimizer.zero_grad() # Zero - gradients, keep summing them
    loss.backward() #Accumulate gradients
    # optimizer.step() Do not update weights

    fc1_grad += ( (net.fc.weight.grad.data).cpu().numpy() )**2
    fc2_grad += ( (net.fc2.weight.grad.data).cpu().numpy() ) **2


# Average gradients in the first layer for the test set
num_batches = len(testloader.dataset.test_labels) / testloader.batch_size
fc1_grad = fc1_grad / num_batches
fc2_grad = fc2_grad / num_batches


############################## PLOTS ####################################


### Gradients Squared ###
counts,bins = np.histogram(fc1_grad, bins=10)
plt.figure()
plt.bar( bins[:-1]+np.diff(bins)/2, counts, np.diff(bins))
plt.title('Histogram of squared gradients in FC1')
plt.show(block=False)

counts,bins = np.histogram(fc2_grad, bins=10)
plt.figure()
plt.bar( bins[:-1]+np.diff(bins)/2, counts, np.diff(bins))
plt.title('Histogram of squared gradients in FC2')
plt.show(block=False)


### Gradients / Weights ###
w_fc1 = (net.fc.weight.data).cpu().numpy()
w_fc2 = (net.fc2.weight.data).cpu().numpy()

fc1_grad_by_w = fc1_grad / np.abs(w_fc1)
counts,bins = np.histogram(fc1_grad_by_w, bins=10)
plt.figure()
plt.bar( bins[:-1]+np.diff(bins)/2, counts, np.diff(bins))
plt.title('Histogram of squared gradients in FC1')
plt.show(block=False)

fc2_grad_by_w = fc2_grad / np.abs(w_fc2)
counts,bins = np.histogram(fc2_grad_by_w, bins=10)
plt.figure()
plt.bar( bins[:-1]+np.diff(bins)/2, counts, np.diff(bins))
plt.title('Squared gradients normalized by |w| in FC2')
plt.show(block=False)


### SQRT (Gradients) / Weights ###

m_round = [1]
e_round = [2]
e_max = [-4, -3, -2, -1, 0, 1, 2, 3, 4] #Do not pass e_max arg if no limit desired
load_dir = os.getcwd() + '/networksv2/networks_dropout/' + model_names[model]
sweet_spot_dir = os.getcwd() + '/networksv2/networks_dropout/SweetSpot/' + sweet_spot[model]
with open(sweet_spot_dir) as f:  # Python 3: open(..., 'rb')
    ss, _, _, _, _ = pickle.load(f)

bin_table = build_binary_table_v2(m_round[0], e_round[0], e_max[int(ss[m_round[0]][e_round[0]])])

#fc1
w_tensor = net.fc.weight.data
w_array = w_tensor.cpu().numpy()                                #Convert torch.cuda.tensor to array
w_array_truncated = stochastic_rounding(w_array, bin_table)     # Truncate
delta_fc1 = np.abs(w_fc1 - w_array_truncated)

#fc2
w_tensor = net.fc2.weight.data
w_array = w_tensor.cpu().numpy()                                #Convert torch.cuda.tensor to array
w_array_truncated = stochastic_rounding(w_array, bin_table)     # Truncate
delta_fc2 = np.abs(w_fc2 - w_array_truncated)

# X-axis
delta_by_w_fc1 = delta_fc1 / np.abs(w_fc1)
delta_by_w_fc2 = delta_fc2 / np.abs(w_fc2)
# Y-axis
prop_error_fc1 = 1 / (np.abs(w_fc1) * np.sqrt(fc1_grad))
prop_error_fc2 = 1 / (np.abs(w_fc2) * np.sqrt(fc2_grad))

plt.figure()
plt.scatter(np.log2(delta_by_w_fc1.flatten()), np.log2(prop_error_fc1.flatten()))
plt.title('log2 plot, FC1')
plt.xlabel('log2 (1 / (|W| sqrt(d^2E/dW^2)))')
plt.ylabel('log2 ()|deltaW| / |W|)')
plt.show(block=False)

plt.figure()
plt.scatter(np.log2(delta_by_w_fc2.flatten()), np.log2(prop_error_fc2.flatten()))
plt.title('log2 plot, FC2')
plt.xlabel('log2 (1 / (|W| sqrt(d^2E/dW^2)))')
plt.ylabel('log2 ()|deltaW| / |W|)')
plt.show(block=False)
