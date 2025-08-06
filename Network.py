import numpy as np
import matplotlib.pyplot as plt
from pickle import dump, load
from copy import deepcopy
from time import time
import joblib as jl
from gc import collect

def relu(x):
    return np.maximum(0, x)

def d_relu(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp( - x.clip(-10, 10)))

def d_sigmoid(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

# write down the network to a file
def write_network(n, name = "network.pkl"):
    with open(name, "wb") as f:
        dump(n, f)

# read the network from a file
def read_network(name = "network.pkl"):
    with open(name, "rb") as f:
        return load(f)

class picture(object):
    '''
    A picture input with its tag for training. Image stored as 320 * 370 matrix, tag stored as an array
    '''

    def __init__(self, image, tag = None):
        self.image = image
        self.tag = tag
        '''
        Tags are: (1 positive 0 negative -1 uncertain nan no label)
        Age, Sex (0 male 1 female), No Finding, Enlarged Cardiomediastinum, Cardiomegaly, Lung Opacity, Lung Lesion,
        Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax,
        Pleural Effusion, Pleural Other, Fracture, Support Devices
        ''' 

    # visualize a single graph.
    def visualize(self):
        image = np.multiply(self.image, 255)
        plt.figure(figsize = (4, 3))
        plt.imshow(image, cmap=plt.cm.gray)
        plt.show()

# a convolutional layer of a neural network. Uses valid padding, ReLU activation, and He initialization.
class convolution(object):
    def __init__(self, image_shape, channel, padding = 0, kernel = (3,3), stride = 1, dilation = 1):
        if not (padding == 0 and dilation == 1):
            raise NotImplementedError

        self.input_height = image_shape[0]
        self.input_width = image_shape[1]
        self.channel = channel
        self.padding = padding
        self.kernel_height = kernel[0]
        self.kernel_width = kernel[1]
        self.stride = stride
        self.dilation = dilation

        self.output_height = (self.input_height - self.kernel_height) // self.stride + 1
        self.output_width = (self.input_width - self.kernel_width) // self.stride + 1
        self.output_shape = (self.output_height, self.output_width)

        # Intialization
        std_dev = np.sqrt(2 / (self.kernel_height * self.kernel_width * self.channel))
        self.kernel_weight = np.random.normal(loc = 0, scale = std_dev, size = kernel)
        self.bias = np.zeros((self.output_height, self.output_width))
        self.kernel_weight_momentum = np.zeros(kernel)
        self.bias_momentum = np.zeros((self.output_height, self.output_width))
        
    # given a batch of input images, return the convluted version, as a 2D array
    def forward_pass(self, input):
        if input.ndim != 3:
            raise ValueError("Input must be of shape (batch_size, input_height, input_width)")
        output = np.zeros((input.shape[0], self.output_height, self.output_width))

        # assumes dilation = 1, no padding
        for i in range(0, self.output_height):
            for j in range(0,self.output_width):
                kernel = input[:, i * self.stride: i * self.stride + self.kernel_height,
                               j * self.stride: j * self.stride + self.kernel_width]
                output[:, i, j] = np.sum(kernel * self.kernel_weight.reshape(1, self.kernel_height, self.kernel_width)
                                         , axis = (1, 2))     

        pre_activation = output + self.bias.reshape(1, self.output_height, self.output_width)
        activation = relu(pre_activation)
        return (pre_activation, activation)

    # given the input, gradients to activations, pre_activation, and the k factor (-lr / batch_size)
    # update the weights and biases, and return the gradients of inputs for the next layer
    # if this is the first layer, use skip_gradient to skip calculating gradient wrt input
    def back_propogation(self, input, pre_activation, d_cost_d_activation, k, m, skip_gradient = False):
        d_relu_pre_activation = d_relu(pre_activation)
        d_cost_d_pre_activation = d_relu_pre_activation * d_cost_d_activation
        grad_bias = np.sum(d_cost_d_pre_activation, axis = 0)

        # compute gradient wrt weights
        grad_kernel_weight = np.zeros((self.kernel_height, self.kernel_width))
        for i in range(0, self.output_height):
            for j in range(0, self.output_width):
                kernel = input[:, i * self.stride: i * self.stride + self.kernel_height,
                               j * self.stride: j * self.stride + self.kernel_width]
                grad_kernel_weight += np.sum(d_cost_d_pre_activation[:, i, j].reshape(-1, 1, 1) * kernel, axis = 0)
        
        self.kernel_weight_momentum = self.kernel_weight_momentum * m + grad_kernel_weight * (1 - m)
        self.kernel_weight += self.kernel_weight_momentum * k
        self.bias_momentum = self.bias_momentum * m + grad_bias * (1 - m)
        self.bias += self.bias_momentum * k

        if skip_gradient:
            return None

        # compute gradient wrt inputs
        d_cost_d_input = np.zeros((input.shape[0], self.input_height, self.input_width))
        flipped_kernel = np.flip(self.kernel_weight, axis = (0, 1)).reshape(1, self.kernel_height, self.kernel_width)
        for i in range(0, self.output_height):
            for j in range(0, self.output_width):
                d_cost_d_input[:, i * self.stride: i * self.stride + self.kernel_height,
                               j * self.stride: j * self.stride + self.kernel_width
                              ] += d_cost_d_pre_activation[:, i, j].reshape(-1, 1, 1) * flipped_kernel
        return d_cost_d_input

# A convolutional layer, which supports multiple channels
class convolution_layer(object):
    def __init__(self, input_channel, output_channel, input_shape, skip_gradient = False):
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.input_shape = input_shape
        self.output_per_input = self.output_channel // self.input_channel
        self.network_list = []
        self.skip_gradient = skip_gradient

        if (self.output_channel % self.input_channel != 0):
            raise ValueError("Number of output channel must be multiple of number of input channels")
        
        for i in range(self.output_channel):
            self.network_list.append(convolution(self.input_shape, self.input_channel))
        self.output_shape = self.network_list[0].output_shape

    # given a list of inputs (input_channel, batch_size, height, width), calculate the list of pre activations and activations
    def forward_pass(self, input_list):
        def process_channel(i):
            local_pre_activation_list = [None] * self.output_per_input
            local_activation_list = [None] * self.output_per_input
            base = i * self.output_per_input
            for conv_id in range(i * self.output_per_input, (i + 1) * self.output_per_input):
                conv = self.network_list[conv_id]
                pre_activation, activation = conv.forward_pass(input_list[i])
                local_pre_activation_list[conv_id - base] = pre_activation
                local_activation_list[conv_id - base] = activation
            return (local_pre_activation_list, local_activation_list)

        result_list = jl.Parallel(n_jobs = self.input_channel)(jl.delayed(process_channel)(i) for i in range(self.input_channel))
        pre_activation_list = []
        activation_list = []
        for local_pre_activation_list, local_activation_list in result_list:
            pre_activation_list += local_pre_activation_list
            activation_list += local_activation_list
        return (activation_list, pre_activation_list)
    
    def back_propogation(self, input_list, pre_activation_list, d_cost_d_activation_list, k, m):
        def process_channel(i):
            d_cost_d_input = np.zeros_like(input_list[i], dtype=np.float64)
            for conv_id in range(i * self.output_per_input, (i + 1) * self.output_per_input):
                conv = self.network_list[conv_id]
                d_cost_d_input_temp = conv.back_propogation(input_list[i], pre_activation_list[conv_id], d_cost_d_activation_list[conv_id], k, m, self.skip_gradient)
                if d_cost_d_input_temp is None:
                    continue
                d_cost_d_input += d_cost_d_input_temp
            return d_cost_d_input

        result_list = jl.Parallel(n_jobs = self.input_channel)(jl.delayed(process_channel)(i) for i in range(self.input_channel))
        
        if self.skip_gradient:
            return None
        
        d_cost_d_input_list = [None] * self.input_channel
        for i, result in enumerate(result_list):
            d_cost_d_input_list[i] = result
        return d_cost_d_input_list

# max pooling layer of a deep network. Only supports 2 * 2 pooling.
class max_pool(object):
    # the input must be both even numbers
    def __init__(self):
        pass

    # a forward pass of the max pooling layer, return the 3D array of activations
    # and a matrix to keep track of the maximum position (0 for top left, 1 for top right, etc.)
    def forward_pass(self, input):
        if input.ndim != 3:
            raise ValueError("Input must be of shape (batch_size, input_height, input_width)")
        if input.shape[1] % 2 != 0 or input.shape[2] % 2 != 0:
            raise ValueError("Input size of a max pooling layer must be even")

        top_left = input[:, ::2, ::2]
        top_right = input[:, ::2, 1::2]
        bottom_left = input[:, 1::2, ::2]
        bottom_right = input[:, 1::2, 1::2]

        stack = np.stack([top_left, top_right, bottom_left, bottom_right])
        activation = np.max(stack, axis = 0)
        indices = np.argmax(stack, axis = 0)
        return (activation, indices)
        
    # given the gradient to the activations, return the gradients to the inputs
    def back_propogation(self, indices, d_cost_d_activation):
        # accumulates gradient based on its origin
        batch_size = indices.shape[0]
        height = indices.shape[1] * 2
        width = indices.shape[2] * 2
        d_cost_d_input = np.zeros((batch_size, height, width))

        # This is a slow but stable version that avoids possible problems with masking
        b, i, j = np.where(indices == 0)
        d_cost_d_input[b, 2 * i, 2 * j] = d_cost_d_activation[b, i, j]
        b, i, j = np.where(indices == 1)
        d_cost_d_input[b, 2 * i, 2 * j + 1] = d_cost_d_activation[b, i, j]
        b, i, j = np.where(indices == 2)
        d_cost_d_input[b, 2 * i + 1, 2 * j] = d_cost_d_activation[b, i, j]
        b, i, j = np.where(indices == 3)
        d_cost_d_input[b, 2 * i + 1, 2 * j + 1] = d_cost_d_activation[b, i, j]
        return d_cost_d_input

# a max pooling layer 
class max_pool_layer(object):
    def __init__(self, input_channel, output_channel, input_shape):
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.input_shape = input_shape
        self.output_per_input = self.output_channel // self.input_channel
        self.network_list = []

        if (self.output_channel % self.input_channel != 0):
            raise ValueError("Number of output channel must be multiple of number of input channels")

        for i in range(self.output_channel):
            self.network_list.append(max_pool())
        self.output_shape = (self.input_shape[0] // 2, self.input_shape[1] // 2)

    def forward_pass(self, input_list):
        activation_list = [None] * self.output_channel
        indices_list = [None] * self.output_channel
        # could be parallelized 
        for i in range(self.input_channel):
            for pool_id in range(i * self.output_per_input, (i + 1) * self.output_per_input):
                pool = self.network_list[pool_id]
                activation, indices = pool.forward_pass(input_list[i])
                activation_list[pool_id] = activation
                indices_list[pool_id] = indices
        return (activation_list, indices_list)
        
    def back_propogation(self, input_list, indices_list, d_cost_d_activation_list):
        def process_channel(i):
            d_cost_d_input = np.zeros_like(input_list[i])
            for pool_id in range(i * self.output_per_input, (i + 1) * self.output_per_input):
                pool = self.network_list[pool_id]
                d_cost_d_input += pool.back_propogation(indices_list[pool_id], d_cost_d_activation_list[pool_id])
            return d_cost_d_input

        d_cost_d_input_list = jl.Parallel(n_jobs = self.input_channel)(jl.delayed(process_channel)(i) for i in range(self.input_channel))
        return d_cost_d_input_list

# the layer before the fully connected network, to reduce channels to 1 (aka 1x1 convolution)
class aggregate_channel(object):
    def __init__(self, input_channel, input_shape):
        self.input_channel = input_channel
        self.height = input_shape[0] 
        self.width = input_shape[1]

        # He initialization
        std_dev = np.sqrt(2 / self.input_channel)
        self.weight = np.random.normal(loc = 0, scale = std_dev, size = (self.input_channel, self.height, self.width))
        self.bias = np.zeros(input_shape)
        self.weight_momentum = np.zeros((self.input_channel, self.height, self.width))
        self.bias_momentum = np.zeros(input_shape)

    # given a list of activations, compute the pre_activation and activation
    def forward_pass(self, input):
        input = np.stack(input, axis = 0) # (input_channel, batch_size, height, width)
        pre_activation = np.sum(input * self.weight.reshape(self.input_channel, 1, self.height, self.width), axis = 0 
                               ) + self.bias.reshape(1, self.height, self.width) # (batch_size, height, width)
        activation = relu(pre_activation)
        return (activation, pre_activation)
        
    # given the gradient to this point, pre_activation, train the network return the gradient wrt inputs
    def back_propogation(self, input, pre_activation, d_cost_d_activation, k, m):
        input = np.stack(input, axis = 0)
        d_cost_d_pre_activation = d_cost_d_activation * d_relu(pre_activation)
        grad_bias = np.sum(d_cost_d_pre_activation, axis = (0, 1))
        self.bias_momentum = self.bias_momentum * m + grad_bias * (1 - m)
        self.bias += self.bias_momentum * k

        grad_weight = np.sum(input * d_cost_d_pre_activation[np.newaxis, :, :, :], axis=1)  # (input_channel, height, width)
        self.weight_momentum = self.weight_momentum * m + grad_weight * (1 - m)
        self.weight += self.weight_momentum * k

        grad_input = self.weight[:, np.newaxis, :, :] * d_cost_d_pre_activation[np.newaxis, :, :, :]  # (input_channel, batch_size, height, width)
        return grad_input

# a fully connected layer of neural network. Given a single 2D array, apply weight and bias, and the sigmoid function for binary classification
class fully_connected(object):
    def __init__(self, input_shape, output_size):
        self.input_height = input_shape[0]
        self.input_width = input_shape[1]
        self.input_size = self.input_height * self.input_width
        self.output_size = output_size

        # Initializationn
        std_dev = np.sqrt(4 / (self.input_size + self.output_size))
        self.weight = np.random.normal(loc = 0, scale = std_dev, size = (self.output_size, self.input_size))
        self.bias = np.zeros((self.output_size, 1)) 
        self.weight_momentum = np.zeros((self.output_size, self.input_size))
        self.bias_momentum = np.zeros((self.output_size, 1)) 

    # given a (batch of) input, return the list of pre_activations and list of activations
    def forward_pass(self, input):
        batch_size = input.shape[0]
        input = input.reshape((-1, self.input_size)).transpose() # (batch_size, width, height) -> (input_size, batch_size)
        pre_activation = np.dot(self.weight, input) + self.bias
        activation = sigmoid(pre_activation)
        return (activation, pre_activation)

    # given a (batch of) input, return the difference and squared difference cost. Can pass in activation list to avoid computing again
    # if some tags are nan, then difference is set to 0.
    def compute_cost(self, input, tag, activation = None):
        if activation is None:
            _, activation = self.forward_pass(input)
        difference = activation - tag.transpose()  # (batch_size, difference)
        difference = np.where(~np.isnan(tag.transpose()), difference, 0)
        cost = np.square(difference).sum()
        num_tags = (~np.isnan(tag)).sum() # adjustment by valid tag count
        return (difference, cost / num_tags) # Note that difference is (output_size, batch_size)
    
    # given a batch of input, pre_activation, and activation, train the network and compute the gradient wrt this layer's input
    def back_propogation(self, input, difference, pre_activation, activation, k, m):
        batch_size = difference.shape[1]
        input = input.reshape((-1, self.input_size)).transpose()
        d_cost_d_activation = 2 * difference
        d_cost_d_pre_activation = d_cost_d_activation * d_sigmoid(pre_activation)

        gradient_bias = np.sum(d_cost_d_pre_activation, axis = 1, keepdims = True)
        gradient_weight = np.dot(d_cost_d_pre_activation, input.transpose())
        d_cost_d_input = np.dot(self.weight.transpose(), d_cost_d_pre_activation)

        self.weight_momentum = self.weight_momentum * m + gradient_weight * (1 - m)
        self.weight += self.weight_momentum * k
        self.bias_momentum = self.bias_momentum * m + gradient_bias * (1 - m)
        self.bias += self.bias_momentum * k
        d_cost_d_input = (d_cost_d_input.transpose()).reshape((batch_size, self.input_height, self.input_width))
        return d_cost_d_input

# This is the outmost structure of the network. It automatically ends with aggregating layer and the fully connected layer
class network(object):
    # the layer_list_temp should be a list of tuples formatted as [('P', 5), ('C', 10), ..] for type and channel count
    # can only be pooling or convolutional.
    def __init__(self, layer_list_temp, input_shape = (320, 368), output_size = 14):
        self.layer_list = []
        self.input_shape = input_shape
        last_shape = input_shape
        last_channel = 1

        for type, channel in layer_list_temp:
            if type == 'C':
                new_layer = convolution_layer(last_channel, channel, last_shape, len(self.layer_list) == 0)
            elif type == 'P':
                new_layer = max_pool_layer(last_channel, channel, last_shape)
            else:
                raise ValueError("The input can only be max pooling layer (P) or convolution layer (C)")
            
            last_shape = new_layer.output_shape
            last_channel = new_layer.output_channel
            self.layer_list.append(new_layer)
        
        self.layer_list.append(aggregate_channel(last_channel, last_shape))
        self.layer_list.append(fully_connected(last_shape, output_size))

    # a forward pass that stores all the activations, pre_activations, and indicies. Also gives the cost and difference, if tags are provided
    def forward_pass(self, input, tags = None):
        input = [input] # convert to one channel
        last_activation = input
        activation_list = []
        arg2_list = []  # pre_activation for convolutional layer and indices for max pooling layer

        for layer in self.layer_list:
            last_activation, arg2 = layer.forward_pass(last_activation)
            activation_list.append(last_activation)
            arg2_list.append(arg2)

        if tags is None:
            return (activation_list, arg2_list, None, None)
        else:
            last_layer = self.layer_list[-1]
            difference, cost = last_layer.compute_cost(input, tags, activation_list[-1])
            return (activation_list, arg2_list, difference, cost)

    # compute the output only in shape (batch_size, output_size)
    def predict(self, input, threshold = 0.6):
        activation_list, _, _, _ = self.forward_pass(input)
        return (activation_list[-1] >= threshold).astype(float).transpose()

    # given a batch object, train the model and output the cost
    def train_batch(self, data, learning_rate, m = 0.8, debug = False):
        activation_list, arg2_list, difference, cost = self.forward_pass(data.images, data.tags)
        k = - learning_rate / data.n_data
        if debug:
            gradient_list = [None] * len(self.layer_list)

        for i, layer in reversed(list(enumerate(self.layer_list))):
            if i == 0:
                input = [data.images]
            else:
                input = activation_list[i - 1]

            if isinstance(layer, fully_connected):
                gradient = layer.back_propogation(input, difference, arg2_list[i], activation_list[i], k, m)
            elif isinstance(layer, aggregate_channel):
                gradient = layer.back_propogation(input, arg2_list[i], gradient, k, m)
            elif isinstance(layer, convolution_layer):
                gradient = layer.back_propogation(input, arg2_list[i], gradient, k, m)
            elif isinstance(layer, max_pool_layer):
                gradient = layer.back_propogation(input, arg2_list[i], gradient)
            else:
                raise ValueError("Layer type error")
        
            if debug:
                gradient_list[i] = gradient

        if debug:
            return gradient_list
        else:
            return cost
    
    # given a large batch of training data, train the model by SGD
    def train(self, data, iteration = 1, learning_rate = 0.01, batch_size = 1000):
        for i in range(iteration):
            training_list = data.segment_batch(batch_size)
            collect()
            sum_cost = 0.0
            for index, batch in enumerate(training_list):
                cost = self.train_batch(batch, learning_rate)
                sum_cost += cost
                print("Finished training mini-batch {} with cost {:.5f}".format(index + 1, float(cost)))
                collect()
            print("Finished training iteration {} with epoch average cost {:.5f}".format(i + 1, sum_cost / (index + 1)))
    
    # given a batch of testing data, return the success rate. Assumes it has all the tags. Only correct if all determination are correct
    def test(self, data, threshold = 0.6):
        prediction = self.predict(data.images, threshold)
        compare = np.isclose(prediction, data.tags)
        num_correct = np.sum(np.all(compare, axis = 0))
        return num_correct / data.n_data

class batch(object):
    def __init__(self, images, tags):
        self.n_data = images.shape[0]
        self.images = images
        self.tags = tags
    
    def __repr__(self):
        return "A batch of {} images and tags".format(self.n_data)

    # given a batch index, convert to a picture object
    def to_picture(self, index):
        return picture(self.images[index], self.tags[index])

    # given a mini-batch size, return a list of batches for SGD
    def segment_batch(self, batch_size):
        indices = np.random.permutation(batch_size * (self.n_data // batch_size))
        shuffled_data = self.images[indices]
        shuffled_tags = self.tags[indices]
        training_list = [batch(shuffled_data[i: i + batch_size],
                        shuffled_tags[i: i + batch_size]) for i in range(0, self.n_data, batch_size)]
        return training_list
    
    # given a list of values return a batch object such that the tag exists in all of them (postive, negative, or uncertain, not nan)
    # has_value and trim_tags generally NOT needed. cheXpert validation has all information, and model can be trained with partial tags
    '''
    0 No Finding, 1 Enlarged Cardiomediastinum, 2 Cardiomegaly, 3 Lung Opacity, 4 Lung Lesion,
    5 Edema, 6 Consolidation, 7 Pneumonia, 8 Atelectasis, 9 Pneumothorax,
    10 Pleural Effusion, 11 Pleural Other, 12 Fracture, 13 Support Devices
    '''
    def has_value(self, value_list):
        if self.tags.shape[1] != 14:
            raise ValueError("Dangerous to select after it has been trimmed down")
        
        mask = ~np.isnan(self.tags[:, np.array(value_list)]).any(axis=1)
        return batch(self.images[mask], self.tags[mask])

    # given a list of values, trim down the tags so that the tags only contain the ones in the list
    def trim_tags(self, value_list):
        if self.tags.shape[1] != 14:
            raise ValueError("Dangerous to select after it has been trimmed down")
        
        value_list = np.array(value_list)
        return batch(self.images[:, value_list], self.tags[:, value_list])

    # convert the uncertain flag (-1) to negative (0) or positive. In place modification
    def convert_uncertain(self, value = 0):
        self.tags[self.tags == -1] = value

# the cheXpert data set. Note that age and sex are removed from the tags
class data_set(object):
    def __init__(self):
        self.train_images = np.load("images_frontal_male_train_small.npy") 
        self.train_tags = np.load("tags_frontal_male_train_small.npy")
        self.test_images = np.load("images_frontal_male_valid.npy")
        self.test_tags = np.load("tags_frontal_male_valid.npy")
    
    # return the entire training set as a batch object
    def train(self):
        return batch(self.train_images, self.train_tags)

    # return the entire test set as a batch object
    def test(self):
        return batch(self.test_images, self.test_tags)

if __name__ == "__main__":
    data = data_set()
    train = data.train()
    train.convert_uncertain(0)
    #value_list = list(range(14))
    test = data.test()   #.has_value(value_list) # ensure it has all tags
    test.convert_uncertain(0) 
    del data

    cnn = network([('C',3), ('C', 6), ('P', 12), ('C',24), ('P', 24), ('C',24)])
    # 320 * 368, 318 * 366, 316 * 364, 158 * 182, 156 * 180, 78 * 90, 76 * 88, 
    cnn.train(train, 3, 0.01)
    print("Success rate {:.2f}".format(cnn.test(test) * 100))
    write_network(cnn)
