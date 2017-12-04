'''
Created Wed Apr 20 2016

@author vipulkhatana

'''
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import gzip 
import math
import random
import itertools

mnist_data_folder = ''

#Name of training and testing files in gunzip format

training_set_file_name =             'train_images_idx3_ubyte_customize.gz'
training_labels_file_name =          'train_labels_idx1_ubyte_customize.gz'
testing_set_file_name =              'test_images_idx3_ubyte_customize.gz'
#testing_labels_file_name =           't10k-labels-idx1-ubyte.gz'

class load_idx:
    
    def __init__(self, file_name=None, fstream=None, file_handler=open):
        self.file_name = file_name
        self.fstream = fstream
        self.file_handler = file_handler
        self.magic_number = 0
        self.header_dtype = np.dtype(np.uint32).newbyteorder('>')            # Defining the header datatype,
                                                                             # '>' specifies big-endian byteorder
                                                                             # so that conversion can be done correctly.
                                                                         
        if not (self.file_name is not None) ^ (self.fstream is not None):    # Condition to check if both input method
                                                                             # are not defined
            raise ValueError('Define either File Name or File Stream')
        elif self.file_name is not None:
            self.fstream = self.file_handler(self.file_name, 'rb')
 
    def get_magic_number(self):
        self.magic_number = np.frombuffer(self.fstream.read(4), dtype=self.header_dtype)
        return self.magic_number
      
    def _extract_header(self):
        mask_dim = int('0x000000ff',16)                                  # Mask for dimensions. Since the information
                                                                         # about the dimensions is present in the last
                                                                         # (fourth) byte.
        
        mask_datatype = int('0x0000ff00',16)                             # Mask for datatype. Since the information
                                                                         # about the datataype is present in the second
                                                                         # last (third) byte
        
        no_of_dimensions = np.bitwise_and(self.magic_number, np.array(mask_dim, dtype=np.uint32))
                                                                         # Extracting the last byte i.e. Number of
                                                                         # dimenstions. And operation with
                                                                         # the created mask
        
        datatype_index = np.right_shift(np.bitwise_and(self.magic_number, np.array(mask_datatype, dtype=np.uint32)),8)
                                                                         # Extracting the second last byte i.e. datatype
                                                                         # index. And operation with Mask and then right
                                                                         # shift by 1 byte (8 bits).
                    
        # Defining the datatype based on the datatype information gathered from the header.
        if datatype_index == int('0x08',16):
            dt = np.dtype(np.uint8)
        elif datatype_index == int('0x09',16):
            dt = np.dtype(np.int8)
        elif datatype_index == int('0x0B',16):
            dt = np.dtype(np.int16)
        elif datatype_index == int('0x0C',16):
            dt = np.dtype(np.int32)
        elif datatype_index == int('0x0D',16):
            dt = np.dtype(np.float32)
        elif datatype_index == int('0x0E',16):
            dt = np.dtype(np.float64)
        
        dimensions = np.empty(no_of_dimensions, dtype=np.uint32)
        
        
        # Extracting the information about dimensions from the file.
        for i in range(no_of_dimensions):
            read_val = np.frombuffer(self.fstream.read(4),dtype=self.header_dtype)
            dimensions[i] = read_val
        
        return dimensions, dt
    
    def load_file(self):
        if self.magic_number == 0:
            self.get_magic_number()
        [dimensions, dt] = self._extract_header()
        total_bytes_to_be_read = np.prod(dimensions, dtype=np.int32)*dt.itemsize
        data = np.frombuffer(self.fstream.read(total_bytes_to_be_read),dtype=dt)
        data = np.reshape(data,dimensions)
        if self.file_name is not None:
            self.fstream.close()
        return data
    

class load_mnist(load_idx):
   
    def __init__(self, file_name, file_type, file_handler=open, convert_to_float = False, display_sample = 0):
        load_idx.__init__(self, file_name = file_name, file_handler=file_handler)
        self.file_type = file_type
        self.convert_to_float = convert_to_float
        self.display_sample = display_sample
        self.mnist_magic_number={'data':2051, 'label':2049}
        if self.file_type == 'label':
            self.display_sample = 0
    

    def load(self):
        self.get_magic_number()
        if self.mnist_magic_number[self.file_type] == self.magic_number:
            self.data = self.load_file()
            if self.convert_to_float:
                self.data = self.data.astype(np.float32)
                self.data = np.multiply(self.data, 1.0/255.0)
            if self.display_sample != 0:
                self.display_samples(self.display_sample)
            return self.data
        else:
            print('Given file is not mnist : (%s,%s)'%(self.file_name, self.file_type))


    def display_samples(self, how_many=5):
        size = self.data.shape[0]
        perm = np.random.permutation(size)
        perm = perm[:how_many]
        images = self.data[perm,:,:]
        for i in range(how_many):
            fig = plt.figure()
            plt.imshow(images[i], cmap='Greys_r')
        
    
    def display_images(self, number):
        if number.shape.__len__() > 1:
            print('Number should be 1D array')
        else:
            for i in number:
                fig = plt.figure()
                plt.imshow(self.data[i], cmap='Greys_r')
#Test bench



# If providing data in the gunzip format
test=load_mnist(training_set_file_name, 'data', file_handler=gzip.GzipFile)
t = test.load()


train_images_obj = load_mnist(mnist_data_folder+training_set_file_name, 'data', file_handler=gzip.GzipFile, display_sample=5)
train_labels_obj = load_mnist(mnist_data_folder+training_labels_file_name, 'label', file_handler=gzip.GzipFile)
test_images_obj = load_mnist(mnist_data_folder+testing_set_file_name, 'data', file_handler=gzip.GzipFile, display_sample=5)
#test_labels_obj = load_mnist(mnist_data_folder+testing_labels_file_name, 'label', file_handler=gzip.GzipFile)

train_images = train_images_obj.load()
train_labels = train_labels_obj.load()
test_images = test_images_obj.load()
# Many learning algorithms accepts images in the vector format. Hence converting images in the vector format.

train_images = train_images.reshape(train_images.shape[0],np.prod(train_images.shape[1:]))
test_images = test_images.reshape(test_images.shape[0], np.prod(test_images.shape[1:]))

def save_pickle(fileName, data):
    with open(fileName+'.pkl', 'w') as fid:
        pkl.dump(data,fid)
    
def load_pickle(fileName):
    with open(fileName, 'rb') as fid:
        data = pkl.load(fid)
    return data

train_data = [train_images, train_labels]
test_data = test_images

pkl.dump(test_images, open('test_images.pkl','wb'), protocol=2)
pkl.dump(train_data, open('train_images_with_labels_mnist.pkl','wb'), protocol=2)
def load_pickle(fileName):
    with open(fileName, 'rb') as fid:
        data = pkl.load(fid)
    return data
        
 #BEGINING THE CODE       
[train_data,train_label] = load_pickle('train_images_with_labels_mnist.pkl')
test = load_pickle('test_images.pkl') 
train_data = train_data/255 
test = test/255
trainv2 = []

for i in range(train_data.shape[0]):
    #print (i)
    trainv2.append([train_data[i]] + [train_label[i]])
testv2 = []
for i in range(test.shape[0]):
    #print (i)
    testv2.append([test[i]] + [0])
    #print (trainv2)   
LearningRate = 25
iteration = 3

# Helps read the input file into a wrokable foramt
def readFile(fileName):
	mat = np.array([line.strip().split(',') for line in open(fileName)])
	return np.array([[line[:-1]] + [line[-1]] for line in mat])

# This helps create a single perceptron unit
def create_unit(num):
    initial_weight = np.array([random.uniform(-0.1, 0.1) for _ in range(num)])
    np.savetxt('initial_weight.txt',initial_weight)

    return np.array([random.uniform(-0.1, 0.1) for _ in range(num)]) # Random Initialized Weights
# Creates a whole layer of perceptrons
def create_layer(num, size):
	layer = np.zeros((num,size))
	for i in range(num):
		layer[i] = create_unit(size)
	return layer
 
def power_e(val):
    if val > 200:
        return float("inf")
    else:
        return math.exp(val)

# The Sigmoid function
def sigmoid(X):
    return np.matrix( [1 / (1 +power_e(-X.item(i))) for i in range(X.shape[1])] )

# The SoftPlus function
def softplus(X):
	return np.matrix( [math.log(1 + math.exp(X.item(i))) for i in range(X.shape[1])] )

# To train the perceptron weights
def train(inputs, hidden, output, map_output):
	count = -1
	global iteration
	for input in inputs:
		count = count + 1
		# Use for changing eta
		# iteration += 1;
		# eta = float(LearningRate / float(iteration)**(1.0/2.0))

		# Use for constant eta
		eta = 0.1

		res = map_output[input[1]]
		test_input = np.matrix(input[0].astype(np.float))
		hid_out = sigmoid(test_input*hidden.T)
		hid_output = np.matrix(np.append([1], hid_out))
		out = sigmoid(hid_output*output.T)
		error_out = np.multiply(np.multiply(out, (1 - out)), (res - out))
		error_hid = np.multiply(np.multiply(hid_out, (1 - hid_out)), (error_out*output)[:,1:])
		output = output + eta*error_out.T*hid_output
		hidden = hidden + (eta*error_hid.T*test_input)
	return hidden, output

# To train the perceptron weights
def train_soft(inputs, hidden, output, map_output):
	count = -1
	global iteration
	for input in inputs:
		count = count + 1
		# Use for changing eta
		# iteration += 1;
		# eta = float(LearningRate / float(iteration)**(1.0/2.0))

		# Use for constant eta
		eta = 0.02

		res = map_output[input[1]]
		test_input = np.matrix(input[0].astype(np.float))
		hid_out = softplus(test_input*hidden.T)
		hid_output = np.matrix(np.append([1], hid_out))
		out = softplus(hid_output*output.T)
		error_out = np.multiply(sigmoid(out), (res - out))
		error_hid = np.multiply(sigmoid(hid_out), (error_out*output)[:,1:])
		output = output + eta*error_out.T*hid_output
		hidden = hidden + (eta*error_hid.T*test_input)
	return hidden, output

# Finds the norm of two vectors
def norm(vec1, vec2):
	return np.linalg.norm(vec1 - vec2)

# Classifies the class based on the output layer values
def classify(output, map_output):
	saved = ''
	min = float("inf")
	for i in range(0, 10): 
		val = map_output[i]     
		if(norm(val, output) < min):
			saved = i
			min = norm(val, output)
	return saved

# Tests on the trained weights
def test(hidden, output, tests, map_output):
	mat = np.array([['a'] + ['b'] for line in tests])
	count = -1
	for test in tests:
		count = count + 1
		test_input = np.matrix(test[0].astype(np.float))
		hid_out = sigmoid(test_input*hidden.T)
		hid_output = np.matrix(np.append([1], hid_out))
		out = sigmoid(hid_output*output.T)
		clsy = classify(out, map_output)
		mat[count][0] = test[1]
		mat[count][1] = clsy
	return mat

# Tests on the trained weights
def test_soft(hidden, output, tests, map_output):
	mat = np.array([['a'] + ['b'] for line in tests])
	count = -1
	for test in tests:
		count = count + 1
		test_input = np.matrix(test[0].astype(np.float))
		hid_out = softplus(test_input*hidden.T)
		hid_output = np.matrix(np.append([1], hid_out))
		out = softplus(hid_output*output.T)
		clsy = classify(out, map_output)
		mat[count][0] = test[1]
		mat[count][1] = clsy
	return mat

# Get input
inputs = np.array(trainv2)
tests = np.array(testv2)

# Append 1 to input layer to give input to hidden layer
for i in range(inputs.shape[0]):
    inputs[i][0] = np.append([1], inputs[i][0])
for i in range(tests.shape[0]):
    tests[i][0] = np.append([1], tests[i][0])

# Create a mapping between different outcomes
list1 = [0,1,2,3,4,5,6,7,8,9]
list2 = [[1,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,1]]
map_output = dict( zip( list1, list2))

# Create train set and validation set
x = int(0.8*inputs.shape[0])
train_set = inputs[:x,:]
validation_set = inputs[x:,:]

print (map_output[1])

# # Define the number of hidden layer units
# # C = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
C = [100]
for i in range(len(C)):
    print ("no of hidden units = ", C[i])
    hidden = create_layer(C[i], 785)
    output = create_layer(10, C[i] + 1)
    saved_validation_acc = [0]
    saved_test_acc = [0]
    saved_train_acc = [0]
    flag_count = 0;
    for i in range (4):
		# Train model for one iteration
        hidden, output = train(train_set, hidden, output, map_output)
		# Test on validation set
        result = test(hidden, output, validation_set, map_output)
        count = 0
        correct = 0
        for line in result:
            count = count + 1 
            if line[0] == line[1]:
                correct = correct + 1
        accuracy = float(correct) / float(count)
        print ("accuracy = ", accuracy)
        saved_validation_acc.extend([accuracy])
        print (saved_validation_acc)
        if accuracy <= saved_validation_acc[i]:
            flag_count += 1
        else:
            flag_count = 0
            saved_hidden = hidden
            saved_output = output
        if flag_count > 3:
            break;
    dump = {'hidden': hidden,'ouput' : output}
    with open('final_weights.pkl','wb') as fid:
        pkl.dump(dump,fid,protocol=2)
	# Classify Test
    result = test(saved_hidden, saved_output, tests, map_output)
    result2 = result[:,1]
    result3 = [int(numeric_string) for numeric_string in result2]
    np.savetxt('result.csv',result3)   
    count = 0
    correct = 0
    for line in result:
        count = count + 1 
        if line[0] == line[1]:
            correct = correct + 1
    accuracy = float(correct) / float(count)
    print ("test set accuracy = ", accuracy)
	# Classify Training
    result = test(saved_hidden, saved_output, inputs, map_output)
    count = 0
    correct = 0
    for line in result:
        count = count + 1 
        if line[0] == line[1]:
            correct = correct + 1
    accuracy = float(correct) / float(count)
    print ("train set accuracy = ", accuracy)

# Define the number of hidden layer units
# C = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
 

