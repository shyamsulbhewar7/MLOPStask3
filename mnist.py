from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.datasets import mnist
from keras.utils import np_utils
import keras
import sys

sys.stdin=open('/input.txt','r')

# loads the MNIST dataset
(X_train, y_train), (X_test, y_test)  = mnist.load_data()

# Lets store the number of rows and columns
img_rows = X_train[0].shape[0]
img_cols = X_train[1].shape[0]

# Getting our date in the right 'shape' needed for Keras
# We need to add a 4th dimenion to our date thereby changing our
# Our original image shape of (60000,28,28) to (60000,28,28,1)
X_train = X_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

# store the shape of a single image 
input_shape = (img_rows, img_cols, 1)

# change our image type to float32 data type
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Normalize our data by changing the range from (0 to 255) to (0 to 1)
X_train /= 255
X_test /= 255

# Now we one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_test.shape[1]
num_pixels = X_train.shape[1] * X_train.shape[2]


# ### Now let's create our layers to replicate LeNet

# In[2]:


# create model
model = Sequential()

# first set of CRP (Convolution, RELU, Pooling)

convlayers = int(input())
first_layer_nfilter = int(input())
first_layer_filter_size = int(input())
first_layer_pool_size = int(input())

this_layer = 'No. of convolve layers : ' + str(convlayers)
this_layer = this_layer + '\nLayer 1'
this_layer = this_layer + '\nNo of filters : ' + str(first_layer_nfilter) + '\nFilter Size : ' + str(first_layer_filter_size) + '\nPool Size : ' + str(first_layer_pool_size)

model.add(Conv2D(first_layer_nfilter, (first_layer_filter_size, first_layer_filter_size),
                 padding = "same", 
                 input_shape = input_shape))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (first_layer_pool_size, first_layer_pool_size)))

#Subsequent CRP sets
for i in range(1,convlayers):
	nfilters = int(input())
	filter_size = int(input())
	pool_size = int(input())
	this_layer = this_layer + '\nLayer ' + str(i+1) + ': '
	this_layer = this_layer + '\nNo of filters : ' + str(nfilters) + '\nFilter Size : ' + str(filter_size) + '\nPool Size : ' + str(pool_size)
	model.add(Conv2D(nfilters, (filter_size, filter_size),padding = "same"))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size = (pool_size, pool_size)))

# Fully connected layers (w/ RELU)
model.add(Flatten())

fc_input = int(input())

this_layer = this_layer + '\nNo. of FC Layers : ' + str(fc_input+1) 

for i in range(0,fc_input):
	no_neurons = int(input())
	this_layer = this_layer + '\nNeurons in Layer ' + str(i+1) + ' : ' + str(no_neurons)
	model.add(Dense(no_neurons))
	model.add(Activation("relu"))

# Softmax (for classification)
model.add(Dense(num_classes))
model.add(Activation("softmax"))
           
this_layer = this_layer + '\nNeurons in Layer ' + str(fc_input + 1) + ' : ' + str(num_classes)

model.compile(loss = 'categorical_crossentropy',
              optimizer = keras.optimizers.Adadelta(),
              metrics = ['accuracy'])
    
print(model.summary())


# ### Now let us train LeNet on our MNIST Dataset

# In[3]:


# Training Parameters
batch_size = 128
epochs = 4

history = model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_test, y_test),
          shuffle=True)

model.save("mlops_auto_model.h5")

# Evaluate the performance of our trained model
scores = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

accuracy_file = open('/accuracy.txt','w')
accuracy_file.write(str(scores[1]))
accuracy_file.close()

display_matter = open('/display_matter.html','r+')
display_matter.read()
display_matter.write('<pre>\n...................................\n')
display_matter.write(this_layer)
display_matter.write('\nAccuracy achieved : ' + str(scores[1])+'\n</pre>')
display_matter.close()
