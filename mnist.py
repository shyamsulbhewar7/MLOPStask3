from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Sequential
from keras import metrics
from keras.layers import Softmax
from keras.optimizers import Adam
import random


def conv1():
    return (Convolution2D(filters=random.randint(30,50), 
                        kernel_size=(3,3), 
                        activation='relu',
                   input_shape=(128, 128, 3)
                       ))

def conv2():
    return (Convolution2D(filters=random.randint(30,50), 
                        kernel_size=random.choice(((3,3),(4,4),(5,5))), 
                        activation='relu',) )

def max1():
    return (MaxPooling2D(pool_size=(2, 2)))

def dense1():
    return (Dense(units=128, activation='relu'))

def dense2():
    return (Dense(units=64, activation='relu'))

def dense3():
    return (Dense(units=32, activation='relu'))

def dense4():
    return (Dense(units=1, activation='relu'))

model=Sequential()

model.add(conv1())
model.add(max1())
X = random.randint(1,5)
if X==1:
    model.add(conv2())
    model.add(max1())
elif X==2:
    model.add(conv2())
    model.add(max1())
    model.add(Flatten())
    model.add(dense1())
elif X==3:
    model.add(conv2())
    model.add(max1())
    model.add(Flatten())
    model.add(dense1())
    model.add(dense3())
elif X==4:
    model.add(conv2())
    model.add(max1())
    model.add(conv2())
    model.add(max1())
    model.add(Flatten())
    model.add(dense1())
    model.add(dense3())
else:
    model.add(conv2())
    model.add(max1())
    model.add(conv2())
    model.add(max1())
    model.add(Flatten())
    model.add(dense1())
    model.add(dense2())
    model.add(dense3())
model.add(dense4())
model.add(Softmax())


print(model.summary())

model.compile(optimizer=Adam(learning_rate=0.0002), loss='binary_crossentropy', metrics=['accuracy'])


from keras_preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        'cnn_dataset/training_set/',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary')
test_set = test_datagen.flow_from_directory(
        'cnn_dataset/test_set/',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary')


# In[10]:


history=model.fit(
        training_set,
        steps_per_epoch=200,
        epochs=10,
        validation_data=test_set,
        validation_steps=80,)



print(max(history.history['accuracy']))
if (max(history.history['accuracy'])) > .80 :
    model.save('model.h5')


fh = open('/root/accuracy.txt','w+')
fh.write (str(history.history['accuracy']))
fh.close()
