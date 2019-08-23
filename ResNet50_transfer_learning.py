
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 00:30:46 2019

@author: sachin
"""

from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import load_model
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm
from IPython.display import SVG
from keras.utils.vis_utils import plot_model, model_to_dot
from keras.models import load_model
from keras.models import model_from_json
from keras.regularizers import l2

%matplotlib inline

classes = 120
image_size = (224, 224)
resNet_pooling = 'avg'
layer_activation = 'relu'
final_layer_activation = 'softmax'
loss = 'categorical_crossentropy'
epochs = 20                         # 20 to 25 is fine.
early_patience = 4
num_images = 20580
batch_size = 128

resNet_weights = 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


model = Sequential()
model.add(ResNet50(include_top = False, weights = resNet_weights,
                   pooling = resNet_pooling))


"""
model.add(Dense(units = 1000, kernel_initializer = 'he_normal',
               use_bias = False))
model.add(BatchNormalization())
model.add(Activation(layer_activation))
model.add(Dropout(rate = 0.2))
"""
model.add(Dense(units = 1000, kernel_initializer = 'he_normal',
               use_bias = False))
model.add(BatchNormalization())
model.add(Activation(layer_activation))
model.add(Dropout(rate = 0.3))

"""
model.add(Dense(units = 150, kernel_initializer = 'he_normal',
               use_bias = False))
model.add(BatchNormalization())
model.add(Activation(layer_activation))
model.add(Dropout(rate = 0.2))
"""

 
model.add(Dense(units = classes, activation = final_layer_activation,
                kernel_initializer = 'he_normal', kernel_regularizer = l2(0.01)))

model.layers[0].trainable = False

adam = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999)
model.compile(optimizer = adam, loss = loss, metrics = ['accuracy'])

data_gen = ImageDataGenerator(preprocessing_function = preprocess_input,
                              validation_split = 0.2)

data_path = '/home/sachin/Desktop/projects/stanford-dogs-dataset/Images'

training_set = data_gen.flow_from_directory(directory = data_path,
                                            target_size = image_size,
                                            class_mode = 'categorical',
                                            batch_size = batch_size,
                                            shuffle = True,
                                            subset = 'training',
                                            seed = 42)

validation_set = data_gen.flow_from_directory(directory = data_path,
                                              target_size = image_size,
                                              class_mode = 'categorical',
                                              batch_size = batch_size,
                                              shuffle = True,
                                              subset = 'validation',
                                              seed = 42)

early_stop = EarlyStopping(monitor = 'val_acc',
                           patience = early_patience,
                           mode = 'max')

file_p = '/home/sachin/Desktop/projects/stanford-dogs-dataset/best_final.hdf5'
model_check = ModelCheckpoint(filepath = file_p,
                              monitor = 'val_acc',
                              save_best_only = True,
                              mode = 'max')

model_history = model.fit_generator(generator = training_set,
                                    steps_per_epoch = training_set.samples // batch_size,
                                    epochs = epochs,
                                    validation_data = validation_set,
                                    validation_steps = validation_set.samples // batch_size,
                                    callbacks = [early_stop, model_check])


json_string = model.to_json()
model = model_from_json(json_string)

# Loading the weights.
model.load_weights('best_final.hdf5')

print(model.history.keys())

# Plotting the model.
plt.figure(1, figsize = (15,8))
plt.subplot(1, 2, 1)
plt.plot(model_history.history['acc'])
plt.plot(model_history.history['val_acc'])
plt.title('Model_accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['train','validation'])

plt.subplot(1, 2, 2)
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('Model_loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train','validation'])

plt.show()

# Visualising the Model.
plot_model(model.layers[0], to_file='model_block1.png')
plot_model(model, to_file='model_block.png')
SVG(model_to_dot(model.layers[0]).create(prog='dot', format='svg'))
SVG(model_to_dot(model).create(prog='dot', format='svg'))
model.layers[0].summary()
model.summary()

# Predictions on random pictures.

c = training_set.class_indices

p_path = 'prediction_img/10.jpg'
test_image = image.load_img(p_path, target_size = (224, 224))
plt.imshow(test_image)
test_image = image.img_to_array(test_image)
test_image = preprocess_input(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict_classes(test_image)
for key, val in c.items():
    if np.squeeze(result) == val:
        print("\t",key.split('-')[1])
        break







 
