#import all files
import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
#from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from resnets_utils import *
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
#matplotlib inline

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

#load dataset
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))


#implement identity function that is used as one residual block
# GRADED FUNCTION: identity_block

def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 3

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X

    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    ### START CODE HERE ###

    # Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = (f,f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)

    ### END CODE HERE ###

    return X
#implement convolution block which is also a risidual block but this one change the size of input data
# GRADED FUNCTION: convolutional_block

def convolutional_block(X, f, filters, stage, block, s = 2):
    """
    Implementation of the convolutional block as defined in Figure 4

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X


    ##### MAIN PATH #####
    # First component of main path
    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    ### START CODE HERE ###

    # Second component of main path (≈3 lines)
    X = Conv2D(F2, (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(F3, (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(F3, (1, 1), strides = (s,s), name = conv_name_base + '1', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)

    ### END CODE HERE ###

    return X
#create resnet with 50 layers using the risidual blocks implemented above
# GRADED FUNCTION: ResNet50

def ResNet50(input_shape = (64, 64, 3), classes = 6):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)


    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    ### START CODE HERE ###

    # Stage 3 (≈4 lines)
    X = convolutional_block(X, f = 3, filters = [128,128,512], stage = 3, block='a', s = 2)
    X = identity_block(X, 3, [128,128,512], stage=3, block='b')
    X = identity_block(X, 3, [128,128,512], stage=3, block='c')
    X = identity_block(X, 3, [128,128,512], stage=3, block='d')

    # Stage 4 (≈6 lines)
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5 (≈3 lines)
    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling2D(pool_size=(2,2), name = 'avg_pool')(X)

    ### END CODE HERE ###

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)

    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model

#create model
model = ResNet50(input_shape = (128, 128, 3), classes = 2)
#compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#fit
model.fit(X_train, Y_train, epochs = 2, batch_size = 32)
#evaluate
preds = model.evaluate(X_test, Y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
#summary
model.summary()


"""
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
input_1 (InputLayer)             (None, 64, 64, 3)     0
____________________________________________________________________________________________________
zero_padding2d_1 (ZeroPadding2D) (None, 70, 70, 3)     0           input_1[0][0]
____________________________________________________________________________________________________
conv1 (Conv2D)                   (None, 32, 32, 64)    9472        zero_padding2d_1[0][0]
____________________________________________________________________________________________________
bn_conv1 (BatchNormalization)    (None, 32, 32, 64)    256         conv1[0][0]
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 32, 32, 64)    0           bn_conv1[0][0]
____________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)   (None, 15, 15, 64)    0           activation_4[0][0]
____________________________________________________________________________________________________
res2a_branch2a (Conv2D)          (None, 15, 15, 64)    4160        max_pooling2d_1[0][0]
____________________________________________________________________________________________________
bn2a_branch2a (BatchNormalizatio (None, 15, 15, 64)    256         res2a_branch2a[0][0]
____________________________________________________________________________________________________
activation_5 (Activation)        (None, 15, 15, 64)    0           bn2a_branch2a[0][0]
____________________________________________________________________________________________________
res2a_branch2b (Conv2D)          (None, 15, 15, 64)    36928       activation_5[0][0]
____________________________________________________________________________________________________
bn2a_branch2b (BatchNormalizatio (None, 15, 15, 64)    256         res2a_branch2b[0][0]
____________________________________________________________________________________________________
activation_6 (Activation)        (None, 15, 15, 64)    0           bn2a_branch2b[0][0]
____________________________________________________________________________________________________
res2a_branch2c (Conv2D)          (None, 15, 15, 256)   16640       activation_6[0][0]
____________________________________________________________________________________________________
res2a_branch1 (Conv2D)           (None, 15, 15, 256)   16640       max_pooling2d_1[0][0]
____________________________________________________________________________________________________
bn2a_branch2c (BatchNormalizatio (None, 15, 15, 256)   1024        res2a_branch2c[0][0]
____________________________________________________________________________________________________
bn2a_branch1 (BatchNormalization (None, 15, 15, 256)   1024        res2a_branch1[0][0]
____________________________________________________________________________________________________
add_2 (Add)                      (None, 15, 15, 256)   0           bn2a_branch2c[0][0]
                                                                   bn2a_branch1[0][0]
____________________________________________________________________________________________________
activation_7 (Activation)        (None, 15, 15, 256)   0           add_2[0][0]
____________________________________________________________________________________________________
res2b_branch2a (Conv2D)          (None, 15, 15, 64)    16448       activation_7[0][0]
____________________________________________________________________________________________________
bn2b_branch2a (BatchNormalizatio (None, 15, 15, 64)    256         res2b_branch2a[0][0]
____________________________________________________________________________________________________
activation_8 (Activation)        (None, 15, 15, 64)    0           bn2b_branch2a[0][0]
____________________________________________________________________________________________________
res2b_branch2b (Conv2D)          (None, 15, 15, 64)    36928       activation_8[0][0]
____________________________________________________________________________________________________
bn2b_branch2b (BatchNormalizatio (None, 15, 15, 64)    256         res2b_branch2b[0][0]
____________________________________________________________________________________________________
activation_9 (Activation)        (None, 15, 15, 64)    0           bn2b_branch2b[0][0]
____________________________________________________________________________________________________
res2b_branch2c (Conv2D)          (None, 15, 15, 256)   16640       activation_9[0][0]
____________________________________________________________________________________________________
bn2b_branch2c (BatchNormalizatio (None, 15, 15, 256)   1024        res2b_branch2c[0][0]
____________________________________________________________________________________________________
add_3 (Add)                      (None, 15, 15, 256)   0           bn2b_branch2c[0][0]
                                                                   activation_7[0][0]
____________________________________________________________________________________________________
activation_10 (Activation)       (None, 15, 15, 256)   0           add_3[0][0]
____________________________________________________________________________________________________
res2c_branch2a (Conv2D)          (None, 15, 15, 64)    16448       activation_10[0][0]
____________________________________________________________________________________________________
bn2c_branch2a (BatchNormalizatio (None, 15, 15, 64)    256         res2c_branch2a[0][0]
____________________________________________________________________________________________________
activation_11 (Activation)       (None, 15, 15, 64)    0           bn2c_branch2a[0][0]
____________________________________________________________________________________________________
res2c_branch2b (Conv2D)          (None, 15, 15, 64)    36928       activation_11[0][0]
____________________________________________________________________________________________________
bn2c_branch2b (BatchNormalizatio (None, 15, 15, 64)    256         res2c_branch2b[0][0]
____________________________________________________________________________________________________
activation_12 (Activation)       (None, 15, 15, 64)    0           bn2c_branch2b[0][0]
____________________________________________________________________________________________________
res2c_branch2c (Conv2D)          (None, 15, 15, 256)   16640       activation_12[0][0]
____________________________________________________________________________________________________
bn2c_branch2c (BatchNormalizatio (None, 15, 15, 256)   1024        res2c_branch2c[0][0]
____________________________________________________________________________________________________
add_4 (Add)                      (None, 15, 15, 256)   0           bn2c_branch2c[0][0]
                                                                   activation_10[0][0]
____________________________________________________________________________________________________
activation_13 (Activation)       (None, 15, 15, 256)   0           add_4[0][0]
____________________________________________________________________________________________________
res3a_branch2a (Conv2D)          (None, 8, 8, 128)     32896       activation_13[0][0]
____________________________________________________________________________________________________
bn3a_branch2a (BatchNormalizatio (None, 8, 8, 128)     512         res3a_branch2a[0][0]
____________________________________________________________________________________________________
activation_14 (Activation)       (None, 8, 8, 128)     0           bn3a_branch2a[0][0]
____________________________________________________________________________________________________
res3a_branch2b (Conv2D)          (None, 8, 8, 128)     147584      activation_14[0][0]
____________________________________________________________________________________________________
bn3a_branch2b (BatchNormalizatio (None, 8, 8, 128)     512         res3a_branch2b[0][0]
____________________________________________________________________________________________________
activation_15 (Activation)       (None, 8, 8, 128)     0           bn3a_branch2b[0][0]
____________________________________________________________________________________________________
res3a_branch2c (Conv2D)          (None, 8, 8, 512)     66048       activation_15[0][0]
____________________________________________________________________________________________________
res3a_branch1 (Conv2D)           (None, 8, 8, 512)     131584      activation_13[0][0]
____________________________________________________________________________________________________
bn3a_branch2c (BatchNormalizatio (None, 8, 8, 512)     2048        res3a_branch2c[0][0]
____________________________________________________________________________________________________
bn3a_branch1 (BatchNormalization (None, 8, 8, 512)     2048        res3a_branch1[0][0]
____________________________________________________________________________________________________
add_5 (Add)                      (None, 8, 8, 512)     0           bn3a_branch2c[0][0]
                                                                   bn3a_branch1[0][0]
____________________________________________________________________________________________________
activation_16 (Activation)       (None, 8, 8, 512)     0           add_5[0][0]
____________________________________________________________________________________________________
res3b_branch2a (Conv2D)          (None, 8, 8, 128)     65664       activation_16[0][0]
____________________________________________________________________________________________________
bn3b_branch2a (BatchNormalizatio (None, 8, 8, 128)     512         res3b_branch2a[0][0]
____________________________________________________________________________________________________
activation_17 (Activation)       (None, 8, 8, 128)     0           bn3b_branch2a[0][0]
____________________________________________________________________________________________________
res3b_branch2b (Conv2D)          (None, 8, 8, 128)     147584      activation_17[0][0]
____________________________________________________________________________________________________
bn3b_branch2b (BatchNormalizatio (None, 8, 8, 128)     512         res3b_branch2b[0][0]
____________________________________________________________________________________________________
activation_18 (Activation)       (None, 8, 8, 128)     0           bn3b_branch2b[0][0]
____________________________________________________________________________________________________
res3b_branch2c (Conv2D)          (None, 8, 8, 512)     66048       activation_18[0][0]
____________________________________________________________________________________________________
bn3b_branch2c (BatchNormalizatio (None, 8, 8, 512)     2048        res3b_branch2c[0][0]
____________________________________________________________________________________________________
add_6 (Add)                      (None, 8, 8, 512)     0           bn3b_branch2c[0][0]
                                                                   activation_16[0][0]
____________________________________________________________________________________________________
activation_19 (Activation)       (None, 8, 8, 512)     0           add_6[0][0]
____________________________________________________________________________________________________
res3c_branch2a (Conv2D)          (None, 8, 8, 128)     65664       activation_19[0][0]
____________________________________________________________________________________________________
bn3c_branch2a (BatchNormalizatio (None, 8, 8, 128)     512         res3c_branch2a[0][0]
____________________________________________________________________________________________________
activation_20 (Activation)       (None, 8, 8, 128)     0           bn3c_branch2a[0][0]
____________________________________________________________________________________________________
res3c_branch2b (Conv2D)          (None, 8, 8, 128)     147584      activation_20[0][0]
____________________________________________________________________________________________________
bn3c_branch2b (BatchNormalizatio (None, 8, 8, 128)     512         res3c_branch2b[0][0]
____________________________________________________________________________________________________
activation_21 (Activation)       (None, 8, 8, 128)     0           bn3c_branch2b[0][0]
____________________________________________________________________________________________________
res3c_branch2c (Conv2D)          (None, 8, 8, 512)     66048       activation_21[0][0]
____________________________________________________________________________________________________
bn3c_branch2c (BatchNormalizatio (None, 8, 8, 512)     2048        res3c_branch2c[0][0]
____________________________________________________________________________________________________
add_7 (Add)                      (None, 8, 8, 512)     0           bn3c_branch2c[0][0]
                                                                   activation_19[0][0]
____________________________________________________________________________________________________
activation_22 (Activation)       (None, 8, 8, 512)     0           add_7[0][0]
____________________________________________________________________________________________________
res3d_branch2a (Conv2D)          (None, 8, 8, 128)     65664       activation_22[0][0]
____________________________________________________________________________________________________
bn3d_branch2a (BatchNormalizatio (None, 8, 8, 128)     512         res3d_branch2a[0][0]
____________________________________________________________________________________________________
activation_23 (Activation)       (None, 8, 8, 128)     0           bn3d_branch2a[0][0]
____________________________________________________________________________________________________
res3d_branch2b (Conv2D)          (None, 8, 8, 128)     147584      activation_23[0][0]
____________________________________________________________________________________________________
bn3d_branch2b (BatchNormalizatio (None, 8, 8, 128)     512         res3d_branch2b[0][0]
____________________________________________________________________________________________________
activation_24 (Activation)       (None, 8, 8, 128)     0           bn3d_branch2b[0][0]
____________________________________________________________________________________________________
res3d_branch2c (Conv2D)          (None, 8, 8, 512)     66048       activation_24[0][0]
____________________________________________________________________________________________________
bn3d_branch2c (BatchNormalizatio (None, 8, 8, 512)     2048        res3d_branch2c[0][0]
____________________________________________________________________________________________________
add_8 (Add)                      (None, 8, 8, 512)     0           bn3d_branch2c[0][0]
                                                                   activation_22[0][0]
____________________________________________________________________________________________________
activation_25 (Activation)       (None, 8, 8, 512)     0           add_8[0][0]
____________________________________________________________________________________________________
res4a_branch2a (Conv2D)          (None, 4, 4, 256)     131328      activation_25[0][0]
____________________________________________________________________________________________________
bn4a_branch2a (BatchNormalizatio (None, 4, 4, 256)     1024        res4a_branch2a[0][0]
____________________________________________________________________________________________________
activation_26 (Activation)       (None, 4, 4, 256)     0           bn4a_branch2a[0][0]
____________________________________________________________________________________________________
res4a_branch2b (Conv2D)          (None, 4, 4, 256)     590080      activation_26[0][0]
____________________________________________________________________________________________________
bn4a_branch2b (BatchNormalizatio (None, 4, 4, 256)     1024        res4a_branch2b[0][0]
____________________________________________________________________________________________________
activation_27 (Activation)       (None, 4, 4, 256)     0           bn4a_branch2b[0][0]
____________________________________________________________________________________________________
res4a_branch2c (Conv2D)          (None, 4, 4, 1024)    263168      activation_27[0][0]
____________________________________________________________________________________________________
res4a_branch1 (Conv2D)           (None, 4, 4, 1024)    525312      activation_25[0][0]
____________________________________________________________________________________________________
bn4a_branch2c (BatchNormalizatio (None, 4, 4, 1024)    4096        res4a_branch2c[0][0]
____________________________________________________________________________________________________
bn4a_branch1 (BatchNormalization (None, 4, 4, 1024)    4096        res4a_branch1[0][0]
____________________________________________________________________________________________________
add_9 (Add)                      (None, 4, 4, 1024)    0           bn4a_branch2c[0][0]
                                                                   bn4a_branch1[0][0]
____________________________________________________________________________________________________
activation_28 (Activation)       (None, 4, 4, 1024)    0           add_9[0][0]
____________________________________________________________________________________________________
res4b_branch2a (Conv2D)          (None, 4, 4, 256)     262400      activation_28[0][0]
____________________________________________________________________________________________________
bn4b_branch2a (BatchNormalizatio (None, 4, 4, 256)     1024        res4b_branch2a[0][0]
____________________________________________________________________________________________________
activation_29 (Activation)       (None, 4, 4, 256)     0           bn4b_branch2a[0][0]
____________________________________________________________________________________________________
res4b_branch2b (Conv2D)          (None, 4, 4, 256)     590080      activation_29[0][0]
____________________________________________________________________________________________________
bn4b_branch2b (BatchNormalizatio (None, 4, 4, 256)     1024        res4b_branch2b[0][0]
____________________________________________________________________________________________________
activation_30 (Activation)       (None, 4, 4, 256)     0           bn4b_branch2b[0][0]
____________________________________________________________________________________________________
res4b_branch2c (Conv2D)          (None, 4, 4, 1024)    263168      activation_30[0][0]
____________________________________________________________________________________________________
bn4b_branch2c (BatchNormalizatio (None, 4, 4, 1024)    4096        res4b_branch2c[0][0]
____________________________________________________________________________________________________
add_10 (Add)                     (None, 4, 4, 1024)    0           bn4b_branch2c[0][0]
                                                                   activation_28[0][0]
____________________________________________________________________________________________________
activation_31 (Activation)       (None, 4, 4, 1024)    0           add_10[0][0]
____________________________________________________________________________________________________
res4c_branch2a (Conv2D)          (None, 4, 4, 256)     262400      activation_31[0][0]
____________________________________________________________________________________________________
bn4c_branch2a (BatchNormalizatio (None, 4, 4, 256)     1024        res4c_branch2a[0][0]
____________________________________________________________________________________________________
activation_32 (Activation)       (None, 4, 4, 256)     0           bn4c_branch2a[0][0]
____________________________________________________________________________________________________
res4c_branch2b (Conv2D)          (None, 4, 4, 256)     590080      activation_32[0][0]
____________________________________________________________________________________________________
bn4c_branch2b (BatchNormalizatio (None, 4, 4, 256)     1024        res4c_branch2b[0][0]
____________________________________________________________________________________________________
activation_33 (Activation)       (None, 4, 4, 256)     0           bn4c_branch2b[0][0]
____________________________________________________________________________________________________
res4c_branch2c (Conv2D)          (None, 4, 4, 1024)    263168      activation_33[0][0]
____________________________________________________________________________________________________
bn4c_branch2c (BatchNormalizatio (None, 4, 4, 1024)    4096        res4c_branch2c[0][0]
____________________________________________________________________________________________________
add_11 (Add)                     (None, 4, 4, 1024)    0           bn4c_branch2c[0][0]
                                                                   activation_31[0][0]
____________________________________________________________________________________________________
activation_34 (Activation)       (None, 4, 4, 1024)    0           add_11[0][0]
____________________________________________________________________________________________________
res4d_branch2a (Conv2D)          (None, 4, 4, 256)     262400      activation_34[0][0]
____________________________________________________________________________________________________
bn4d_branch2a (BatchNormalizatio (None, 4, 4, 256)     1024        res4d_branch2a[0][0]
____________________________________________________________________________________________________
activation_35 (Activation)       (None, 4, 4, 256)     0           bn4d_branch2a[0][0]
____________________________________________________________________________________________________
res4d_branch2b (Conv2D)          (None, 4, 4, 256)     590080      activation_35[0][0]
____________________________________________________________________________________________________
bn4d_branch2b (BatchNormalizatio (None, 4, 4, 256)     1024        res4d_branch2b[0][0]
____________________________________________________________________________________________________
activation_36 (Activation)       (None, 4, 4, 256)     0           bn4d_branch2b[0][0]
____________________________________________________________________________________________________
res4d_branch2c (Conv2D)          (None, 4, 4, 1024)    263168      activation_36[0][0]
____________________________________________________________________________________________________
bn4d_branch2c (BatchNormalizatio (None, 4, 4, 1024)    4096        res4d_branch2c[0][0]
____________________________________________________________________________________________________
add_12 (Add)                     (None, 4, 4, 1024)    0           bn4d_branch2c[0][0]
                                                                   activation_34[0][0]
____________________________________________________________________________________________________
activation_37 (Activation)       (None, 4, 4, 1024)    0           add_12[0][0]
____________________________________________________________________________________________________
res4e_branch2a (Conv2D)          (None, 4, 4, 256)     262400      activation_37[0][0]
____________________________________________________________________________________________________
bn4e_branch2a (BatchNormalizatio (None, 4, 4, 256)     1024        res4e_branch2a[0][0]
____________________________________________________________________________________________________
activation_38 (Activation)       (None, 4, 4, 256)     0           bn4e_branch2a[0][0]
____________________________________________________________________________________________________
res4e_branch2b (Conv2D)          (None, 4, 4, 256)     590080      activation_38[0][0]
____________________________________________________________________________________________________
bn4e_branch2b (BatchNormalizatio (None, 4, 4, 256)     1024        res4e_branch2b[0][0]
____________________________________________________________________________________________________
activation_39 (Activation)       (None, 4, 4, 256)     0           bn4e_branch2b[0][0]
____________________________________________________________________________________________________
res4e_branch2c (Conv2D)          (None, 4, 4, 1024)    263168      activation_39[0][0]
____________________________________________________________________________________________________
bn4e_branch2c (BatchNormalizatio (None, 4, 4, 1024)    4096        res4e_branch2c[0][0]
____________________________________________________________________________________________________
add_13 (Add)                     (None, 4, 4, 1024)    0           bn4e_branch2c[0][0]
                                                                   activation_37[0][0]
____________________________________________________________________________________________________
activation_40 (Activation)       (None, 4, 4, 1024)    0           add_13[0][0]
____________________________________________________________________________________________________
res4f_branch2a (Conv2D)          (None, 4, 4, 256)     262400      activation_40[0][0]
____________________________________________________________________________________________________
bn4f_branch2a (BatchNormalizatio (None, 4, 4, 256)     1024        res4f_branch2a[0][0]
____________________________________________________________________________________________________
activation_41 (Activation)       (None, 4, 4, 256)     0           bn4f_branch2a[0][0]
____________________________________________________________________________________________________
res4f_branch2b (Conv2D)          (None, 4, 4, 256)     590080      activation_41[0][0]
____________________________________________________________________________________________________
bn4f_branch2b (BatchNormalizatio (None, 4, 4, 256)     1024        res4f_branch2b[0][0]
____________________________________________________________________________________________________
activation_42 (Activation)       (None, 4, 4, 256)     0           bn4f_branch2b[0][0]
____________________________________________________________________________________________________
res4f_branch2c (Conv2D)          (None, 4, 4, 1024)    263168      activation_42[0][0]
____________________________________________________________________________________________________
bn4f_branch2c (BatchNormalizatio (None, 4, 4, 1024)    4096        res4f_branch2c[0][0]
____________________________________________________________________________________________________
add_14 (Add)                     (None, 4, 4, 1024)    0           bn4f_branch2c[0][0]
                                                                   activation_40[0][0]
____________________________________________________________________________________________________
activation_43 (Activation)       (None, 4, 4, 1024)    0           add_14[0][0]
____________________________________________________________________________________________________
res5a_branch2a (Conv2D)          (None, 2, 2, 512)     524800      activation_43[0][0]
____________________________________________________________________________________________________
bn5a_branch2a (BatchNormalizatio (None, 2, 2, 512)     2048        res5a_branch2a[0][0]
____________________________________________________________________________________________________
activation_44 (Activation)       (None, 2, 2, 512)     0           bn5a_branch2a[0][0]
____________________________________________________________________________________________________
res5a_branch2b (Conv2D)          (None, 2, 2, 512)     2359808     activation_44[0][0]
____________________________________________________________________________________________________
bn5a_branch2b (BatchNormalizatio (None, 2, 2, 512)     2048        res5a_branch2b[0][0]
____________________________________________________________________________________________________
activation_45 (Activation)       (None, 2, 2, 512)     0           bn5a_branch2b[0][0]
____________________________________________________________________________________________________
res5a_branch2c (Conv2D)          (None, 2, 2, 2048)    1050624     activation_45[0][0]
____________________________________________________________________________________________________
res5a_branch1 (Conv2D)           (None, 2, 2, 2048)    2099200     activation_43[0][0]
____________________________________________________________________________________________________
bn5a_branch2c (BatchNormalizatio (None, 2, 2, 2048)    8192        res5a_branch2c[0][0]
____________________________________________________________________________________________________
bn5a_branch1 (BatchNormalization (None, 2, 2, 2048)    8192        res5a_branch1[0][0]
____________________________________________________________________________________________________
add_15 (Add)                     (None, 2, 2, 2048)    0           bn5a_branch2c[0][0]
                                                                   bn5a_branch1[0][0]
____________________________________________________________________________________________________
activation_46 (Activation)       (None, 2, 2, 2048)    0           add_15[0][0]
____________________________________________________________________________________________________
res5b_branch2a (Conv2D)          (None, 2, 2, 512)     1049088     activation_46[0][0]
____________________________________________________________________________________________________
bn5b_branch2a (BatchNormalizatio (None, 2, 2, 512)     2048        res5b_branch2a[0][0]
____________________________________________________________________________________________________
activation_47 (Activation)       (None, 2, 2, 512)     0           bn5b_branch2a[0][0]
____________________________________________________________________________________________________
res5b_branch2b (Conv2D)          (None, 2, 2, 512)     2359808     activation_47[0][0]
____________________________________________________________________________________________________
bn5b_branch2b (BatchNormalizatio (None, 2, 2, 512)     2048        res5b_branch2b[0][0]
____________________________________________________________________________________________________
activation_48 (Activation)       (None, 2, 2, 512)     0           bn5b_branch2b[0][0]
____________________________________________________________________________________________________
res5b_branch2c (Conv2D)          (None, 2, 2, 2048)    1050624     activation_48[0][0]
____________________________________________________________________________________________________
bn5b_branch2c (BatchNormalizatio (None, 2, 2, 2048)    8192        res5b_branch2c[0][0]
____________________________________________________________________________________________________
add_16 (Add)                     (None, 2, 2, 2048)    0           bn5b_branch2c[0][0]
                                                                   activation_46[0][0]
____________________________________________________________________________________________________
activation_49 (Activation)       (None, 2, 2, 2048)    0           add_16[0][0]
____________________________________________________________________________________________________
res5c_branch2a (Conv2D)          (None, 2, 2, 512)     1049088     activation_49[0][0]
____________________________________________________________________________________________________
bn5c_branch2a (BatchNormalizatio (None, 2, 2, 512)     2048        res5c_branch2a[0][0]
____________________________________________________________________________________________________
activation_50 (Activation)       (None, 2, 2, 512)     0           bn5c_branch2a[0][0]
____________________________________________________________________________________________________
res5c_branch2b (Conv2D)          (None, 2, 2, 512)     2359808     activation_50[0][0]
____________________________________________________________________________________________________
bn5c_branch2b (BatchNormalizatio (None, 2, 2, 512)     2048        res5c_branch2b[0][0]
____________________________________________________________________________________________________
activation_51 (Activation)       (None, 2, 2, 512)     0           bn5c_branch2b[0][0]
____________________________________________________________________________________________________
res5c_branch2c (Conv2D)          (None, 2, 2, 2048)    1050624     activation_51[0][0]
____________________________________________________________________________________________________
bn5c_branch2c (BatchNormalizatio (None, 2, 2, 2048)    8192        res5c_branch2c[0][0]
____________________________________________________________________________________________________
add_17 (Add)                     (None, 2, 2, 2048)    0           bn5c_branch2c[0][0]
                                                                   activation_49[0][0]
____________________________________________________________________________________________________
activation_52 (Activation)       (None, 2, 2, 2048)    0           add_17[0][0]
____________________________________________________________________________________________________
avg_pool (AveragePooling2D)      (None, 1, 1, 2048)    0           activation_52[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 2048)          0           avg_pool[0][0]
____________________________________________________________________________________________________
fc6 (Dense)                      (None, 6)             12294       flatten_1[0][0]
====================================================================================================
Total params: 23,600,006
Trainable params: 23,546,886
Non-trainable params: 53,120
"""
