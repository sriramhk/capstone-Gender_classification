from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout
from keras.preprocessing.image import ImageDataGenerator
#from keras.initializers import he_uniform 
#from keras.utils.vis_utils import plot_model
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers import Input
 
# function for creating a projected inception module
def inception_module(layer_in, f1, f2_in, f2_out, f3_in, f3_out, f4_in, f4_out, f5_out):
	# 1x1 conv
	conv1 = Conv2D(f1, (1,1), padding='same', activation='relu')(layer_in)
	# 3x3 conv
	conv3 = Conv2D(f2_in, (1,1), padding='same', activation='relu')(layer_in)
	conv3 = Conv2D(f2_out, (3,3), padding='same', activation='relu', kernel_initializer='he_uniform')(conv3)
	# 5x5 conv
	conv5 = Conv2D(f3_in, (1,1), padding='same', activation='relu')(layer_in)
	conv5 = Conv2D(f3_out, (5,5), padding='same', activation='relu', kernel_initializer='he_uniform')(conv5)
    # 7x7 conv
	conv7 = Conv2D(f4_in, (1,1), padding='same', activation='relu')(layer_in)
	conv7 = Conv2D(f4_out, (7,7), padding='same', activation='relu', kernel_initializer='he_uniform')(conv7)
	# 3x3 max pooling
	pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(layer_in)
	pool = Conv2D(f5_out, (1,1), padding='same', activation='relu', kernel_initializer='he_uniform')(pool)
	# concatenate filters, assumes filters/channels last
	layer_out = concatenate([conv1, conv3, conv5, conv7, pool], axis=-1)
	return layer_out
	
# define model input
input_layer = Input(shape=(256, 256, 3))
# add inception block 1
layer = inception_module(input_layer, 64, 96, 128, 16, 32, 16, 32, 32)
#normalization
layer = BatchNormalization(axis=-1)(layer)
# Dropout
layer = Dropout(0.5)(layer)
#flatten
flat = Flatten()(layer)
#dense layers
dense_1 = Dense(units = 1024, activation='relu', kernel_initializer='he_uniform')(flat)
dense_2 = Dense(units = 128, activation='relu', kernel_initializer='he_uniform')(dense_1)
# last layer
output = Dense(1, activation='softmax')(dense_2)

# NN model
model = Model(inputs=input_layer, outputs=output)
model.summary()
#plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)


# Compiling the CNN
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Dataset TODO
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

train_set = train_datagen.flow_from_directory('dataset/train_set',
                                                 target_size = (256, 256),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('datset/test_set',
                                            target_size = (256, 256),
                                            batch_size = 32,
                                            class_mode = 'binary')


## Actual classifier 
model.fit_generator(train_set,
                         epochs = 50,
                         validation_data = test_set)#no of test images) ##to be calculated##

model.save_weights("model.h5")
print("Saved model to disk")
