import pickle
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
from keras import initializers
import helper

tf.python.control_flow_ops = tf

# Load pickled data
training_file = 'train.p'
validation_file = 'valid.p'
testing_file = 'test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

# pre-process data
X_train_norm,  X_valid_norm, X_test_norm = helper.normalize(X_train, X_valid, X_test)

label_binarizer = LabelBinarizer()
y_one_hot_train = label_binarizer.fit_transform(y_train)
y_one_hot_valid = label_binarizer.fit_transform(y_valid)
y_one_hot_test = label_binarizer.fit_transform(y_test)

# fix the seed for reducing as much as possible variability in the results
seed = 10
np.random.seed(seed)

# Build Deep Network Model in Keras
def create_model():
    # create model
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(32, 32, 3))) # by default it initializes the weights to Xavier
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Conv2D(100, (1, 1)))  # by default it initializes the weights to Xavier
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(1024, kernel_initializer=initializers.random_normal(stddev=0.01)))  # 128
    model.add(Activation('relu'))
    model.add(Dense(256, kernel_initializer=initializers.random_normal(stddev=0.01))) #128
    model.add(Activation('relu'))
    model.add(Dense(43))
    model.add(Activation('softmax'))

    # Compile model
    model.compile('adam', 'categorical_crossentropy', ['accuracy'])
    return model

model = create_model()

# serialize model to JSON
model_json = model.to_json()
with open("keras-traffic-sign-model.json", "w") as json_file:
    json_file.write(model_json)

# checkpoint
filepath = "weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Fit the model
history = model.fit(X_train_norm, y_one_hot_train, batch_size=128, epochs=50,
                    validation_data=(X_valid_norm, y_one_hot_valid), callbacks=callbacks_list, verbose=0)

# serialize weights to HDF5
model.save_weights("keras-traffic-sign-weights.h5")
print("Saved model to disk")

print("Testing")
# load json and create model
json_file = open('keras-traffic-sign-model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("keras-traffic-sign-weights.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
score = loaded_model.evaluate(X_test_norm, y_one_hot_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))
