from matplotlib.image import imread
import matplotlib.pyplot as plt
import cv2
from keras.models import model_from_json
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from scipy import misc

# read the downloaded signs and resize to 32x32x3
signs = []
fig, ax = plt.subplots(2, 5)
fig.set_size_inches(2, 5)

for i in range(5):
    signs.append(imread("../traffic-signs-data/sign"+str(i+1)+".jpg"))

for i in range(5):
    ax[0][i].imshow(signs[i])

rsigns = []
X_test = np.array(32, 32, 3)

for i in range(5):
    rsigns.append(cv2.resize(signs[i], (32, 32)))
    X_test[i] = misc.imread(rsigns[i])  # 32x32x3 array

for i in range(5):
    ax[1][i].imshow(rsigns[i])
plt.show()



y_test = [3, 2, 9, 14, 12]
label_binarizer = LabelBinarizer()
y_one_hot_test = label_binarizer.fit_transform(y_test)

X_test_norm = (X_test / 255.) - 0.5

print("Testing on 5 images")
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