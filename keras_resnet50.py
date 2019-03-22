from keras.engine.training import Model
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
import numpy as np
import os
import sys
import tensorflow as tf

# Data preprocessing
X = []
Y = []

i = 0

for root, dirs, files in os.walk("./Images/"):

	for filename in files:

		img = image.load_img(os.path.join(root, filename), target_size=(224, 224))
		img = image.img_to_array(img)
		#print(img.shape)
		#img = np.expand_dims(img, axis=0)
		#print(img.shape)
		#img = preprocess_input(img)
		#print(img.shape)
		
		X.append(img)
		Y.append(i)

	i += 1


X = np.asarray(X)
Y = np_utils.to_categorical(np.asarray(Y), i)

mask = np.random.rand(len(X)) < 0.9

X_trn, X_tst = X[mask], X[~mask]
Y_trn, Y_tst = Y[mask], Y[~mask]


# ResNet 50 model
model = tf.keras.applications.MobileNetV2(weights=None, classes=i)

model = tf.contrib.tpu.keras_to_tpu_model(model,
	strategy=tf.contrib.tpu.TPUDistributionStrategy(
		tf.contrib.cluster_resolver.TPUClusterResolver(tpu='grpc://10.240.1.2:8470')
    )
)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

early_stopping = EarlyStopping(monitor="val_loss", patience=3)
model.fit(X_trn, Y_trn, batch_size=32, epochs=500, verbose=2, validation_data=(X_tst, Y_tst))


# Results
loss, acc = model.evaluate(X_tst, Y_tst, verbose=0)
print("Test score: {:.2f} accuracy: {:.2f}%".format(loss, acc*100))
