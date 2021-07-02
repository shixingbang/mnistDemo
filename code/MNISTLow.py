import numpy as np
from tensorflow.python.keras.utils import np_utils as nputil
from tensorflow.python.keras.layers import Add, Conv2D, Input, Lambda, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.datasets import mnist

num_classes = 10


(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype(np.float32) / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype(np.float32) / 255

y_train = nputil.to_categorical(y_train)
y_test = nputil.to_categorical(y_test)

model = Sequential()

model.add(Conv2D(1, (28, 28), input_shape=(28, 28, 1), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.5))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))
# model.add(Conv2D(128, (1, 1), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))
model.add(Flatten())
# model.add(Dense(784, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
print("test")
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
# Training
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=200, verbose=2)

# Prepare model for inference
for k in model.layers:
    if type(k) is Dropout:
        model.layers.remove(k)

# Print model summary
print(model.summary())

# Save the model
model.save('mnistLow.h5')
