import numpy as np
from tensorflow.python.keras.utils import np_utils as nputil
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.datasets import mnist

num_classes = 10
# 读取mnist训练数据
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshape数据
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype(np.float32) / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype(np.float32) / 255

y_train = nputil.to_categorical(y_train)
y_test = nputil.to_categorical(y_test)
# 创建模型
model = Sequential()
# 声明模型结构
model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(128, (1, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
# 编译模型，定义损失函数
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 开始训练
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=200, verbose=2)

# 丢弃dropout层
for k in model.layers:
    if type(k) is Dropout:
        model.layers.remove(k)

# Print model summary
print(model.summary())

# 保存模型
model.save('mnistCNN.h5')
