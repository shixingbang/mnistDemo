
from tensorflow.python.keras.layers import Add, Conv2D, Input, Lambda
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.datasets import mnist

def MNIST():
    x_in = Input(shape=(None, None, 3))
    x = Conv2D(filters=32, kernel_size=5, padding='same', activation='relu')(x_in)