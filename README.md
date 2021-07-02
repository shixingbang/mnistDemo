# mnistDemo


运行demo：
  1. pod install

训练模型：
  1. 执行 MNIST.py， 得到 mnistCNN.h5 （keras model）
  2. 执行 convertor.py， 得到 mnist.tflite （转化为更轻量的移动端模型）

安卓端也可用同样的 tflite 模型进行预测
