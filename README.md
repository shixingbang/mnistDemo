# mnistDemo


运行demo：
  1. 连上手机，换一下签名证书
  2. 可以尝试直接 run
  3. 缺东西的话可以 pod install

训练模型：
  1. 执行 MNIST.py， 得到 mnistCNN.h5 （keras model）
  2. 执行 convertor.py， 得到 mnist.tflite （转化为更轻量的移动端模型）
test2
安卓端也可用同样的 tflite 模型进行预测

test1
