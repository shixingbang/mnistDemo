import coremltools
import keras2onnx
import onnx
from tensorflow.python.keras.models import load_model
import tensorflow as tf

# tf.compat.v1.disable_v2_behavior()
# tf.compat.v1.disable_eager_execution()

model = load_model('./mnistLow.h5')
# onnx_model = keras2onnx.convert_keras(model, model.name)
# temp_model_file = 'onnx_model.onnx'
# onnx.save_model(onnx_model, temp_model_file)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.inference_type = tf.uint8    #tf.lite.constants.QUANTIZED_UINT8
# converter.allow_custom_ops = True

converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_types = [tf.float16]
# converter.post_training_quantize = True

# converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS]


# print(model.summary())

tflite_model = converter.convert()

# Save the model.
with open('mnistlow.tflite', 'wb') as f:
  f.write(tflite_model)