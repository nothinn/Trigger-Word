import tensorflow as tf

saved_model_dir = "trains/saved-model-40-0.044.hdf5"

converter = tf.contrib.lite.TFLiteConverter.from_keras_model_file(saved_model_dir)
converter.optimizations = [tf.contrib.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.lite.constants.FLOAT16]
tflite_quant_model = converter.convert()

print(tflite_quant_model)