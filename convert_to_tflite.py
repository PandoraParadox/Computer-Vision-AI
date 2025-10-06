import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
import os
from models.backbone import ClassificationModel

class TorchToTensorFlow:
    def __init__(self, pytorch_model: nn.Module, input_shape: tuple = (1, 128, 128)):
        self.pytorch_model = pytorch_model
        self.input_shape = input_shape

    def convert_conv_bn_layer(self, conv_layer: nn.Conv2d, bn_layer: nn.BatchNorm2d):
        conv_weight = conv_layer.weight.data
        bn_weight = bn_layer.weight.data
        bn_bias = bn_layer.bias.data
        bn_mean = bn_layer.running_mean.data
        bn_var = bn_layer.running_var.data
        bn_eps = bn_layer.eps

        scale = bn_weight / torch.sqrt(bn_var + bn_eps)
        fused_weight = conv_weight * scale.view(-1, 1, 1, 1)
        fused_bias = bn_bias - bn_mean * scale
        tf_weight = fused_weight.permute(2, 3, 1, 0).numpy()
        tf_bias = fused_bias.numpy()
        return tf_weight, tf_bias

    def copy_weights_manually(self, tf_model):
        self.pytorch_model.eval()
        state_dict = self.pytorch_model.state_dict()

        def get_fused_conv_weights(conv_weight, bn_weight, bn_bias, bn_mean, bn_var, bn_eps=1e-5):
            scale = bn_weight / torch.sqrt(bn_var + bn_eps)
            fused_weight = conv_weight * scale.view(-1, 1, 1, 1)
            fused_bias = bn_bias - bn_mean * scale
            tf_weight = fused_weight.permute(2, 3, 1, 0).numpy()
            tf_bias = fused_bias.numpy()
            return tf_weight, tf_bias

        conv1_w, conv1_b = get_fused_conv_weights(
            state_dict['features.0.weight'], state_dict['features.1.weight'],
            state_dict['features.1.bias'], state_dict['features.1.running_mean'],
            state_dict['features.1.running_var']
        )
        tf_model.get_layer('conv1').set_weights([conv1_w, conv1_b])

        conv2_w, conv2_b = get_fused_conv_weights(
            state_dict['features.4.weight'], state_dict['features.5.weight'],
            state_dict['features.5.bias'], state_dict['features.5.running_mean'],
            state_dict['features.5.running_var']
        )
        tf_model.get_layer('conv2').set_weights([conv2_w, conv2_b])

        conv3_w, conv3_b = get_fused_conv_weights(
            state_dict['features.8.weight'], state_dict['features.9.weight'],
            state_dict['features.9.bias'], state_dict['features.9.running_mean'],
            state_dict['features.9.running_var']
        )
        tf_model.get_layer('conv3').set_weights([conv3_w, conv3_b])

        conv4_w, conv4_b = get_fused_conv_weights(
            state_dict['features.12.weight'], state_dict['features.13.weight'],
            state_dict['features.13.bias'], state_dict['features.13.running_mean'],
            state_dict['features.13.running_var']
        )
        tf_model.get_layer('conv4').set_weights([conv4_w, conv4_b])

        classifier_w = state_dict['classifier.1.weight'].numpy().T
        classifier_b = state_dict['classifier.1.bias'].numpy()
        tf_model.get_layer('classifier').set_weights([classifier_w, classifier_b])

    def build_tensorflow_model(self):
        inputs = tf.keras.Input(shape=(128, 128, 1))
        x = tf.keras.layers.Conv2D(64, 7, strides=2, padding='same', use_bias=True, activation=None, name='conv1')(inputs)
        x = tf.keras.layers.ReLU(name='relu1')(x)
        x = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same', name='pool1')(x)

        x = tf.keras.layers.Conv2D(128, 5, strides=2, padding='same', use_bias=True, activation=None, name='conv2')(x)
        x = tf.keras.layers.ReLU(name='relu2')(x)
        x = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same', name='pool2')(x)

        x = tf.keras.layers.Conv2D(256, 3, strides=2, padding='same', use_bias=True, activation=None, name='conv3')(x)
        x = tf.keras.layers.ReLU(name='relu3')(x)
        x = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same', name='pool3')(x)

        x = tf.keras.layers.Conv2D(512, 3, strides=2, padding='same', use_bias=True, activation=None, name='conv4')(x)
        x = tf.keras.layers.ReLU(name='relu4')(x)

        x = tf.keras.layers.GlobalAveragePooling2D(name='avgpool')(x)
        x = tf.keras.layers.Dropout(0.2, name='dropout')(x)
        outputs = tf.keras.layers.Dense(self.pytorch_model.num_classes, activation=None, name='classifier')(x)
        return tf.keras.Model(inputs, outputs)
    
    def convert_to_tflite(self, tf_model, tflite_path, quantize=False):
        converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"TFLite model saved to {tflite_path}")



def convert_pytorch_to_tflite(model_path: str, num_classes: int = 3, output_dir: str = "converted_models", quantize: bool = False):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cpu')
    pytorch_model = ClassificationModel(num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        pytorch_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        pytorch_model.load_state_dict(checkpoint)
    pytorch_model.eval()
    converter = TorchToTensorFlow(pytorch_model)
    tf_model = converter.build_tensorflow_model()
    converter.copy_weights_manually(tf_model)
    tf_model_path = os.path.join(output_dir, "model.h5")
    tf_model.save(tf_model_path)
    tflite_path = os.path.join(output_dir, "model_quantized.tflite") if quantize else os.path.join(output_dir, "model.tflite")
    print(f"Saving TFLite model to: {os.path.abspath(tflite_path)}")
    converter.convert_to_tflite(tf_model, tflite_path, quantize=quantize)
    return tflite_path

if __name__ == "__main__":
    MODEL_PATH = "weights/best_model.pth"
    NUM_CLASSES = 3
    OUTPUT_DIR = "converted_models"
    QUANTIZE = False

    tflite_path = convert_pytorch_to_tflite(
        model_path=MODEL_PATH,
        num_classes=NUM_CLASSES,
        output_dir=OUTPUT_DIR,
        quantize=QUANTIZE
    )
    print("TFLite model saved at:", tflite_path)
