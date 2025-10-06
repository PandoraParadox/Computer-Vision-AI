import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
from models.backbone import ClassificationModel


def test_layer_padding():
    """Test if explicit padding matches PyTorch behavior"""
    print("Testing padding fix for Conv1...")
    
    # Load model
    device = torch.device('cpu')
    model = ClassificationModel(num_classes=3)
    ckpt = torch.load("weights/best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    model.eval()
    
    # Get weights for first layer
    state_dict = model.state_dict()
    conv_w = state_dict['features.0.weight']
    bn_w = state_dict['features.1.weight']
    bn_b = state_dict['features.1.bias']
    bn_mean = state_dict['features.1.running_mean']
    bn_var = state_dict['features.1.running_var']
    
    # Fuse conv + bn
    scale = bn_w / torch.sqrt(bn_var + 1e-5)
    fused_w = conv_w * scale.view(-1, 1, 1, 1)
    fused_b = bn_b - bn_mean * scale
    tf_w = fused_w.permute(2, 3, 1, 0).numpy()
    tf_b = fused_b.numpy()
    
    # Create test input
    x_test = torch.randn(1, 1, 128, 128)
    x_tf = x_test.permute(0, 2, 3, 1).numpy()
    
    print(f"Input shape - PyTorch: {x_test.shape}, TF: {x_tf.shape}")
    
    # PyTorch forward (first layer only)
    with torch.no_grad():
        # Conv + BN + ReLU
        conv_out = torch.nn.functional.conv2d(x_test, conv_w, bias=None, stride=2, padding=3)
        bn_out = torch.nn.functional.batch_norm(
            conv_out, bn_mean, bn_var, bn_w, bn_b, training=False, eps=1e-5
        )
        pytorch_result = torch.relu(bn_out)
        print(f"PyTorch output: {pytorch_result.shape} | range: [{pytorch_result.min():.3f}, {pytorch_result.max():.3f}]")
    
    # TensorFlow version with explicit padding
    print("\nTesting TensorFlow with explicit padding...")
    
    # Method 1: ZeroPadding2D + valid conv
    padded_input = tf.keras.layers.ZeroPadding2D(padding=3)(x_tf)
    print(f"After padding: {padded_input.shape}")
    
    tf_conv = tf.keras.layers.Conv2D(64, 7, strides=2, padding='valid', use_bias=True)
    tf_conv.build((None, 134, 134, 1))  # 128 + 2*3 = 134
    tf_conv.set_weights([tf_w, tf_b])
    
    conv_out = tf_conv(padded_input)
    tf_result = tf.nn.relu(conv_out)
    
    print(f"TensorFlow output: {tf_result.shape} | range: [{tf_result.numpy().min():.3f}, {tf_result.numpy().max():.3f}]")
    
    # Compare
    pytorch_for_comparison = pytorch_result.permute(0, 2, 3, 1).numpy()
    diff = np.abs(pytorch_for_comparison - tf_result.numpy())
    
    print(f"\nComparison:")
    print(f"Max difference: {np.max(diff):.6f}")
    print(f"Mean difference: {np.mean(diff):.6f}")
    
    if np.max(diff) < 0.01:
        print("✅ Padding fix successful!")
        return True
    else:
        print("❌ Still have differences. Need to check other parameters.")
        return False


def test_full_model_with_padding_fix():
    """Test full model conversion with padding fix"""
    print("\n" + "="*60)
    print("Testing full model with padding fix")
    print("="*60)
    
    # Import the fixed converter
    from convert_to_tflite import TorchToTensorFlow
    
    # Load PyTorch model
    device = torch.device('cpu')
    model = ClassificationModel(num_classes=3)
    ckpt = torch.load("weights/best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    model.eval()
    
    # Build TF model with padding fix
    converter = TorchToTensorFlow(model)
    tf_model = converter.build_tensorflow_model()
    
    print("TensorFlow model with explicit padding:")
    tf_model.summary()
    
    # Copy weights
    converter.copy_weights(tf_model)
    
    # Test with multiple inputs
    print("\nTesting with multiple random inputs...")
    max_diffs = []
    
    for i in range(5):
        x_test = torch.randn(1, 1, 128, 128)
        x_tf = x_test.permute(0, 2, 3, 1).numpy()
        
        # PyTorch
        with torch.no_grad():
            pytorch_out = model(x_test).numpy()
        
        # TensorFlow
        tf_out = tf_model(x_tf, training=False).numpy()
        
        diff = np.abs(pytorch_out - tf_out)
        max_diff = np.max(diff)
        max_diffs.append(max_diff)
        
        print(f"Test {i+1}: Max diff = {max_diff:.6f}")
    
    avg_max_diff = np.mean(max_diffs)
    print(f"\nAverage max difference: {avg_max_diff:.6f}")
    
    if avg_max_diff < 0.01:
        print("✅ Full model conversion successful!")
        return True
    else:
        print("❌ Still have issues in full model.")
        return False


if __name__ == "__main__":
    # Test individual layer first
    layer_success = test_layer_padding()
    
    if layer_success:
        # Test full model
        full_success = test_full_model_with_padding_fix()
        
        if full_success:
            print("\n🎉 Ready to convert to TFLite!")
            print("Run: python convert_to_tflite.py")
        else:
            print("\n❌ Need more debugging for full model.")
    else:
        print("\n❌ Fix the individual layer first.")