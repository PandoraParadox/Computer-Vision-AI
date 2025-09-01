import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import json
from pathlib import Path

# Import your models (adjust as needed)
from models.backbone import MyDetector, ImprovedYOLODetector

class ComprehensiveYOLODebugger:
    def __init__(self, model_path, num_classes=6, model_type='original'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_classes = num_classes
        self.img_size = 416
        self.S = 7
        self.num_anchors = 2
        
        print(f"🚀 Starting comprehensive YOLO debugging...")
        print(f"📱 Device: {self.device}")
        print(f"🎯 Model type: {model_type}")
        
        # Load model based on type
        self.model = self.load_model(model_path, model_type)
        
        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
        ])
        
        # Class names
        self.class_names = ['car', 'truck', 'bus', 'motorbike', 'bicycle', 'person']
        
        # Debug results storage
        self.debug_results = {
            'model_info': {},
            'weight_analysis': {},
            'prediction_analysis': {},
            'coordinate_analysis': {}
        }
    
    def load_model(self, model_path, model_type):
        """Load model with comprehensive analysis"""
        print(f"📂 Loading model from: {model_path}")
        
        # Initialize model based on type
        if model_type == 'fixed':
            model = DetectionLoss(num_classes=self.num_classes, S=self.S, num_anchors=self.num_anchors)
        elif model_type == 'improved':
            model = ImprovedYOLODetector(num_classes=self.num_classes, S=self.S, num_anchors=self.num_anchors)
        else:
            model = MyDetector(num_classes=self.num_classes, S=self.S, num_anchors=self.num_anchors)
        
        try:
            if torch.cuda.is_available():
                checkpoint = torch.load(model_path)
            else:
                checkpoint = torch.load(model_path, map_location='cpu')
            
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    # Store checkpoint info
                    self.debug_results['model_info'] = {
                        'epoch': checkpoint.get('epoch', 'unknown'),
                        'train_loss': checkpoint.get('train_loss', 'N/A'),
                        'val_metrics': checkpoint.get('val_metrics', {}),
                        'checkpoint_keys': list(checkpoint.keys())
                    }
                    print(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
                else:
                    model.load_state_dict(checkpoint)
                    self.debug_results['model_info'] = {'type': 'state_dict_only'}
            else:
                model.load_state_dict(checkpoint)
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
        
        model.to(self.device)
        model.eval()
        
        # Analyze model architecture
        self.analyze_model_architecture(model)
        
        return model
    
    def analyze_model_architecture(self, model):
        """Analyze model architecture and parameters"""
        print(f"\n🏗️ Model Architecture Analysis:")
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   📊 Total parameters: {total_params:,}")
        print(f"   🎯 Trainable parameters: {trainable_params:,}")
        
        # Test forward pass with dummy input
        dummy_input = torch.randn(1, 3, self.img_size, self.img_size).to(self.device)
        with torch.no_grad():
            try:
                output = model(dummy_input)
                print(f"   ✅ Forward pass successful")
                print(f"   📐 Output shape: {output.shape}")
                
                expected_shape = (1, self.num_anchors, 5 + self.num_classes, self.S, self.S)
                if output.shape == expected_shape:
                    print(f"   ✅ Output shape matches expected: {expected_shape}")
                else:
                    print(f"   ⚠️ Output shape mismatch! Expected: {expected_shape}, Got: {output.shape}")
                    
            except Exception as e:
                print(f"   ❌ Forward pass failed: {e}")
        
        self.debug_results['model_info'].update({
            'total_params': total_params,
            'trainable_params': trainable_params,
            'output_shape': str(output.shape) if 'output' in locals() else 'Failed'
        })
    
    def analyze_weights(self):
        """Comprehensive weight analysis"""
        print(f"\n🏋️ Weight Analysis:")
        
        weight_stats = {}
        problematic_layers = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Basic statistics
                stats = {
                    'shape': list(param.shape),
                    'mean': param.mean().item(),
                    'std': param.std().item(),
                    'min': param.min().item(),
                    'max': param.max().item(),
                    'num_zeros': (param == 0).sum().item(),
                    'num_nan': torch.isnan(param).sum().item(),
                    'num_inf': torch.isinf(param).sum().item()
                }
                
                weight_stats[name] = stats
                
                # Check for problems
                if stats['num_nan'] > 0:
                    problematic_layers.append(f"{name}: {stats['num_nan']} NaN values")
                if stats['num_inf'] > 0:
                    problematic_layers.append(f"{name}: {stats['num_inf']} Inf values")
                if stats['std'] > 10:
                    problematic_layers.append(f"{name}: Very high std ({stats['std']:.2f})")
                if stats['std'] < 1e-6:
                    problematic_layers.append(f"{name}: Very low std ({stats['std']:.6f})")
        
        # Print summary
        if problematic_layers:
            print(f"   ⚠️ Found {len(problematic_layers)} problematic layers:")
            for issue in problematic_layers[:5]:  # Show first 5
                print(f"      - {issue}")
        else:
            print(f"   ✅ No problematic weights found")
        
        # Check prediction layer specifically
        pred_layers = [name for name in weight_stats.keys() if 'prediction' in name or 'detection' in name]
        if pred_layers:
            print(f"   🎯 Prediction layer analysis:")
            for layer in pred_layers:
                stats = weight_stats[layer]
                print(f"      {layer}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
        
        self.debug_results['weight_analysis'] = {
            'total_layers': len(weight_stats),
            'problematic_layers': problematic_layers,
            'prediction_layers': {name: weight_stats[name] for name in pred_layers}
        }
    
    def analyze_predictions_detailed(self, image_path):
        """Detailed prediction analysis"""
        print(f"\n🔍 Detailed Prediction Analysis for: {os.path.basename(image_path)}")
        
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Get raw predictions
        with torch.no_grad():
            outputs = self.model(img_tensor)
        
        pred = outputs[0]  # [num_anchors, num_classes+5, S, S]
        
        # Analyze confidence distribution
        conf_values = pred[:, 0, :, :].cpu().numpy()  # [num_anchors, S, S]
        
        conf_stats = {
            'min': float(conf_values.min()),
            'max': float(conf_values.max()),
            'mean': float(conf_values.mean()),
            'std': float(conf_values.std()),
            'median': float(np.median(conf_values))
        }
        
        print(f"   📊 Confidence Statistics:")
        print(f"      Range: [{conf_stats['min']:.6f}, {conf_stats['max']:.6f}]")
        print(f"      Mean: {conf_stats['mean']:.6f}, Std: {conf_stats['std']:.6f}")
        
        # Count predictions at different thresholds
        thresholds = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
        threshold_counts = {}
        for thresh in thresholds:
            count = (conf_values > thresh).sum()
            threshold_counts[thresh] = int(count)
            print(f"      Above {thresh}: {count}")
        
        # Analyze bounding box predictions
        box_values = pred[:, 1:5, :, :].cpu().numpy()  # [num_anchors, 4, S, S]
        
        box_stats = {}
        for i, coord in enumerate(['x', 'y', 'w', 'h']):
            coord_values = box_values[:, i, :, :]
            box_stats[coord] = {
                'min': float(coord_values.min()),
                'max': float(coord_values.max()),
                'mean': float(coord_values.mean()),
                'std': float(coord_values.std())
            }
        
        print(f"   📐 Bounding Box Statistics:")
        for coord, stats in box_stats.items():
            print(f"      {coord}: [{stats['min']:.2f}, {stats['max']:.2f}], mean={stats['mean']:.2f}")
        
        # Analyze class predictions
        class_values = pred[:, 5:, :, :].cpu().numpy()  # [num_anchors, num_classes, S, S]
        class_stats = {
            'min': float(class_values.min()),
            'max': float(class_values.max()),
            'mean': float(class_values.mean()),
            'entropy': self.calculate_entropy(class_values)
        }
        
        print(f"   🎯 Class Prediction Statistics:")
        print(f"      Range: [{class_stats['min']:.4f}, {class_stats['max']:.4f}]")
        print(f"      Mean entropy: {class_stats['entropy']:.4f}")
        
        # Find most confident predictions
        high_conf_predictions = []
        for a in range(self.num_anchors):
            for y in range(self.S):
                for x in range(self.S):
                    conf = conf_values[a, y, x]
                    if conf > 0.01:  # Very low threshold
                        class_probs = pred[a, 5:, y, x].cpu().numpy()
                        class_id = np.argmax(class_probs)
                        class_conf = class_probs[class_id]
                        final_conf = conf * class_conf
                        
                        box = box_values[a, :, y, x]
                        
                        high_conf_predictions.append({
                            'anchor': a,
                            'grid': (x, y),
                            'confidence': float(conf),
                            'class_id': int(class_id),
                            'class_conf': float(class_conf),
                            'final_conf': float(final_conf),
                            'box': [float(b) for b in box]
                        })
        
        # Sort by final confidence
        high_conf_predictions.sort(key=lambda x: x['final_conf'], reverse=True)
        
        print(f"   🎯 Top 5 Predictions:")
        for i, pred_info in enumerate(high_conf_predictions[:5]):
            print(f"      {i+1}. Class {pred_info['class_id']} ({self.class_names[pred_info['class_id']]})")
            print(f"         Conf: {pred_info['confidence']:.4f}, Final: {pred_info['final_conf']:.4f}")
            print(f"         Grid: {pred_info['grid']}, Box: {pred_info['box']}")
        
        self.debug_results['prediction_analysis'] = {
            'image': os.path.basename(image_path),
            'confidence_stats': conf_stats,
            'threshold_counts': threshold_counts,
            'box_stats': box_stats,
            'class_stats': class_stats,
            'top_predictions': high_conf_predictions[:10]
        }
        
        return high_conf_predictions, img
    
    def calculate_entropy(self, class_probs):
        """Calculate average entropy of class predictions"""
        # Flatten and calculate entropy for each prediction
        flat_probs = class_probs.reshape(-1, class_probs.shape[1])
        entropies = []
        
        for probs in flat_probs:
            # Add small epsilon to avoid log(0)
            probs = probs + 1e-8
            entropy = -np.sum(probs * np.log(probs))
            entropies.append(entropy)
        
        return np.mean(entropies)
    
    def visualize_predictions(self, image_path, conf_threshold=0.001):
        """Visualize predictions with very detailed information"""
        print(f"\n🎨 Creating detailed visualization...")
        
        predictions, original_img = self.analyze_predictions_detailed(image_path)
        
        # Convert PIL to OpenCV
        img_cv = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)
        img_cv = cv2.resize(img_cv, (self.img_size, self.img_size))
        
        # Create debug image with grid overlay
        debug_img = img_cv.copy()
        
        # Draw grid
        grid_size = self.img_size // self.S
        for i in range(1, self.S):
            x = i * grid_size
            cv2.line(debug_img, (x, 0), (x, self.img_size), (128, 128, 128), 1)
            y = i * grid_size
            cv2.line(debug_img, (0, y), (self.img_size, y), (128, 128, 128), 1)
        
        # Colors for different anchors
        anchor_colors = [(255, 0, 0), (0, 255, 0)]  # Red, Green
        
        drawn_count = 0
        for pred_info in predictions:
            if pred_info['final_conf'] > conf_threshold and drawn_count < 20:  # Limit to top 20
                anchor = pred_info['anchor']
                x_grid, y_grid = pred_info['grid']
                box = pred_info['box']
                
                # Convert box coordinates (assuming they're in image coordinates)
                center_x, center_y, width, height = box
                
                # Clamp coordinates
                x1 = max(0, int(center_x - width/2))
                y1 = max(0, int(center_y - height/2))
                x2 = min(self.img_size, int(center_x + width/2))
                y2 = min(self.img_size, int(center_y + height/2))
                
                if x2 > x1 and y2 > y1:
                    color = anchor_colors[anchor % len(anchor_colors)]
                    
                    # Draw bounding box
                    cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw grid cell indicator
                    grid_x_pixel = x_grid * grid_size + grid_size // 2
                    grid_y_pixel = y_grid * grid_size + grid_size // 2
                    cv2.circle(debug_img, (grid_x_pixel, grid_y_pixel), 3, color, -1)
                    
                    # Label with detailed info
                    class_name = self.class_names[pred_info['class_id']]
                    label = f"A{anchor}:{class_name}:{pred_info['final_conf']:.3f}"
                    
                    # Background for text
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
                    )
                    cv2.rectangle(debug_img, (x1, y1-text_height-5), 
                                (x1+text_width, y1), color, -1)
                    
                    # Text
                    cv2.putText(debug_img, label, (x1, y1-2), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    drawn_count += 1
        
        # Add info overlay
        info_lines = [
            f"Image: {os.path.basename(image_path)}",
            f"Predictions shown: {drawn_count}",
            f"Threshold: {conf_threshold}",
            f"Grid: {self.S}x{self.S}, Anchors: {self.num_anchors}"
        ]
        
        y_pos = 20
        for line in info_lines:
            cv2.putText(debug_img, line, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y_pos += 20
        
        # Save debug visualization
        output_path = f"comprehensive_debug_{os.path.basename(image_path)}"
        cv2.imwrite(output_path, debug_img)
        print(f"   💾 Detailed visualization saved: {output_path}")
        
        return debug_img
    
    def test_with_multiple_images(self, image_dir, max_images=5):
        """Test model with multiple images"""
        print(f"\n🖼️ Testing with multiple images from: {image_dir}")
        
        if not os.path.exists(image_dir):
            print(f"❌ Directory not found: {image_dir}")
            return
        
        # Get image files
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in extensions:
            image_files.extend(Path(image_dir).glob(f"*{ext}"))
            image_files.extend(Path(image_dir).glob(f"*{ext.upper()}"))
        
        image_files = list(image_files)[:max_images]
        
        if not image_files:
            print(f"❌ No image files found in {image_dir}")
            return
        
        all_results = []
        for img_path in image_files:
            print(f"\n--- Processing: {img_path.name} ---")
            try:
                predictions, _ = self.analyze_predictions_detailed(str(img_path))
                self.visualize_predictions(str(img_path), conf_threshold=0.005)
                
                # Store results
                result = {
                    'image': img_path.name,
                    'total_predictions': len(predictions),
                    'high_conf_predictions': len([p for p in predictions if p['final_conf'] > 0.1])
                }
                all_results.append(result)
                
            except Exception as e:
                print(f"❌ Error processing {img_path.name}: {e}")
        
        # Summary
        print(f"\n📊 Multi-image Test Summary:")
        total_preds = sum(r['total_predictions'] for r in all_results)
        total_high_conf = sum(r['high_conf_predictions'] for r in all_results)
        
        print(f"   📋 Images processed: {len(all_results)}")
        print(f"   🎯 Total predictions: {total_preds}")
        print(f"   ⭐ High confidence (>0.1): {total_high_conf}")
        print(f"   📈 Average predictions per image: {total_preds/len(all_results) if all_results else 0:.1f}")
        
        return all_results
    
    def generate_comprehensive_report(self, output_file="yolo_debug_report.json"):
        """Generate comprehensive debug report"""
        print(f"\n📄 Generating comprehensive report...")
        
        # Add summary statistics
        self.debug_results['summary'] = {
            'total_model_params': self.debug_results['model_info'].get('total_params', 0),
            'has_weight_issues': len(self.debug_results['weight_analysis'].get('problematic_layers', [])) > 0,
            'prediction_quality': 'needs_analysis'  # This would be determined by actual testing
        }
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(self.debug_results, f, indent=2)
        
        print(f"   💾 Report saved to: {output_file}")
        
        # Print key findings
        print(f"\n🔍 Key Findings:")
        if self.debug_results['weight_analysis'].get('problematic_layers'):
            print(f"   ⚠️ Found weight issues in {len(self.debug_results['weight_analysis']['problematic_layers'])} layers")
        else:
            print(f"   ✅ No major weight issues detected")
        
        if 'prediction_analysis' in self.debug_results:
            pred_stats = self.debug_results['prediction_analysis']
            print(f"   📊 Confidence range: [{pred_stats['confidence_stats']['min']:.4f}, {pred_stats['confidence_stats']['max']:.4f}]")
            print(f"   🎯 Top prediction confidence: {pred_stats['top_predictions'][0]['final_conf']:.4f}" if pred_stats['top_predictions'] else "No strong predictions")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive YOLO Model Debugging')
    parser.add_argument('--model', required=True, help='Path to model file')
    parser.add_argument('--test-img', help='Path to test image')
    parser.add_argument('--train-dir', default='data/images/train', help='Training images directory')
    parser.add_argument('--num-classes', type=int, default=6, help='Number of classes')
    parser.add_argument('--model-type', choices=['original', 'fixed', 'improved'], default='original', help='Model type')
    parser.add_argument('--max-images', type=int, default=5, help='Maximum number of images to test')
    parser.add_argument('--conf-threshold', type=float, default=0.001, help='Confidence threshold for visualization')
    
    args = parser.parse_args()
    
    # Create debugger
    debugger = ComprehensiveYOLODebugger(args.model, args.num_classes, args.model_type)
    
    # Run comprehensive analysis
    print("🚀 Starting comprehensive debugging...")
    
    # 1. Analyze weights
    debugger.analyze_weights()
    
    # 2. Test with training images
    if os.path.exists(args.train_dir):
        debugger.test_with_multiple_images(args.train_dir, args.max_images)
    
    # 3. Test with specific image
    if args.test_img and os.path.exists(args.test_img):
        print(f"\n🎯 Testing with specific image: {args.test_img}")
        debugger.visualize_predictions(args.test_img, args.conf_threshold)
    
    # 4. Generate report
    debugger.generate_comprehensive_report()
    
    print(f"\n✅ Comprehensive debugging completed!")
    print(f"Check the generated visualization images and debug report for detailed analysis.")


if __name__ == "__main__":
    main()