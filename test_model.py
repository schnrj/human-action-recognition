
import torch
import torch.nn as nn
import numpy as np
from models.multimodal_mamba import MultimodalMamba, StageConfig, count_parameters, estimate_model_flops
from models.multimodal_mamba import HMCNBlock, FBAM, LightweightVimBlock, FBAMMambaBlock, HaarDWT



def load_multimodal_input():
    """Load multimodal input tensors - now concatenated for MultimodalMamba."""
    try:
        # Load individual modality tensors
        x_rgb = torch.load('x_rgb.pt')     # shape: (1, 3, 224, 224)
        x_depth = torch.load('x_depth.pt') # shape: (1, 1, 224, 224)  
        x_imu = torch.load('x_imu.pt')     # shape: (1, 3, 224, 224)
        
        print(f"Loaded tensors - RGB: {x_rgb.shape}, Depth: {x_depth.shape}, IMU: {x_imu.shape}")
        
        # Concatenate along channel dimension for MultimodalMamba
        x_multimodal = torch.cat([x_rgb, x_depth, x_imu], dim=1)  # shape: (1, 7, 224, 224)
        
        print(f"Concatenated multimodal tensor shape: {x_multimodal.shape}")
        return x_multimodal
        
    except FileNotFoundError as e:
        print(f"Warning: Could not load saved tensors ({e})")
        print("Creating synthetic multimodal data for testing...")
        # Create synthetic multimodal data (RGB + Depth + IMU = 7 channels)
        x_multimodal = torch.randn(1, 7, 224, 224)
        print(f"Synthetic multimodal tensor shape: {x_multimodal.shape}")
        return x_multimodal


def create_test_models():
    """Create different model configurations for testing."""
    NUM_CLASSES = 27
    
    configurations = {
        'small': {
            'name': 'Small MultimodalMamba',
            'model': MultimodalMamba(
                in_channels=7,
                num_classes=NUM_CLASSES,
                stages=[StageConfig(1, 32), StageConfig(1, 64)],
                hmcn_k=[3, 5, 7],
                patch=7,
                d_model=128,
                d_hidden=64,
                use_mamba=False,
                rank_ratio=0.5
            )
        },
        'medium': {
            'name': 'Medium MultimodalMamba',
            'model': MultimodalMamba(
                in_channels=7,
                num_classes=NUM_CLASSES,
                stages=[StageConfig(1, 64), StageConfig(1, 128)],
                hmcn_k=[3, 5, 7, 9],
                patch=7,
                d_model=256,
                d_hidden=128,
                use_mamba=False,
                rank_ratio=0.5
            )
        },
        'large': {
            'name': 'Large MultimodalMamba',
            'model': MultimodalMamba(
                in_channels=7,
                num_classes=NUM_CLASSES,
                stages=[StageConfig(1, 64), StageConfig(1, 128), StageConfig(1, 256)],
                hmcn_k=[3, 5, 7, 9],
                patch=7,
                d_model=256,
                d_hidden=128,
                use_mamba=False,
                rank_ratio=0.5
            )
        }
    }
    
    return configurations


def test_model_inference(model_config, x_multimodal):
    """Test model inference and analyze performance."""
    print("=" * 80)
    print(f"Testing {model_config['name']} - Inference")
    print("=" * 80)
    
    model = model_config['model']
    model.eval()
    
    # Test inference
    try:
        with torch.no_grad():
            start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            
            if torch.cuda.is_available():
                model = model.cuda()
                x_multimodal = x_multimodal.cuda()
                start_time.record()
            
            logits = model(x_multimodal)
            
            if torch.cuda.is_available():
                end_time.record()
                torch.cuda.synchronize()
                inference_time = start_time.elapsed_time(end_time)
                print(f"GPU Inference time: {inference_time:.2f} ms")
            
        print(f"Model output shape: {logits.shape}")
        print(f"Expected shape: (1, 27)")
        
        # Get predictions
        probabilities = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(logits, dim=1).item()
        confidence = probabilities[0, pred_class].item()
        
        print(f"Predicted class: {pred_class}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Top-3 predictions: {torch.topk(probabilities, 3, dim=1).indices.squeeze().tolist()}")
        
        # Model analysis
        print(f"\n📊 MODEL ANALYSIS:")
        print("-" * 50)
        analysis = model.analyze_model(input_shape=x_multimodal.shape, verbose=False)
        
        print(f"Total Parameters: {analysis['parameters']['total_parameters']:,}")
        print(f"Trainable Parameters: {analysis['parameters']['trainable_parameters']:,}")
        print(f"Model Size: {analysis['model_size_mb']:.2f} MB")
        print(f"FLOPs: {analysis['flops']:,}")
        print(f"GFLOPs: {analysis['gflops']:.2f}")
        print(f"MFLOPs: {analysis['mflops']:.1f}")
        
        print("✅ Inference test passed!\n")
        return analysis
        
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        raise


def test_model_training_step(model_config, x_multimodal):
    """Test model training step."""
    print("=" * 80)
    print(f"Testing {model_config['name']} - Training Step")
    print("=" * 80)
    
    model = model_config['model']
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Create target
    target = torch.tensor([0])  # Class 0 for testing
    
    try:
        model.train()
        
        if torch.cuda.is_available():
            model = model.cuda()
            x_multimodal = x_multimodal.cuda()
            target = target.cuda()
        
        # Training step
        optimizer.zero_grad()
        logits = model(x_multimodal)
        loss = criterion(logits, target)
        
        print(f"Training loss: {loss.item():.4f}")
        print(f"Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        total_grad_norm = 0
        param_count = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm().item() ** 2
                param_count += 1
        
        total_grad_norm = total_grad_norm ** 0.5
        print(f"Gradient norm: {total_grad_norm:.4f}")
        print(f"Parameters with gradients: {param_count}")
        
        optimizer.step()
        
        print("✅ Training step test passed!\n")
        return loss.item()
        
    except Exception as e:
        print(f"❌ Training step test failed: {e}")
        raise


def test_model_components():
    """Test individual model components."""
    print("=" * 80)
    print("Testing Individual Model Components")
    print("=" * 80)
    
   
    
    batch_size, channels, height, width = 2, 64, 56, 56
    x = torch.randn(batch_size, channels, height, width)
    
    # Test HMCNBlock
    print("Testing HMCNBlock...")
    hmcn = HMCNBlock(in_ch=channels, width=channels, stages_k=[3, 5, 7])
    out_hmcn = hmcn(x)
    print(f"✓ HMCNBlock: {x.shape} -> {out_hmcn.shape}")
    
    # Test FBAM
    print("Testing FBAM...")
    fbam = FBAM(channels=channels)
    out_fbam = fbam(x)
    print(f"✓ FBAM: {x.shape} -> {out_fbam.shape}")
    
    # Test HaarDWT
    print("Testing HaarDWT...")
    dwt = HaarDWT(max_channels=channels)
    LL, LH, HL, HH = dwt(x)
    print(f"✓ HaarDWT: {x.shape} -> 4 x {LL.shape}")
    
    # Test FBAMMambaBlock
    print("Testing FBAMMambaBlock...")
    fbam_mamba = FBAMMambaBlock(
        channels=channels, patch=7, d_model=256, d_hidden=128
    )
    out_fbam_mamba = fbam_mamba(x)
    print(f"✓ FBAMMambaBlock: {x.shape} -> {out_fbam_mamba.shape}")
    
    print("✅ All component tests passed!\n")


def run_comprehensive_analysis():
    """Run comprehensive model analysis and comparison."""
    print("=" * 80)
    print("COMPREHENSIVE MULTIMODAL MAMBA ANALYSIS")
    print("=" * 80)
    
    # Load input data
    x_multimodal = load_multimodal_input()
    
    # Test individual components first
    test_model_components()
    
    # Create model configurations
    model_configs = create_test_models()
    
    results = {}
    
    # Test each configuration
    for config_name, config in model_configs.items():
        print(f"\n🔍 TESTING CONFIGURATION: {config_name.upper()}")
        print("=" * 80)
        
        # Inference test
        analysis = test_model_inference(config, x_multimodal.clone())
        
        # Training test
        loss = test_model_training_step(config, x_multimodal.clone())
        
        # Store results
        results[config_name] = {
            'analysis': analysis,
            'training_loss': loss,
            'name': config['name']
        }
    
    # Print comparison
    print("\n" + "=" * 100)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 100)
    print(f"{'Model':<20} {'Params (M)':<12} {'Size (MB)':<12} {'GFLOPs':<10} {'Loss':<10}")
    print("-" * 100)
    
    for config_name, result in results.items():
        params_m = result['analysis']['parameters']['total_parameters'] / 1e6
        size_mb = result['analysis']['model_size_mb']
        gflops = result['analysis']['gflops']
        loss = result['training_loss']
        
        print(f"{config_name:<20} {params_m:<12.2f} {size_mb:<12.2f} {gflops:<10.2f} {loss:<10.4f}")
    
    print("=" * 100)
    
    # Efficiency analysis
    print(f"\nEFFICIENCY ANALYSIS:")
    print("-" * 50)
    for config_name, result in results.items():
        params_m = result['analysis']['parameters']['total_parameters'] / 1e6
        gflops = result['analysis']['gflops']
        efficiency = gflops / params_m if params_m > 0 else 0
        print(f"{config_name:<20}: {efficiency:.2f} GFLOPs/M params")
    
    return results


def benchmark_inference_speed(num_runs=100):
    """Benchmark inference speed for different model sizes."""
    print("\n" + "=" * 80)
    print("INFERENCE SPEED BENCHMARK")
    print("=" * 80)
    
    x_multimodal = torch.randn(1, 7, 224, 224)
    model_configs = create_test_models()
    
    if torch.cuda.is_available():
        x_multimodal = x_multimodal.cuda()
        print("Using CUDA for benchmarking")
    else:
        print("Using CPU for benchmarking")
    
    for config_name, config in model_configs.items():
        model = config['model']
        model.eval()
        
        if torch.cuda.is_available():
            model = model.cuda()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(x_multimodal)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                if torch.cuda.is_available():
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    _ = model(x_multimodal)
                    end.record()
                    torch.cuda.synchronize()
                    times.append(start.elapsed_time(end))
                else:
                    import time
                    start = time.time()
                    _ = model(x_multimodal)
                    end = time.time()
                    times.append((end - start) * 1000)  # Convert to ms
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"{config_name:<15}: {avg_time:.2f} ± {std_time:.2f} ms")


if __name__ == "__main__":
    print("🚀 Starting MultimodalMamba Comprehensive Testing")
    print("=" * 80)
    
    try:
        # Run comprehensive analysis
        results = run_comprehensive_analysis()
        
        # Run speed benchmark
        benchmark_inference_speed(num_runs=50)
        
        print(f"\n🎉 All tests completed successfully!")
        print("=" * 80)
        print("✅ Model is ready for training and deployment!")
        
    except Exception as e:
        print(f"❌ Testing failed with error: {e}")
        raise
    
    print("\n📝 RECOMMENDATIONS:")
    print("-" * 50)
    print("• Use 'medium' configuration for balanced performance/efficiency")
    print("• Monitor gradient norms during training (should be < 10.0)")
    print("• Consider mixed precision training for faster inference")
    print("• Batch size can be increased based on available GPU memory")
    print("=" * 80)