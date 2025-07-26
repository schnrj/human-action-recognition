import torch
from models.pyramid_model import MultimodalFrequencyAwareHAR

def load_multimodal_input():
    # Load tensors saved from preprocessing
    x_rgb = torch.load('x_rgb.pt')     # shape: (1, 3, 224, 224)
    x_depth = torch.load('x_depth.pt') # shape: (1, 1, 224, 224)
    x_imu = torch.load('x_imu.pt')     # shape: (1, 3, 224, 224)
    return x_rgb, x_depth, x_imu

def test_model_inference():
    print("=" * 60)
    print("Testing Multimodal HAR Model Inference")
    print("=" * 60)
    x_rgb, x_depth, x_imu = load_multimodal_input()
    NUM_CLASSES = 27
    model = MultimodalFrequencyAwareHAR(3, 1, 3, NUM_CLASSES)
    model.eval()
    with torch.no_grad():
        logits = model(x_rgb, x_depth, x_imu)
    print(f"Model output shape: {logits.shape}")
    pred_class = torch.argmax(logits, dim=1).item()
    print(f"Predicted class: {pred_class}")
    print("✅ Inference test passed!\n")

def test_model_training_step():
    print("=" * 60)
    print("Testing Model Training Step")
    print("=" * 60)
    x_rgb, x_depth, x_imu = load_multimodal_input()
    NUM_CLASSES = 27
    model = MultimodalFrequencyAwareHAR(3, 1, 3, NUM_CLASSES)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    target = torch.tensor([0])
    model.train()
    optimizer.zero_grad()
    logits = model(x_rgb, x_depth, x_imu)
    loss = criterion(logits, target)
    print(f"Training loss: {loss.item():.4f}")
    loss.backward()
    optimizer.step()
    print("✅ Training step test passed!\n")

if __name__ == "__main__":
    test_model_inference()
    test_model_training_step()
    print("🎉 All tests completed successfully!")
