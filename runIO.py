import os
import torch
from torchvision import transforms, models
from PIL import Image
import sys

# If you have a GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class names in the same order as training
class_names = ['Benign', '[Malignant] Pre-B', '[Malignant] Pro-B', '[Malignant] early Pre-B']

# Define image transforms (must match training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def load_model(model_path):
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_image(model, image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)
        pred_label = class_names[pred.item()]

    return pred_label

def main():
    if len(sys.argv) < 3:
        print("Usage: python run.py <model_path> <image_path>")
        print("\nExample:")
        print(" python run.py ./model/best_model.pth ./test_images/sample.jpg\n")
        sys.exit(1)

    # ======= Add your downloaded model path here as first argument =======
    model_path = sys.argv[1]  # <-- Path to your trained model weights (e.g., './model/best_model.pth')
    
    image_path = sys.argv[2]  # Path to the image you want to classify

    model = load_model(model_path)

    prediction = predict_image(model, image_path)
    print(f"Prediction for image '{image_path}': {prediction}")

if __name__ == '__main__':
    main()
