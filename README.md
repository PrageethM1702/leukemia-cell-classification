# 🧬 Blood Cell Cancer Classification using PyTorch

This repository provides a deep learning model built with PyTorch to classify blood cell images into four types related to acute lymphoblastic leukemia (ALL). The model is based on a fine-tuned ResNet18 architecture.

---

## 📁 Dataset

- **Source**: [Kaggle - Blood Cell Cancer (ALL) 4 Class](https://www.kaggle.com/datasets/mohammadamireshraghi/blood-cell-cancer-all-4class)
- **Classes**:
  - `Benign`
  - `[Malignant] Pre-B`
  - `[Malignant] Pro-B`
  - `[Malignant] early Pre-B`

---

## 🔧 Setup

### 1. Install dependencies

torch  
torchvision  
Pillow

Install using pip:

pip install torch torchvision Pillow

---

## 🚀 Running Inference

Step 1: Download or place your trained model file (e.g., best_model.pth) in any directory you prefer.  
Step 2: Place your test image anywhere accessible, such as a test_images/ folder.  
Step 3: Run the inference script:

python run.py ./model/best_model.pth ./test_images/sample_image.jpg

---

## 💡 Notes

- The model expects images resized to 224×224 and normalized using ImageNet statistics (handled in code).
- Automatically uses GPU if available, otherwise defaults to CPU.
- Accepts .jpg, .png, or any common image format.

---

## 📄 How to Use `run.py` Programmatically

import sys  
sys.argv = [  
  'run.py',  
  './model/best_model.pth',  
  './test_images/sample_image.jpg'  
]  

from run import main  
main()

---

## 📊 Model Performance

Test Accuracy: 98.77%  
Weighted F1 Score: 0.9876

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Benign | 0.97 | 0.95 | 0.96 |
| [Malignant] Pre-B | 1.00 | 0.99 | 1.00 |
| [Malignant] Pro-B | 0.99 | 1.00 | 1.00 |
| [Malignant] early Pre-B | 0.98 | 0.99 | 0.99 |

---

## 📥 Download Model

import zipfile  
with zipfile.ZipFile('model.zip', 'w') as zipf:  
  zipf.write('best_model.pth')  

from IPython.display import FileLink  
FileLink('model.zip')

---

## 🧠 Future Improvements

- Integrate Grad-CAM for model explainability
- Add web interface using Streamlit or Flask
- Expand dataset with more diverse samples
- Fine-tune deeper layers for better generalization

---

## 📌 License

This project is released under the MIT License.

---

## 🤝 Contributions

Pull requests and forks are welcome. If you find a bug or have a feature request, feel free to open an issue.
