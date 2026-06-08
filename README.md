# 🌿 AI Crop Disease Detector

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)
![Gradio](https://img.shields.io/badge/Gradio-UI-FF7C00.svg)

> **🚀 Try the Live Demo:** [(https://huggingface.co/spaces/Aziz2209/Crop-Disease-Detection.1)]

An advanced, end-to-end Machine Learning web application designed to detect and diagnose plant diseases from leaf images. Built using a fine-tuned MobileNetV2 Convolutional Neural Network and a beautiful, custom-styled Gradio Blocks interface.

## ✨ Key Features
- **Highly Accurate Detection:** Classifies 15 different crop conditions (healthy and diseased) across Tomatoes, Potatoes, and Bell Peppers.
- **Out-Of-Distribution (OOD) Guard AI:** Uses a secondary mathematical entropy detector to automatically reject invalid uploads (like random objects, logos, and animals).
- **Beautiful Dashboard UI:** Custom-styled glassmorphism UI with side-by-side detection panels, dynamic severity coloring, and expandable treatment recommendations.
- **Live Analytics Tracker:** Automatically logs user scans into a lightweight CSV database to display real-time usage statistics and most detected diseases.

## 🛠️ Tech Stack
- **Deep Learning Framework:** PyTorch & Torchvision
- **Model Architecture:** MobileNetV2 (Transfer Learning from ImageNet)
- **Web Framework & UI:** Gradio (`gr.Blocks`), custom HTML/CSS
- **Data Processing:** Pillow, Torchvision Transforms

## 💻 How to Run Locally

1. Clone the repository:
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/crop-disease-detection.git
cd crop-disease-detection
```

2. Install the lightweight CPU dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```
Open `http://localhost:7860` in your web browser.

## 📈 Model Training
The model was fine-tuned on the PlantVillage dataset. Training code is available in `train.py`, featuring automated learning rate scheduling, dropout layers for regularization, and early stopping.

---
*Created as an advanced AI portfolio project.*
