

# 🐱🐶 Pet Breed Classifier

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![Gradio](https://img.shields.io/badge/Gradio-4.x-green)](https://www.gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modern image classification project that identifies **37 different cat and dog breeds** from photos using **transfer learning** with **MobileNetV2** on the Oxford-IIIT Pet dataset.

Achieves **~92–95% validation accuracy** with fine-tuning — far better than training from scratch!

## ✨ Features

- Transfer learning using **MobileNetV2** pre-trained on ImageNet
- Data loading & augmentation via **TensorFlow Datasets** (`oxford_iiit_pet:3.*.*`)
- Efficient training pipeline with caching, prefetching & early stopping
- Interactive web demo with **Gradio** (upload photo → get top-5 breed predictions)
- Clean, modular notebook structure for easy experimentation
- Ready for deployment on **Hugging Face Spaces** (permanent public link)

## 📊 Dataset

- **Oxford-IIIT Pet Dataset** (via `tensorflow_datasets`)
- 37 fine-grained classes (cat & dog breeds)
- ~3,680 training images + ~3,669 test images
- Real-world photos with varying poses, lighting, backgrounds

## 🛠️ Tech Stack

- Python 3.8+
- TensorFlow 2.x + Keras
- TensorFlow Datasets
- Gradio (for interactive UI)
- MobileNetV2 (transfer learning backbone)
- NumPy, Matplotlib

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/YOUR-USERNAME/pet-breed-classifier.git
cd pet-breed-classifier
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

(or manually:)

```bash
pip install tensorflow tensorflow-datasets gradio matplotlib numpy
```

### 3. Run the notebook

Open `Image_Classification_Pets.ipynb` in Jupyter / Colab / VS Code

- Training takes ~10–30 minutes on a free Colab GPU (T4)
- After training → launch the Gradio demo at the bottom


## 📈 Results

| Model                  | Validation Accuracy | Training Time (GPU) | Notes                          |
|------------------------|----------------------|----------------------|--------------------------------|
| Custom small CNN       | ~55–65%             | Fast                | From scratch – poor on fine-grained task |
| MobileNetV2 (frozen)   | ~88–91%             | ~10–15 min          | Transfer learning head only    |
| MobileNetV2 + fine-tune| **~92–95%**         | ~20–40 min          | Best results – recommended     |

(Confusion matrix, sample predictions, and training curves are shown in the notebook.)

## 🖼️ Demo Screenshots


<p align="center">
  <img src="screenshots/Gradio Interface Cat.png" alt="BreedNet Gradio Interface" width="800">
</p>

<p align="center">
  <img src="screenshots/Gradio Interface Dog2.png" alt="BreedNet Gradio Interface" width="800">
</p>

## 📂 Project Structure

```
pet-breed-classifier/
├── Image_Classification_Pets.ipynb     # Main notebook (training + demo)
├── requirements.txt                    # Dependencies
├── model/                              # (optional) saved model files
│   └── mobilenetv2_pets.h5
├── screenshots/                        # Demo images
└── README.md
```


## 🚀 Deployment (Permanent Link)

The temporary Gradio link expires after ~72 hours.

For always-online access:

1. Create a **Hugging Face Space** → https://huggingface.co/new-space
2. Choose **Gradio** SDK
3. Convert notebook logic → `app.py` (load model + define predict + launch demo)
4. Add `requirements.txt`
5. Commit → auto-deploys in minutes


## 📜 License

MIT License – feel free to use, modify, and share!

## 🙏 Acknowledgments

- [Oxford-IIIT Pet Dataset](http://www.robots.ox.ac.uk/~vgg/data/pets/)
- [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/oxford_iiit_pet)
- [Gradio](https://www.gradio.app/)
- Inspired by transfer learning tutorials from TensorFlow & Hugging Face


Star ⭐ the repo if you find it helpful!
```

