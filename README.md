# ImageToSpectrogramPrediction

This project to convert audio to spectrogram images and train models that predict spectrograms from images (or vice-versa). This repository contains preprocessing utilities, dataset folders, and an example script `AudioToSpectrogram.py`.

## Features

- Convert audio files to spectrogram images
- Organized dataset folders for source/target and train/test
- Starter script for conversion and experimentation

## Repository structure

- `AudioToSpectrogram.py` - Example script to generate spectrograms from audio (project entrypoint)
- `gearbox/`
  - `train/` - training audio files
  - `source_test/` - source test audio files
  - `target_test/` - target test audio files
- `SpectrogramImage/`
  - `train/` - generated spectrogram images for training

## Requirements

- Python 3.8+
- Typical packages used in audio & ML workflows (install as needed):

```bash
pip install numpy matplotlib librosa pillow tqdm
# plus your ML framework (tensorflow or torch) as required
```

If you prefer a pinned set of dependencies, create a `requirements.txt` and install with `pip install -r requirements.txt`.

## Quick start

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install numpy matplotlib librosa pillow tqdm
```

2. Convert audio to spectrogram images (example):

```bash
python AudioToSpectrogram.py
```

Note: `AudioToSpectrogram.py` may accept arguments depending on your local edits. Open the file and follow the usage comments near its top.

## Dataset layout

- Place raw audio files to be converted inside `gearbox/train/` for training, and `gearbox/source_test/` / `gearbox/target_test/` for test examples.
- Generated spectrogram images should be saved under `SpectrogramImage/train/` (the project may include code that automatically writes here).

## Training & Evaluation

- Use the prepared spectrogram images as input/targets for your model.
- Training scripts are not included by default — adapt your own training loop using your preferred framework (TensorFlow / PyTorch).

## Tips

- Use `librosa` for audio loading and spectrogram generation.
- Normalize spectrograms before feeding them to a neural network.
- Keep train/test splits consistent between `gearbox/` and `SpectrogramImage/` folders.

## Contributing

Contributions welcome — please open issues or pull requests for improvements, bug fixes, or additional scripts (training, evaluation, or dataset utilities).


## Contact

If you have questions or want help extending this project, open an issue or contact the repository owner.
# ImageToSpectrogramPrediction

Project to convert audio and image inputs into spectrogram representations and to train/predict spectrogram images using an autoencoder-style model.

**Table of contents**
- Overview
- Repository structure
- Dataset layout
- Pretrained model
- Installation
- Usage
  - Generate spectrograms from audio
  - Train / Notebooks
  - Inference (use pretrained model)
- Model & training details
- Tips and troubleshooting
- Contributing
- License

**Overview**

This repo contains code and notebooks for working with spectrogram images: creating spectrograms from audio, training an image-to-spectrogram model autoencoder

The goal is to predict or reconstruct spectrogram images from input data (images or audio-derived inputs) using convolutional autoencoder approaches. The repository includes example datasets, training notebooks, a utility script to create spectrogram images from audio, and a pretrained model file.

**Repository structure**

- AudioToSpectrogram.py — utility script to convert audio files to spectrogram images (example usage in notebooks).
- spectrogram_autoencoder_section00.h5 — provided pretrained Keras model weights.
- train.ipynb — primary training notebook (data preparation, model definition, training loop).
- traditional_train.ipynb — alternate training notebook (traditional approach / variations).
- test.ipynb — quick testing / evaluation examples.
- SpectrogramImage/ — folder containing spectrogram image datasets (split folders inside).
- gearbox/ — another dataset collection (similar structure to SpectrogramImage).

Inside each dataset folder (e.g., `SpectrogramImage/` and `gearbox/`):
- `train/` — training images.
- `source_test/` — source inputs for test/validation.
- `target_test/` — ground-truth spectrograms for test/validation.

**Dataset layout and conventions**

- Input images (source) are typically single-channel (grayscale) spectrogram images or other image representations.
- Target images are spectrogram images to be predicted/reconstructed by the model.
- Filenames for source/target pairs should match where applicable so the notebooks' loader can pair them for supervised training.


**Installation**

Recommended: create a Python virtual environment and install the packages below. The project targets Python 3.8+.

Example (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install tensorflow numpy matplotlib librosa pillow opencv-python scikit-image h5py tqdm jupyter
```

If you prefer, install a single-line requirements list:

```powershell
pip install tensorflow numpy matplotlib librosa pillow opencv-python scikit-image h5py tqdm jupyter
```

Note: `tensorflow` will pull the correct CPU/GPU wheel if available. For GPU support on Windows, install the compatible CUDA/cuDNN for your TensorFlow version.

**Usage**

Generate spectrograms from audio
- Use `AudioToSpectrogram.py` or the notebooks to convert WAV (or other audio) files into spectrogram images. Notebooks include example conversion code and visualization.

Training (notebooks)
- Open `train.ipynb` or `traditional_train.ipynb` in Jupyter to run full training pipelines. The notebooks contain the data-loading, preprocessing, model definition, training loops, and checkpoints.

Inference (load pretrained model)

Example Python snippet to load the included model and predict on a single input image:

```python
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load model
model = load_model('spectrogram_autoencoder_section00.h5')

# Load a single grayscale image (example path)
img = Image.open('SpectrogramImage/source_test/example.png').convert('L')
arr = np.array(img).astype('float32') / 255.0

# reshape to (1, H, W, 1) — adjust depending on the model's expected input shape
arr = arr[np.newaxis, ..., np.newaxis]

# Predict
pred = model.predict(arr)

# Convert prediction back to image and save
pred_img = (pred[0, ..., 0] * 255.0).clip(0, 255).astype('uint8')
Image.fromarray(pred_img).save('predicted_spectrogram.png')
```

Adjust shape manipulation above if your model expects a different input size or channel order; the notebooks show exact preprocessing used during training.

**Model & training details**

- Architecture: convolutional encoder-decoder (autoencoder) — several Conv2D + Downsampling layers in the encoder and mirrored Conv2DTranspose / Upsampling layers in the decoder.
- Loss: mean squared error (MSE) between predicted and ground-truth spectrogram images.
- Optimizer: Adam (commonly used; initial LR ~1e-3 or 1e-4 for fine-tuning).
- Metrics: visual inspection, PSNR, or structural similarity (SSIM) can be useful for evaluation.

Training tips
- Normalize input and target images to [0, 1] (float32) before training.
- Use data augmentation where helpful (small random crops, flips, time/frequency augmentations for audio-derived spectrograms).
- Start with small batches and reduce LR if the validation loss plateaus.

**Tips and troubleshooting**

- Out-of-memory (OOM) errors: lower `batch_size` or reduce image resolution.
- If predictions look noisy, try training longer, adding regularization, or using a lower learning rate.
- Ensure the image preprocessing during inference exactly matches the preprocessing in the notebooks (resizing, scaling, channel order).

**Contributing**

Contributions are welcome. Good first steps:
- Add `requirements.txt` listing exact versions used for reproducibility.
- Provide small sample audio files and expected spectrograms in a `samples/` folder for quick testing.
- Add a small script or CLI wrapper around `AudioToSpectrogram.py` to make conversion easier.

**License**

This repository does not include a license file. If you plan to share or publish, add a `LICENSE` file describing the intended license.

---

If you'd like, I can:
- add a `requirements.txt` with pinned versions,
- add a small inference script `predict.py` that wraps the load/predict/save snippet,
- or update one of the notebooks to include a step-by-step quickstart using the included pretrained model.# [Project Name] – Audio-to-Spectrogram Anomaly Prediction

> **Deep Learning Pipeline for Detecting Anomalies in Audio Through Spectrogram Representations**

This project converts raw audio signals into spectrogram images and applies deep-learning–based autoencoder models to **detect anomalies** through reconstruction error & Mahalanobis distance–based score evaluation. The system is designed for applications such as fault detection in machinery, environmental monitoring, equipment sound diagnostics, and pattern irregularity discovery in audio data.

---

## 🚀 Key Features

* **Audio → Spectrogram conversion** using Short-Time Fourier Transform (STFT)
* **Spectrogram preprocessing pipeline**: grayscale conversion, resizing, normalization
* **Autoencoder training pipeline** to learn normal patterns
* **Mahalanobis-based anomaly scoring** for robust irregularity detection
* **Visualization of anomaly clusters** using KMeans
* **3-level severity classification**: Normal, Medium anomaly, Severe anomaly
* Modular, extensible **data and model structure**

---



---

## 🔄 Data Flow Pipeline

```
Raw Audio (.wav, .mp3)
        │
        ▼
Short-Time Fourier Transform (STFT)
        │
        ▼
Spectrogram Matrix → Image Conversion
        │
        ▼
Grayscale Normalized Spectrograms
        │                 │
        │                 └─► Test Data → Autoencoder Reconstruction → Error
        ▼
Autoencoder Training (Normal Patterns)
        │
        ▼
Latent Space Extraction → PCA → Mahalanobis Score
        │
        ▼
KMeans / Thresholding → 3-level Classification
```

---

## 🧹 Preprocessing Steps

| Step          | Operation                   | Purpose                         |
| ------------- | --------------------------- | ------------------------------- |
| Load audio    | `librosa.load()`            | signal extraction               |
| STFT          | `librosa.stft()`            | frequency-time representation   |
| Convert to dB | `librosa.amplitude_to_db()` | visual clarity                  |
| Resize        | `cv2.resize()`              | model input spatial consistency |
| Normalize     | `img/255.0`                 | training stability              |
| Expand dims   | `(H,W,1)`                   | channel format for CNN          |

---

## 🧠 Model Architecture (Autoencoder)

```
Input → Conv2D → MaxPool → Conv2D → MaxPool → Latent Space
        ↓                                           ↓
   Conv2DTranspose ← Upsample ← Conv2DTranspose ← Output
```

* Loss function: **Mean Squared Error (MSE)**
* Optimizer: **Adam**, lr=1e-3
* Output: reconstructed spectrogram image

---

## 📊 Anomaly Scoring Logic

| Metric                   | Description                  | Notes                     |
| ------------------------ | ---------------------------- | ------------------------- |
| Reconstruction error     | pixel-wise diff              | unstable alone            |
| **Mahalanobis Distance** | distance in latent PCA space | **strong anomaly signal** |
| PCA reduction            | dimensionality stabilization | avoids covariance issues  |

Anomaly range example from dataset:

```
Normal: 3.7 – 9.0
Medium anomaly: 9.0 – 15.0
Severe anomaly: >15.0
```

---

## 📈 Cluster Visualization

Cluster centers help determine anomaly separation:

```
Cluster 0 → Low (Normal)
Cluster 1 → Mid  (Medium Anomaly)
Cluster 2 → High (Severe Anomaly)
```

Visualized via 1D scatter of scores vs. cluster labels.

---

## 🔎 Evaluation Example

```
Image: section_00_target_test_anomaly.png
Anomaly Score: 18.62
Classification: SEVERE ANOMALY.
```

---

## 🧪 Usage Guide

**Convert audio → spectrogram**

```
python scripts/AudioToSpectrogram.py
```

**Train model** — via notebook

```
notebooks/train.ipynb
```

**Run inference**

```
notebooks/test.ipynb
```

---

## 📌 Future Enhancements

* Add GRU-based audio domain prediction
* Integrate real-time anomaly alert system
* Add attention bottleneck for saliency mapping
* Deploy with FastAPI + stream inference

---

## 🏷 Hashtags

```
#DeepLearning #AudioAnalysis #Spectrogram #AnomalyDetection #Mahalanobis #Autoencoder #MachineLearning #AudioProcessing
```

---

## 🤝 Contributing

Contributions are welcome — submit PRs for model improvements, dataset scripts, visualization tools, or deployment setups.

---

## 👨‍💻 Maintainer

Feel free to open issues for collaboration, debugging, or improvements.


File updated: [README.md](README.md)
