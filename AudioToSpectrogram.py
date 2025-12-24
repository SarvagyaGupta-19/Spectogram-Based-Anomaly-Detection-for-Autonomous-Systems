import os
import librosa
import numpy as np
import cv2
from tqdm import tqdm

# -------- PATHS --------
INPUT_FOLDER = "C:\\Users\\praka\\OneDrive\\Desktop\\Machine Learning\\Deep-Learning\\ImageToSpectrogramPrediction\\gearbox\\train"
OUTPUT_FOLDER = "C:\\Users\\praka\\OneDrive\\Desktop\\Machine Learning\\Deep-Learning\\ImageToSpectrogramPrediction\\SpectrogramImage\\train"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -------- PARAMETERS --------
SAMPLE_RATE = 22050
N_MELS = 128
FMAX = 8000

def audio_to_mel_spectrogram(audio_path, save_path):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=N_MELS,
        fmax=FMAX
    )

    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Normalize to 0-255 for image saving
    mel_norm = cv2.normalize(mel_db, None, 0, 255, cv2.NORM_MINMAX)
    mel_norm = mel_norm.astype(np.uint8)

    cv2.imwrite(save_path, mel_norm)

# -------- PROCESS ALL FILES --------
audio_files = [
    f for f in os.listdir(INPUT_FOLDER)
    if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg"))
]

for file in tqdm(audio_files, desc="Converting Audio → Spectrogram"):
    input_path = os.path.join(INPUT_FOLDER, file)
    output_path = os.path.join(
        OUTPUT_FOLDER,
        os.path.splitext(file)[0] + ".png"
    )

    audio_to_mel_spectrogram(input_path, output_path)

print("✅ All spectrograms saved successfully!")
