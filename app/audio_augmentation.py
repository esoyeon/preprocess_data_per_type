import librosa
import soundfile as sf
import numpy as np
import io
import random
import matplotlib.pyplot as plt


def time_stretch(y: np.ndarray, rate: float = None) -> np.ndarray:
    """Time stretching"""
    if rate is None:
        rate = random.uniform(0.8, 1.2)
    return librosa.effects.time_stretch(y, rate=rate)


def pitch_shift(y: np.ndarray, sr: int, steps: int = None) -> np.ndarray:
    """Pitch shifting"""
    if steps is None:
        steps = random.randint(-4, 4)
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=steps)


def add_noise(y: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
    """Add background noise"""
    noise = np.random.randn(len(y))
    return y + noise_factor * noise


def augment_audio(audio_bytes: bytes) -> tuple:
    """
    Augment the audio using various techniques:
    1. Time stretching
    2. Pitch shifting
    3. Adding noise
    """
    try:
        # Load audio from bytes
        audio_io = io.BytesIO(audio_bytes)
        y, sr = librosa.load(audio_io)
        
        # Apply augmentations
        y = time_stretch(y)
        y = pitch_shift(y, sr)
        y = add_noise(y)
        
        # Generate MFCC spectrogram
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Create spectrogram image
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfccs, sr=sr, x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Augmented MFCC Spectrogram')
        
        # Save plot to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        spectrogram_bytes = buf.getvalue()
        
        # Convert audio to bytes
        output_io = io.BytesIO()
        sf.write(output_io, y, sr, format='WAV')
        
        return output_io.getvalue(), spectrogram_bytes
    
    except Exception as e:
        print(f"Error augmenting audio: {str(e)}")
        return audio_bytes, None
