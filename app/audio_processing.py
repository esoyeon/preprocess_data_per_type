import librosa
import soundfile as sf
import numpy as np
import io
import matplotlib.pyplot as plt
import base64


def resample_audio(
    audio_data: np.ndarray, original_rate: int, target_rate: int = 44100
) -> np.ndarray:
    """Resample audio to target sampling rate"""
    if original_rate == target_rate:
        return audio_data

    duration = len(audio_data) / original_rate
    new_length = int(duration * target_rate)
    return np.interp(
        np.linspace(0, duration, new_length),
        np.linspace(0, duration, len(audio_data)),
        audio_data,
    )


def extract_mfcc_features(audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
    """Extract MFCC features from audio"""
    # Simple implementation of MFCC-like features
    # Using short-time Fourier transform
    frame_length = int(0.025 * sample_rate)  # 25ms frames
    hop_length = int(0.010 * sample_rate)  # 10ms hop

    # Compute spectrogram
    frames = np.array(
        [
            audio_data[i : i + frame_length]
            for i in range(0, len(audio_data) - frame_length, hop_length)
        ]
    )
    windows = np.hanning(frame_length)
    windowed_frames = frames * windows

    # Compute FFT
    fft_frames = np.fft.rfft(windowed_frames)
    return np.abs(fft_frames)


def is_wav_file(audio_bytes: bytes) -> bool:
    """Check if the audio file is in WAV format"""
    try:
        with io.BytesIO(audio_bytes) as audio_io:
            with wave.open(audio_io, "rb") as wav_file:
                return True
    except:
        return False


def preprocess_audio(audio_bytes: bytes) -> tuple:
    """
    Preprocess the audio and generate MFCC spectrogram:
    1. Load and convert to mono
    2. Resample to standard rate
    3. Noise reduction
    4. Normalize
    5. Generate MFCC spectrogram
    """
    try:
        # Load audio from bytes
        audio_io = io.BytesIO(audio_bytes)
        y, sr = librosa.load(audio_io, mono=True, sr=22050)  # Resample to 22050Hz
        
        # Noise reduction using spectral gating
        y_denoised = librosa.effects.preemphasis(y)
        
        # Normalize audio
        y_normalized = librosa.util.normalize(y_denoised)
        
        # Generate MFCC spectrogram
        mfccs = librosa.feature.mfcc(y=y_normalized, sr=sr, n_mfcc=13)
        
        # Create spectrogram image
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfccs, sr=sr, x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title('MFCC Spectrogram')
        
        # Save plot to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        spectrogram_bytes = buf.getvalue()
        
        # Convert processed audio to bytes
        output_io = io.BytesIO()
        sf.write(output_io, y_normalized, sr, format='WAV')
        
        return output_io.getvalue(), spectrogram_bytes
    
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        return audio_bytes, None


def bandpass_filter(y: np.ndarray, sr: int, lowcut: int, highcut: int) -> np.ndarray:
    """Apply bandpass filter to keep only frequencies between lowcut and highcut"""
    nyquist = sr // 2
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Get window length (odd number)
    window_length = 2049
    
    # Create bandpass filter
    b = librosa.filters.get_window('hann', window_length)
    
    # Apply filter
    y_filtered = librosa.stft(y)
    freq_bins = librosa.fft_frequencies(sr=sr, n_fft=window_length)
    
    mask = np.logical_and(freq_bins >= lowcut, freq_bins <= highcut)
    y_filtered[~mask] = 0
    
    return librosa.istft(y_filtered)


def apply_dynamic_range_compression(y: np.ndarray, threshold_db: float = -20.0, ratio: float = 4.0) -> np.ndarray:
    """Apply dynamic range compression"""
    # Convert to dB
    y_db = librosa.amplitude_to_db(np.abs(y))
    
    # Apply compression
    mask = y_db > threshold_db
    y_db[mask] = threshold_db + (y_db[mask] - threshold_db) / ratio
    
    # Convert back to amplitude
    return librosa.db_to_amplitude(y_db) * np.sign(y)
