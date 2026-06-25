import librosa
import numpy as np

def load_single_audio(audio_path, target_sr=16000, target_seconds=10):
    """
    Load an audio file and make it a fixed length.

    Args:
        audio_path (str): Path to the audio file.
        target_sr (int): Target sample rate.
        target_seconds (int): Desired duration in seconds.

    Returns:
        tuple: (audio, sample_rate)
    """

    # Load and resample
    audio, sr = librosa.load(audio_path, sr=target_sr)

    target_length = target_sr * target_seconds

    # If audio is longer than target length, cut it
    if len(audio) > target_length:
        audio = audio[:target_length]

    # If audio is shorter, repeat it
    elif len(audio) < target_length:
        repeat_count = int(np.ceil(target_length / len(audio)))

        audio = np.tile(audio, repeat_count)
        audio = audio[:target_length]

    print("Type:", type(audio))
    print("Shape:", audio.shape)
    print("Sample Rate:", sr)
    print("Duration:", len(audio) / sr)

    return audio, sr