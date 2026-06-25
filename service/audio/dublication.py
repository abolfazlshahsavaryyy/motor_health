import numpy as np
import librosa
def add_noise(audio, noise_factor=0.005):
    noise = np.random.randn(len(audio))
    return audio + noise_factor * noise

def shift_audio(audio, shift_max=0.2):
    shift = np.random.randint(
        int(len(audio) * shift_max)
    )

    return np.roll(audio, shift)

def change_volume(audio, factor):
    return audio * factor



def generate_augmented_samples(audio, sr):

    samples = [audio.copy()]

    # 5 noisy versions
    for _ in range(5):

        noise_factor = np.random.uniform(
            0.001,
            0.02
        )

        augmented = add_noise(
            audio.copy(),
            noise_factor=noise_factor
        )

        samples.append(augmented)

    # 5 shifted versions
    for _ in range(5):

        shift_max = np.random.uniform(
            0.05,
            0.3
        )

        augmented = shift_audio(
            audio.copy(),
            shift_max=shift_max
        )

        samples.append(augmented)

    # 5 volume changed versions
    for _ in range(5):

        factor = np.random.uniform(
            0.7,
            1.3
        )

        augmented = change_volume(
            audio.copy(),
            factor=factor
        )

        samples.append(augmented)

    return samples