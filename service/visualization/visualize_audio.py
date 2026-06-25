import matplotlib.pyplot as plt

def visualize_audio(audio,title):
    """
    Plot the waveform of an audio signal.

    Args:
        audio (numpy.ndarray): Audio samples.
    """
    plt.figure(figsize=(12, 4))
    plt.plot(audio)

    plt.title(title)
    plt.xlabel("Sample Number")
    plt.ylabel("Amplitude")

    plt.grid(True)
    plt.show()