import librosa

def extract_mfcc(audio, sr, n_mfcc=10):
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc
    )

    print("MFCC Shape:", mfcc.shape)

    return mfcc