import os
import numpy as np
import pandas as pd

from service.audio.read_audio import load_single_audio
from service.processing.audio_processing import extract_mfcc
from service.audio.dublication import generate_augmented_samples

def build_dataset():
    classes = {
        "Healthy": 1,
        "Bearing": 2,
        "Propeller": 3
    }

    dataset = []

    for class_name, label in classes.items():

        file_counter = 1

        while True:

            audio_path = f"data/{class_name}/{file_counter}.wav"

            if not os.path.exists(audio_path):
                print(f"Finished {class_name}")
                break

            print(f"Processing {audio_path}")

            try:
                audio, sr = load_single_audio(audio_path)
                augmented_audios = generate_augmented_samples(
                audio,
                sr
                )


                for aug_audio in augmented_audios:

                    mfcc = extract_mfcc(
                        aug_audio,
                        sr,
                        n_mfcc=13
                    )
                
                    features = mfcc.flatten()
                
                    row = features.tolist()
                    row.append(label)
                
                    dataset.append(row)

            except Exception as e:
                print(f"Error reading {audio_path}")
                print(e)

            file_counter += 1

    return dataset

