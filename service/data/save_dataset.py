import os
import numpy as np
import pandas as pd

from service.audio.read_audio import load_single_audio
from service.processing.audio_processing import extract_mfcc


def save_dataset_to_csv(dataset,
                        output_path="data/engine_dataset.csv"):

    if len(dataset) == 0:
        print("Dataset is empty.")
        return

    feature_count = len(dataset[0]) - 1

    columns = [
        f"feature_{i}"
        for i in range(feature_count)
    ]

    columns.append("label")

    df = pd.DataFrame(
        dataset,
        columns=columns
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(f"Saved dataset to {output_path}")
    print(f"Dataset shape: {df.shape}")
