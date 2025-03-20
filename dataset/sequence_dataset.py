import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os

from dataset.dataset import (
    FEATURE_COLS_SIR,
    FEATURE_COLS_INTERVENTIONS,
    FEATURE_COLS_STATIC,
    LABEL_COLS,
)


class SequenceDataset(Dataset):
    def __init__(self, file_indices, data_directory, is_deltas=False):
        self.sequences = []
        self.sequence_lengths = []

        # Load all data files based on indices and organize by sequence
        for idx in file_indices:
            # Load file
            file_path = os.path.join(data_directory, f"{idx + 1}.csv")
            df = pd.read_csv(file_path)

            # Convert to numpy arrays
            sir_seq = df[FEATURE_COLS_SIR].values.astype(np.float32)
            interventions_seq = df[FEATURE_COLS_INTERVENTIONS].values.astype(np.float32)
            static_seq = df[FEATURE_COLS_STATIC].values.astype(np.float32)

            # For static features, they're the same for the entire sequence,
            # so we can just repeat the first row
            static_seq = np.repeat(static_seq[0:1], len(sir_seq), axis=0)

            # Get labels (next SIR state)
            labels_seq = df[LABEL_COLS].values.astype(np.float32)

            # If using deltas, compute the changes
            if is_deltas:
                for i, (label_col, feature_col) in enumerate(
                    zip(LABEL_COLS, FEATURE_COLS_SIR)
                ):
                    col_idx = i % 6  # Column index within the feature group
                    labels_seq[:, col_idx] = (
                        labels_seq[:, col_idx] - sir_seq[:, col_idx]
                    )

            # Store this sequence
            self.sequences.append(
                {
                    "sir": sir_seq,
                    "interventions": interventions_seq,
                    "static": static_seq,
                    "labels": labels_seq,
                }
            )
            self.sequence_lengths.append(len(sir_seq))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        return (
            torch.tensor(sequence["sir"]),
            torch.tensor(sequence["interventions"]),
            torch.tensor(sequence["static"]),
            torch.tensor(sequence["labels"]),
        )
