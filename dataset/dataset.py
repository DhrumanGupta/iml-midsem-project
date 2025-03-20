import numpy as np
from torch.utils.data import Dataset

FEATURE_COLS_SIR = [
    "S_Students",
    "I_Students",
    "R_Students",
    "S_Adults",
    "I_Adults",
    "R_Adults",
]

FEATURE_COLS_INTERVENTIONS = [
    "School_Lockdown_Intensity",
    "Office_Lockdown_Intensity",
]

FEATURE_COLS_STATIC = [
    "Adult_Ratio",
    "Student_Ratio",
    "Home_Size",
    "School_Size",
    "Work_Size",
]

LABEL_COLS = [
    "Label_S_Students",
    "Label_I_Students",
    "Label_R_Students",
    "Label_S_Adults",
    "Label_I_Adults",
    "Label_R_Adults",
]

FEATURE_SIZE = (
    len(FEATURE_COLS_SIR) + len(FEATURE_COLS_INTERVENTIONS) + len(FEATURE_COLS_STATIC)
)


class SimulationDataset(Dataset):
    def __init__(self, dataframe, sequence_length: int = 1):
        assert sequence_length >= 1, "sequence_length must be at least 1"
        self.sequence_length = sequence_length
        # Reset index to preserve sequential order
        self.df = dataframe.reset_index(drop=True)
        # Precompute numpy arrays for faster access
        self.X_sir = self.df[FEATURE_COLS_SIR].values.astype(np.float32)
        self.X_interventions = self.df[FEATURE_COLS_INTERVENTIONS].values.astype(
            np.float32
        )
        self.X_static = self.df[FEATURE_COLS_STATIC].values.astype(np.float32)
        self.Y = self.df[LABEL_COLS].values.astype(np.float32)

    def __len__(self):
        # With sequence mode, we use a sliding window; note that the last available index is len(df)-sequence_length.
        if self.sequence_length == 1:
            return len(self.df)
        else:
            return len(self.df) - self.sequence_length

    def __getitem__(self, idx):
        if self.sequence_length == 1:
            return (
                self.X_sir[idx],
                self.X_interventions[idx],
                self.X_static[idx],
                self.Y[idx],
            )
        else:
            # Return a window (of any chosen length) for each modality.
            x_sir_seq = self.X_sir[idx : idx + self.sequence_length]
            x_interventions_seq = self.X_interventions[idx : idx + self.sequence_length]
            x_static_seq = self.X_static[idx : idx + self.sequence_length]
            # The target is always the next row after the window.
            target = self.Y[idx + self.sequence_length]
            return (x_sir_seq, x_interventions_seq, x_static_seq, target)
