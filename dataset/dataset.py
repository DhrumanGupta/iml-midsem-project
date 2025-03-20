from torch.utils.data import Dataset
import numpy as np


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


# Custom Dataset to load the DataFrame
class SimulationDataset(Dataset):
    def __init__(self, dataframe):
        # Define the feature and label columns
        self.X_sir = dataframe[FEATURE_COLS_SIR].values.astype(np.float32)
        self.X_interventions = dataframe[FEATURE_COLS_INTERVENTIONS].values.astype(
            np.float32
        )
        self.X_static = dataframe[FEATURE_COLS_STATIC].values.astype(np.float32)
        self.Y = dataframe[LABEL_COLS].values.astype(np.float32)

    def __len__(self):
        return len(self.X_sir)

    def __getitem__(self, idx):
        return (
            self.X_sir[idx],
            self.X_interventions[idx],
            self.X_static[idx],
            self.Y[idx],
        )
