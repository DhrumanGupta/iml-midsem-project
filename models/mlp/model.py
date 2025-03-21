import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from tqdm import tqdm

IS_PYTORCH = True
AUTOREGRESSIVE = False
device = "cpu"


class Model(nn.Module):
    def __init__(
        self,
        input_size,
        is_deltas,
        config={
            "hidden_sizes": [256, 512, 1024, 512, 512, 256],
            "lr": 0.01,
        },
    ):
        if config is None or "hidden_sizes" not in config:
            raise ValueError("hidden_sizes must be provided in config")

        if not config.get("lr"):
            config["lr"] = 0.01

        super(Model, self).__init__()

        # Create a list to hold all layers
        layers = []

        self.is_deltas = is_deltas

        # Input layer
        prev_size = input_size

        self.lr = config["lr"]

        # Add hidden layers
        for hidden_size in config["hidden_sizes"]:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))  # Add batch normalization
            prev_size = hidden_size

        # Store layers as ModuleList
        self.shared_layers = nn.ModuleList(layers)

        # Two separate heads for students and adults
        # self.fc_students = nn.Linear(hidden_sizes[-1], 3)
        # self.fc_adults = nn.Linear(hidden_sizes[-1], 3)

        self.fc = nn.Linear(config["hidden_sizes"][-1], 6)

        self.to(device)

    def forward(self, x_sir, x_interventions, x_static):
        x = torch.cat((x_sir, x_interventions, x_static), dim=1).to(device)

        # Pass through all shared layers
        for layer in self.shared_layers:
            x = layer(x)

        # Get outputs for each group
        logits = self.fc(x)
        students_logits = logits[:, :3]
        adults_logits = logits[:, 3:]

        if self.is_deltas:
            students_probs = students_logits
            adults_probs = adults_logits
        else:
            students_logits = nn.functional.sigmoid(students_logits)
            adults_logits = nn.functional.sigmoid(adults_logits)

            students_probs = students_logits / students_logits.sum(dim=1, keepdim=True)
            adults_probs = adults_logits / adults_logits.sum(dim=1, keepdim=True)

        res = torch.cat((students_probs, adults_probs), dim=1).to(device)
        return res


def train_model(model, train_loader, val_loader, num_epochs):
    optimizer = optim.Adam(model.parameters(), lr=model.lr)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        # Add progress bar for training loop
        train_pbar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False
        )
        for x_sir, x_interventions, x_static, labels in train_pbar:
            x_sir = x_sir.to(device)
            x_interventions = x_interventions.to(device)
            x_static = x_static.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            preds = model(x_sir, x_interventions, x_static)

            loss = criterion(preds, labels)

            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            # Update progress bar with current loss
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        avg_train_loss = running_train_loss / len(train_loader)

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            # Add progress bar for validation loop
            val_pbar = tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False
            )
            for x_sir, x_interventions, x_static, labels in val_pbar:
                # Move tensors to device
                x_sir = x_sir.to(device)
                x_interventions = x_interventions.to(device)
                x_static = x_static.to(device)
                labels = labels.to(device)

                preds = model(x_sir, x_interventions, x_static)

                loss = criterion(preds, labels)
                running_val_loss += loss.item()
                # Update progress bar with current loss
                val_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        avg_val_loss = running_val_loss / len(val_loader)

        yield avg_train_loss, avg_val_loss, epoch


def predict(model, x_sir, x_interventions, x_static) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x_sir = torch.tensor(x_sir, dtype=torch.float32).to(device)
        x_interventions = torch.tensor(x_interventions, dtype=torch.float32).to(device)
        x_static = torch.tensor(x_static, dtype=torch.float32).to(device)
        preds = model(x_sir, x_interventions, x_static)
        return preds.cpu().numpy()


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))
