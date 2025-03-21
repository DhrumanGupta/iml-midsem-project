import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

IS_PYTORCH = True
device = "cuda" if torch.cuda.is_available() else "cpu"
AUTOREGRESSIVE = True


class Model(nn.Module):
    def __init__(
        self,
        input_size,
        is_deltas,
        hidden_size=128,
        num_layers=8,
        dropout=0.2,
    ):
        super(Model, self).__init__()
        self.is_deltas = is_deltas
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM expects input shape: [batch, seq_len, input_size]
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 6),  # 6 outputs: S, I, R for students and adults
        )

        self.to(device)

    def forward(self, x_sir, x_interventions, x_static):
        # Old (pointwise) mode: x_sir is [batch, features]
        if x_sir.dim() == 2:
            x = torch.cat((x_sir, x_interventions, x_static), dim=1).to(device)
            x = x.unsqueeze(1)  # Make sequence length = 1
        # Sequence mode: x_sir is [batch, seq_len, features]
        elif x_sir.dim() == 3:
            x = torch.cat((x_sir, x_interventions, x_static), dim=2).to(device)
        else:
            raise ValueError("Invalid input dimensions")

        lstm_out, _ = self.lstm(x)
        lstm_last = lstm_out[:, -1, :]  # use the output from the last time step

        logits = self.fc(lstm_last)
        students_logits = logits[:, :3]
        adults_logits = logits[:, 3:]

        if self.is_deltas:
            students_probs = students_logits
            adults_probs = adults_logits
        else:
            students_logits = torch.sigmoid(students_logits)
            adults_logits = torch.sigmoid(adults_logits)
            students_probs = students_logits / students_logits.sum(dim=1, keepdim=True)
            adults_probs = adults_logits / adults_logits.sum(dim=1, keepdim=True)

        res = torch.cat((students_probs, adults_probs), dim=1).to(device)
        return res

    def init_hidden(self, batch_size):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return (h0, c0)


def train_model(model, train_loader, val_loader, num_epochs):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0

        train_pbar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False
        )
        for batch in train_pbar:
            x_sir, x_interventions, x_static, labels = batch
            x_sir = x_sir.to(device)
            x_interventions = x_interventions.to(device)
            x_static = x_static.to(device)

            # For autoregressive training, we need sequence data
            seq_length = x_sir.shape[1] if x_sir.dim() == 3 else 1

            # Reset gradients once per batch
            optimizer.zero_grad()

            # Initialize total loss for this batch
            batch_loss = torch.tensor(0.0, device=device, requires_grad=True)

            # For each step in the sequence
            for t in range(seq_length - 1):
                # Get current timestep data
                curr_x_sir = x_sir[:, t : t + 1, :] if seq_length > 1 else x_sir
                curr_x_interventions = (
                    x_interventions[:, t : t + 1, :]
                    if seq_length > 1
                    else x_interventions
                )
                curr_x_static = (
                    x_static[:, t : t + 1, :] if seq_length > 1 else x_static
                )

                # Get next timestep's ground truth
                next_sir = x_sir[:, t + 1, :] if seq_length > 1 else labels

                # Make prediction for next timestep
                preds = model(curr_x_sir, curr_x_interventions, curr_x_static)

                # Calculate loss against next timestep's values
                step_loss = criterion(preds, next_sir)

                # Accumulate loss for this timestep
                batch_loss = batch_loss + step_loss

            # Backpropagate the accumulated loss
            # if seq_length > 1:
            #     batch_loss = batch_loss / float(
            #         seq_length - 1
            #     )  # Ensure we keep tensor type
            batch_loss.backward()
            optimizer.step()

            running_train_loss += batch_loss.item()
            train_pbar.set_postfix({"loss": f"{batch_loss.item():.4f}"})

        avg_train_loss = running_train_loss / len(train_loader)
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            val_pbar = tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False
            )
            for batch in val_pbar:
                x_sir, x_interventions, x_static, labels = batch
                x_sir = x_sir.to(device)
                x_interventions = x_interventions.to(device)
                x_static = x_static.to(device)

                # Same autoregressive approach for validation
                batch_size = x_sir.shape[0]
                seq_length = x_sir.shape[1] if x_sir.dim() == 3 else 1

                batch_loss = torch.tensor(0.0, device=device, requires_grad=True)

                for t in range(seq_length - 1):
                    curr_x_sir = x_sir[:, t : t + 1, :] if seq_length > 1 else x_sir
                    curr_x_interventions = (
                        x_interventions[:, t : t + 1, :]
                        if seq_length > 1
                        else x_interventions
                    )
                    curr_x_static = (
                        x_static[:, t : t + 1, :] if seq_length > 1 else x_static
                    )

                    next_sir = x_sir[:, t + 1, :] if seq_length > 1 else labels

                    preds = model(curr_x_sir, curr_x_interventions, curr_x_static)
                    step_loss = criterion(preds, next_sir)
                    batch_loss = batch_loss + step_loss

                # if seq_length > 1:
                #     batch_loss = batch_loss / float(
                #         seq_length - 1
                #     )  # Ensure we keep tensor type
                running_val_loss += batch_loss.item()
                val_pbar.set_postfix({"loss": f"{batch_loss.item():.4f}"})

        avg_val_loss = running_val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        yield avg_train_loss, avg_val_loss, epoch


def predict(model, x_sir, x_interventions, x_static) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        # Convert numpy arrays to tensors if not already
        if not torch.is_tensor(x_sir):
            x_sir = torch.tensor(x_sir, dtype=torch.float32)
        if not torch.is_tensor(x_interventions):
            x_interventions = torch.tensor(x_interventions, dtype=torch.float32)
        if not torch.is_tensor(x_static):
            x_static = torch.tensor(x_static, dtype=torch.float32)
        x_sir = x_sir.to(device)
        x_interventions = x_interventions.to(device)
        x_static = x_static.to(device)
        preds = model(x_sir, x_interventions, x_static)
        return preds.cpu().numpy()


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location=device))
