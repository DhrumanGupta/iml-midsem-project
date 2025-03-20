import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

IS_PYTORCH = True

device = "cuda" if torch.cuda.is_available() else "cpu"


class Model(nn.Module):
    def __init__(
        self,
        input_size,
        is_deltas,
        hidden_size=256,
        num_layers=4,
        dropout=0.2,
    ):
        super(Model, self).__init__()

        self.is_deltas = is_deltas
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Fully connected layer for output
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 6),  # 6 outputs: S, I, R for students and adults
        )

        self.to(device)

    def forward(self, x_sir, x_interventions, x_static, lengths=None):
        # Combine inputs - shape: [batch_size, seq_len, features]
        x = torch.cat((x_sir, x_interventions, x_static), dim=2).to(device)

        # Pack padded sequence if lengths are provided
        if lengths is not None:
            x_packed = torch.nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=True
            )

            # Pass through LSTM
            lstm_out, _ = self.lstm(x_packed)

            # Unpack sequence
            lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True
            )
        else:
            # If no lengths provided, just run through LSTM normally
            lstm_out, _ = self.lstm(x)

        # Reshape for batch processing through FC layers
        batch_size, seq_len, _ = lstm_out.size()
        lstm_out_reshaped = lstm_out.contiguous().view(
            batch_size * seq_len, self.hidden_size
        )

        # Pass through fully connected layers
        logits = self.fc(lstm_out_reshaped)

        # Reshape back
        logits = logits.view(batch_size, seq_len, -1)

        # Split outputs for students and adults
        students_logits = logits[:, :, :3]
        adults_logits = logits[:, :, 3:]

        if self.is_deltas:
            students_probs = students_logits
            adults_probs = adults_logits
        else:
            # Apply softmax to get probabilities
            students_logits_flat = students_logits.view(-1, 3)
            adults_logits_flat = adults_logits.view(-1, 3)

            students_logits_flat = torch.sigmoid(students_logits_flat)
            adults_logits_flat = torch.sigmoid(adults_logits_flat)

            students_probs_flat = students_logits_flat / students_logits_flat.sum(
                dim=1, keepdim=True
            )
            adults_probs_flat = adults_logits_flat / adults_logits_flat.sum(
                dim=1, keepdim=True
            )

            students_probs = students_probs_flat.view(batch_size, seq_len, 3)
            adults_probs = adults_probs_flat.view(batch_size, seq_len, 3)

        # Combine probabilities for students and adults
        probs = torch.cat((students_probs, adults_probs), dim=2)
        return probs

    def init_hidden(self, batch_size):
        # Initialize hidden state and cell state
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

        # Add progress bar for training loop
        train_pbar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False
        )

        for batch in train_pbar:
            x_sir, x_interventions, x_static, labels, lengths = batch

            # Move tensors to device
            x_sir = x_sir.to(device)
            x_interventions = x_interventions.to(device)
            x_static = x_static.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            optimizer.zero_grad()

            # Forward pass
            preds = model(x_sir, x_interventions, x_static, lengths)

            # Calculate loss - mask padded areas
            loss = 0
            batch_size = x_sir.size(0)
            for i in range(batch_size):
                seq_len = lengths[i]
                # Only compute loss on actual sequence (not padding)
                seq_preds = preds[i, :seq_len]
                seq_labels = labels[i, :seq_len]
                loss += criterion(seq_preds, seq_labels)
            loss = loss / batch_size  # Average loss across batch

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
            for batch in val_pbar:
                x_sir, x_interventions, x_static, labels, lengths = batch

                # Move tensors to device
                x_sir = x_sir.to(device)
                x_interventions = x_interventions.to(device)
                x_static = x_static.to(device)
                labels = labels.to(device)
                lengths = lengths.to(device)

                # Forward pass
                preds = model(x_sir, x_interventions, x_static, lengths)

                # Calculate loss
                loss = 0
                batch_size = x_sir.size(0)
                for i in range(batch_size):
                    seq_len = lengths[i]
                    # Only compute loss on actual sequence (not padding)
                    seq_preds = preds[i, :seq_len]
                    seq_labels = labels[i, :seq_len]
                    loss += criterion(seq_preds, seq_labels)
                loss = loss / batch_size  # Average loss across batch

                running_val_loss += loss.item()
                # Update progress bar with current loss
                val_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_val_loss = running_val_loss / len(val_loader)

        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)

        yield avg_train_loss, avg_val_loss, epoch


def predict(model, x_sir, x_interventions, x_static) -> np.ndarray:
    """
    Predict the entire sequence

    Args:
        x_sir: Initial SIR state (single time step) or sequence of SIR states
        x_interventions: Intervention parameters (sequence)
        x_static: Static parameters (sequence)
    """
    model.eval()
    with torch.no_grad():
        # Check if x_sir is a single time step or a sequence
        if len(x_sir.shape) == 2:  # Single time step [batch, features]
            # We need to predict step by step
            seq_len = x_interventions.shape[1]
            batch_size = x_interventions.shape[0]

            # Initialize output tensor to store all predictions
            outputs = np.zeros((batch_size, seq_len, 6))

            # Initial SIR state
            current_sir = torch.tensor(x_sir, dtype=torch.float32).to(device)

            # For each time step
            for t in range(seq_len):
                # Get current intervention and static
                current_int = torch.tensor(
                    x_interventions[:, t : t + 1], dtype=torch.float32
                ).to(device)
                current_static = torch.tensor(
                    x_static[:, t : t + 1], dtype=torch.float32
                ).to(device)

                # Make prediction for this step
                x = torch.cat(
                    (current_sir.unsqueeze(1), current_int, current_static), dim=2
                )
                pred = model.lstm(x)[0]

                # Process through fully connected layer
                pred = pred.view(-1, model.hidden_size)
                pred = model.fc(pred)

                # Split and process for students and adults
                students_logits = pred[:, :3]
                adults_logits = pred[:, 3:]

                if model.is_deltas:
                    students_probs = students_logits
                    adults_probs = adults_logits
                else:
                    students_logits = torch.sigmoid(students_logits)
                    adults_logits = torch.sigmoid(adults_logits)

                    students_probs = students_logits / students_logits.sum(
                        dim=1, keepdim=True
                    )
                    adults_probs = adults_logits / adults_logits.sum(
                        dim=1, keepdim=True
                    )

                pred = torch.cat((students_probs, adults_probs), dim=1)
                pred_np = pred.cpu().numpy()

                # Store prediction
                outputs[:, t] = pred_np

                # Update current_sir for next step
                current_sir = pred.detach()

            return outputs
        else:  # Full sequence [batch, sequence, features]
            # Convert inputs to tensors
            x_sir = torch.tensor(x_sir, dtype=torch.float32).to(device)
            x_interventions = torch.tensor(x_interventions, dtype=torch.float32).to(
                device
            )
            x_static = torch.tensor(x_static, dtype=torch.float32).to(device)

            # Make prediction for the entire sequence
            preds = model(x_sir, x_interventions, x_static)
            return preds.cpu().numpy()


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location=device))
