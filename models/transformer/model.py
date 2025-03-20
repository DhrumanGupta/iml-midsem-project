import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

IS_PYTORCH = True


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: [batch, seq_len, d_model]
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class Model(nn.Module):
    def __init__(
        self,
        input_size,
        is_deltas,
        hidden_size=128,
        num_layers=4,
        num_heads=8,
        dropout=0.2,
    ):
        """
        Parameters:
          input_size: Total dimension after concatenating x_sir, x_interventions, and x_static.
          is_deltas: Whether to output raw deltas or normalized probabilities.
        """
        super(Model, self).__init__()
        self.is_deltas = is_deltas
        self.hidden_size = hidden_size

        # Project the concatenated input to the hidden dimension
        self.input_linear = nn.Linear(input_size, hidden_size)

        # Positional encoding (only used when seq_len > 1)
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)

        # Transformer encoder; batch_first=True expects input shape [batch, seq_len, hidden_size]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Fully connected layers: output 6 values (e.g., S, I, R for students and adults)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 6),
        )

        self.to(device)

    def forward(self, x_sir, x_interventions, x_static, return_sequence=False):
        """
        Parameters:
          x_sir, x_interventions, x_static:
              Inputs that are either 2D [batch, features] or 3D [batch, seq_len, features].
          return_sequence:
              If True and in sequence mode, return the predictions at every time step.
              Otherwise, return only the final prediction.
        """
        if x_sir.dim() == 2:
            # Pointwise mode: combine features and add sequence dimension of 1.
            x = torch.cat((x_sir, x_interventions, x_static), dim=1).to(device)
            x = self.input_linear(x)  # [batch, hidden_size]
            x = x.unsqueeze(1)  # [batch, 1, hidden_size]
            # No positional encoding or mask needed for a single timestep.
            x = self.transformer_encoder(x)
            x_out = self.fc(x)  # [batch, 1, 6]
        elif x_sir.dim() == 3:
            # Sequence mode: concatenate along the feature dimension.
            x = torch.cat((x_sir, x_interventions, x_static), dim=2).to(device)
            x = self.input_linear(x)  # [batch, seq_len, hidden_size]
            if x.size(1) > 1:
                x = self.pos_encoder(x)
                # Create a causal mask so each time step only attends to previous ones.
                seq_len = x.size(1)
                mask = torch.triu(
                    torch.full((seq_len, seq_len), float("-inf")), diagonal=1
                ).to(device)
                x = self.transformer_encoder(x, mask=mask)
            else:
                x = self.transformer_encoder(x)
            x_out = self.fc(x)  # [batch, seq_len, 6]
        else:
            raise ValueError("Invalid input dimensions")

        # Post-process outputs based on the is_deltas flag.
        if self.is_deltas:
            processed = x_out
        else:
            if x_out.dim() == 3:
                # Split into students and adults components.
                students_logits = torch.sigmoid(x_out[..., :3])
                adults_logits = torch.sigmoid(x_out[..., 3:])
                # Normalize along the last dimension (for each time step)
                students_probs = students_logits / (
                    students_logits.sum(dim=2, keepdim=True) + 1e-8
                )
                adults_probs = adults_logits / (
                    adults_logits.sum(dim=2, keepdim=True) + 1e-8
                )
                processed = torch.cat((students_probs, adults_probs), dim=2)
            else:
                students_logits = torch.sigmoid(x_out[:, :3])
                adults_logits = torch.sigmoid(x_out[:, 3:])
                students_probs = students_logits / (
                    students_logits.sum(dim=1, keepdim=True) + 1e-8
                )
                adults_probs = adults_logits / (
                    adults_logits.sum(dim=1, keepdim=True) + 1e-8
                )
                processed = torch.cat((students_probs, adults_probs), dim=1)

        if return_sequence:
            return processed
        else:
            # In sequence mode, return only the final prediction.
            return processed[:, -1, :] if processed.dim() == 3 else processed


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
            optimizer.zero_grad()

            # If in sequence mode, we use vectorized computation:
            if x_sir.dim() == 3:
                # Forward pass returning predictions at every time step.
                preds_seq = model(
                    x_sir, x_interventions, x_static, return_sequence=True
                )
                # Use outputs from time steps 0 to seq_len-2 to predict the next SIR state.
                preds = preds_seq[:, :-1, :]  # shape: [batch, seq_len-1, 6]
                targets = x_sir[:, 1:, :]  # shape: [batch, seq_len-1, 6]
                loss = criterion(preds, targets)
            else:
                # Pointwise mode: predict directly.
                preds = model(x_sir, x_interventions, x_static)
                loss = criterion(preds, labels)

            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        avg_train_loss = running_train_loss / len(train_loader)

        # Validation phase (vectorized similar to training)
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
                if x_sir.dim() == 3:
                    preds_seq = model(
                        x_sir, x_interventions, x_static, return_sequence=True
                    )
                    preds = preds_seq[:, :-1, :]
                    targets = x_sir[:, 1:, :]
                    loss = criterion(preds, targets)
                else:
                    preds = model(x_sir, x_interventions, x_static)
                    loss = criterion(preds, labels)
                running_val_loss += loss.item()
                val_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            avg_val_loss = running_val_loss / len(val_loader)
            scheduler.step(avg_val_loss)
        yield avg_train_loss, avg_val_loss, epoch


def predict(model, x_sir, x_interventions, x_static) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        # Convert inputs to tensors if needed.
        if not torch.is_tensor(x_sir):
            x_sir = torch.tensor(x_sir, dtype=torch.float32)
        if not torch.is_tensor(x_interventions):
            x_interventions = torch.tensor(x_interventions, dtype=torch.float32)
        if not torch.is_tensor(x_static):
            x_static = torch.tensor(x_static, dtype=torch.float32)
        x_sir = x_sir.to(device)
        x_interventions = x_interventions.to(device)
        x_static = x_static.to(device)
        # For prediction we return only the final output.
        preds = model(x_sir, x_interventions, x_static, return_sequence=False)
        return preds.cpu().numpy()


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location=device))
