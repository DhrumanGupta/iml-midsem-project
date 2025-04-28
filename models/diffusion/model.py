import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import math
from einops import rearrange

IS_PYTORCH = True
AUTOREGRESSIVE = False
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Helper Functions & Modules ---

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        # Input shape: [batch_size]
        # Output shape: [batch_size, dim]
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Linear(dim, dim_out)
        # Use LayerNorm for stability with potentially small batch sizes during prediction
        self.norm = nn.LayerNorm(dim_out)
        # self.norm = nn.GroupNorm(groups, dim_out) if groups > 1 and dim_out % groups == 0 else nn.LayerNorm(dim_out)
        self.act = nn.SiLU() # Swish activation

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            # Ensure scale and shift have the correct dimension for broadcasting
            # Expected shape: [batch_size, dim_out]
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    """Residual block with time embedding integration."""
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)) # Output scale and shift
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Linear(dim, dim_out) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            # Reshape time_emb to [batch_size, 2, dim_out] to separate scale and shift
            time_emb = rearrange(time_emb, 'b (c d) -> b c d', c=2)
            scale_shift = time_emb.unbind(dim=1) # Split into two tensors: [batch_size, dim_out]

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h) # No scale/shift on second block
        return h + self.res_conv(x)

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x) # Self-attention if context is None
        k = self.to_k(context)
        v = self.to_v(context)

        # Reshape for multi-head attention
        # Assume input x is [batch, features] -> needs sequence dim
        # For non-sequence data, treat feature dim as sequence of length 1? Or apply attention across features?
        # Let's assume we want attention across features, but MultiheadAttention expects sequence.
        # We can add a dummy sequence dimension.
        # x shape: [b, d_q] -> q shape: [b, inner_dim]
        # context shape: [b, d_c] -> k,v shape: [b, inner_dim]

        # Add sequence dim: [b, 1, d]
        q, k, v = map(lambda t: rearrange(t, 'b d -> b 1 d'), (q, k, v))

        # Split heads: [b, 1, h*d_h] -> [b, h, 1, d_h]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        # Attention scores: [b, h, 1, d_h] x [b, h, d_h, 1] -> [b, h, 1, 1]
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)

        # Weighted values: [b, h, 1, 1] x [b, h, 1, d_h] -> [b, h, 1, d_h]
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)

        # Combine heads: [b, h, 1, d_h] -> [b, 1, h*d_h]
        out = rearrange(out, 'b h n d -> b n (h d)')

        # Remove sequence dim: [b, 1, inner_dim] -> [b, inner_dim]
        out = rearrange(out, 'b 1 d -> b d')
        return self.to_out(out)

# --- Denoising Network with ResNet and Attention ---

class DenoisingNetwork(nn.Module):
    """Predicts noise using ResNet blocks and cross-attention for conditioning."""
    def __init__(self, data_dim, condition_dim, model_dim=64, dim_mults=(1, 2, 4), attn_heads=4, attn_dim_head=32):
        super().__init__()

        self.data_dim = data_dim
        self.condition_dim = condition_dim

        # --- Time Embedding ---
        time_dim = model_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(model_dim),
            nn.Linear(model_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # --- Condition Encoder ---
        # Process the condition vector into an embedding space suitable for attention context
        cond_emb_dim = model_dim * 4 # Can be different from time_dim
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, cond_emb_dim),
            nn.GELU(),
            nn.Linear(cond_emb_dim, cond_emb_dim)
        )

        # --- Main Network (U-Net like structure without down/upsampling for non-image data) ---
        dims = [model_dim, *map(lambda m: model_dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # Initial projection for the noisy data
        self.init_conv = nn.Linear(data_dim, model_dim)

        self.blocks = nn.ModuleList([])
        num_resolutions = len(in_out)

        # Build blocks
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.blocks.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_emb_dim=time_dim),
                ResnetBlock(dim_out, dim_out, time_emb_dim=time_dim),
                CrossAttention(query_dim=dim_out, context_dim=cond_emb_dim, heads=attn_heads, dim_head=attn_dim_head)
            ]))

        # Middle block (optional, can add more complexity here)
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = CrossAttention(query_dim=mid_dim, context_dim=cond_emb_dim, heads=attn_heads, dim_head=attn_dim_head)
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)

        # Final layers
        self.final_res_block = ResnetBlock(mid_dim, model_dim, time_emb_dim=time_dim)
        self.final_conv = nn.Linear(model_dim, data_dim)

    def forward(self, x_noisy, t, condition):
        # x_noisy: [batch, data_dim]
        # t: [batch]
        # condition: [batch, condition_dim]

        # 1. Embed time and condition
        time_emb = self.time_mlp(t) # [batch, time_dim]
        cond_emb = self.condition_encoder(condition) # [batch, cond_emb_dim]

        # 2. Initial projection of noisy data
        x = self.init_conv(x_noisy) # [batch, model_dim]

        # 3. Apply blocks
        # Since we don't have spatial dimensions/downsampling like a U-Net,
        # we just pass through the blocks sequentially. Skip connections are internal to ResNet blocks.
        for resnet1, resnet2, attn in self.blocks:
            x = resnet1(x, time_emb)
            x = resnet2(x, time_emb)
            x = attn(x, context=cond_emb) + x # Add residual connection for attention

        # 4. Middle blocks
        x = self.mid_block1(x, time_emb)
        x = self.mid_attn(x, context=cond_emb) + x # Add residual connection for attention
        x = self.mid_block2(x, time_emb)

        # 5. Final blocks
        x = self.final_res_block(x, time_emb)
        return self.final_conv(x) # Predict noise: [batch, data_dim]

# --- Diffusion Model Class ---

class Model(nn.Module):
    def __init__(self, input_size, is_deltas, config=None):
        super().__init__()

        # input_size = 15 (prev_sir(6) + interventions(2) + static(7))
        self.data_dim = 6 # Dimension of the data being diffused (current SIR state)
        # Condition includes previous SIR state + interventions + static features
        self.condition_dim = input_size # The condition is the full input vector

        default_config = {
            "timesteps": 100,
            "lr": 0.001,
            # Denoising Network specific config
            "model_dim": 64,
            "dim_mults": (1, 2, 4),
            "attn_heads": 4,
            "attn_dim_head": 32,
            # NOTE: Deprecated keys from previous MLP based denoiser
            # "denoising_hidden_sizes": [128, 256, 256, 128],
            # "time_cond_emb_dim": 128,
        }
        if config is None:
            config = default_config
        else:
            # Merge defaults, ensuring new keys exist even if old config is passed
            config = {**default_config, **config}

        self.input_size = input_size # Keep original input_size if needed elsewhere
        self.is_deltas = is_deltas
        self.timesteps = config["timesteps"]
        self.lr = config["lr"]
        self.config = config # Store config

        # Initialize the refactored Denoising Network
        self.denoising_net = DenoisingNetwork(
            data_dim=self.data_dim,
            condition_dim=self.condition_dim,
            model_dim=config["model_dim"],
            dim_mults=config["dim_mults"],
            attn_heads=config["attn_heads"],
            attn_dim_head=config["attn_dim_head"]
        ).to(device)

        # Precompute diffusion variables
        betas = linear_beta_schedule(self.timesteps).to(device)
        alphas = 1. - betas
        self.betas = betas # Need betas for p_sample
        self.alphas = alphas
        self.alphas_cumprod = torch.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = nn.functional.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        # Ensure posterior variance is > 0 for sqrt
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-20)


        self.to(device)

    # --- Diffusion Utils ---
    def _extract(self, a, t, x_shape):
        # Extract coefficients at specified timesteps t and reshape for broadcasting
        batch_size = t.shape[0]
        # Ensure t is on the correct device before gather
        out = a.gather(-1, t.to(a.device))
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def q_sample(self, x_start, t, noise=None):
        """Forward process: Add noise to data."""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x_start, t, condition, noise=None):
        """Calculate loss for a single timestep t, conditioned."""
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start, t, noise=noise)
        predicted_noise = self.denoising_net(x_noisy, t, condition)

        loss = nn.functional.mse_loss(noise, predicted_noise)
        return loss

    @torch.no_grad()
    def p_sample(self, x_noisy, t, condition):
        """Reverse process: Sample x_{t-1} from x_t, conditioned."""
        betas_t = self._extract(self.betas, t, x_noisy.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_noisy.shape)
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x_noisy.shape)

        # Predict noise using the network
        predicted_noise = self.denoising_net(x_noisy, t, condition)

        # Equation 11 in DDPM paper: Model mean calculation
        model_mean = sqrt_recip_alphas_t * (
            x_noisy - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )

        # Check if we are at the last step (t=0)
        is_last_step = (t == 0)
        # Create a mask for non-last steps to add noise
        nonzero_mask = (~is_last_step).float().view(-1, *([1] * (len(x_noisy.shape) - 1)))

        posterior_variance_t = self._extract(self.posterior_variance, t, x_noisy.shape)
        noise = torch.randn_like(x_noisy)
        # Only add noise if not the last step
        return model_mean + nonzero_mask * torch.sqrt(posterior_variance_t) * noise


    @torch.no_grad()
    def sample(self, shape, condition):
        """Generate samples starting from noise, conditioned."""
        batch_size = shape[0]
        # Start from pure noise
        x_t = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.timesteps)), desc="Sampling", total=self.timesteps, leave=False):
            # Create tensor for timestep i for the whole batch
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            # Sample x_{t-1} using the reverse step, passing the condition
            x_t = self.p_sample(x_t, t, condition)

        # x_t now holds the denoised sample x_0
        return x_t

    # Forward pass used during training to calculate loss
    def forward(self, x_sir_prev, x_interventions, x_static, x_sir_target):
         # x_sir_target is the ground truth next state (x_start for diffusion)
         # x_sir_prev, x_interventions, x_static form the condition
        x_start = x_sir_target.to(device) # This is the target state we want to diffuse/denoise
        batch_size = x_start.shape[0]

        # Construct condition tensor (prev_sir + interventions + static)
        condition = torch.cat([
            x_sir_prev.to(device),
            x_interventions.to(device),
            x_static.to(device)
        ], dim=1)

        # Sample random timesteps for this batch
        t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()

        # Calculate noise prediction loss for these timesteps
        return self.p_losses(x_start, t, condition) # Pass target state, time, and condition

# --- Training, Prediction, Save/Load Functions (Unchanged) ---

def train_model(model, train_loader, val_loader, num_epochs):
    optimizer = optim.Adam(model.parameters(), lr=model.lr)

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        train_pbar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False
        )
        # Data loader yields: x_sir (prev), x_interventions, x_static, labels (target sir)
        for x_sir_prev, x_interventions, x_static, labels in train_pbar:
            x_start = labels.to(device) # Target state (next SIR) is x_start
            batch_size = x_start.shape[0]

            # Construct condition from current inputs
            condition = torch.cat([
                x_sir_prev.to(device),
                x_interventions.to(device),
                x_static.to(device)
            ], dim=1)

            optimizer.zero_grad()

            # Sample timesteps
            t = torch.randint(0, model.timesteps, (batch_size,), device=device).long()

            # Calculate loss (model internally calls p_losses)
            loss = model.p_losses(x_start, t, condition) # Pass x_start, t, and condition

            loss.backward()
            # Optional: Gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_train_loss += loss.item()
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        avg_train_loss = running_train_loss / len(train_loader)

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            val_pbar = tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False
            )
            for x_sir_prev, x_interventions, x_static, labels in val_pbar:
                x_start = labels.to(device)
                batch_size = x_start.shape[0]
                condition = torch.cat([
                    x_sir_prev.to(device),
                    x_interventions.to(device),
                    x_static.to(device)
                ], dim=1)
                t = torch.randint(0, model.timesteps, (batch_size,), device=device).long()
                loss = model.p_losses(x_start, t, condition)
                running_val_loss += loss.item()
                val_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        avg_val_loss = running_val_loss / len(val_loader)

        yield avg_train_loss, avg_val_loss, epoch


# Prediction now uses the conditional sampling
def predict(model, x_sir_prev, x_interventions, x_static) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        # Convert inputs to tensors
        x_sir_prev_t = torch.tensor(x_sir_prev, dtype=torch.float32).to(device)
        x_interventions_t = torch.tensor(x_interventions, dtype=torch.float32).to(device)
        x_static_t = torch.tensor(x_static, dtype=torch.float32).to(device)

        batch_size = x_sir_prev_t.shape[0]
        data_shape = (batch_size, model.data_dim) # Shape of the SIR state to predict

        # Construct the condition tensor (prev_sir + interventions + static)
        condition = torch.cat([x_sir_prev_t, x_interventions_t, x_static_t], dim=1)

        # Generate the next state prediction using the conditional reverse diffusion process.
        predicted_next_sir = model.sample(shape=data_shape, condition=condition)

        # --- Post-processing (Clamping/Normalization) ---
        predicted_next_sir = torch.clamp(predicted_next_sir, 0.0, 1.0) # Clamp ratios

        # Renormalize S+I+R to sum to 1 for students and adults separately
        # Ensure sums are not zero before dividing
        students_sum = predicted_next_sir[:, :3].sum(dim=1, keepdim=True) + 1e-8
        adults_sum = predicted_next_sir[:, 3:].sum(dim=1, keepdim=True) + 1e-8

        predicted_next_sir[:, :3] /= students_sum
        predicted_next_sir[:, 3:] /= adults_sum

        # Ensure clamping again after normalization might slightly push values outside [0,1]
        predicted_next_sir = torch.clamp(predicted_next_sir, 0.0, 1.0)

        return predicted_next_sir.cpu().numpy()


def save_model(model, path):
    """Saves the model state dictionary and config."""
    # Ensure model is on CPU before saving to avoid GPU info in state_dict
    model.cpu()
    save_dict = {
        'model_state_dict': model.state_dict(),
        'config': model.config # Save the config used for this model
    }
    torch.save(save_dict, path)
    model.to(device) # Move back to original device
    print(f"Model state dict and config saved to {path}")


def load_model(model, path, map_location=None):
    """Loads the model state dictionary. Assumes model instance is already created with correct config."""
    if map_location is None:
        map_location = device # Load to default device if not specified
    try:
        checkpoint = torch.load(path, map_location=map_location)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            # Update config on the loaded model instance IF the config affects architecture
            # This ensures consistency if loading into a differently configured model shell
            if hasattr(model, 'config') and 'config' in checkpoint:
                 # Only update if necessary and keys match expected structure
                 # Careful not to overwrite essential parts if checkpoint config is partial/old
                 # A safer approach might be to re-instantiate the model using loaded config
                 # model.config.update(checkpoint['config'])
                 pass # For now, assume model is instantiated correctly before loading

        else:
             # Assume it's just the state dict if the key is missing (older save format?)
             model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()
        print(f"Model state dict loaded from {path} to {device}")
    except Exception as e:
        print(f"Error loading model from {path}: {e}")
        # Depending on requirements, either raise e or handle gracefully
        raise e 