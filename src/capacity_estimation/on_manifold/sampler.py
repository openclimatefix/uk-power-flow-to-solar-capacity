"""Manifold learning and generative weather sampling using a cVAE."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class _CondVAE(nn.Module):
    """Conditional VAE for generating on-manifold weather scenarios."""

    def __init__(
        self,
        x_dim: int,
        n_locations: int,
        z_dim: int = 8,
        hidden: int = 128,
        emb_loc: int = 16,
        emb_month: int = 4,
        emb_hour: int = 4,
    ) -> None:
        super().__init__()
        self.z_dim = z_dim

        # Categorical embeddings for temporal/spatial context
        self.loc_emb = nn.Embedding(n_locations, emb_loc)
        self.month_emb = nn.Embedding(13, emb_month)
        self.hour_emb = nn.Embedding(24, emb_hour)

        cond_dim = emb_loc + emb_month + emb_hour

        # Encoder: x + context -> latent distribution parameters (mu, logvar)
        self.encoder = nn.Sequential(
            nn.Linear(x_dim + cond_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, z_dim)
        self.logvar = nn.Linear(hidden, z_dim)

        # Decoder: z + context -> reconstructed x
        self.decoder = nn.Sequential(
            nn.Linear(z_dim + cond_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, x_dim),
        )

    def _cond(self, loc: torch.Tensor, mo: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        """Concatenate embeddings into a single conditioning vector."""
        return torch.cat(
            [self.loc_emb(loc), self.month_emb(mo), self.hour_emb(hr)],
            dim=-1,
        )

    def forward(
        self,
        x: torch.Tensor,
        loc: torch.Tensor,
        mo: torch.Tensor,
        hr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the cVAE."""
        h = self.encoder(torch.cat([x, self._cond(loc, mo, hr)], dim=-1))
        mu, logvar = self.mu(h), self.logvar(h)

        # Reparameterisation trick: z = mu + sigma * epsilon, epsilon ~ N(0, I)
        # Allows gradients to flow through the stochastic sampling step
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)

        x_hat = self.decoder(torch.cat([z, self._cond(loc, mo, hr)], dim=-1))
        return x_hat, mu, logvar

    @staticmethod
    def loss_fn(
        x: torch.Tensor,
        x_hat: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 1.0,
    ) -> torch.Tensor:
        """ELBO loss combining reconstruction (MSE) and KL divergence."""
        # Reconstruction term: penalises deviation from input features
        recon = nn.functional.mse_loss(x_hat, x, reduction="mean")
        # KL term: regularises latent space toward unit Gaussian prior N(0, I)
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + beta * kld


class OnManifoldSampler:
    """Handles manifold learning and generative weather sampling for TFT scenarios."""

    def __init__(
        self,
        parquet_path: Path,
        feature_list: list[str],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        save_dir: Path = Path("sampler_ckpt"),
        max_rows: int = 1_000_000,
        epochs: int = 8,
        batch_size: int = 2048,
        lr: float = 2e-3,
    ) -> None:
        self.parquet_path = Path(parquet_path)
        self.features = list(feature_list)
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.max_rows = max_rows
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

        self.scaler: StandardScaler | None = None
        self.model: _CondVAE | None = None
        self.loc2id: dict[str, int] = {}
        self.minmax: dict[str, tuple[float, float]] = {}

    def _get_paths(self) -> tuple[Path, Path]:
        """Generate deterministic paths for model artifacts."""
        tag = "_".join(sorted(self.features))
        h = hash(tag) & 0xFFFFFFF
        return self.save_dir / f"cvae_{h}.pt", self.save_dir / f"scaler_{h}.json"

    def _prepare_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and normalise feature data for training."""
        logger.info("Loading sampler data from %s", self.parquet_path)
        cols = ["location", "month", "hour", *self.features]
        df = pd.read_parquet(self.parquet_path, columns=cols).dropna()
        df["location"] = df["location"].astype(str)

        sites = sorted(df["location"].unique())
        self.loc2id = {s: i for i, s in enumerate(sites)}

        x_raw = df[self.features].astype(float).to_numpy()
        self.scaler = StandardScaler().fit(x_raw)

        # Record empirical bounds per feature for post-generation clipping in sample_vectors
        self.minmax = {
            f: (float(x_raw[:, i].min()), float(x_raw[:, i].max()))
            for i, f in enumerate(self.features)
        }

        loc_ids = df["location"].map(self.loc2id).to_numpy()
        months = pd.to_numeric(df["month"]).clip(1, 12).to_numpy()
        hours = pd.to_numeric(df["hour"]).clip(0, 23).to_numpy()
        x_scaled = self.scaler.transform(x_raw)

        if len(x_scaled) > self.max_rows:
            logger.info("Sub-sampling to %d rows", self.max_rows)
            idx = np.random.choice(len(x_scaled), self.max_rows, replace=False)
            return x_scaled[idx], loc_ids[idx], months[idx], hours[idx]

        return x_scaled, loc_ids, months, hours

    def train_or_load(self) -> None:
        """Load a checkpoint or train a new cVAE manifold model."""
        m_path, s_path = self._get_paths()

        if m_path.exists() and s_path.exists():
            logger.info("Loading existing cVAE artifacts.")
            with open(s_path) as f:
                obj = json.load(f)

            self.scaler = StandardScaler()
            self.scaler.mean_ = np.array(obj["m"])
            self.scaler.scale_ = np.array(obj["s"])
            self.scaler.var_ = self.scaler.scale_**2
            self.minmax = obj["mm"]

            sites = sorted(
                pd.read_parquet(self.parquet_path, columns=["location"])["location"]
                .astype(str)
                .unique()
            )
            self.loc2id = {s: i for i, s in enumerate(sites)}

            self.model = _CondVAE(len(self.features), len(self.loc2id)).to(self.device)
            self.model.load_state_dict(torch.load(m_path, map_location=self.device))
            self.model.eval()
            return

        logger.info("Starting manifold training.")
        xs, locs, months, hours = self._prepare_data()
        self.model = _CondVAE(len(self.features), len(self.loc2id)).to(self.device)

        ds = TensorDataset(
            torch.from_numpy(xs.astype(np.float32)),
            torch.from_numpy(locs.astype(np.int64)),
            torch.from_numpy(months.astype(np.int64)),
            torch.from_numpy(hours.astype(np.int64)),
        )
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        # Adam with default betas; lr tuned for fast convergence on weather distributions
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()

        # Standard ELBO optimisation — minimise reconstruction + KL simultaneously
        for ep in range(1, self.epochs + 1):
            total_loss = 0.0
            for xb, lb, mb, hb in dl:
                xb, lb, mb, hb = (t.to(self.device) for t in (xb, lb, mb, hb))
                x_hat, mu, logvar = self.model(xb, lb, mb, hb)
                loss = _CondVAE.loss_fn(xb, x_hat, mu, logvar)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)

            logger.info("Epoch %d/%d - Loss: %.6f", ep, self.epochs, total_loss / len(ds))

        # Persist weights and scaler separately so scaler can be inspected without loading torch
        torch.save(self.model.state_dict(), m_path)
        with open(s_path, "w") as f:
            json.dump(
                {
                    "m": self.scaler.mean_.tolist(),
                    "s": self.scaler.scale_.tolist(),
                    "mm": self.minmax,
                },
                f,
            )

        self.model.eval()

    @torch.inference_mode()
    def sample_vectors(
        self,
        location: str,
        month: int,
        hour: int,
        k: int,
    ) -> list[dict[str, float]]:
        """Generate weather samples constrained to the learned manifold."""
        loc_id = self.loc2id.get(str(location), 0)
        loc = torch.full((k,), loc_id, dtype=torch.long, device=self.device)
        mo = torch.full((k,), int(month), dtype=torch.long, device=self.device).clamp(1, 12)
        hr = torch.full((k,), int(hour), dtype=torch.long, device=self.device).clamp(0, 23)

        # Sample from latent prior z ~ N(0, I) and decode through conditioned generator
        z = torch.randn(k, self.model.z_dim, device=self.device)
        xs_scaled = (
            self.model.decoder(torch.cat([z, self.model._cond(loc, mo, hr)], dim=-1)).cpu().numpy()
        )

        # Invert standardisation to recover physical feature magnitudes
        x_inv = self.scaler.inverse_transform(xs_scaled)

        # Clip to empirical training bounds to prevent physically implausible extrapolation
        samples = []
        for i in range(k):
            vec = {}
            for j, f in enumerate(self.features):
                lo, hi = self.minmax[f]
                vec[f] = float(np.clip(x_inv[i, j], lo, hi))
            samples.append(vec)

        return samples
