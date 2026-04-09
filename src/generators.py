from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from .workflows import VIP_COLUMNS, add_vip_status

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    nn = None


class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, samples: int, **kwargs) -> pd.DataFrame:
        pass


class RandomGenerator(BaseGenerator):
    """Independent baseline generator for business-facing VIP features."""

    def generate(self, samples: int, **kwargs) -> pd.DataFrame:
        random_state = kwargs.get("random_state")
        rng = np.random.default_rng(random_state)
        credit_score = np.clip(rng.normal(700, 65, samples), 300, 850)
        frequency = rng.poisson(10, samples)
        avg_transaction = np.clip(rng.lognormal(mean=3.5, sigma=0.55, size=samples), 5, None)
        df = pd.DataFrame(
            {
                "avg_transaction": avg_transaction.round(2),
                "frequency": frequency.astype(int),
                "credit_score": credit_score.round(0),
            }
        )
        return add_vip_status(df)


class BayesianNetworkGenerator(BaseGenerator):
    """Approximate Bayesian generator using posterior predictive multivariate sampling."""

    def generate(self, samples: int, **kwargs) -> pd.DataFrame:
        random_state = kwargs.get("random_state")
        seed_df = kwargs.get("seed_df")
        rng = np.random.default_rng(random_state)

        if seed_df is not None and all(col in seed_df.columns for col in VIP_COLUMNS):
            base = seed_df[VIP_COLUMNS].dropna().astype(float)
            if len(base) >= 5:
                mean = base.mean().to_numpy()
                cov = np.cov(base.to_numpy().T)
            else:
                mean = np.array([85.0, 12.0, 720.0])
                cov = np.diag([30.0**2, 4.0**2, 45.0**2])
        else:
            mean = np.array([85.0, 12.0, 720.0])
            cov = np.array(
                [
                    [35.0**2, 18.0, 420.0],
                    [18.0, 4.0**2, 10.0],
                    [420.0, 10.0, 45.0**2],
                ]
            )

        cov = cov + np.eye(3) * 1e-6
        raw = rng.multivariate_normal(mean, cov, size=samples)
        df = pd.DataFrame(
            {
                "avg_transaction": np.clip(raw[:, 0], 5, None).round(2),
                "frequency": np.clip(raw[:, 1], 0, None).round(0),
                "credit_score": np.clip(raw[:, 2], 300, 850).round(0),
            }
        )
        return add_vip_status(df)


class GANGenerator(BaseGenerator):
    """Neural-network GAN generator; falls back to a structured simulator if torch is unavailable."""

    def generate(self, samples: int, **kwargs) -> pd.DataFrame:
        random_state = kwargs.get("random_state", 42)
        seed_df = kwargs.get("seed_df")
        epochs = kwargs.get("epochs", 200)

        if torch is None or nn is None:
            return self._fallback_generate(samples=samples, random_state=random_state)

        training = self._prepare_training_data(seed_df, random_state)
        generated = self._train_and_sample(training, samples=samples, random_state=random_state, epochs=epochs)
        df = pd.DataFrame(generated, columns=VIP_COLUMNS)
        df["avg_transaction"] = df["avg_transaction"].clip(lower=5).round(2)
        df["frequency"] = df["frequency"].clip(lower=0).round(0)
        df["credit_score"] = df["credit_score"].clip(lower=300, upper=850).round(0)
        return add_vip_status(df)

    def _prepare_training_data(self, seed_df: pd.DataFrame | None, random_state: int) -> np.ndarray:
        if seed_df is not None and all(col in seed_df.columns for col in VIP_COLUMNS):
            base = seed_df[VIP_COLUMNS].dropna().astype(float)
            if len(base) >= 10:
                return base.to_numpy(dtype=np.float32)
        baseline = BayesianNetworkGenerator().generate(1000, random_state=random_state)
        return baseline[VIP_COLUMNS].to_numpy(dtype=np.float32)

    def _train_and_sample(self, training: np.ndarray, samples: int, random_state: int, epochs: int) -> np.ndarray:
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        mean = training.mean(axis=0, keepdims=True)
        std = training.std(axis=0, keepdims=True) + 1e-6
        normalized = (training - mean) / std
        data = torch.tensor(normalized, dtype=torch.float32)

        latent_dim = 8

        class GeneratorNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(latent_dim, 32),
                    nn.ReLU(),
                    nn.Linear(32, 32),
                    nn.ReLU(),
                    nn.Linear(32, 3),
                )

            def forward(self, z):
                return self.net(z)

        class DiscriminatorNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(3, 32),
                    nn.LeakyReLU(0.2),
                    nn.Linear(32, 16),
                    nn.LeakyReLU(0.2),
                    nn.Linear(16, 1),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                return self.net(x)

        generator = GeneratorNet()
        discriminator = DiscriminatorNet()
        loss_fn = nn.BCELoss()
        g_opt = torch.optim.Adam(generator.parameters(), lr=1e-3)
        d_opt = torch.optim.Adam(discriminator.parameters(), lr=1e-3)

        batch_size = min(64, len(data))
        for _ in range(max(50, epochs)):
            idx = torch.randint(0, len(data), (batch_size,))
            real_batch = data[idx]
            noise = torch.randn(batch_size, latent_dim)
            fake_batch = generator(noise)

            d_opt.zero_grad()
            real_loss = loss_fn(discriminator(real_batch), torch.ones(batch_size, 1))
            fake_loss = loss_fn(discriminator(fake_batch.detach()), torch.zeros(batch_size, 1))
            d_loss = real_loss + fake_loss
            d_loss.backward()
            d_opt.step()

            g_opt.zero_grad()
            noise = torch.randn(batch_size, latent_dim)
            generated = generator(noise)
            g_loss = loss_fn(discriminator(generated), torch.ones(batch_size, 1))
            g_loss.backward()
            g_opt.step()

        noise = torch.randn(samples, latent_dim)
        generated = generator(noise).detach().numpy()
        return generated * std + mean

    def _fallback_generate(self, samples: int, random_state: int) -> pd.DataFrame:
        rng = np.random.default_rng(random_state)
        latent = rng.normal(size=(samples, 3))
        transformation = np.array(
            [
                [45, 2.0, 75],
                [10, 1.5, 25],
                [25, 0.8, 40],
            ]
        )
        generated = latent @ transformation + np.array([120, 11, 705])
        df = pd.DataFrame(
            {
                "avg_transaction": np.clip(generated[:, 0], 5, None).round(2),
                "frequency": np.clip(generated[:, 1], 0, None).round(0),
                "credit_score": np.clip(generated[:, 2], 300, 850).round(0),
            }
        )
        return add_vip_status(df)


def get_generator(method: str) -> BaseGenerator:
    mapping = {
        "random": RandomGenerator(),
        "specialized": BayesianNetworkGenerator(),
        "gan": GANGenerator(),
    }
    generator = mapping.get(method.lower())
    if generator is None:
        raise ValueError(f"Unknown generation method: {method}")
    return generator
