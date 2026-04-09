from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from .workflows import add_vip_status


class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, samples: int, **kwargs) -> pd.DataFrame:
        pass


class RandomGenerator(BaseGenerator):
    """Generate independent but business-facing VIP customer features."""

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


class SpecializedGenerator(BaseGenerator):
    """Generate correlated VIP customer features using a multivariate normal core."""

    def generate(self, samples: int, **kwargs) -> pd.DataFrame:
        random_state = kwargs.get("random_state")
        rng = np.random.default_rng(random_state)
        mean = np.array([720, 12, 90])
        cov = np.array(
            [
                [50**2, 20, 900],
                [20, 4**2, 12],
                [900, 12, 45**2],
            ]
        )
        raw = rng.multivariate_normal(mean, cov, samples)
        df = pd.DataFrame(
            {
                "credit_score": np.clip(raw[:, 0], 300, 850).round(0),
                "frequency": np.clip(raw[:, 1], 0, None).round(0),
                "avg_transaction": np.clip(raw[:, 2], 5, None).round(2),
            }
        )
        return add_vip_status(df[["avg_transaction", "frequency", "credit_score"]])


class GANGenerator(BaseGenerator):
    """Lightweight GAN-style simulator that preserves business-facing columns."""

    def generate(self, samples: int, **kwargs) -> pd.DataFrame:
        random_state = kwargs.get("random_state")
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
        "specialized": SpecializedGenerator(),
        "gan": GANGenerator(),
    }
    generator = mapping.get(method.lower())
    if generator is None:
        raise ValueError(f"Unknown generation method: {method}")
    return generator
