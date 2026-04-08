import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, samples: int, **kwargs) -> pd.DataFrame:
        pass

class RandomGenerator(BaseGenerator):
    """Basic stochastic generation using independent distributions."""
    def generate(self, samples: int, **kwargs) -> pd.DataFrame:
        data = {
            'feature_a': np.random.normal(0, 1, samples),
            'feature_b': np.random.exponential(1, samples),
            'feature_c': np.random.poisson(5, samples)
        }
        return pd.DataFrame(data)

class SpecializedGenerator(BaseGenerator):
    """Bayesian-inspired generation using a Multivariate Normal (Copula-lite)."""
    def generate(self, samples: int, **kwargs) -> pd.DataFrame:
        mean = [0, 5]
        cov = [[1, 0.8], [0.8, 1]]  # High correlation between features
        data = np.random.multivariate_normal(mean, cov, samples)
        return pd.DataFrame(data, columns=['feature_a', 'feature_b'])

class GANGenerator(BaseGenerator):
    """
    Simplified GAN Wrapper. In a production environment, this would load a 
    PyTorch/TF model. Here, it simulates the learned distribution.
    """
    def generate(self, samples: int, **kwargs) -> pd.DataFrame:
        # Simulating a GAN by adding structured noise to a latent manifold
        latent = np.random.randn(samples, 2)
        transformation = np.array([[1.2, 0.5], [0.1, 0.9]])
        data = np.dot(latent, transformation) + 0.5
        return pd.DataFrame(data, columns=['feature_a', 'feature_b'])

def get_generator(method: str) -> BaseGenerator:
    mapping = {
        'random': RandomGenerator(),
        'specialized': SpecializedGenerator(),
        'gan': GANGenerator()
    }
    return mapping.get(method.lower())
