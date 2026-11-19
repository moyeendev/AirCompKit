"""Over-the-air federated learning (FEATURES §7): datasets, partitioners, NumPy models,
the OTA-FL loop and the gradient bank.

    import aircompkit as ac
    from aircompkit.learning import FederatedConfig, train

    sc = ac.Scenario(devices={"name": "fixed", "K": 20}, power_control="channel_inversion")
    res = train(sc, FederatedConfig(dataset="synthetic", rounds=50))
"""

from aircompkit.learning.aggregate import aircomp_aggregate
from aircompkit.learning.bank import GradientBank
from aircompkit.learning.data import Dataset, fashion_mnist, mnist, synthetic
from aircompkit.learning.fl import FederatedConfig, train
from aircompkit.learning.models import MLP, logreg
from aircompkit.learning.partition import IID, Dirichlet, Shards

__all__ = [
    "IID",
    "MLP",
    "Dataset",
    "Dirichlet",
    "FederatedConfig",
    "GradientBank",
    "Shards",
    "aircomp_aggregate",
    "fashion_mnist",
    "logreg",
    "mnist",
    "synthetic",
    "train",
]
