from .kernels import DotProductTanimoto, MinMaxTanimoto

# Alias
Tanimoto = MinMaxTanimoto
TanimotoBinary = DotProductTanimoto

__all__ = [
    "MinMaxTanimoto",
    "DotProductTanimoto",
    "Tanimoto",
    "TanimotoBinary",
]
