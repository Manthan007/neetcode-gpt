import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        #
        # Pre-activation: z = dot(x, w) + b
        z = (x @ w) + b
        if activation == "sigmoid":
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
            acti = np.sum(1/(1 + np.exp(-z)))
        else:
        # ReLU: max(0, z)
            acti = np.maximum(0, z)
        # return round(your_answer, 5)
        return np.round(acti, 5)
        pass
