import torch
import torch.nn
from torchtyping import TensorType
import numpy as np

# Round all answers to 4 decimal places: torch.round(tensor, decimals=4)
class Solution:
    def reshape(self, to_reshape: TensorType[float]) -> TensorType[float]:
        # Reshape (M, N) tensor to (M*N/2, 2)
        M, N = to_reshape.shape
        return np.round(torch.reshape(to_reshape, (M*N//2, 2)), 4)
        # Use torch.reshape(tensor, new_shape)  

    def average(self, to_avg: TensorType[float]) -> TensorType[float]:
        # Compute column-wise mean (average across rows)
        # Use torch.mean(tensor, dim=0)
        return np.round(torch.mean(to_avg, dim=0), 4)

    def concatenate(self, cat_one: TensorType[float], cat_two: TensorType[float]) -> TensorType[float]:
        # Join two tensors side-by-side along dim=1
        # Use torch.cat((a, b), dim=1)
        return np.round(torch.cat((cat_one, cat_two), dim=1), 4)

    def get_loss(self, prediction: TensorType[float], target: TensorType[float]) -> TensorType[float]:
        # Compute Mean Squared Error between prediction and target
        # Use torch.nn.functional.mse_loss(prediction, target)
        return np.round(torch.nn.functional.mse_loss(prediction, target), 4)
