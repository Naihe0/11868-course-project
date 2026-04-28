"""
For additional transformer related

Sequential
Embedding

"""
import numpy as np
import numba
from numba import cuda
from math import sqrt
from .module import Module, Parameter
from .tensor_functions import (MatMul ,Add, Mul, PowerScalar ,zeros, ones, rand, tensor, tensor_from_numpy, zeros_tensor_from_numpy, ones_tensor_from_numpy)
from .nn import one_hot
from .tensor_ops import TensorBackend
from .tensor import Tensor
import random
from .operators import prod

from typing import Any, Dict, Optional, Sequence, Tuple
from .tensor_functions import LayerNorm


@cuda.jit
def _embedding_lookup_kernel(out, token_ids, weights, total, embedding_dim):
    pos = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if pos < total:
        emb_dim = pos % embedding_dim
        token_pos = pos // embedding_dim
        token_id = int(token_ids[token_pos])
        out[pos] = weights[token_id * embedding_dim + emb_dim]

class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, backend: TensorBackend):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Args:
            num_embeddings : The vocabulary size
            embedding_dim : The size of each embedding vector

        Attributes:
            weights : The learnable weights of shape (num_embeddings, embedding_dim) initialized from N(0, 1).
        """
        self.backend = backend
        self.num_embeddings = num_embeddings # Vocab size
        self.embedding_dim  = embedding_dim  # Embedding Dimension
        ### BEGIN ASSIGN3_2
        shape = (num_embeddings, embedding_dim)
        vals = [random.gauss(0, 1) for _ in range(int(prod(shape)))]
        tensor = Tensor.make(vals, shape, backend=backend)
        tensor.requires_grad_(True)
        self.weights = Parameter(tensor)
        ### END ASSIGN3_2
    
    def forward(self, x: Tensor):
        """Maps word indices to one-hot vectors, and projects to embedding vectors.

        Args:
            x : Tensor of shape (batch_size, seq_len)

        Returns:
            output : Tensor of shape (batch_size, seq_len, embedding_dim)
        """
        bs, seq_len = x.shape
        if getattr(x.backend, "cuda", False):
            out = x.zeros((bs, seq_len, self.embedding_dim))
            total = bs * seq_len * self.embedding_dim
            threads = 128
            blocks = (total + threads - 1) // threads
            token_storage = (
                x._tensor._storage
                if x._tensor.is_contiguous()
                else x.contiguous()._tensor._storage
            )
            weight_tensor = self.weights.value
            weight_storage = (
                weight_tensor._tensor._storage
                if weight_tensor._tensor.is_contiguous()
                else weight_tensor.contiguous()._tensor._storage
            )
            _embedding_lookup_kernel[blocks, threads](
                out._tensor._storage,
                token_storage,
                weight_storage,
                total,
                self.embedding_dim,
            )
            return out

        token_ids = x.to_numpy().astype(np.int64)
        embedded = self.weights.value.to_numpy()[token_ids]
        return tensor_from_numpy(embedded.astype(np.float32), backend=self.backend)


    
class Dropout(Module):
    def __init__(self, p_dropout: float=0.1):
        super().__init__()
        """During training, randomly zeroes some of the elements of the input tensor with probability :attr:`p_dropout`.

        Attributes: 
            p_dropout : Probability an element will be zeroed.
        """
        self.p_dropout = p_dropout

    def forward(self, x: Tensor) -> Tensor: 
        """During training, randomly zero out elements of a tensor and scale by (1 - p_dropout)
        
        Args: 
            x : Tensor of shape (*)
        
        Returns: 
            output : Tensor of shape (*)

        Note: If p_dropout is 0, directly return the input tensor. Otherwise, the random seed may cause problems
        """
        if self.p_dropout == 0.0 or (not self.training):
            return x
        mask = tensor_from_numpy(
            (np.random.rand(*x.shape) > self.p_dropout).astype(np.float32),
            backend=x.backend,
        )
        return x * mask / (1 - self.p_dropout)
            


class Linear(Module):
    def __init__(self, in_size: int, out_size: int, bias: bool, backend: TensorBackend):
        super().__init__()
        """Applies a linear transformation to the incoming data. (Same as PyTorch)

        Parameters:
            in_size  - The size of the dimension the transformation will be applied to
            out_size - The size of the resulting transformation's dimension
            bias     - If True, then add an additive bias

        Attributes:
            weights - The learnable weights of shape (in_size, out_size) initialized from Uniform(-1/sqrt(in_size), 1/sqrt(in_size)).
            bias   - The learnable weights of shape (out_size, ) initialized from Uniform(-1/sqrt(in_size), 1/sqrt(in_size)).
        """
        self.out_size = out_size
        self.use_bias = bias
        self.weights = Parameter((rand((in_size, out_size),backend=backend)-0.5)*2*(1/sqrt(in_size)))
        if self.use_bias:
            self.bias = Parameter((rand((out_size,), backend=backend)-0.5)*2*(1/sqrt(in_size)))
        ### END ASSIGN3_2

    def forward(self, x: Tensor):
        """Applies a linear transformation to the incoming data.
        
        Args: 
            x : Tensor of shape (n, in_size)
        
        Returns:
            output : Tensor of shape (n, out_size)
        """
        batch, in_size = x.shape
        x = x.view(batch,in_size)        
        weight_reshape = self.weights.value.view(in_size, self.out_size)
        output = MatMul.apply(x, weight_reshape)
        if self.use_bias:
            bias = self.bias.value.view(1, self.out_size)
            output = Add.apply(output, bias)
        return output
        ### END ASSIGN3_2


class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float, backend: TensorBackend):
        super().__init__()
        """Applies Layer Normalization over a mini-batch of 1-dimensional inputs.
        
        Args: 
            dim : Expected size of the last dimension to apply layer normalization.
            eps : A value added for numerical stability.
        
        Attributes: 
            weights : the learnable weights of the module of shape (self.dim, ) initialized to 1.
            bias    : the learnable bias of the module of shape (self.dim, ) initialized to 0.
        """
        self.dim = dim
        self.eps = eps
        ### BEGIN ASSIGN3_2
        self.weights = Parameter(ones((self.dim,), backend=backend))
        self.bias = Parameter(zeros((self.dim,), backend = backend))
        ### END ASSIGN3_2

    def forward(self, x: Tensor) -> Tensor:
        """Applies Layer Normalization over a mini-batch of inputs. 
        NOTE: You can assume the input to this layer is a 2D tensor of shape (batch_size, dim)
        You will use implicit broadcasting in miniTorch to use the weight and bias.
        
        Input: 
            x - Tensor of shape (bs, dim)
        
        Output: 
            output - Tensor of shape (bs, dim)
        """
        batch, dim = x.shape

        bias = self.bias.value.view(1, self.dim)
        weights = self.weights.value.view(1, self.dim)
        use_fused = getattr(x.backend, "use_fused_kernel", False)

        if use_fused:
            return LayerNorm.apply(x, weights, bias)
        else:
            x_mean = x.mean(dim = 1)
            x_var = x.var(dim = 1)
            var = x_var + self.eps
            tep = Mul.apply(weights,( (x - x_mean)/ var**0.5))
            out =  Add.apply(tep, bias)
        return out
