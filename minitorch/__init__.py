# MiniTorch v0.5 - PagedAttention Project
#
# Base modules (copied from hw3)
import minitorch.scalar_functions as scalar_functions  # noqa: F401,F403

from .autodiff import *  # noqa: F401,F403
from .cuda_ops import *  # noqa: F401,F403
from .datasets import datasets  # noqa: F401,F403
from .fast_conv import *  # noqa: F401,F403
from .fast_ops import *  # noqa: F401,F403
from .module import *  # noqa: F401,F403
from .nn import *  # noqa: F401,F403
from .optim import *  # noqa: F401,F403
from .scalar import Scalar, ScalarHistory, derivative_check  # noqa: F401,F403
from .scalar_functions import ScalarFunction  # noqa: F401,F403
from .tensor import *  # noqa: F401,F403
from .tensor_data import *  # noqa: F401,F403
from .tensor_ops import *  # noqa: F401,F403
from .testing import MathTest, MathTestVariable  # type: ignore # noqa: F401,F403

try:
    from .tensor_functions import *  # noqa: F401,F403
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise

# Project-specific modules
from .transformer import *  # noqa: F401,F403
from .block_manager import *  # noqa: F401,F403
from .paged_attention import *  # noqa: F401,F403

version = "0.5"
