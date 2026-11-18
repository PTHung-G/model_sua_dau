"""Exit nonzero if the pinned portable dependencies are missing or broken."""
import numpy
import torch
assert numpy.__version__ == '2.2.6', 'Expected NumPy 2.2.6'
assert torch.__version__.split('+')[0] == '2.6.0', 'Expected PyTorch 2.6.0'
print('Dependencies ready.')
