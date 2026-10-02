"""
NumPy 2 compatibility for imgaug (unmaintained since 2020): restore the few module attributes it reads at import or call
time and that NumPy 1.20-2.0 removed. Imported first by synthtiger/__init__.py; a no-op on NumPy 1.x.
"""

import warnings

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning, module=__name__)   # hasattr(np, "object") warns on 1.x

if not hasattr(np, "sctypes"):
    np.sctypes = {
        "int": [np.int8, np.int16, np.int32, np.int64],
        "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
        "float": [np.float16, np.float32, np.float64],
        "complex": [np.complex64, np.complex128],
        "others": [bool, object, bytes, str, np.void],
    }
for name, value in (("bool", bool), ("int", int), ("float", float), ("complex", complex), ("object", object), ("str", str)):
    if not hasattr(np, name):
        setattr(np, name, value)
for name, target in (("product", "prod"), ("cumproduct", "cumprod"), ("alltrue", "all"), ("sometrue", "any"), ("round_", "round")):
    if not hasattr(np, name):
        setattr(np, name, getattr(np, target))
