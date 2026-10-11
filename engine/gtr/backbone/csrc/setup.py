"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Build a PyTorch extension that exposes the hand-written chunk_gla kernel.
"""

import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

here = os.path.dirname(os.path.abspath(__file__))

setup(
    name='gla_torch_ext',
    ext_modules=[
        CUDAExtension(
            name='gla_torch_ext',
            sources=[os.path.join(here, 'gla_torch_ext.cu')],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3', '-std=c++17',
                    '--use_fast_math',
                    '-gencode=arch=compute_120,code=sm_120',  # RTX 6000D / RTX PRO 6000 (Blackwell)
                ],
            },
            include_dirs=[here],
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
