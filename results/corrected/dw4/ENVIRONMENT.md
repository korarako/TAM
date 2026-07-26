# Runtime environment

The corrected reference and formal rerun were executed on
`TUMWMFM-GPU04` with:

```text
Linux 5.15.0-179-generic x86_64
NVIDIA GeForce RTX 3090
NVIDIA driver 535.309.01
Python 3.11.15
JAX 0.7.1
jaxlib 0.7.1
NumPy 1.26.4
SciPy 1.17.1
Matplotlib 3.10.9
Flax 0.12.7
Optax 0.2.8
```

The runtime interpreter was:

```text
${CONDA_ROOT}/envs/ab/bin/python
```

The exact project requirements and executed source tree are preserved in
`code_snapshot/`. The formal wrapper exported:

```text
PYTHONPATH=${ADTM_ROOT}/src
XLA_PYTHON_CLIENT_PREALLOCATE=false
JAX_ENABLE_X64=true
```
