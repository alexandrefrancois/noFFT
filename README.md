# noFFT

A reference implementation of the [Resonate](https://alexandrefrancois.org/Resonate)
algorithm in C++ for Python,
using [pybind11](https://pybind11.readthedocs.io/en/stable/).
The C++ code uses the Accelerate framework and will only work on Mac/iOS platforms.

This is a crude version to demonstrate the capabilities of the algorithm through Jupyter notebooks.

The goal is to turn this into a proper Python package - contributions welcome!

Author: Alexandre R.J. François

## Installation

- download, checkout or clone this repository
- pip install .

- see notebooks and additional Python functions for usage

## Resonate Functions

The C++ code implements a resonator bank class similar to that provided in the
[Oscillators Swift package](https://github.com/alexandrefrancois/Oscillators)
with vectorized update per sample.

Python functions:
- `resonate`: wraps creating the bank with the parameters provided and running the updates for an input signal.
- `resonate_wrapper`: computes a resonator bank outputs from an input signal, using the C++ implementation.
- `resonate_python`: computes a resonator bank outputs from a single frequency sinusoidal input signal (impulse). The loop over samples is done in Python, so much slower than the C++ counterpart.


## Jupyter Notebooks

- **SpectralAnalysisExperiments**: code to analyze and plot resonator and resonator bank properties.
- **Spectrograms**: code to compute and plot spectrograms of audio signals, using [Librosa](https://librosa.org)
- **Chromas**: code to compute and plot chromas and chromagrams on audio signals, using [Librosa](https://librosa.org)
- **MFCCs**: code to compute and plot mel frequency scale spectrograms and chromagrams on audio signals, using [Librosa](https://librosa.org)


## License

MIT License

Copyright (c) 2025 Alexandre R.J. Francois
