import numpy as np
from noFFT import resonate


fmin = 32.70
n_bins = 84
bins_per_octave = 12

sr = 44100.0  # in Hz
duration = 0.1  # in s
n_points = int(sr * duration)

# Sinusoid input signal
ifreq = 440
signal = np.cos(2 * np.pi * ifreq * np.linspace(0.0, duration, num=n_points))

float_y = np.array(signal, dtype=np.float32)
# print(float_y.dtype)

freqs = fmin * 2.0 ** (np.r_[0:n_bins] / float(bins_per_octave))
float_fs = np.array(freqs, dtype=np.float32)

# alphas = 0.001 * np.ones_like(freqs)
alphas = 1 - np.exp(-(1 / sr) * freqs / np.log10(freqs))
float_as = np.array(alphas, dtype=np.float32)

hop_length = 256
float_R = resonate(float_y, sr, float_fs, float_as, float_as, hop_length)

R = np.array(float_R, dtype=np.float64)
R = R.reshape((-1, n_bins)).T

# shape of R:
# each slice of size twoNumResonators has
# numResonators Re values and numResonators Im values (split complex)
# print(R.shape)
# print(R.dtype)

assert R.shape[0] == n_bins
assert R.shape[1] == int(2 * n_points / hop_length)
