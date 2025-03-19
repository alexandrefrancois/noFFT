import numpy as np
from noFFT import resonate


def log_frequencies(fmin=32.7, n_freqs=84, freqs_per_octave=12):
	return fmin * 2.0 ** (np.r_[0:n_freqs] / float(freqs_per_octave))

def alphas_heuristic(frequencies, sr, k = 1):
  return 1 - np.exp(- (1/sr) * frequencies / (k * np.log10(1+frequencies)))


# Compute a resonator bank outputs from an input signal
# Uses C++ implementation of update - betas = alphas
# y: input signal
# sr: sampling rate
# frequencies: resonant frequencies for each resonator
# alphas: EMWA parameters for each resonator
# hop_length: number of samples between ech output sample
def resonate_wrapper(y, sr, frequencies, alphas, hop_length=1, output_type='powers'):
	float_y = np.array(y, dtype=np.float32)
	float_fs = np.array(frequencies, dtype=np.float32)
	float_as = np.array(alphas, dtype=np.float32)

	# betas = alphas
	float_Rsc = resonate(float_y, sr, float_fs, float_as, float_as, hop_length)

	Rsc = np.array(float_Rsc, dtype=np.float64)
	nfreqs = frequencies.shape[0]
	Rsc = Rsc.reshape((-1, (2 * nfreqs)))
	# compute complex values in the right shape
	re = Rsc[...,:nfreqs]
	im = Rsc[...,nfreqs:]

	if output_type == 'powers':
		# max powers is 0.25
		return re * re + im * im
	if output_type == 'amplitudes':
		# max amplitude is 0.5
		return np.sqrt(re * re + im * im)
	
	# default: return complex vector
	Rcx = re + im * 1j
	Rcx = Rcx
	return Rcx

# Compute a resonator bank outputs from a single frequency sinusoidal input signal (impulse)
# Uses C++ implementation of update - betas = alphas
# ifreq: input signal frequency
# rfreqs: resonant frequencies for each resonator
# alphas: EMWA parameters for each resonator
# sr: sampling rate
# duration: duration of synthesyzed signal
def frequency_response(ifreq, rfreqs, alphas, sr, duration, output_type = 'powers'):
	# make input signal: single frequency step
	numpoints = int(sr*duration)
	signal = np.cos(2*np.pi*ifreq * np.linspace(0.0, duration, num=numpoints)) 
	hop_length = 1

	return resonate_wrapper(
		y=signal, sr=sr, frequencies=rfreqs, alphas=alphas, hop_length=hop_length, output_type=output_type)


# Compute a resonator bank's equalizer coefficients by performing a frequency sweep
# Calls frequency_response for each resonator frequency
# Uses C++ implementation of update - betas = alphas
# frequencies: resonant frequencies for each resonator
# alphas: EMWA parameters for each resonator
# sr: sampling rate
# duration: duration of synthesyzed signal
def frequency_sweep(frequencies, alphas, sr):
	taus = -1.0 / (np.log(1 - alphas) * sr)
	areas_p = np.zeros_like(frequencies)
	for idx, ifreq in enumerate(frequencies):
		duration = 40 * taus[idx]
		r = frequency_response(ifreq=ifreq, rfreqs=frequencies, alphas=alphas, sr=sr, duration=duration, output_type='powers')
		areas_p[idx] = np.sum(r[-1], axis=0)
	return 0.25 / np.sqrt(areas_p)

