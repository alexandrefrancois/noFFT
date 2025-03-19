import numpy as np

# Compute a resonator bank outputs from a single frequency sinusoidal input signal (impulse)
# ifreq: input signal frequency
# rfreqs: resonant frequencies for each resonator
# alphas: EMWA parameters for each resonator
# betas: EMWA parameters for smoothing step
# sr: sampling rate
# duration: duration of synthesyzed signal
def resonate_python(ifreq, rfreqs, alphas, betas, sr, duration):
	numpoints = int(sr*duration)
	signal = np.cos(2*np.pi*ifreq * np.linspace(0.0, duration, num=numpoints)) 
	nfreqs = rfreqs.shape[0]
 
	# Oscillator values
	start = np.zeros(nfreqs)
	stop = duration * np.ones(nfreqs)
	linsp = np.linspace(start=start, stop=stop, num=numpoints)
	omegasm = -2*np.pi*rfreqs * linsp

	# components of the resonator complex value, initialized to e^(i*omega)
	# this corresponds to the phasor in the online implementation
	cosines = (signal * np.cos(omegasm).T).T
	sines = (signal * np.sin(omegasm).T).T

	# initial value 0
	cosines[0] = 0.0
	sines[0] = 0.0
 
	# smoothed output
	outputs = np.zeros(cosines.shape)
	angles = np.zeros(cosines.shape)

	omas = 1.0 - alphas
	omb = 1.0 - betas
 
	# only do one loop and compute powers
	for i in range(1, numpoints):
		cosines[i] = omas * cosines[i-1] + alphas * cosines[i]
		sines[i] = omas * sines[i-1] + alphas * sines[i]
		# output is powers
		outputs[i] = omb * outputs[i-1] + betas * (cosines[i] * cosines[i] + sines[i] * sines[i])
		angles[i] = np.atan2(sines[i], cosines[i])

	# print("outputs shape:", outputs.shape)
 
	# max power is 0.25 - max amplitude is 0.5
	return outputs, angles
