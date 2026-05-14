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

def generate_signal(frequencies: list[float], durations: list[float], transition_duration: float, sr: float) -> np.ndarray:
    """
    Generates a continuous signal composed of sinusoidal segments with smooth 
    frequency transitions, using a single transition duration for all segments.

    Args:
        frequencies: List of segment frequencies (in Hz).
        durations: List of segment steady-state durations (in seconds).
        transition_duration: The duration for ALL transitions between segments (in seconds).
        sr: The sampling rate (in samples per second).

    Returns:
        A continuous 1D NumPy array representing the generated signal.
    """
    
    # 1. Input Validation (Ensure arrays have the same size)
    n_segments = len(frequencies)
    if n_segments != len(durations):
        raise ValueError("Input arrays (frequencies and durations) must have the same size.")
    
    all_samples = []

    # Initialize phase for continuity
    current_phase = 0.0

    # Iterate through segments
    for i in range(n_segments):
        f1 = frequencies[i]
        t_steady = durations[i]
        
        # --- 2. Segment Generation (Steady State) ---
        
        # Calculate time vector for the steady-state segment
        t_steady_vec = np.arange(0, t_steady, 1.0 / sr)
        
        # Generate the segment signal using the constant frequency f1
        segment_signal = np.sin(2 * np.pi * f1 * t_steady_vec + current_phase)
        all_samples.append(segment_signal)
        
        # Update phase for the next part (end of steady state segment)
        if len(t_steady_vec) > 0:
            current_phase = (2 * np.pi * f1 * t_steady_vec[-1] + current_phase) % (2 * np.pi)
        
        # --- 3. Transition Generation (If not the last segment) ---
        
        # Apply the single transition duration parameter here
        t_transition = transition_duration
        
        if i < n_segments - 1:
            f2 = frequencies[i + 1]
            
            if t_transition > 0:
                # Calculate time vector for the transition
                t_trans_vec = np.arange(0, t_transition, 1.0 / sr)
                
                # Frequency sweep function: linear interpolation from f1 to f2
                # f(t) = f1 + (f2 - f1) * (t / T_trans)
                
                # Phase is the integral of 2*pi*f(t) dt.
                # Integral of f(t) is: f1*t + (f2-f1)/2 * (t^2 / T_trans)
                phase_increment = (
                    2 * np.pi * (
                        f1 * t_trans_vec 
                        + (f2 - f1) * 0.5 * (t_trans_vec**2 / t_transition)
                    )
                )
                
                # Generate the transition signal
                transition_signal = np.sin(phase_increment + current_phase)
                all_samples.append(transition_signal)
                
                # Update phase for the next steady-state segment
                if len(phase_increment) > 0:
                    current_phase = (phase_increment[-1] + current_phase) % (2 * np.pi)
            
            else:
                # If transition_duration is 0, the next segment starts immediately.
                pass
                
    # 4. Concatenate all segments and transitions
    if not all_samples:
        return np.array([])
        
    final_signal = np.concatenate(all_samples)
    
    return final_signal