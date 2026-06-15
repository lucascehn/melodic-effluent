import numpy as np

def generate_sine_wave(frequency=440.0, duration=5.0, fs=44100):
    # duration in seconds
    samples = (np.sin(2 * np.pi * np.arange(fs * duration) * frequency / fs)).astype(np.float32)
    return samples

def synthesize_controllable_mix(frequencies, partial_amplitudes, duration=5.0, fs=44100, onset=0.0):
    # frequencies is a list of frequencies to mix
    samples = np.zeros(int(fs * duration), dtype=np.float32)
    onset_samples = np.zeros(int(fs * onset), dtype=np.float32)
    for freq, amp in zip(frequencies, partial_amplitudes):  
        samples += amp * generate_sine_wave(frequency=freq, duration=duration, fs=fs)
    samples = np.concatenate((onset_samples, samples))[:int(fs * duration)]  # Ensure total length is duration
    return samples

def vibrato_effect(frequency=440.0, vibrato_rate=5.0, vibrato_depth=0.01, duration=5.0, fs=44100):
    # vibrato rate - how fast the pitch wobbles
    # vibrato depth - fraction of f0, so 1% pitch deviation

    t = np.arange(fs * duration) / fs
    f_instantaneous = frequency * (1 + vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t))
    samples = np.sin(2 * np.pi * np.cumsum(f_instantaneous) / fs)
    return samples.astype(np.float32)