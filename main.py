import numpy as np
from sound_player import play_sound
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hann
from audio_generator import generate_sine_wave, synthesize_controllable_mix, vibrato_effect

def main_stft_test():
    samples = generate_sine_wave()

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ShortTimeFFT.html#scipy.signal.ShortTimeFFT
    hop_size = 512
    fs = 44100
    window_size = 4 * hop_size
    w = hann(window_size, sym=True)
    SFT = ShortTimeFFT(w, hop=hop_size, fs=fs)
    Sx = SFT.stft(samples)

    print(SFT.invertible and "STFT is invertible" or "STFT is not invertible")
    # from istft. dtype = float == float64, but play_sound expects float32
    reversed_samples = SFT.istft(Sx, k1=len(samples)).astype(np.float32)
    print(np.allclose(samples, reversed_samples) and "Perfect reconstruction" or "Reconstruction differs")

    volume = 0.5
    play_sound(reversed_samples, fs=fs, volume=volume)
    play_sound(samples, fs=fs, volume=volume)

def main():
    harmonic_combs = [440 * (i + 1) for i in range(4)]
    partial_amplitudes = [0.5, 0.5, 0.5, 0.5]

    a4_samples = synthesize_controllable_mix(harmonic_combs, partial_amplitudes, duration=1.0, fs=44100)
    fs = 44100
    volume = 0.5
    print("playing a4")
    play_sound(a4_samples, fs=fs, volume=volume)

    harmonic_combs = [440 * (i + 1) for i in range(4)]
    partial_amplitudes = [1.0, 0.2, 0.1, 0.05]
    samples = synthesize_controllable_mix(harmonic_combs, partial_amplitudes, duration=1.0, fs=44100, onset=0.25)
    fs = 44100
    volume = 0.5
    print("playing a4 different timbre (different partials with onset)")
    play_sound(samples, fs=fs, volume=volume)

    harmonic_combs = [329.63 * (i + 1) for i in range(4)]
    partial_amplitudes = [1.0, 0.2, 0.1, 0.05]
    e4_samples = synthesize_controllable_mix(harmonic_combs, partial_amplitudes, duration=1.0, fs=44100)
    fs = 44100
    volume = 0.5
    print("playing e4")
    play_sound(e4_samples, fs=fs, volume=volume)

    vibrato_samples = vibrato_effect(frequency=440.0, vibrato_rate=5.0, vibrato_depth=0.01, duration=1.0, fs=44100)
    print("playing a4 vibrato")
    play_sound(vibrato_samples, fs=fs, volume=volume)

    mixed_samples = (e4_samples + a4_samples) / 2
    print("playing mix of a4 and e4")
    play_sound(mixed_samples, fs=fs, volume=volume)

if __name__ == "__main__":
    main()
