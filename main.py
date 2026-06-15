import numpy as np
from sound_player import play_sound
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hann

def main():
    volume = 0.5  # range [0.0, 1.0]
    fs = 44100  # sampling rate, Hz, must be integer
    duration = 5.0  # in seconds, may be float
    f = 440.0  # sine frequency, Hz, may be float

    # generate 440 hz sine wave
    samples = (np.sin(2 * np.pi * np.arange(fs * duration) * f / fs)).astype(np.float32)

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ShortTimeFFT.html#scipy.signal.ShortTimeFFT
    hop_size = 512
    window_size = 4 * hop_size
    w = hann(window_size, sym=True)
    SFT = ShortTimeFFT(w, hop=hop_size, fs=fs)
    Sx = SFT.stft(samples)

    print(SFT.invertible and "STFT is invertible" or "STFT is not invertible")
    # from istft. dtype = float == float64, but play_sound expects float32
    reversed_samples = SFT.istft(Sx, k1=len(samples)).astype(np.float32)
    print(np.allclose(samples, reversed_samples) and "Perfect reconstruction" or "Reconstruction differs")

    play_sound(reversed_samples, fs=fs, volume=volume)
    play_sound(samples, fs=fs, volume=volume)


if __name__ == "__main__":
    main()
