import pyaudio
import time

def play_sound(samples, fs=44100, volume=0.5):
    # Source - https://stackoverflow.com/a/27978895
    p = pyaudio.PyAudio()
    output_bytes = (volume * samples).tobytes()

    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=fs,
                    output=True)

    start_time = time.time()
    stream.write(output_bytes)
    print("Played sound for {:.2f} seconds".format(time.time() - start_time))

    stream.stop_stream()
    stream.close()

    p.terminate()