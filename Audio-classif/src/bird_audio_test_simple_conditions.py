import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio
import os


def load_wav_make_simple(filename):
    # Load encoded wav file
    file_contents = tf.io.read_file(filename)  # Returns a byte encoded string
    # Decode wav (tensors by channels)
    wav, sample_rate = tf.audio.decode_wav(
        file_contents, desired_channels=1
    )  # We decode the string and specify the channel that we want: mono or stereo
    # Removes trailing axis
    wav = tf.squeeze(
        wav, axis=-1
    )  # Squeezing so that we remove the trailing axis from wav. Wav shape was (n, 1), we only need the n
    sample_rate = tf.cast(
        sample_rate, dtype=tf.int64
    )  # Format of int instead of string
    # Goes from 44100Hz to 16000hz - amplitude of the audio signal
    wav = tfio.audio.resample(
        wav, rate_in=sample_rate, rate_out=16000
    )  # Resampling the wave in the format of Int and 16000Hz, so we significantly reduced the size of the wav file.
    return wav


def preprocess(file_path, label):
    if os.path.splitext(file_path)[1] != ".wav":
        raise Exception("Sorry, only .wav files are allowed!")
    wav = load_wav_make_simple(file_path)  # Load our data into waveform
    wav = wav[
        :80000
    ]  # We are going to take a bit less than 10 seconds of each waveform
    zero_padding = tf.zeros(
        [80000] - tf.shape(wav), dtype=tf.float32
    )  # We are padding a file with zeros if it is too short in length
    wav = tf.concat([zero_padding, wav], 0)
    spectrogram = tf.signal.stft(
        wav, frame_length=1000, frame_step=50
    )  # Short time fourier transform
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram, label


source = r"C:\Users\laury\OneDrive\Stalinis kompiuteris\SE4CSAI\Data\My own\Audio detection\Audio files\warblrb10k_public_wav\wav\0a4ef72d-611f-4adc-9cf1.wav"
myreturn = preprocess(source, 1)
print(myreturn[0].shape)

if myreturn[0].shape == (1581, 513, 1):
    print("Test passed")
else:
    print("Test failed")
