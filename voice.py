import librosa
import IPython.display as ipd
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from glob import glob
from scipy import signal
import random
import cv2
import os

import pyaudio
import wave

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 3  #设置录音的时间长度


def load_audio(filename=None, sr=16000, second=3, samples=None):
    samples, sample_rate = librosa.load(filename, sr=sr)
    
    if second is not None and len(samples) < sr * second:
        samples = np.pad(samples, (0, sr * second - len(samples)), 'constant')
    if second is not None and len(samples) > sr * second:
        samples = samples[0:sr * second]
    return samples


def predict_voice(sample):
    beat = librosa.beat.beat_track(sample)
    v_max = sample.max()
    w = beat[0] + 250*v_max
    
    if w > 270:
        label = 'angry'
    elif w < 140:
        label = 'thinking'
    else:
        label = 'normal'
    return label


def play(sample):
    return ipd.Audio(sample, 16000, autoplay=True)

def recorder():
    #WAVE_OUTPUT_FILENAME = "./voice/thinking5.wav"
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    print("* recording")
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    
    return frames