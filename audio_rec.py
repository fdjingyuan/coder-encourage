import threading
import pyaudio
import wave
import librosa
import cv2
import numpy as np

global_image = None

class StoppableThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition."""
    def __init__(self):
        super(StoppableThread, self).__init__()
        self._stop_event = threading.Event()
    def stop(self):
        self._stop_event.set()
    @property
    def stopped(self):
        return self._stop_event.is_set()

class AudioRec(StoppableThread):

    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = 3  #设置录音的时间长度
    WAVE_OUTPUT_FILENAME = "./test.wav"
    lock = threading.Lock()
    image = None
    samples = None

    def __init__(self):
        super(AudioRec, self).__init__()
    
    def record(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        frames_per_buffer=self.CHUNK)
        print("* recording")
        frames = []
        for i in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
            data = stream.read(self.CHUNK)
            frames.append(data)
        print("* done recording")

        stream.stop_stream()
        stream.close()
        p.terminate()
        wf = wave.open(self.WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
    
    def get_pic(self):
        samples, sample_rate = librosa.load(self.WAVE_OUTPUT_FILENAME)
        self.samples = samples.copy()
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from matplotlib.figure import Figure
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.gca()
        ax.plot(samples)
        ax.axis('off')
        canvas.draw()       # draw the canvas, cache the renderer
        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
        return image
    
    def run(self):
        while True:
            if self.stopped:
                return
            self.record()
            print('record ok!')
            self.image = self.get_pic()
