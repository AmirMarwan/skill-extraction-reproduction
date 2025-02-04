
def audio_passthrough_handler(start_audio_passthru):

    from time import sleep
    from threading import Thread

    class AudioStreamer:
        def __init__(self,
                    desired_input_device_name = 'Intel® Smart Sound pour microphones numériques), Windows DirectSound',
                    desired_output_device_name = 'CABLE-A Input (VB-Audio Cable A), Windows DirectSound'):
            import sounddevice as sd
            self.sd = sd

            self.sd.default.device = desired_input_device_name, desired_output_device_name
            
            self.is_passthrough = False
            self.sd.default.samplerate = 48000
            self.sd.default.channels = 2
            self.sd.default.dtype = 'int16'
            self.sd.default.blocksize = 0
            self.sd.default.latency = 'low'

        def start_audio_stream(self):
            self.is_passthrough = True
            Thread(target=self.audio_passthrough, daemon=True).start()

        def stop_audio_stream(self):
            self.is_passthrough = False

        def audio_passthrough(self):
            def callback(indata, outdata, frames, time, status):
                if status:
                    print(status)
                outdata[:] = indata
            with self.sd.RawStream(callback=callback):
                while self.is_passthrough:
                    sleep(0.5)

    audio_streamer = AudioStreamer()
    
    import psutil, os
    parent = psutil.Process(os.getpid())
    parent.nice(psutil.HIGH_PRIORITY_CLASS)
    for child in parent.children():
        child.nice(psutil.HIGH_PRIORITY_CLASS)
    while True:
        flag = start_audio_passthru.get()
        if flag:
            audio_streamer.start_audio_stream()
        else:
            audio_streamer.stop_audio_stream()

if __name__ == '__main__':
    from multiprocessing import Process, Queue

    start_audio_passthru = Queue()

    Process(target=audio_passthrough_handler, args=(start_audio_passthru,)).start()
    while True:
        start_audio_passthru.put(0)
        input('stop')
        start_audio_passthru.put(1)
        input('start')
