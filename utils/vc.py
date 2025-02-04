from multiprocessing import Process

class Harvest(Process):
    def __init__(self, inp_q, opt_q):
        Process.__init__(self)
        self.inp_q = inp_q
        self.opt_q = opt_q
    def run(self):
        import numpy as np
        import pyworld
        while 1:
            idx, x, res_f0, n_cpu, ts = self.inp_q.get()
            f0, t = pyworld.harvest(
                x.astype(np.double),
                fs=16000,
                f0_ceil=1100,
                f0_floor=50,
                frame_period=10,
            )
            res_f0[idx] = f0
            if len(res_f0.keys()) >= n_cpu:
                self.opt_q.put(ts)

def vc_handler(start_vc):
    
    from time import sleep, time
    from threading import Thread
    import torch
    import numpy as np
    import sounddevice as sd
    import torch.nn.functional as F
    import torchaudio.transforms as tat
    import sys
    import librosa
    import rvc.tools.rvc_for_realtime as rvc_for_realtime
    # from rvc.lib import rtrvc as rvc_for_realtime
    from rvc.configs.config import Config
    from rvc.tools.torchgate import TorchGate
    from multiprocessing import Queue, cpu_count
    import faiss.loader

    inp_q = Queue()
    opt_q = Queue()
    n_cpu = min(cpu_count(), 2)
    for _ in range(n_cpu):
        Harvest(inp_q, opt_q).start()
    

    class VoiceChanger:

        def __init__(self,
                     desired_input_device_name = 'Réseau de microphones (Technolo',
                     desired_output_device_name = 'CABLE-A Input (VB-Audio Cable A, MME'):
            
            sd.default.device = desired_input_device_name, desired_output_device_name

            self.config = Config()
            self.config.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
            self.pth_path: str = "model_assets/kikiV1.pth"
            self.index_path: str = "model_assets/kikiV1.index"
            self.pitchg: int = 0
            self.samplerate: int = 16000
            self.block_time: float = 0.5
            self.buffer_num: int = 1
            self.threhold: int = -60
            self.crossfade_time: float = 0.1
            self.extra_time: float = 1.0
            self.index_rate = 0.3
            self.rms_mix_rate = 0.7
            self.n_cpu = 2
            self.f0method = "rmvpe"
            self.function = "vc"
            self.flag_vc = 0
        
        def start_vc(self):
            if self.flag_vc:
                return
            torch.cuda.empty_cache()
            self.flag_vc = True
            # print(self.config.__dict__)
            self.config.use_jit = True
            self.rvc = rvc_for_realtime.RVC(self.pitchg,self.pth_path,self.index_path,self.index_rate,self.n_cpu,inp_q,opt_q,self.config,self.rvc if hasattr(self, "rvc") else None,)
            self.samplerate = self.rvc.tgt_sr
            self.zc = self.rvc.tgt_sr // 100
            self.block_frame = (int(np.round(self.block_time* self.samplerate/ self.zc)) * self.zc)
            self.block_frame_16k = 160 * self.block_frame // self.zc
            self.crossfade_frame = (int(np.round(self.crossfade_time* self.samplerate/ self.zc)) * self.zc)
            self.sola_search_frame = self.zc
            self.extra_frame = (int(np.round(self.extra_time* self.samplerate/ self.zc)) * self.zc)
            self.input_wav: torch.Tensor = torch.zeros(self.extra_frame + self.crossfade_frame + self.sola_search_frame+ self.block_frame, device=self.config.device, dtype=torch.float32,)
            self.input_wav_res: torch.Tensor = torch.zeros(160 * self.input_wav.shape[0] // self.zc,device=self.config.device,dtype=torch.float32,)
            self.pitch: np.ndarray = np.zeros(self.input_wav.shape[0] // self.zc,dtype="int32",)
            self.pitchf: np.ndarray = np.zeros(self.input_wav.shape[0] // self.zc,dtype="float64",)
            self.sola_buffer: torch.Tensor = torch.zeros(self.crossfade_frame, device=self.config.device, dtype=torch.float32)
            self.nr_buffer: torch.Tensor = self.sola_buffer.clone()
            self.output_buffer: torch.Tensor = self.input_wav.clone()
            self.res_buffer: torch.Tensor = torch.zeros(2 * self.zc, device=self.config.device, dtype=torch.float32)
            self.valid_rate = 1 - (self.extra_frame - 1) / self.input_wav.shape[0]
            self.fade_in_window: torch.Tensor = (torch.sin(0.5 * np.pi * torch.linspace(0.0,1.0,steps=self.crossfade_frame,device=self.config.device,dtype=torch.float32,)) ** 2)
            self.fade_out_window: torch.Tensor = 1 - self.fade_in_window
            self.resampler = tat.Resample(orig_freq=self.samplerate,new_freq=16000,dtype=torch.float32,).to(self.config.device)
            self.tg = TorchGate(sr=self.samplerate, n_fft=4 * self.zc, prop_decrease=0.9).to(self.config.device)
            thread_vc = Thread(target=self.soundinput)
            thread_vc.start()

        def soundinput(self):
            channels = 2
            with sd.Stream(channels=channels,callback=self.audio_callback,blocksize=self.block_frame,samplerate=self.samplerate,dtype="float32",) as stream:
                # stream_latency = stream.latency[-1]
                while self.flag_vc:
                    sleep(self.block_time)
                    # print("Audio block passed.")
            # print("ENDing VC")

        def audio_callback(self, indata: np.ndarray, outdata: np.ndarray, frames, times, status):
            indata = librosa.to_mono(indata.T)
            if self.threhold > -60:
                rms = librosa.feature.rms(y=indata, frame_length=4 * self.zc, hop_length=self.zc)
                db_threhold = (librosa.amplitude_to_db(rms, ref=1.0)[0] < self.threhold)
                for i in range(db_threhold.shape[0]):
                    if db_threhold[i]:
                        indata[i * self.zc : (i + 1) * self.zc] = 0
            self.input_wav[: -self.block_frame] = self.input_wav[self.block_frame :].clone()
            self.input_wav[-self.block_frame :] = torch.from_numpy(indata).to(self.config.device)
            self.input_wav_res[: -self.block_frame_16k] = self.input_wav_res[self.block_frame_16k :].clone()
            self.input_wav_res[-self.block_frame_16k - 160 :] = self.resampler(self.input_wav[-self.block_frame - 2 * self.zc :])[160:]
            # infer
            f0_extractor_frame = self.block_frame_16k + 800
            if self.f0method == "rmvpe":
                f0_extractor_frame = (5120 * ((f0_extractor_frame - 1) // 5120 + 1) - 160)
            infer_wav = self.rvc.infer(self.input_wav_res,self.input_wav_res[-f0_extractor_frame:].cpu().numpy(),self.block_frame_16k,self.valid_rate,self.pitch,self.pitchf,self.f0method,)
            infer_wav = infer_wav[-self.crossfade_frame - self.sola_search_frame - self.block_frame :]
            # volume envelop mixing
            if self.rms_mix_rate < 1 and self.function == "vc":
                # start_time = time()
                rms1 = librosa.feature.rms(y=self.input_wav_res[-160 * infer_wav.shape[0] // self.zc :].cpu().numpy(),frame_length=640,hop_length=160,)
                rms1 = torch.from_numpy(rms1).to(self.config.device)
                rms1 = F.interpolate(rms1.unsqueeze(0),size=infer_wav.shape[0] + 1,mode="linear",align_corners=True,)[0, 0, :-1]
                rms2 = librosa.feature.rms(y=infer_wav[:].cpu().numpy(),frame_length=4 * self.zc,hop_length=self.zc,)
                rms2 = torch.from_numpy(rms2).to(self.config.device)
                rms2 = F.interpolate(rms2.unsqueeze(0),size=infer_wav.shape[0] + 1,mode="linear",align_corners=True,)[0, 0, :-1]
                rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-3)
                infer_wav *= torch.pow(rms1 / rms2, torch.tensor(1 - self.rms_mix_rate))
                # print("rms_mix_rate", time() - start_time)
            conv_input = infer_wav[None, None, : self.crossfade_frame + self.sola_search_frame]
            cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
            cor_den = torch.sqrt(F.conv1d(conv_input**2,torch.ones(1, 1, self.crossfade_frame, device=self.config.device),) + 1e-8)
            sola_offset = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])
            infer_wav = infer_wav[sola_offset : sola_offset + self.block_frame + self.crossfade_frame]
            infer_wav[: self.crossfade_frame] *= self.fade_in_window
            infer_wav[: self.crossfade_frame] += self.sola_buffer * self.fade_out_window
            self.sola_buffer[:] = infer_wav[-self.crossfade_frame :]
            outdata[:] = (infer_wav[: -self.crossfade_frame].repeat(2, 1).t().cpu().numpy())

        def stop_vc(self):
            self.flag_vc = False

    voice_changer = VoiceChanger()
    
    import psutil, os
    parent = psutil.Process(os.getpid())
    parent.nice(psutil.HIGH_PRIORITY_CLASS)
    for child in parent.children():
        child.nice(psutil.HIGH_PRIORITY_CLASS)
    while True:
        flag = start_vc.get()
        if flag:
            voice_changer.start_vc()
        else:
            voice_changer.stop_vc()

if __name__ == '__main__':
    import os
    if not os.getcwd().lower().endswith('res3'):
        os.chdir('Res3')
    # os.environ["USE_FLASH_ATTENTION"] = "1"

    from multiprocessing import Queue
    start_vc = Queue()

    Process(target=vc_handler, args=(start_vc,)).start()
    while True:
        start_vc.put(0)
        input('stop')
        start_vc.put(1)
        input('start')
