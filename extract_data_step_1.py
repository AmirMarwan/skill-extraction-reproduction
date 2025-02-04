
from time import sleep, time
import os
if not os.getcwd().lower().endswith('res3'):
    os.chdir('Res3')

from multiprocessing import Process, Event, Queue
from threading import Thread

from utils import microphone_server
from utils import browser_instance_handler, app_webpage
from utils import get_custom_logger
logger = get_custom_logger(__name__)
import uvicorn

START_TIME = 0

def save_camera_info(file_name_queue=Queue(), output_file_queue=Queue(), mic_message_queue=Queue(), start_time_queue=Queue(), playback_finished_event=Event()):
    from PIL import Image
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    import os
    os.add_dll_directory('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin')
    os.add_dll_directory('C:/OpenCV_Build/opencv/build/install/x64/vc17/bin')
    import cv2

    model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", torch_dtype=torch.bfloat16).to("cuda")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    conversation = [{"role": "user",
        "content": [{"type": "image",},{"type": "text", "text": "In front of the bakery at the mall, describe the customers and what they are wearing."},],}]
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    start_time = start_time_queue.get()
    video_file = file_name_queue.get()
    video = cv2.VideoCapture(video_file)
    fps = video.get(cv2.CAP_PROP_FPS)
    output_file = output_file_queue.get()
    reset_counter = 0
    while True:
        sleep(0.01)
        if not start_time_queue.empty():
            start_time = start_time_queue.get()
        if not file_name_queue.empty():
            video_file = file_name_queue.get()
            video = cv2.VideoCapture(video_file)
            fps = video.get(cv2.CAP_PROP_FPS)
        if not output_file_queue.empty():
            output_file = output_file_queue.get()
        try:
            message_tmp = mic_message_queue.get(timeout=0.1)
            # start = time()
            message_mic = message_tmp[1]
            time_stamp = message_tmp[3]
            time_stamp_end = message_tmp[4]
            time_delta = time_stamp - start_time + START_TIME*2
            time_delta_end = time_stamp_end - start_time + START_TIME*2
            frame_number = int(fps * time_delta)
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            success, frame = video.read()
            frame = cv2.resize(frame, (960,540), interpolation=cv2.INTER_AREA)
            # cv2.imshow('frame', frame); cv2.waitKey(1)
            if not success:
                logger.error("Failed to extract the frame. Check the video file and timestamp.")
            im_pil = Image.fromarray(frame)
            inputs = processor(text=[text_prompt], images=[im_pil], padding=True, return_tensors="pt").to("cuda")
            output_ids = model.generate(**inputs, max_new_tokens=256)
            generated_ids = [output_ids[len(input_ids) :]for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
            output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].replace('\n\n','\n')
            # print(f'Elapsed time: {time()-start}')
            with open(output_file, 'a') as fh:
                fh.write(f'Camera timestamp: {time_delta:.1f}\nEnd of utterance: {time_delta_end:.1f}\n[Camera]\n{output_text}\n[Context]\nSalesman: {message_mic}\n\n\n')
        except Exception as e:
            if playback_finished_event.is_set():
                if mic_message_queue.empty():
                    reset_counter += 1
                    if reset_counter > 30:
                        playback_finished_event.clear()
                        reset_counter = 0
                else:
                    reset_counter = 0

def play_audio(file_name_queue=Queue(), start_time_queue_e=Queue(), playback_finished_event=Event()):
    import sounddevice as sd
    from pydub import AudioSegment
    import numpy as np

    device_name = 'CABLE-B Input (VB-Audio Cable B), Windows DirectSound'
    sd.default.device = device_name

    while True:
        aac_file_path = file_name_queue.get()
        audio = AudioSegment.from_file(aac_file_path, format="aac")
        start_ms = int(START_TIME * 1000)
        if start_ms > 0:
            audio = audio[start_ms:]
        samples = np.array(audio.get_array_of_samples())
        if audio.channels == 2:
            samples = samples.reshape((-1, 2))
        samples = samples / (2 ** 15)

        while playback_finished_event.is_set():
            sleep(0.1)
        sd.play(samples, samplerate=audio.frame_rate)
        start_time = time() + START_TIME
        start_time_queue_e.put(start_time)
        sd.wait()
        playback_finished_event.set()



RESOURCE_FILES = [
                #   'Videos/data/0/2024-08-01 11-12-04', #test
                #   'Videos/data/6/2024-08-01 11-12-04', #test
                  'Videos/data/1/2024-08-04 15-40-56',
                  'Videos/data/2/2024-08-04 18-12-59',
                  'Videos/data/3/2024-08-23 15-51-03',
                  'Videos/data/4/2024-08-27 15-58-36',
                  'Videos/data/5/2024-09-03 16-18-56',
                  ]
    
if __name__ == '__main__':

    mic_start_queue_e = Queue()
    mic_message_queue_e = Queue()
    start_time_queue_e = Queue()
    file_name_video_queue_e = Queue()
    file_name_audio_queue_e = Queue()
    output_file_queue_e = Queue()
    playback_finished_event_e = Event()

    Thread(target=microphone_server, args=(mic_start_queue_e, mic_message_queue_e, 1234)).start()
    Thread(target=uvicorn.run, args=(app_webpage,), kwargs={'host':"127.0.0.1", 'port':5004, 'access_log':False}).start()
    Process(target=browser_instance_handler, kwargs={'mode':'extract_data'}).start()

    Process(target=play_audio, args=(file_name_audio_queue_e, start_time_queue_e, playback_finished_event_e,)).start()
    Process(target=save_camera_info, args=(file_name_video_queue_e, output_file_queue_e, mic_message_queue_e, start_time_queue_e, playback_finished_event_e)).start()

    sleep(5)

    for file_name in RESOURCE_FILES:
        output_file = 'resources/raw_timestamped/raw_t'
        if not os.path.exists(output_file[:output_file.rfind('/')]):
            os.makedirs(output_file[:output_file.rfind('/')])
        file_name_video_queue_e.put(file_name + '.mkv')
        file_name_audio_queue_e.put(file_name + '.aac')
        output_file_queue_e.put(output_file + f'_{os.path.basename(os.path.dirname(file_name))}.txt')
        mic_start_queue_e.put(1)
        while not playback_finished_event_e.is_set():
            sleep(0.5)
        while playback_finished_event_e.is_set():
            sleep(0.5)