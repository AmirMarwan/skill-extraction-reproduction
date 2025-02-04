
from time import sleep, time
import os
if not os.getcwd().lower().endswith('res3'):
    os.chdir('Res3')
from utils import get_custom_logger
logger = get_custom_logger(__name__)

import csv
from tqdm import tqdm
from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
import os
os.add_dll_directory('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin')
os.add_dll_directory('C:/OpenCV_Build/opencv/build/install/x64/vc17/bin')
import cv2

RAW_DATA_FOLDER = 'resources/raw_timestamped/'
RAW_DATA_FILES = [RAW_DATA_FOLDER+f for f in os.listdir(RAW_DATA_FOLDER) if f.endswith('.txt')]

RESOURCE_FILES = {'1':'Videos/data/1/2024-08-04 15-40-56',
                  '2':'Videos/data/2/2024-08-04 18-12-59',
                  '3':'Videos/data/3/2024-08-23 15-51-03',
                  '4':'Videos/data/4/2024-08-27 15-58-36',
                  '5':'Videos/data/5/2024-09-03 16-18-56',}

def parse_input_file(file_path):
    data_entries = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        current_entry = {}
        camera_info = []
        context_info = []
        inside_camera = False
        inside_context = False
        for line in lines:
            line = line.strip()
            if line.startswith("Camera timestamp:"):
                current_entry['Camera timestamp'] = float(line.split(":")[1].strip())
            elif line.startswith("End of utterance:"):
                current_entry['End of utterance'] = float(line.split(":")[1].strip())
            elif "[Camera]" in line:
                inside_camera = True
                inside_context = False
            elif "[Context]" in line:
                inside_context = True
                inside_camera = False
            elif inside_camera:
                camera_info.append(line)
            elif inside_context:
                context_info.append(line)
            if not line:
                if current_entry:
                    current_entry['[Camera]'] = ' '.join(camera_info).strip() or 'NaN'
                    current_entry['[Context]'] = ' '.join(context_info).strip() or 'no utterance'
                    data_entries.append(current_entry)
                    current_entry = {}
                    camera_info = []
                    context_info = []
    return data_entries

def calculate_time_delta_and_generate_output(data_entries, output_path):
    video_file = RESOURCE_FILES[file_name.split('_')[-1].split('.')[0]]+'.mkv'
    video = cv2.VideoCapture(video_file)
    fps = video.get(cv2.CAP_PROP_FPS)
    
    output_data = []

    last_processed_timestamp = None
    if os.path.exists(output_path):
        with open(output_path, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            output_data.append(next(csvreader, None))
            for row in csvreader:
                output_data.append(row)
                last_processed_timestamp = row[0]

    start_index = 0
    if last_processed_timestamp:
        for i, entry in enumerate(data_entries):
            if str(entry['Camera timestamp']) == last_processed_timestamp:
                start_index = i + 1
                break

    with open(output_path, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        if not output_data:
            csvwriter.writerow(['Camera timestamp', 'End of utterance', 'Time delta', '[Camera]', '[Context]'])
        if start_index == 0:
            current = data_entries[0]
            csvwriter.writerow([current['Camera timestamp'],
                                current['End of utterance'],
                                f'{0.0:.1f}',
                                current['[Camera]'],
                                current['[Context]']])
            start_index = 1

    for i in tqdm(range(start_index, len(data_entries)), desc="Processing"):
        current = data_entries[i]
        previous = data_entries[i - 1]
        
        median_time = (previous['End of utterance'] + current['Camera timestamp']) / 2
        delta_time1 = median_time - previous['End of utterance']
        delta_time2 = current['Camera timestamp'] - median_time

        frame_number = int(fps * median_time)
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
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].replace('\n','')

        with open(output_path, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([f'{median_time:.1f}',
                                f'{median_time:.1f}',
                                f"{delta_time1:.1f}",
                                output_text,
                                'no utterance is needed'])

            csvwriter.writerow([current['Camera timestamp'],
                                current['End of utterance'],
                                f"{delta_time2:.1f}",
                                current['[Camera]'],
                                current['[Context]']])

if __name__ == '__main__':
    model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", torch_dtype=torch.bfloat16).to("cuda")
    processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    conversation = [{"role": "user",
        "content": [{"type": "image",},{"type": "text", "text": "In front of the bakery at the mall, describe the customers and what they are wearing."},],}]
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    for file_name in tqdm(RAW_DATA_FILES, desc="Files"):
        output_file = file_name.replace('raw_timestamped', 'raw').replace('.txt', '.csv')
        entries = parse_input_file(file_name)
        if not os.path.exists(output_file[:output_file.rfind('/')]):
            os.makedirs(output_file[:output_file.rfind('/')])
        calculate_time_delta_and_generate_output(entries, output_file)
