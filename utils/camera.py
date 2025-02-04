try:
    from . import get_custom_logger
    logger = get_custom_logger(__name__)
except:
    pass

def camera_process_handler(camera_frame_queue, cropped_frame_queue, cropped_frame_queue_2, frame_gui_queue,
                           human_detected_queue, human_detected_bool, human_current_queue,
                           costumer_name_queue, costumer_name_bool,
                           image_info_queue, image_info_bool, **camera_kwargs):
    import os
    from threading import Thread
    from time import sleep, time
    os.add_dll_directory('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin')
    os.add_dll_directory('C:/OpenCV_Build/opencv/build/install/x64/vc17/bin')
    import cv2
    import numpy as np
    from collections import deque
    # print(logger)
    try:
        global logger
        assert logger
    except:
        if not os.getcwd().lower().endswith('res3'):
            os.chdir('Res3')
        import sys
        sys.path.append('.')
        from utils import get_custom_logger
        logger = get_custom_logger(__name__)
    
    capture_camera_feed_fps = camera_kwargs.pop('capture_camera_feed_fps', 20)
    cam_feed_res = camera_kwargs.pop('cam_feed_res', (640, 360))
    human_detection_fps = camera_kwargs.pop('human_detection_fps', 3)
    costumer_name_fps = camera_kwargs.pop('costumer_name_fps', 2)
    periodic_camera_fps = camera_kwargs.pop('periodic_camera_fps', 2)

    def capture_camera_feed(camera_frame_queue, human_current_queue, frame_gui_queue):
        CAMERA_FEED = 1
        cap = cv2.VideoCapture(CAMERA_FEED)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_feed_res[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_feed_res[1])
        xy0 = (0, 0, 0, 0)
        xy = []
        draw_counter = 0
        while True:
            sleep(1/capture_camera_feed_fps)
            try:
                ret, frame = cap.read()
                if not ret:
                    raise Exception("Failed to capture image")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                while True:
                    try:
                        if camera_frame_queue.full():
                            camera_frame_queue.get()
                        camera_frame_queue.put(frame)
                        break
                    except:
                        pass
            except Exception as e:
                logger.error(f'Camera feed exception {e}')
            else:
                try:
                    if not human_current_queue.empty():
                        draw_counter = 0
                        human_current = human_current_queue.get()
                        xy0 = human_current[0]
                        xy = []
                        for idx in range(1, len(human_current)):
                            xy.append(human_current[idx])
                    frame_gui = frame.copy()
                    if draw_counter < 15:
                        draw_counter += 1
                        frame_gui = cv2.rectangle(frame_gui, (xy0[0], xy0[1]), (xy0[2], xy0[3]), (0, 255, 0), 2)
                        for idx in range(1, len(xy)):
                            frame_gui = cv2.rectangle(frame_gui, (xy[idx][0], xy[idx][1]), (xy[idx][2], xy[idx][3]), (255, 0, 0), 2)
                    # if frame_gui_queue.full():
                    #     frame_gui_queue.get()
                    
                    while True:
                        try:
                            if frame_gui_queue.full():
                                frame_gui_queue.get(timeout=0.01)
                            frame_gui_queue.put(frame_gui)
                            break
                        except:
                            pass
                    
                except Exception as e:
                    logger.error(f'Camera GUI frame {e}')

    def human_detection(camera_frame_queue, human_detected_queue, human_detected_bool, cropped_frame_queue, cropped_frame_queue_2, human_current_queue):
        from ultralytics import YOLO
        model = YOLO('model_assets/yolo11s.pt')

        moving_max = deque(maxlen=3)
        cropped_list = deque(maxlen=3)
        human_current = deque(maxlen=3)
        while True:
            sleep(1/human_detection_fps)
            try:
                if not human_detected_bool.empty():
                    while not human_detected_bool.empty():
                        bb = human_detected_bool.get()
                    if bb:
                        image = camera_frame_queue.get()
                        # start_time = time()
                        results = model.predict(image, classes=[0], verbose=False, half=True)
                        if len(results[0].boxes) > 0:
                            max_conf = max(results[0].boxes[0].conf)
                            if max_conf > 0.8:
                                human_current = []
                                moving_max.append(max_conf)
                                cropped_list.clear()
                                for i in range(min(len(results[0].boxes),3)):
                                    if results[0].boxes[i].conf > 0.7:
                                        x1, y1, x2, y2 = map(int, results[0].boxes[i].xyxy[0].tolist())
                                        cropped_list.append(image[y1:y2, x1:x2])
                                        human_current.append((x1, y1, x2, y2))
                                while True:
                                    try:
                                        if cropped_frame_queue.full():
                                            cropped_frame_queue.get(timeout=0.01)
                                        cropped_frame_queue.put(image)
                                        # cropped_frame_queue.put(cropped_list[0])
                                        break
                                    except:
                                        pass
                                while True:
                                    try:
                                        if cropped_frame_queue_2.full():
                                            cropped_frame_queue_2.get(timeout=0.01)
                                        cropped_frame_queue_2.put(cropped_list)
                                        break
                                    except:
                                        pass
                                while True:
                                    try:
                                        if human_current_queue.full():
                                            human_current_queue.get()
                                        human_current_queue.put(human_current)
                                        break
                                    except:
                                        pass
                            else:
                                moving_max.append(0)
                        else:
                            moving_max.append(0)
                        while True:
                            try:
                                if human_detected_queue.full():
                                    human_detected_queue.get(timeout=0.01)
                                human_detected_queue.put(len(cropped_list) if max(moving_max) else False)
                                break
                            except:
                                pass
                        # print(f'Human detection time: {time() - start_time}')
            except Exception as e:
                logger.error(f'Human detection exception {e}')

    def costumer_name(cropped_frame_queue_2, costumer_name_queue, costumer_name_bool):
        import face_recognition
        import pickle as pk
        try:
            with open('secrets/faces.pk', 'rb') as ff:
                known_face_encodings, known_face_names = pk.load(ff)
        except:
            known_face_encodings = [np.zeros(128),]
            known_face_names = ['0',]
        moving_detected_names = deque(maxlen=4)
        # face_detector = cv2.CascadeClassifier('secrets/haarcascade_frontalface_alt.xml')

        while True:
            try:
                sleep(1/costumer_name_fps)
                if not costumer_name_bool.empty():
                    while not costumer_name_bool.empty():
                        bb = costumer_name_bool.get()
                    if bb:
                        # start_time = time()
                        if not cropped_frame_queue_2.empty():
                            frame_list = cropped_frame_queue_2.get()
                        else:
                            frame_list = []
                        detected_names = set()
                        for frame in frame_list:
                            rgb_frame = frame[:, :, ::-1]
                            rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
                            face_locations = face_recognition.face_locations(rgb_frame, 3, model="cnn")
                            # gray_img = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
                            # face_locations = face_detector.detectMultiScale(rgb_frame, 1.1, 2)
                            # print(face_locations)
                            if len(face_locations):
                                # face_locations_2 = []
                                # for x, y, w, h in face_locations:
                                #     face_locations_2.append((x, y, w, h))
                                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, model="large")
                                for face_encoding in face_encodings:
                                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.8)
                                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                                    best_match_index = np.argmin(face_distances)
                                    if matches[best_match_index]:
                                        name = known_face_names[best_match_index]
                                    else:
                                        name = f"Costumer {len(known_face_encodings) + 1}"
                                        known_face_encodings.append(face_encoding)
                                        known_face_names.append(name)
                                        with open('secrets/faces.pk', 'wb') as ff:
                                            pk.dump((known_face_encodings, known_face_names), ff, protocol=pk.HIGHEST_PROTOCOL)
                                    detected_names.add(name)
                        moving_detected_names.append(detected_names)
                        output = set().union(*moving_detected_names)
                        if len(output) > 0:
                            while True:
                                try:
                                    if costumer_name_queue.full():
                                        costumer_name_queue.get(timeout=0.01)
                                    costumer_name_queue.put(output)
                                    break
                                except:
                                    pass
                        # print(f'Costumer name time: {time() - start_time}')
            except Exception as e:
                logger.error(f'Costumer name exception {e}')

    def periodic_camera(cropped_frame_queue, image_info_queue, image_info_bool):
        import torch
        from PIL import Image
        from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig

        # quantization_config = BitsAndBytesConfig(load_in_8bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
        processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)

        while True:
            try:
                sleep(1/periodic_camera_fps)
                if not image_info_bool.empty():
                    while not image_info_bool.empty():
                        bb = image_info_bool.get()
                    if bb:
                        rgb_frame = cropped_frame_queue.get()
                        # start_time = time()
                        # rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
                        im_pil = Image.fromarray(rgb_frame)
                        # ratio = 600/im_pil.height
                        # ratio = 1
                        # im_pil = im_pil.resize((int(im_pil.height*ratio), int(im_pil.width*ratio)))
                        # inputs = processor(images=im_pil, text=f'The costumer is wearing', return_tensors="pt").to("cuda", torch.float16)
                        # inputs = processor(images=im_pil, text=f'<CAPTION>', return_tensors="pt").to("cuda", torch.float16)
                        inputs = processor(images=im_pil, text=f'<DETAILED_CAPTION>', return_tensors="pt").to("cuda", torch.bfloat16)
                        # inputs = processor(images=im_pil, return_tensors="pt").to("cuda", torch.float16)
                        out = model.generate(**inputs, max_new_tokens=50, num_beams=3, early_stopping=True)
                        result = processor.decode(out[0], skip_special_tokens=True)
                        while True:
                            try:
                                if image_info_queue.full():
                                    image_info_queue.get(timeout=0.01)
                                image_info_queue.put(result)
                                break
                            except:
                                pass
                        # print(f'Periodic camera time: {time() - start_time}')
            except Exception as e:
                logger.error(f'Periodic camera exception {e}')
    

    camera_feed_thread = Thread(target=capture_camera_feed, args=(camera_frame_queue, human_current_queue, frame_gui_queue))
    camera_feed_thread.daemon = True
    camera_feed_thread.start()
    logger.info("Camera feed thread started.")
    human_detection_thread = Thread(target=human_detection, args=(camera_frame_queue, human_detected_queue, human_detected_bool, cropped_frame_queue, cropped_frame_queue_2, human_current_queue))
    human_detection_thread.daemon = True
    human_detection_thread.start()
    logger.info("Human detection thread started.")
    costumer_name_thread = Thread(target=costumer_name, args=(cropped_frame_queue_2, costumer_name_queue, costumer_name_bool))
    costumer_name_thread.daemon = True
    costumer_name_thread.start()
    logger.info("Costumer name thread started.")
    periodic_camera_thread = Thread(target=periodic_camera, args=(cropped_frame_queue, image_info_queue, image_info_bool))
    periodic_camera_thread.daemon = True
    periodic_camera_thread.start()
    logger.info("Periodic camera thread started.")
    while True:
        sleep(2)


if __name__ == '__main__':

    from multiprocessing import Process, Queue
    from threading import Thread

    camera_frame_queue = Queue(maxsize=2)
    cropped_frame_queue = Queue(maxsize=1)
    cropped_frame_queue_2 = Queue(maxsize=1)
    frame_gui_queue = Queue(maxsize=2)
    human_detected_queue = Queue()
    human_detected_bool = Queue()
    human_current_queue = Queue()
    costumer_name_queue = Queue()
    costumer_name_bool = Queue()
    image_info_queue = Queue()
    image_info_bool = Queue()
    image_info_queue = Queue()
    image_info_bool = Queue()
    camera_kwargs={'capture_camera_feed_fps': 20, 'cam_feed_res': (640, 360), 'human_detection_fps': 3, 'costumer_name_fps': 2, 'periodic_camera_fps': 2}
    
    camera_thread = Process(target=camera_process_handler, args=(camera_frame_queue, cropped_frame_queue, cropped_frame_queue_2, frame_gui_queue,
                                                                 human_detected_queue, human_detected_bool, human_current_queue,
                                                                 costumer_name_queue, costumer_name_bool,
                                                                 image_info_queue, image_info_bool,), kwargs=camera_kwargs)
    camera_thread.daemon = True
    camera_thread.start()
    
    def show_camera_feed(camera_frame_queue, cropped_frame_queue, frame_gui_queue):
        import os
        os.add_dll_directory('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin')
        os.add_dll_directory('C:/OpenCV_Build/opencv/build/install/x64/vc17/bin')
        import cv2
        while True:
            if not camera_frame_queue.empty():
                frame = camera_frame_queue.get()
                if frame is not None:
                    cv2.imshow("Camera Feed", frame)
            if not frame_gui_queue.empty():
                frame = frame_gui_queue.get()
                if frame is not None:
                    cv2.imshow("Camera Feed gui", frame)
            # if not cropped_frame_queue.empty():
            #     frame_list = cropped_frame_queue.get()
            #     for i in range(len(frame_list)):
            #         frame = frame_list[i]
            #         if frame is not None:
            #             cv2.imshow(f"Cropped Feed {i}", frame)
            # if cv2.waitKey(100) & 0xFF == ord('q'):
            #     break
            cv2.waitKey(100)
        # cv2.destroyAllWindows()

    Thread(target=show_camera_feed, args=(camera_frame_queue, cropped_frame_queue, frame_gui_queue)).start()
    
    while True:
        if human_detected_bool.empty():
            human_detected_bool.put(1)
        if not human_detected_queue.empty():
            while not human_detected_queue.empty():
                human_detected = human_detected_queue.get()
                # print(human_detected)
                
        if costumer_name_bool.empty():
            costumer_name_bool.put(1)
        if not costumer_name_queue.empty():
            while not costumer_name_queue.empty():
                costumer_name = costumer_name_queue.get()
                # print(costumer_name)

        if image_info_bool.empty():
            image_info_bool.put(1)
        if not image_info_queue.empty():
            while not image_info_queue.empty():
                image_info = image_info_queue.get()
                print(image_info)