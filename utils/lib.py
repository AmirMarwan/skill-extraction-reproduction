import openai
from llama_index.llms.openai import OpenAI as OpenAI_llama
from llama_index.core.llms import ChatMessage
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from time import sleep
from enum import IntEnum
import json
import os
from . import browser_instance_handler, app_webpage
from . import camera_process_handler
from . import tts_server
from . import periodic_documentation
from . import microphone_server
from . import audio_passthrough_handler
from . import vc_handler

import uvicorn
import asyncio
import websockets

from threading import Thread
from multiprocessing import Process
from multiprocessing import Queue

from . import get_custom_logger
logger = get_custom_logger(__name__)

class OperatingMode(IntEnum):
    AUTONOMOUS = 0
    STANDALONE = 1
    SUPPORT = 2
    SELECTIVE = 3

def load_api_key(secrets_file="secrets/OpenAI_keys.json"):
    with open(secrets_file) as f:
        secrets = json.load(f)
    return secrets["OPENAI_API_KEY"]

api_keys = load_api_key()
key_number = len(api_keys)

class Util_lib:
    def __init__(self, *args, **kwargs):
        self.key_index = -1
        # self.__dict__.update({key: value for key, value in locals().items() if key != 'self'})
        self.operating_mode = kwargs.pop('operating_mode', OperatingMode.SELECTIVE)
        self.resources_folder = kwargs.pop('resources_folder', f'resources/raw/')
        self.use_llamaindex = kwargs.pop('use_llamaindex', True)
        self.use_mic = kwargs.pop('use_mic', True)
        self.mic_start_queue_1 = kwargs.pop('mic_start_queue_1', Queue())
        self.mic_message_queue_1 = kwargs.pop('mic_message_queue_1', Queue())
        self.mic_start_queue_2 = kwargs.pop('mic_start_queue_2', Queue())
        self.mic_message_queue_2 = kwargs.pop('mic_message_queue_2', Queue())
        self.camera_frame_queue = kwargs.pop('camera_frame_queue', Queue())
        self.frame_gui_queue = kwargs.pop('frame_gui_queue', Queue())
        self.cropped_frame_queue = kwargs.pop('cropped_frame_queue', Queue())
        self.cropped_frame_queue_2 = kwargs.pop('cropped_frame_queue_2', Queue())
        self.image_info_queue = kwargs.pop('image_info_queue', Queue())
        self.image_info_bool = kwargs.pop('image_info_bool', Queue())
        self.documentation_queue = kwargs.pop('documentation_queue', Queue())
        self.documentation_input = kwargs.pop('documentation_input', Queue())
        self.human_detected_queue = kwargs.pop('human_detected_queue', Queue())
        self.human_detected_bool = kwargs.pop('human_detected_bool', Queue())
        self.human_current_queue = kwargs.pop('human_current_queue', Queue())
        self.costumer_name_queue = kwargs.pop('costumer_name_queue', Queue())
        self.costumer_name_bool = kwargs.pop('costumer_name_bool', Queue())
        self.start_vc = kwargs.pop('start_vc', Queue())
        self.start_passthrough = kwargs.pop('start_passthrough', Queue())

    def initialize(self, **kwargs):
        camera_kwargs = kwargs.pop('camera_kwargs', dict())
        if self.operating_mode == OperatingMode.SELECTIVE:
            self.__set_voice_changer()
        self.__set_audio_passthrough()
        self.__set_camera_thread(**camera_kwargs)
        self.__set_tts()
        self.open_browser_instances()
        self.__set_openai_client()
        self.__set_mic_thread()
        if self.use_mic:
            self.__set_recorder()
        if self.use_llamaindex:
            self.__set_llama_index()
        self.__set_periodic_documentation_thread()
        self.loop = asyncio.get_event_loop()
        self.loop.run_until_complete(self.tts_connect())
        self.tts('こんにちは')
        logger.info("Util_lib initialized.")

    def __set_llama_index(self):
        PERSIST_DIR = self.resources_folder + "index_storage"
        if not os.path.exists(PERSIST_DIR):
            documents = SimpleDirectoryReader(input_dir=self.resources_folder).load_data()
            self.doc_index = VectorStoreIndex.from_documents(documents)
            self.doc_index.storage_context.persist(persist_dir=PERSIST_DIR)
        else:
            storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            self.doc_index = load_index_from_storage(storage_context)
        # self.query_engine = self.doc_index.as_query_engine()
        self.chat_model = OpenAI_llama(temperature=0, model="gpt-4o")
        logger.info("LlamaIndex loaded.")

    def __set_openai_client(self):
        self.key_index = (self.key_index+1) % key_number
        openai.api_key = api_keys[self.key_index]
        self.client = openai.OpenAI(api_key=api_keys[self.key_index])
        logger.info("OpenAI client set up.")

    def __set_recorder(self):
        self.webpage_thread = Thread(target=uvicorn.run, args=(app_webpage,), kwargs={'host':"127.0.0.1", 'port':5004, 'access_log':False})
        self.webpage_thread.daemon = True
        self.webpage_thread.start()
        logger.info('Webpage servers started at 5004.')

    def __set_mic_thread(self):
        self.mic_thread_1 = Thread(target=microphone_server, args=(self.mic_start_queue_1, self.mic_message_queue_1, 1234))
        self.mic_thread_1.daemon = True
        self.mic_thread_1.start()
        self.mic_thread_2 = Thread(target=microphone_server, args=(self.mic_start_queue_2, self.mic_message_queue_2, 1233))
        self.mic_thread_2.daemon = True
        self.mic_thread_2.start()
        logger.info("Mic servers started at 1234 and 1233.")
    
    def __set_voice_changer(self):
        self.voice_changer_thread = Process(target=vc_handler, args=(self.start_vc,))
        self.voice_changer_thread.daemon = False
        self.voice_changer_thread.start()
        logger.info("Voice changer process started.")

    def __set_audio_passthrough(self):
        self.audio_passthrough_thread = Process(target=audio_passthrough_handler, args=(self.start_passthrough,))
        self.audio_passthrough_thread.daemon = True
        self.audio_passthrough_thread.start()
        logger.info("Audio passthrough process started.")
    
    def __set_tts(self):
        self.tts_thread = Process(target=tts_server, args=(2001,))
        self.tts_thread.daemon = True
        self.tts_thread.start()
        logger.info("TTS server started at 2001")
        # sleep(1)

    def __set_camera_thread(self, **camera_kwargs):
        self.camera_thread = Process(target=camera_process_handler, args=(self.camera_frame_queue, self.cropped_frame_queue, self.cropped_frame_queue_2, self.frame_gui_queue,
                                                                          self.human_detected_queue, self.human_detected_bool, self.human_current_queue,
                                                                          self.costumer_name_queue, self.costumer_name_bool,
                                                                          self.image_info_queue, self.image_info_bool,), kwargs=camera_kwargs)
        self.camera_thread.daemon = True
        self.camera_thread.start()
        logger.info('Camera process started')
    
    def __set_periodic_documentation_thread(self):
        self.documentation_thread = Thread(target=periodic_documentation, args=(openai.api_key, self.doc_index, self.documentation_queue, self.documentation_input))
        self.documentation_thread.daemon = True
        self.documentation_thread.start()
        logger.info("Periodic documentation fetching server started.")

    async def tts_connect(self):
        logger.info('Connecting to TTS websocket...')
        while True:
            try:
                self.websocket_tts = await websockets.connect('ws://localhost:2001/ws', ping_interval=3600, ping_timeout=3600)
                break
            except:
                sleep(0.1)
        logger.info('TTS websocket connected.')

    async def __tts(self, text):
        if self.websocket_tts is None or self.websocket_tts.closed:
            await self.tts_connect()
        try:
            await self.websocket_tts.send(text)
            length_in_seconds = await self.websocket_tts.recv()
            return float(length_in_seconds)
        except (websockets.exceptions.ConnectionClosed, OSError) as e:
            logger.warning(f"Connection lost during send: {e}")
            await self.tts_connect()
            await self.websocket_tts.send(text)
            length_in_seconds = await self.websocket_tts.recv()
            return float(length_in_seconds)

    def tts(self, text):
        return self.loop.run_until_complete(self.__tts(text))

    def get_response_llamaindex(self, messages, **kwargs):
        if self.use_llamaindex:
            stop = kwargs.get('stop', [])
            stream = kwargs.get('stream', [])
            messages_user = ChatMessage(role='user', content=messages[1]['content'])
            self.chat_model.additional_kwargs = {"stop": stop}
            self.chat_engine = self.doc_index.as_chat_engine(llm=self.chat_model, prefix_messages=[ChatMessage(role='system', content=messages[0]['content'])])
            if stream:
                resources = self.chat_engine.stream_chat(str(messages_user))
                return resources.response_gen
            else:
                resources = str(self.chat_engine.chat(str(messages_user)))
        else:
            resources = self.get_response(messages, **kwargs)
        return resources

    def get_response(self, messages, **kwargs):
        stop = kwargs.get('stop', [])
        stream = kwargs.get('stream', False)
        while True:
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    stop=stop,
                    stream=True
                )
                if stream:
                    return response
                else:
                    answer = ''
                    for event in response: 
                        event_text = event.choices[0].delta
                        if event_text.content != None:
                            answer += str(event_text.content)
                    return answer
            except Exception as e:
                logger.error(f"Error: {e}")
                self.__set_openai_client()
                logger.info("Retrying after 5s...")
                sleep(5)
            
    def open_browser_instances(self):
        self.browser_thread = Process(target=browser_instance_handler)
        self.browser_thread.daemon = True
        self.browser_thread.start()