
from fastapi import FastAPI, WebSocket
import uvicorn
import asyncio
from time import sleep, time
from datetime import datetime, timezone

async def mic_server_callback(websocket, start_queue, message_queue):
    await websocket.accept()
    mic_start = False
    while True:
        sleep(0.1)
        while not start_queue.empty():
            mic_start = start_queue.get()
            if mic_start:
                await websocket.send_text('start')
            else:
                await websocket.send_text('stop')
        if mic_start:
            while True:
                    message = await websocket.receive_text()
                    if message.startswith('result'):
                        message_content = message[7:message.index('\nconfidence:')]
                        confidence = message[message.index('\nconfidence:')+12:message.index('\ntime_stamp:')]
                        time_stamp_str = message[message.index('\ntime_stamp:')+12:-1]
                        try:
                            time_stamp_start = datetime.strptime(time_stamp_str, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc).astimezone().timestamp()
                        except:
                            time_stamp_start = time()
                        time_stamp_end = time()
                        message_queue.put((1, message_content, confidence, time_stamp_start, time_stamp_end))
                        break
                    else:
                        message_queue.put((2, message))

def microphone_server(mic_start_queue, mic_message_queue, port):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    appc = FastAPI()
    @appc.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await mic_server_callback(websocket, mic_start_queue, mic_message_queue)

    config = uvicorn.Config(appc, host="127.0.0.1", port=port, access_log=False)
    server = uvicorn.Server(config)
    loop.run_until_complete(server.serve())
    loop.run_forever()
    loop.close()


if __name__=='__main__':
    from multiprocessing import  Queue
    from threading import Thread

    mic_start_queue_1 = Queue()
    mic_message_queue_1 = Queue()
    mic_start_queue_2 = Queue()
    mic_message_queue_2 = Queue()

    Thread(target=microphone_server, args=(mic_start_queue_1, mic_message_queue_1, 1234)).start()
    Thread(target=microphone_server, args=(mic_start_queue_2, mic_message_queue_2, 1233)).start()

    def show_messages(mic_message_queue_1=Queue(), mic_message_queue_2=Queue()):
        while True:
            if not mic_message_queue_1.empty():
                while not mic_message_queue_1.empty():
                    message = mic_message_queue_1.get()
                print(message)
            if not mic_message_queue_2.empty():
                while not mic_message_queue_2.empty():
                    message = mic_message_queue_2.get()
                print(message)


    Thread(target=show_messages, args=(mic_message_queue_1, mic_message_queue_2)).start()
    
    def save_messages(mic_message_queue=Queue()):
        message = ''
        while True:
            sleep(0.5)
            if not mic_message_queue.empty():
                while not mic_message_queue.empty():
                    message += mic_message_queue.get()[1]
                with open('mic_messages.txt', 'a') as fh:
                    fh.write(message+'\n')
                message = ''

    # Thread(target=save_messages, args=(mic_message_queue_2, )).start() # Do not use at the same time as show_messages
    
    while True:
        mic_start_queue_1.put(0)
        mic_start_queue_2.put(0)
        input('stop')
        mic_start_queue_1.put(1)
        mic_start_queue_2.put(1)
        input('start')