
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

EDGE_DRIVER_PATH = 'utils/html/msedgedriver.exe'

try:
    from . import get_custom_logger
    logger = get_custom_logger(__name__)
except:
    import os
    if not os.getcwd().lower().endswith('res3'):
        os.chdir('Res3')
    import sys
    sys.path.append('.')
    from utils import get_custom_logger
    logger = get_custom_logger(__name__)

app_webpage = FastAPI()
app_webpage.mount("/static", StaticFiles(directory="utils/html/static"), name="static")
app_webpage.mount("/pics", StaticFiles(directory="utils/html/static"), name="pics")

@app_webpage.get("/client", response_class=HTMLResponse)
async def read_root(request: Request):
    templates = Jinja2Templates(directory="utils/html/templates")
    return templates.TemplateResponse("STT.html", {"request": request})

@app_webpage.get("/operator", response_class=HTMLResponse)
async def read_root(request: Request):
    templates = Jinja2Templates(directory="utils/html/templates")
    return templates.TemplateResponse("STT.html", {"request": request})

def browser_instance_handler(mode=None):
    from selenium import webdriver
    from selenium.webdriver.edge.service import Service
    from selenium.webdriver.edge.options import Options
    from selenium.webdriver.common.by import By
    from time import sleep

    def browser_instance(page_url, mic_device='', soket_url='ws://127.0.0.1:1233/ws', speaker_text='オペレータ'):

        edge_options = Options()
        edge_prefs = {"profile.content_settings.exceptions.media_stream_mic": {page_url: {"setting": 1}}}
        edge_options.add_experimental_option("prefs", edge_prefs)
        edge_service = Service(EDGE_DRIVER_PATH)
        driver = webdriver.Edge(service=edge_service, options=edge_options)
        driver.get(page_url)
        font_element = driver.find_element(By.CSS_SELECTOR, '#operator_text font')
        driver.execute_script(f"arguments[0].innerText = '{speaker_text}';", font_element)
        socket_url_field = driver.find_element(By.ID, "socket_url")
        driver.execute_script("arguments[0].value = arguments[1];", socket_url_field, soket_url)

        driver.execute_script("window.open('');")
        driver.switch_to.window(driver.window_handles[1])
        driver.get('edge://settings/content/microphone')
        driver.implicitly_wait(2)
        microphone_dropdown_button = driver.find_element(By.XPATH, '//div[@aria-label="Microphone"]//button')
        microphone_dropdown_button.click()
        driver.implicitly_wait(1)
        dropdown_options = driver.find_elements(By.XPATH, '//*[@role="listbox"]/*')
        for option in dropdown_options:
            if mic_device in option.text.lower():
                option.click()
                break
        driver.implicitly_wait(1)
        driver.refresh()
        driver.switch_to.window(driver.window_handles[0])

        return driver
    
    if mode == 'extract_data':
        browser_instance_e = browser_instance('http://127.0.0.1:5004/client', mic_device='cable-b', soket_url='ws://127.0.0.1:1234/ws', speaker_text='Data Extractor')
    else:
        # browser_instance_1 = browser_instance('http://127.0.0.1:5004/operator', mic_device='intel', soket_url='ws://127.0.0.1:1233/ws', speaker_text='オペレータ')
        browser_instance_2 = browser_instance('http://127.0.0.1:5004/client', mic_device='cable-b', soket_url='ws://127.0.0.1:1234/ws', speaker_text='お客さん')
        # browser_instance_2 = browser_instance('http://127.0.0.1:5004/client', mic_device='cable output', soket_url='ws://127.0.0.1:1234/ws', speaker_text='お客さん')
    
    logger.info('Browser instances opened.')
    while True:
        sleep(2)

if __name__ == '__main__':
    # import os
    # if not os.getcwd().lower().endswith('res3'):
    #     os.chdir('Res3')
    # import sys
    # sys.path.append('.')
    from multiprocessing import Process
    from threading import Thread
    import uvicorn
    Thread(target=uvicorn.run, args=(app_webpage,), kwargs={'host':"127.0.0.1", 'port':5004, 'access_log':False}).start()
    # Process(target=browser_instance_handler, kwargs={'mode':'extract_data'}).start()