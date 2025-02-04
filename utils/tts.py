import asyncio
import uvicorn
from fastapi import FastAPI, WebSocket

from . import get_custom_logger
logger = get_custom_logger(__name__)

async def tts_handler(websocket, model, sd, wavfile):
    await websocket.accept()
    while True:
        try:
            message = await asyncio.wait_for(websocket.receive_text(), timeout=3600)
            if message == ' ':
                length_in_seconds = 0
                sd.stop()
            else:
                sr, audio = model.infer(message)
                length_in_seconds = len(audio) / sr
                sd.play(audio, sr)
            # wavfile.write("output_audio.wav", sr, audio)
            await websocket.send_text(f"{length_in_seconds}")
        except Exception as e:
            logger.error(f"Connection closed or error occurred: {e}")
            break

def tts_server(port):
    from style_bert_vits2.nlp import bert_models
    from style_bert_vits2.constants import Languages
    import scipy.io.wavfile as wavfile
    import sounddevice as sd
    from pathlib import Path
    from huggingface_hub import hf_hub_download
    from style_bert_vits2.tts_model import TTSModel

    desired_output_device_name = 'CABLE-A Input (VB-Audio Cable A), Windows DirectSound'
    sd.default.device = desired_output_device_name

    bert_models.load_model(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")
    bert_models.load_tokenizer(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")
    
    model_file = "jvnv-F1-jp/jvnv-F1-jp_e160_s14000.safetensors"
    config_file = "jvnv-F1-jp/config.json"
    style_file = "jvnv-F1-jp/style_vectors.npy"
    for file in [model_file, config_file, style_file]:
        hf_hub_download("litagin/style_bert_vits2_jvnv", file, local_dir="model_assets")
    
    assets_root = Path("model_assets")
    model = TTSModel(
        model_path=assets_root / model_file,
        config_path=assets_root / config_file,
        style_vec_path=assets_root / style_file,
        device="cuda",
    )
    _ = model.infer('こんにちは')
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    appt = FastAPI()

    @appt.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await tts_handler(websocket, model=model, sd=sd, wavfile=wavfile)

    config = uvicorn.Config(appt, host="127.0.0.1", port=port, log_level="info", access_log=True, ws_ping_interval=3600, ws_ping_timeout=3600)
    server = uvicorn.Server(config)
    loop.run_until_complete(server.serve())
    loop.run_forever()
    loop.close()
