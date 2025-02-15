###############################################################################
#  Copyright (C) 2024 LiveTalking@lipku https://github.com/lipku/LiveTalking
#  email: lipku@foxmail.com
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  
#       http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################

# server.py
from flask import Flask, render_template, send_from_directory, request, jsonify
#from flask_sockets import Sockets   # Si usás websockets, descomentar
import base64
import time
import json
#import gevent
#from gevent import pywsgi
#from geventwebsocket.handler import WebSocketHandler
import os
import re
import numpy as np
from threading import Thread, Event
#import multiprocessing
import torch.multiprocessing as mp

from aiohttp import web
import aiohttp
import aiohttp_cors
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.rtcrtpsender import RTCRtpSender
from webrtc import HumanPlayer

import argparse
import random

import shutil
import asyncio
import torch

app = Flask(__name__)
#sockets = Sockets(app)  # Descomentar si se utiliza

nerfreals = {}
opt = None
model = None
avatar = None

def llm_response(message, nerfreal):
    start = time.perf_counter()
    from openai import OpenAI
    client = OpenAI(
        # Si no configuraste las variables de entorno, reemplazá con tu API Key
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    end = time.perf_counter()
    print(f"llm Time init: {end-start}s")
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[{'role': 'system', 'content': 'You are a helpful assistant.'},
                  {'role': 'user', 'content': message}],
        stream=True,
        stream_options={"include_usage": True}
    )
    result = ""
    first = True
    for chunk in completion:
        if len(chunk.choices) > 0:
            if first:
                end = time.perf_counter()
                print(f"llm Time to first chunk: {end-start}s")
                first = False
            msg = chunk.choices[0].delta.content
            lastpos = 0
            for i, char in enumerate(msg):
                if char in ",.!;:，。！？：；":
                    result = result + msg[lastpos:i+1]
                    lastpos = i+1
                    if len(result) > 10:
                        print(result)
                        nerfreal.put_msg_txt(result)
                        result = ""
            result = result + msg[lastpos:]
    end = time.perf_counter()
    print(f"llm Time to last chunk: {end-start}s")
    nerfreal.put_msg_txt(result)

##### Funciones y manejo de WebRTC ###############################
pcs = set()

def randN(N):
    '''Genera un número aleatorio de N dígitos'''
    min_val = pow(10, N - 1)
    max_val = pow(10, N)
    return random.randint(min_val, max_val - 1)

def build_nerfreal(sessionid):
    opt.sessionid = sessionid
    if opt.model == 'wav2lip':
        from lipreal import LipReal
        nerfreal = LipReal(opt, model, avatar)
    elif opt.model == 'musetalk':
        from musereal import MuseReal
        nerfreal = MuseReal(opt, model, avatar)
    elif opt.model == 'ernerf':
        from nerfreal import NeRFReal
        nerfreal = NeRFReal(opt, model, avatar)
    elif opt.model == 'ultralight':
        from lightreal import LightReal
        nerfreal = LightReal(opt, model, avatar)
    return nerfreal

# Función para descargar y configurar los archivos necesarios (solo para wav2lip)
def setup_files():
    import wget
    import tarfile

    # Crear directorios de destino si no existen
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/avatars", exist_ok=True)

    # --- Descarga del modelo wav2lip384.pth ---
    url1 = ("https://drive.usercontent.google.com/download?"
            "id=10nOKATfuaxsmDU8sG2dQkQyq68KLslvZ&export=download&authuser=0&confirm=t&"
            "uuid=81dce160-e018-4e35-8db0-dd3c5926ab62&at=AIrpjvP5eJw6nrTkmuSu8S5iIJkW:1739582743480")
    archivo_modelo = "wav2lip384.pth"
    if not os.path.exists(archivo_modelo):
        print("Descargando wav2lip384.pth...")
        wget.download(url1, archivo_modelo)
        print("\nDescarga completada.")
    else:
        print("El archivo wav2lip384.pth ya existe.")

    # Copiar y renombrar a la carpeta models (wav2lip.pth)
    destino_modelo = os.path.join("models", "wav2lip.pth")
    shutil.copy(archivo_modelo, destino_modelo)
    print(f"Archivo copiado y renombrado a: {destino_modelo}")

    # --- Descarga del archivo wav2lip384_avatar1.tar.gz ---
    url2 = ("https://drive.usercontent.google.com/download?"
            "id=1RRWbPEvxNzPbuplIqDrhrZZR8WzBlFVv&export=download&authuser=0&confirm=t&"
            "uuid=76191218-ce67-4568-bae0-b63705301eb7&at=AIrpjvPq7iI5yCHAaISE9g2SeZAN:1739582732553")
    archivo_avatar = "wav2lip384_avatar1.tar.gz"
    if not os.path.exists(archivo_avatar):
        print("Descargando wav2lip384_avatar1.tar.gz...")
        wget.download(url2, archivo_avatar)
        print("\nDescarga completada.")
    else:
        print("El archivo wav2lip384_avatar1.tar.gz ya existe.")

    # Extraer el archivo tar.gz en un directorio temporal
    directorio_temp = "temp_extracted"
    os.makedirs(directorio_temp, exist_ok=True)
    with tarfile.open(archivo_avatar, "r:gz") as tar:
        tar.extractall(path=directorio_temp)
    print("Archivo wav2lip384_avatar1.tar.gz extraído.")

    # Copiar el contenido extraído a data/avatars
    for item in os.listdir(directorio_temp):
        origen = os.path.join(directorio_temp, item)
        destino = os.path.join("data/avatars", item)
        if os.path.isdir(origen):
            shutil.copytree(origen, destino, dirs_exist_ok=True)
        else:
            shutil.copy2(origen, destino)
    print("Contenido copiado a data/avatars.")

    # Eliminar el directorio temporal
    shutil.rmtree(directorio_temp)
    print("Directorio temporal eliminado.")

##### Rutas y funciones asíncronas para la API ###############
async def offer(request):
    params = await request.json()
    offer_desc = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    if len(nerfreals) >= opt.max_session:
        print('Se alcanzó el máximo de sesiones')
        return -1
    sessionid = randN(6)
    print('sessionid =', sessionid)
    nerfreals[sessionid] = None
    nerfreal = await asyncio.get_event_loop().run_in_executor(None, build_nerfreal, sessionid)
    nerfreals[sessionid] = nerfreal

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)
            del nerfreals[sessionid]
        if pc.connectionState == "closed":
            pcs.discard(pc)
            del nerfreals[sessionid]

    player = HumanPlayer(nerfreals[sessionid])
    audio_sender = pc.addTrack(player.audio)
    video_sender = pc.addTrack(player.video)
    capabilities = RTCRtpSender.getCapabilities("video")
    preferences = list(filter(lambda x: x.name == "H264", capabilities.codecs))
    preferences += list(filter(lambda x: x.name == "VP8", capabilities.codecs))
    preferences += list(filter(lambda x: x.name == "rtx", capabilities.codecs))
    transceiver = pc.getTransceivers()[1]
    transceiver.setCodecPreferences(preferences)

    await pc.setRemoteDescription(offer_desc)

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type, "sessionid": sessionid}
        ),
    )

async def human(request):
    params = await request.json()
    sessionid = params.get('sessionid', 0)
    if params.get('interrupt'):
        nerfreals[sessionid].flush_talk()

    if params['type'] == 'echo':
        nerfreals[sessionid].put_msg_txt(params['text'])
    elif params['type'] == 'chat':
        await asyncio.get_event_loop().run_in_executor(None, llm_response, params['text'], nerfreals[sessionid])
    return web.Response(
        content_type="application/json",
        text=json.dumps({"code": 0, "data": "ok"}),
    )

async def humanaudio(request):
    try:
        form = await request.post()
        sessionid = int(form.get('sessionid', 0))
        fileobj = form["file"]
        filename = fileobj.filename
        filebytes = fileobj.file.read()
        nerfreals[sessionid].put_audio_file(filebytes)
        return web.Response(
            content_type="application/json",
            text=json.dumps({"code": 0, "msg": "ok"}),
        )
    except Exception as e:
        return web.Response(
            content_type="application/json",
            text=json.dumps({"code": -1, "msg": "err", "data": str(e)}),
        )

async def set_audiotype(request):
    params = await request.json()
    sessionid = params.get('sessionid', 0)
    nerfreals[sessionid].set_curr_state(params['audiotype'], params['reinit'])
    return web.Response(
        content_type="application/json",
        text=json.dumps({"code": 0, "data": "ok"}),
    )

async def record(request):
    params = await request.json()
    sessionid = params.get('sessionid', 0)
    if params['type'] == 'start_record':
        nerfreals[sessionid].start_recording()
    elif params['type'] == 'end_record':
        nerfreals[sessionid].stop_recording()
    return web.Response(
        content_type="application/json",
        text=json.dumps({"code": 0, "data": "ok"}),
    )

async def is_speaking(request):
    params = await request.json()
    sessionid = params.get('sessionid', 0)
    return web.Response(
        content_type="application/json",
        text=json.dumps({"code": 0, "data": nerfreals[sessionid].is_speaking()}),
    )

async def on_shutdown(app):
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

async def post(url, data):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data) as response:
                return await response.text()
    except aiohttp.ClientError as e:
        print(f'Error: {e}')

async def run(push_url, sessionid):
    nerfreal = await asyncio.get_event_loop().run_in_executor(None, build_nerfreal, sessionid)
    nerfreals[sessionid] = nerfreal

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    player = HumanPlayer(nerfreals[sessionid])
    audio_sender = pc.addTrack(player.audio)
    video_sender = pc.addTrack(player.video)

    await pc.setLocalDescription(await pc.createOffer())
    answer = await post(push_url, pc.localDescription.sdp)
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer, type='answer'))

##########################################
if __name__ == '__main__':
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    # Opciones de entrada
    parser.add_argument('--pose', type=str, default="data/data_kf.json", help="Fuente de poses (transforms.json)")
    parser.add_argument('--au', type=str, default="data/au.csv", help="Área de parpadeo de ojos")
    parser.add_argument('--torso_imgs', type=str, default="", help="Ruta de imágenes de torso")
    parser.add_argument('-O', action='store_true', help="Equivale a --fp16 --cuda_ray --exp_eye")
    parser.add_argument('--data_range', type=int, nargs='*', default=[0, -1], help="Rango de datos a usar")
    parser.add_argument('--workspace', type=str, default='data/video')
    parser.add_argument('--seed', type=int, default=0)
    ### Opciones de entrenamiento
    parser.add_argument('--ckpt', type=str, default='data/pretrained/ngp_kf.pth')
    parser.add_argument('--num_rays', type=int, default=4096 * 16, help="Número de rayos por imagen en cada paso")
    parser.add_argument('--cuda_ray', action='store_true', help="Usar CUDA raymarching en vez de pytorch")
    parser.add_argument('--max_steps', type=int, default=16, help="Máximo número de pasos por rayo (válido solo con --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=16, help="Número de pasos por rayo (válido sin --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=0, help="Pasos de upsample por rayo (válido sin --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16, help="Intervalo para actualizar estado extra (válido con --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="Batch de rayos para evitar OOM (válido sin --cuda_ray)")
    ### Configuración de pérdida
    parser.add_argument('--warmup_step', type=int, default=10000, help="Pasos de warm up")
    parser.add_argument('--amb_aud_loss', type=int, default=1, help="Usar pérdida ambiental para audio")
    parser.add_argument('--amb_eye_loss', type=int, default=1, help="Usar pérdida ambiental para ojos")
    parser.add_argument('--unc_loss', type=int, default=1, help="Usar pérdida de incertidumbre")
    parser.add_argument('--lambda_amb', type=float, default=1e-4, help="Lambda para la pérdida ambiental")
    ### Opciones de red
    parser.add_argument('--fp16', action='store_true', help="Usar precisión mixta AMP")
    parser.add_argument('--bg_img', type=str, default='white', help="Imagen de fondo")
    parser.add_argument('--fbg', action='store_true', help="Fondo por frame")
    parser.add_argument('--exp_eye', action='store_true', help="Control explícito de ojos")
    parser.add_argument('--fix_eye', type=float, default=-1, help="Fijar área de ojos (valor entre 0 y 0.3)")
    parser.add_argument('--smooth_eye', action='store_true', help="Suavizar secuencia de ojos")
    parser.add_argument('--torso_shrink', type=float, default=0.8, help="Contraer coordenadas del fondo para mayor flexibilidad")
    ### Opciones de dataset
    parser.add_argument('--color_space', type=str, default='srgb', help="Espacio de color (linear, srgb)")
    parser.add_argument('--preload', type=int, default=0, help="0: cargar on-the-fly, 1: preload en CPU, 2: preload en GPU")
    parser.add_argument('--bound', type=float, default=1, help="Escena limitada en box[-bound, bound]^3")
    parser.add_argument('--scale', type=float, default=4, help="Escala para la cámara")
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="Offset de la cámara")
    parser.add_argument('--dt_gamma', type=float, default=1/256, help="dt_gamma para ray marching adaptativo")
    parser.add_argument('--min_near', type=float, default=0.05, help="Distancia mínima de la cámara")
    parser.add_argument('--density_thresh', type=float, default=10, help="Umbral de densidad (sigma)")
    parser.add_argument('--density_thresh_torso', type=float, default=0.01, help="Umbral de densidad para torso (alpha)")
    parser.add_argument('--patch_size', type=int, default=1, help="Renderizar parches (usar [64, 32, 16] para LPIPS)")
    parser.add_argument('--init_lips', action='store_true', help="Inicializar región de labios")
    parser.add_argument('--finetune_lips', action='store_true', help="Afinar labios con LPIPS y landmarks")
    parser.add_argument('--smooth_lips', action='store_true', help="Suavizar secuencia de labios")
    parser.add_argument('--torso', action='store_true', help="Fijar cabeza y entrenar torso")
    parser.add_argument('--head_ckpt', type=str, default='', help="Modelo para cabeza")
    ### Opciones de GUI
    parser.add_argument('--gui', action='store_true', help="Iniciar GUI")
    parser.add_argument('--W', type=int, default=450, help="Ancho de la GUI")
    parser.add_argument('--H', type=int, default=450, help="Alto de la GUI")
    parser.add_argument('--radius', type=float, default=3.35, help="Radio de la cámara en la GUI")
    parser.add_argument('--fovy', type=float, default=21.24, help="Campo de visión en la GUI")
    parser.add_argument('--max_spp', type=int, default=1, help="Máximo de samples por pixel en la GUI")
    ### Otros parámetros
    parser.add_argument('--att', type=int, default=2, help="Modo de atención de audio (0, 1, 2)")
    parser.add_argument('--aud', type=str, default='', help="Fuente de audio (ruta a archivo npy o vacío)")
    parser.add_argument('--emb', action='store_true', help="Usar clase de audio + embedding")
    parser.add_argument('--ind_dim', type=int, default=4, help="Dimensión de código individual")
    parser.add_argument('--ind_num', type=int, default=10000, help="Cantidad de códigos individuales")
    parser.add_argument('--ind_dim_torso', type=int, default=8, help="Dimensión de código para torso")
    parser.add_argument('--amb_dim', type=int, default=2, help="Dimensión ambiental")
    parser.add_argument('--part', action='store_true', help="Usar parte del dataset (1/10)")
    parser.add_argument('--part2', action='store_true', help="Usar parte del dataset (primeros 15s)")
    parser.add_argument('--train_camera', action='store_true', help="Optimizar la cámara")
    parser.add_argument('--smooth_path', action='store_true', help="Suavizar trayectoria de cámara")
    parser.add_argument('--smooth_path_window', type=int, default=7, help="Ventana para suavizar trayectoria")
    # ASR
    parser.add_argument('--asr', action='store_true', help="Cargar ASR para la app en tiempo real")
    parser.add_argument('--asr_wav', type=str, default='', help="Ruta del wav para ASR")
    parser.add_argument('--asr_play', action='store_true', help="Reproducir audio de ASR")
    parser.add_argument('--asr_model', type=str, default='cpierse/wav2vec2-large-xlsr-53-esperanto')
    parser.add_argument('--asr_save_feats', action='store_true')
    parser.add_argument('--fps', type=int, default=50, help="FPS de audio")
    parser.add_argument('-l', type=int, default=10, help="Ventana izquierda (20ms)")
    parser.add_argument('-m', type=int, default=8, help="Ventana central (20ms)")
    parser.add_argument('-r', type=int, default=10, help="Ventana derecha (20ms)")
    # Fullbody
    parser.add_argument('--fullbody', action='store_true', help="Humano fullbody")
    parser.add_argument('--fullbody_img', type=str, default='data/fullbody/img')
    parser.add_argument('--fullbody_width', type=int, default=580)
    parser.add_argument('--fullbody_height', type=int, default=1080)
    parser.add_argument('--fullbody_offset_x', type=int, default=0)
    parser.add_argument('--fullbody_offset_y', type=int, default=0)
    # Opciones específicas para musetalk
    parser.add_argument('--avatar_id', type=str, default='avator_1')
    parser.add_argument('--bbox_shift', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--customvideo_config', type=str, default='')
    parser.add_argument('--tts', type=str, default='edgetts')  # xtts, gpt-sovits, cosyvoice
    parser.add_argument('--REF_FILE', type=str, default=None)
    parser.add_argument('--REF_TEXT', type=str, default=None)
    parser.add_argument('--TTS_SERVER', type=str, default='http://127.0.0.1:9880')
    parser.add_argument('--model', type=str, default='ernerf')  # Opciones: ernerf, musetalk, wav2lip, ultralight
    parser.add_argument('--transport', type=str, default='rtcpush')  # Opciones: rtmp, webrtc, rtcpush
    parser.add_argument('--push_url', type=str, default='http://localhost:1985/rtc/v1/whip/?app=live&stream=livestream')
    parser.add_argument('--max_session', type=int, default=1)
    parser.add_argument('--listenport', type=int, default=8010)

    opt = parser.parse_args()
    opt.customopt = []
    if opt.customvideo_config != '':
        with open(opt.customvideo_config, 'r') as file:
            opt.customopt = json.load(file)

    # Cargar modelo y avatar según la opción elegida
    if opt.model == 'ernerf':
        from nerfreal import NeRFReal, load_model, load_avatar
        model = load_model(opt)
        avatar = load_avatar(opt)
    elif opt.model == 'musetalk':
        from musereal import MuseReal, load_model, load_avatar, warm_up
        print(opt)
        model = load_model()
        avatar = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size, model)
    elif opt.model == 'wav2lip':
        # Descargar archivos y configurar para wav2lip
        setup_files()
        from lipreal import LipReal, load_model, load_avatar, warm_up
        print(opt)
        model = load_model("./models/wav2lip.pth")
        avatar = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size, model, 384)
    elif opt.model == 'ultralight':
        from lightreal import LightReal, load_model, load_avatar, warm_up
        print(opt)
        model = load_model(opt)
        avatar = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size, avatar, 160)

    # Si se usa RTMP, se inicia el render en un hilo separado
    if opt.transport == 'rtmp':
        thread_quit = Event()
        nerfreals[0] = build_nerfreal(0)
        rendthrd = Thread(target=nerfreals[0].render, args=(thread_quit,))
        rendthrd.start()

    #############################################################################
    appasync = web.Application()
    appasync.on_shutdown.append(on_shutdown)
    appasync.router.add_post("/offer", offer)
    appasync.router.add_post("/human", human)
    appasync.router.add_post("/humanaudio", humanaudio)
    appasync.router.add_post("/set_audiotype", set_audiotype)
    appasync.router.add_post("/record", record)
    appasync.router.add_post("/is_speaking", is_speaking)
    appasync.router.add_static('/', path='web')

    # Configurar CORS
    cors = aiohttp_cors.setup(appasync, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
        )
    })
    for route in list(appasync.router.routes()):
        cors.add(route)

    pagename = 'webrtcapi.html'
    if opt.transport == 'rtmp':
        pagename = 'echoapi.html'
    elif opt.transport == 'rtcpush':
        pagename = 'rtcpushapi.html'
    print('Iniciando servidor http; http://<serverip>:' + str(opt.listenport) + '/' + pagename)

    def run_server(runner):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, '0.0.0.0', opt.listenport)
        loop.run_until_complete(site.start())
        if opt.transport == 'rtcpush':
            for k in range(opt.max_session):
                push_url = opt.push_url
                if k != 0:
                    push_url = opt.push_url + str(k)
                loop.run_until_complete(run(push_url, k))
        loop.run_forever()

    run_server(web.AppRunner(appasync))

