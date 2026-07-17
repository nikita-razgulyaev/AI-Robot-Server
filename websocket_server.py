"""WebSocket сервер - связь с ESP32 и веб-интерфейсом + голосовое общение"""
import os
import asyncio
import json
import logging
import base64
import io
import wave
import tempfile
from typing import Set, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from modules.robot_brain import RobotBrain
from config.settings import SERVER_HOST, SERVER_PORT

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Robot AI Server - Soren", version="3.0")

robot_brain: RobotBrain = None
active_connections: Set[WebSocket] = set()

# ===== РАЗДЕЛЬНЫЕ РЕЖИМЫ АУДИО =====
audio_input_mode = "robot"   # "robot" = ESP32 микрофон, "local" = микрофон ноутбука
audio_output_mode = "robot"  # "robot" = ESP32 динамик, "local" = наушники ноутбука

# ===== LIFECYCLE =====

@app.on_event("startup")
async def startup():
    global robot_brain
    logger.info("🦉 Запуск сервера Сорена...")
    robot_brain = RobotBrain()
    logger.info(f"✅ Сервер готов: ws://{SERVER_HOST}:{SERVER_PORT}")

@app.on_event("shutdown")
async def shutdown():
    if robot_brain:
        robot_brain.shutdown()
    logger.info("Сервер остановлен")

# ===== HTTP ENDPOINTS =====

@app.get("/")
async def root():
    return {
        "status": "Robot AI Server - Soren is running",
        "version": "3.0",
        "audio_input_mode": audio_input_mode,
        "audio_output_mode": audio_output_mode,
        "ai_modes": robot_brain.get_modes() if robot_brain else {}
    }

@app.get("/status")
async def status():
    if robot_brain is None:
        return {"status": "initializing"}
    return {
        "status": "ready",
        "audio_input_mode": audio_input_mode,
        "audio_output_mode": audio_output_mode,
        "ai_modes": robot_brain.get_modes(),
        "connections": len(active_connections),
        "servo_angles": robot_brain.servos.get_current_angles(),
        "vision_context": robot_brain.vision_context,
        "current_emotion": robot_brain.current_emotion
    }

@app.post("/audio_mode")
async def set_audio_mode(mode: str = Form(...), type: str = Form("output")):
    global audio_input_mode, audio_output_mode

    if mode not in ["robot", "local"]:
        return JSONResponse(
            {"status": "error", "message": "Invalid mode. Use 'robot' or 'local'"},
            status_code=400
        )

    if type == "input":
        audio_input_mode = mode
        logger.info(f"🎤 Режим ВВОДА аудио изменён: {mode}")
    elif type == "output":
        audio_output_mode = mode
        logger.info(f"🔊 Режим ВЫВОДА аудио изменён: {mode}")
    else:
        return JSONResponse(
            {"status": "error", "message": "Invalid type. Use 'input' or 'output'"},
            status_code=400
        )

    return JSONResponse({
        "status": "ok",
        "audio_input_mode": audio_input_mode,
        "audio_output_mode": audio_output_mode
    })

@app.post("/ai_mode")
async def set_ai_mode(module: str = Form(...), mode: str = Form(...)):
    """
    Переключение режима AI модулей.
    module: "stt", "tts", "llm"
    mode: "local" или "cloud"
    """
    if module not in ["stt", "tts", "llm"]:
        return JSONResponse(
            {"status": "error", "message": "Invalid module. Use 'stt', 'tts' or 'llm'"},
            status_code=400
        )

    if mode not in ["local", "cloud"]:
        return JSONResponse(
            {"status": "error", "message": "Invalid mode. Use 'local' or 'cloud'"},
            status_code=400
        )

    if robot_brain is None:
        return JSONResponse({"status": "error", "message": "Сервер ещё загружается"})

    result = await robot_brain.handle_command({"type": "set_mode", "module": module, "mode": mode})
    logger.info(f"🧠 Режим {module.upper()} изменён: {mode}")

    return JSONResponse(result)

@app.post("/speak")
async def speak_text(text: str = Form(...)):
    logger.info(f"/speak вызван: '{text}'")

    if robot_brain is None:
        return JSONResponse({"status": "error", "message": "Сервер ещё загружается"})

    if robot_brain.llm is None:
        return JSONResponse({"status": "error", "message": "LLM не инициализирован"})

    try:
        try:
            from modules.fuzzy_matcher import correct_speech_text
            raw_text = text
            corrected_text = correct_speech_text(raw_text)
            if corrected_text != raw_text:
                logger.info(f"🎯 Fuzzy (/speak): '{raw_text}' → '{corrected_text}'")
            user_text = corrected_text
        except ImportError:
            user_text = text

        llm_result = robot_brain.llm.generate(user_text, robot_brain.vision_context)
        response_text = llm_result.get("text", "")
        action = llm_result.get("action")
        emotion = llm_result.get("emotion", "calm")

        if not response_text:
            return JSONResponse({"status": "error", "message": "LLM вернул пустой ответ"})

        servo_angles = robot_brain.emotion_engine.get_servo_angles(emotion)
        eye_led = robot_brain.emotion_engine.get_eye_led(emotion)

        tts_audio = robot_brain.tts.synthesize(response_text)
        if not tts_audio:
            return JSONResponse({"status": "error", "message": "Ошибка синтеза речи"})

        if action:
            asyncio.create_task(robot_brain.servos.play_animation(action))
        else:
            robot_brain.servos.set_all_servos(servo_angles)

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(48000)
            wf.writeframes(tts_audio)
        wav_bytes = wav_buffer.getvalue()
        audio_b64 = base64.b64encode(wav_bytes).decode('utf-8')

        return JSONResponse({
            "status": "ok",
            "text": user_text,
            "raw_text": text,
            "response": response_text,
            "action": action,
            "emotion": emotion,
            "servo_angles": servo_angles,
            "eye_led": eye_led,
            "audio_base64": audio_b64,
            "audio_input_mode": audio_input_mode,
            "audio_output_mode": audio_output_mode,
            "ai_modes": robot_brain.get_modes()
        })

    except Exception as e:
        logger.error(f"Ошибка в /speak: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({"status": "error", "message": str(e)})


@app.post("/voice")
async def voice_input(
    audio: UploadFile = File(...),
    audio_output_mode_param: str = Form("")
):
    current_output_mode = audio_output_mode_param if audio_output_mode_param in ["robot", "local"] else audio_output_mode

    logger.info(f"🎤 Голосовой ввод: {audio.filename}, {audio.size} bytes, output_mode={current_output_mode}")

    if robot_brain is None:
        return JSONResponse({"status": "error", "message": "Сервер ещё загружается"})

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await audio.read()
            tmp.write(content)
            tmp_path = tmp.name

        logger.info("Распознавание речи...")
        stt_result = robot_brain.stt.transcribe_from_file(tmp_path)
        os.unlink(tmp_path)

        if not stt_result.get("success"):
            return JSONResponse({"status": "error", "message": "Речь не распознана", "stt_error": stt_result.get("error")})

        raw_text = stt_result["text"]
        user_text = stt_result.get("corrected_text", raw_text) or raw_text

        if user_text != raw_text:
            logger.info(f"🎯 Fuzzy (/voice): '{raw_text}' → '{user_text}'")

        logger.info(f"👤 Распознано: '{user_text}'")

        if not user_text.strip():
            return JSONResponse({"status": "error", "message": "Пустой текст"})

        llm_result = robot_brain.llm.generate(user_text, robot_brain.vision_context)
        response_text = llm_result.get("text", "")
        action = llm_result.get("action")
        emotion = llm_result.get("emotion", "calm")

        servo_angles = robot_brain.emotion_engine.get_servo_angles(emotion)
        eye_led = robot_brain.emotion_engine.get_eye_led(emotion)

        tts_audio = robot_brain.tts.synthesize(response_text)

        if action:
            asyncio.create_task(robot_brain.servos.play_animation(action))
        else:
            robot_brain.servos.set_all_servos(servo_angles)

        audio_b64 = ""
        if current_output_mode == "local" and tts_audio:
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(48000)
                wf.writeframes(tts_audio)
            wav_bytes = wav_buffer.getvalue()
            audio_b64 = base64.b64encode(wav_bytes).decode('utf-8')

        return JSONResponse({
            "status": "ok",
            "user_text": user_text,
            "raw_text": raw_text,
            "response": response_text,
            "action": action,
            "emotion": emotion,
            "servo_angles": servo_angles,
            "eye_led": eye_led,
            "audio_base64": audio_b64,
            "audio_input_mode": audio_input_mode,
            "audio_output_mode": current_output_mode,
            "ai_modes": robot_brain.get_modes()
        })

    except Exception as e:
        logger.error(f"Ошибка в /voice: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({"status": "error", "message": str(e)})


# ===== WEBSOCKET =====

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.add(websocket)
    client_info = f"{websocket.client.host}:{websocket.client.port}"
    logger.info(f"ESP32 подключен: {client_info}")

    try:
        while True:
            message = await websocket.receive()
            if "text" in message:
                await handle_text_message(websocket, message["text"])
            elif "bytes" in message:
                await handle_binary_message(websocket, message["bytes"])
    except WebSocketDisconnect:
        logger.info(f"ESP32 отключен: {client_info}")
    except Exception as e:
        logger.error(f"Ошибка WebSocket: {e}")
    finally:
        active_connections.discard(websocket)


async def handle_text_message(websocket: WebSocket, text: str):
    try:
        data = json.loads(text)
        msg_type = data.get("type")

        if msg_type in ["servo", "servo_multi", "animation", "text", "get_status", "clear_history", "set_mode"]:
            result = await robot_brain.handle_command(data)
            await websocket.send_json(result)
        elif msg_type == "ping":
            await websocket.send_json({"type": "pong", "timestamp": data.get("timestamp")})
        elif msg_type == "audio_mode":
            await websocket.send_json({
                "type": "audio_mode",
                "input_mode": audio_input_mode,
                "output_mode": audio_output_mode
            })
        elif msg_type == "ai_mode":
            await websocket.send_json({
                "type": "ai_mode",
                "modes": robot_brain.get_modes()
            })
        else:
            await websocket.send_json({"status": "error", "message": f"Неизвестный тип: {msg_type}"})
    except json.JSONDecodeError:
        await websocket.send_json({"status": "error", "message": "Неверный JSON"})
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        await websocket.send_json({"status": "error", "message": str(e)})


async def handle_binary_message(websocket: WebSocket, data: bytes):
    try:
        if len(data) < 5:
            return
        data_type = data[:4].decode('ascii')
        payload = data[4:]

        if data_type == "AUDI":
            result = await robot_brain.process_audio_chunk(payload)
            if result:
                response = {
                    "type": "response",
                    "user_text": result["text"],
                    "robot_text": result["response"],
                    "action": result.get("action"),
                    "emotion": result.get("emotion", "calm"),
                    "servo_angles": result.get("servo_angles", [90]*18),
                    "eye_led": result.get("eye_led", "soft_white_low"),
                    "ai_modes": robot_brain.get_modes()
                }
                await websocket.send_json(response)

                if result["audio"] and audio_output_mode == "robot":
                    audio_packet = b"AUDI" + result["audio"]
                    await websocket.send_bytes(audio_packet)

        elif data_type == "VIDE":
            vision_result = await robot_brain.process_video_frame(payload)
            servo_cmd = {
                "type": "servo_update",
                "angles": vision_result["servo_angles"],
                "face_detected": vision_result["face_detected"],
                "face_offset": vision_result["face_offset"]
            }
            await websocket.send_json(servo_cmd)
    except Exception as e:
        logger.error(f"Ошибка бинарных данных: {e}")


# ===== ПАНЕЛЬ УПРАВЛЕНИЯ =====

@app.get("/panel", response_class=HTMLResponse)
async def control_panel():
    return PANEL_HTML


PANEL_HTML = """<!DOCTYPE html>
<html>
<head>
    <title>Soren Control Panel v3.0</title>
    <meta charset="utf-8">
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; background: #1a1a2e; color: #eee; padding: 20px; max-width: 1400px; margin: 0 auto; }
        h1 { color: #00d4ff; text-align: center; }
        h2 { color: #00d4ff; font-size: 18px; margin-top: 0; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .section { background: #16213e; border-radius: 10px; padding: 20px; margin: 10px 0; }
        .section-full { grid-column: 1 / -1; }
        .servo-grid { display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px; }
        .servo { background: #0f3460; padding: 8px; border-radius: 5px; text-align: center; font-size: 12px; }
        .servo input { width: 100%; }
        button { background: #00d4ff; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin: 5px; font-weight: bold; color: #000; }
        button:hover { background: #0099cc; }
        button.danger { background: #ff4444; color: #fff; }
        button.success { background: #00ff88; color: #000; }
        button.mic { background: #ff6b00; color: #fff; font-size: 18px; padding: 15px 30px; }
        button.mic.recording { background: #ff0000; animation: pulse 1s infinite; }
        button.mode-btn { font-size: 12px; padding: 8px 16px; }
        button.mode-btn.active { background: #00ff88; color: #000; }
        button.mode-btn.inactive { background: #444; color: #888; }
        button.mode-btn:hover { opacity: 0.8; }
        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
        #log { background: #0a0a0a; padding: 10px; height: 250px; overflow-y: scroll; font-family: monospace; font-size: 12px; border-radius: 5px; }
        .chat-container { display: flex; flex-direction: column; gap: 10px; }
        .chat-input { display: flex; gap: 10px; }
        .chat-input input { flex: 1; padding: 12px; border-radius: 5px; border: none; background: #0f3460; color: #fff; font-size: 14px; }
        .message { padding: 10px; border-radius: 8px; margin: 5px 0; }
        .message.user { background: #0f3460; }
        .message.robot { background: #1a4a3e; }
        .emotion-badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 11px; margin-left: 10px; }
        .emotion-calm { background: #4a90d9; }
        .emotion-sad { background: #5a5a8a; }
        .emotion-angry { background: #d94a4a; }
        .emotion-loving { background: #d9a04a; }
        .emotion-determined { background: #4ad94a; }
        .emotion-surprised { background: #d94aff; }
        .emotion-tired { background: #8a8a8a; }
        .audio-controls { display: flex; align-items: center; gap: 15px; padding: 10px; background: #0f3460; border-radius: 8px; margin-bottom: 15px; flex-wrap: wrap; }
        .audio-control-group { display: flex; align-items: center; gap: 10px; padding: 8px 15px; background: #0a1628; border-radius: 8px; border: 1px solid #1a3a5c; }
        .audio-control-group label { font-size: 13px; color: #aaa; min-width: 120px; }
        .toggle-switch { position: relative; display: inline-block; width: 60px; height: 30px; }
        .toggle-switch input { opacity: 0; width: 0; height: 0; }
        .slider { position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background-color: #ff4444; transition: .4s; border-radius: 30px; }
        .slider:before { position: absolute; content: ""; height: 22px; width: 22px; left: 4px; bottom: 4px; background-color: white; transition: .4s; border-radius: 50%; }
        input:checked + .slider { background-color: #00ff88; }
        input:checked + .slider:before { transform: translateX(30px); }
        .mode-robot { color: #ff4444; }
        .mode-local { color: #00ff88; }
        .voice-section { text-align: center; padding: 20px; }
        .voice-status { font-size: 14px; color: #888; margin-top: 10px; min-height: 20px; }
        .recording-indicator { display: none; color: #ff0000; font-weight: bold; }
        .recording-indicator.active { display: inline; }
        .audio-visualizer { width: 100%; height: 40px; background: #0a0a0a; border-radius: 5px; margin: 10px 0; display: none; }
        .audio-visualizer.active { display: block; }
        .mode-label { font-size: 12px; font-weight: bold; }
        .ai-mode-panel { display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; }
        .ai-module { background: #0a1628; border: 1px solid #1a3a5c; border-radius: 8px; padding: 15px; text-align: center; }
        .ai-module h3 { margin: 0 0 10px 0; color: #00d4ff; font-size: 16px; }
        .ai-module .status { font-size: 13px; margin: 8px 0; font-weight: bold; }
        .ai-module .status.cloud { color: #00ff88; }
        .ai-module .status.local { color: #4a90d9; }
        .ai-module .status.error { color: #ff4444; }
        .ai-module .desc { font-size: 11px; color: #666; margin-top: 8px; }
    </style>
</head>
<body>
    <h1>🦉 Soren Control Panel v3.0</h1>

    <!-- Режимы AI модулей -->
    <div class="section section-full">
        <h2>🧠 Режимы AI модулей (Local / Cloud)</h2>
        <div class="ai-mode-panel" id="ai-mode-panel">
            <div class="ai-module">
                <h3>🎤 STT — Распознавание речи</h3>
                <div class="status" id="stt-status">Загрузка...</div>
                <button class="mode-btn" id="stt-local-btn" onclick="setAIMode('stt', 'local')">💻 Локально</button>
                <button class="mode-btn" id="stt-cloud-btn" onclick="setAIMode('stt', 'cloud')">☁️ Облако</button>
                <div class="desc">Локально: faster-whisper (~500MB)<br>Облако: недоступно (нужен VPN)</div>
            </div>
            <div class="ai-module">
                <h3>🧠 LLM — Языковая модель</h3>
                <div class="status" id="llm-status">Загрузка...</div>
                <button class="mode-btn" id="llm-local-btn" onclick="setAIMode('llm', 'local')">💻 Локально</button>
                <button class="mode-btn" id="llm-cloud-btn" onclick="setAIMode('llm', 'cloud')">☁️ Облако</button>
                <div class="desc">Локально: Qwen 7B (~4.5GB)<br>Облако: GitHub Models</div>
            </div>
            <div class="ai-module">
                <h3>🔊 TTS — Синтез речи</h3>
                <div class="status" id="tts-status">Загрузка...</div>
                <button class="mode-btn" id="tts-local-btn" onclick="setAIMode('tts', 'local')">💻 Локально</button>
                <button class="mode-btn" id="tts-cloud-btn" onclick="setAIMode('tts', 'cloud')">☁️ Облако</button>
                <div class="desc">Локально: Silero (~120MB)<br>Облако: FreeTTS (edge-tts)</div>
            </div>
        </div>
        <div style="font-size: 12px; color: #888; margin-top: 12px; text-align: center;">
            ⚠️ Для облачного LLM нужен <b>GITHUB_MODELS_KEY</b> (токен GitHub)
        </div>
    </div>

    <!-- Режимы аудио -->
    <div class="section section-full">
        <h2>🔊 Режимы аудио (Robot / Local)</h2>
        <div class="audio-controls">
            <div class="audio-control-group">
                <label>🎤 <b>Ввод звука:</b></label>
                <label class="toggle-switch">
                    <input type="checkbox" id="audio-input-toggle" onchange="toggleAudioInputMode()">
                    <span class="slider"></span>
                </label>
                <span id="audio-input-mode-text" class="mode-robot mode-label">🔊 Робот (ESP32 микрофон)</span>
            </div>
            <div class="audio-control-group">
                <label>🔊 <b>Вывод звука:</b></label>
                <label class="toggle-switch">
                    <input type="checkbox" id="audio-output-toggle" onchange="toggleAudioOutputMode()">
                    <span class="slider"></span>
                </label>
                <span id="audio-output-mode-text" class="mode-robot mode-label">🔊 Робот (ESP32 динамик)</span>
            </div>
            <span id="connection-status" style="margin-left: auto; font-size: 14px;">● OFFLINE</span>
        </div>
    </div>

    <!-- Голосовое общение -->
    <div class="section section-full">
        <h2>🎤 Голосовое общение с Сореном</h2>
        <div class="voice-section">
            <button id="mic-btn" class="mic" onclick="toggleRecording()">🎤 Нажми и говори</button>
            <div class="voice-status">
                <span id="voice-status-text">Нажмите кнопку микрофона и говорите</span>
                <span id="recording-indicator" class="recording-indicator"> 🔴 ЗАПИСЬ</span>
            </div>
            <canvas id="audio-visualizer" class="audio-visualizer"></canvas>
        </div>
    </div>

    <!-- Чат -->
    <div class="grid">
        <div class="section">
            <h2>💬 Чат с Сореном</h2>
            <div class="chat-container">
                <div id="chat-history" style="max-height: 300px; overflow-y: auto;"></div>
                <div class="chat-input">
                    <input type="text" id="chat-input" placeholder="Напиши Сорену..." onkeypress="if(event.key==='Enter') sendChat()">
                    <button onclick="sendChat()">Отправить</button>
                </div>
            </div>
        </div>
        <div class="section">
            <h2>🎬 Анимации</h2>
            <button onclick="sendCmd({type:'animation',name:'wave'})">👋 Помахать</button>
            <button onclick="sendCmd({type:'animation',name:'nod'})">🙂 Кивнуть</button>
            <button onclick="sendCmd({type:'animation',name:'shake_head'})">😕 Качнуть</button>
            <button onclick="sendCmd({type:'animation',name:'idle'})">😶 Покой</button>
            <br><br>
            <button onclick="sendCmd({type:'clear_history'})" class="danger">🗑 Очистить историю</button>
        </div>
    </div>

    <!-- Сервоприводы -->
    <div class="section section-full">
        <h2>🦾 Сервоприводы (18 шт)</h2>
        <div class="servo-grid" id="servo-grid"></div>
        <br>
        <button onclick="setAllServos()">Применить все</button>
        <button onclick="resetServos()">Сбросить в 90°</button>
    </div>

    <!-- Лог -->
    <div class="section section-full">
        <h2>📋 Лог</h2>
        <div id="log"></div>
    </div>

    <audio id="audio-player" style="display:none;"></audio>

    <script>
        const ws = new WebSocket(`ws://${window.location.host}/ws`);
        let audioInputMode = 'robot';
        let audioOutputMode = 'robot';
        let aiModes = { stt: 'local', tts: 'local', llm: 'local' };

        ws.onopen = () => {
            document.getElementById('connection-status').textContent = '● ONLINE';
            document.getElementById('connection-status').style.color = '#00ff88';
            log('WebSocket подключен');
            ws.send(JSON.stringify({type: 'audio_mode'}));
            ws.send(JSON.stringify({type: 'ai_mode'}));
        };

        ws.onclose = () => {
            document.getElementById('connection-status').textContent = '● OFFLINE';
            document.getElementById('connection-status').style.color = '#ff4444';
            log('WebSocket отключен');
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            log('← ' + JSON.stringify(data));
            if (data.angles) updateServoDisplay(data.angles);
            if (data.emotion) log('Эмоция: ' + data.emotion);
            if (data.type === 'audio_mode') {
                if (data.input_mode) audioInputMode = data.input_mode;
                if (data.output_mode) audioOutputMode = data.output_mode;
                updateAudioModeUI();
            }
            if (data.type === 'ai_mode' && data.modes) {
                aiModes = data.modes;
                updateAIModeUI();
            }
            if (data.modes && !data.type) {
                aiModes = data.modes;
                updateAIModeUI();
            }
        };

        // ===== AI MODE SWITCHING =====
        async function setAIMode(module, mode) {
            log(`🧠 Переключение ${module.toUpperCase()} → ${mode}...`);
            const formData = new FormData();
            formData.append('module', module);
            formData.append('mode', mode);

            try {
                const response = await fetch('/ai_mode', {method: 'POST', body: formData});
                const data = await response.json();
                if (data.status === 'ok') {
                    if (data.modes) {
                        aiModes = data.modes;
                        updateAIModeUI();
                    }
                    log(`✅ ${module.toUpperCase()} переключён на ${mode}`);
                } else {
                    log('❌ Ошибка: ' + (data.message || 'Неизвестная'));
                    // Обновим UI на случай если сервер вернул текущие режимы
                    if (data.modes) {
                        aiModes = data.modes;
                        updateAIModeUI();
                    }
                }
            } catch (e) {
                log('❌ Сетевая ошибка: ' + e.message);
            }
        }

        function updateAIModeUI() {
            const modules = ['stt', 'llm', 'tts'];
            modules.forEach(mod => {
                const mode = aiModes[mod] || 'local';
                const statusEl = document.getElementById(mod + '-status');
                const localBtn = document.getElementById(mod + '-local-btn');
                const cloudBtn = document.getElementById(mod + '-cloud-btn');

                if (statusEl) {
                    if (mode === 'cloud') {
                        statusEl.textContent = '☁️ Облачный';
                        statusEl.className = 'status cloud';
                    } else {
                        statusEl.textContent = '💻 Локальный';
                        statusEl.className = 'status local';
                    }
                }
                if (localBtn) {
                    localBtn.className = 'mode-btn ' + (mode === 'local' ? 'active' : 'inactive');
                }
                if (cloudBtn) {
                    cloudBtn.className = 'mode-btn ' + (mode === 'cloud' ? 'active' : 'inactive');
                }
            });
        }

        // ===== AUDIO MODES =====
        async function toggleAudioInputMode() {
            const toggle = document.getElementById('audio-input-toggle');
            const newMode = toggle.checked ? 'local' : 'robot';
            const formData = new FormData();
            formData.append('mode', newMode);
            formData.append('type', 'input');
            const response = await fetch('/audio_mode', {method: 'POST', body: formData});
            const data = await response.json();
            if (data.status === 'ok') {
                audioInputMode = data.audio_input_mode;
                updateAudioModeUI();
            }
        }

        async function toggleAudioOutputMode() {
            const toggle = document.getElementById('audio-output-toggle');
            const newMode = toggle.checked ? 'local' : 'robot';
            const formData = new FormData();
            formData.append('mode', newMode);
            formData.append('type', 'output');
            const response = await fetch('/audio_mode', {method: 'POST', body: formData});
            const data = await response.json();
            if (data.status === 'ok') {
                audioOutputMode = data.audio_output_mode;
                updateAudioModeUI();
            }
        }

        function updateAudioModeUI() {
            const inputToggle = document.getElementById('audio-input-toggle');
            const inputText = document.getElementById('audio-input-mode-text');
            inputToggle.checked = (audioInputMode === 'local');
            if (audioInputMode === 'local') {
                inputText.textContent = '🎧 Локально (микрофон ПК)';
                inputText.className = 'mode-local mode-label';
            } else {
                inputText.textContent = '🔊 Робот (ESP32 микрофон)';
                inputText.className = 'mode-robot mode-label';
            }

            const outputToggle = document.getElementById('audio-output-toggle');
            const outputText = document.getElementById('audio-output-mode-text');
            outputToggle.checked = (audioOutputMode === 'local');
            if (audioOutputMode === 'local') {
                outputText.textContent = '🎧 Локально (наушники ПК)';
                outputText.className = 'mode-local mode-label';
            } else {
                outputText.textContent = '🔊 Робот (ESP32 динамик)';
                outputText.className = 'mode-robot mode-label';
            }
        }

        // ===== VOICE =====
        let mediaRecorder = null;
        let audioChunks = [];
        let isRecording = false;
        let audioContext = null;
        let analyser = null;
        let visualizerCanvas = null;
        let visualizerCtx = null;

        async function toggleRecording() {
            const btn = document.getElementById('mic-btn');
            const statusText = document.getElementById('voice-status-text');
            const indicator = document.getElementById('recording-indicator');
            const visualizer = document.getElementById('audio-visualizer');

            if (!isRecording) {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    mediaRecorder = new MediaRecorder(stream);
                    audioChunks = [];
                    mediaRecorder.ondataavailable = (e) => { audioChunks.push(e.data); };
                    mediaRecorder.onstop = async () => {
                        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                        await sendVoiceToServer(audioBlob);
                        stream.getTracks().forEach(track => track.stop());
                    };
                    mediaRecorder.start();
                    isRecording = true;
                    btn.textContent = '⏹ Остановить';
                    btn.classList.add('recording');
                    statusText.textContent = 'Идёт запись... говорите';
                    indicator.classList.add('active');
                    visualizer.classList.add('active');
                    setupVisualizer(stream);
                    log('🎤 Начало записи голоса');
                } catch (err) {
                    log('❌ Ошибка доступа к микрофону: ' + err.message);
                    alert('Разрешите доступ к микрофону в настройках браузера');
                }
            } else {
                mediaRecorder.stop();
                isRecording = false;
                btn.textContent = '🎤 Нажми и говори';
                btn.classList.remove('recording');
                statusText.textContent = 'Обработка...';
                indicator.classList.remove('active');
                visualizer.classList.remove('active');
                log('🎤 Конец записи, отправка...');
            }
        }

        function setupVisualizer(stream) {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            analyser = audioContext.createAnalyser();
            const source = audioContext.createMediaStreamSource(stream);
            source.connect(analyser);
            analyser.fftSize = 256;
            visualizerCanvas = document.getElementById('audio-visualizer');
            visualizerCtx = visualizerCanvas.getContext('2d');
            visualizerCanvas.width = visualizerCanvas.offsetWidth;
            visualizerCanvas.height = visualizerCanvas.offsetHeight;
            drawVisualizer();
        }

        function drawVisualizer() {
            if (!isRecording || !analyser) return;
            requestAnimationFrame(drawVisualizer);
            const bufferLength = analyser.frequencyBinCount;
            const dataArray = new Uint8Array(bufferLength);
            analyser.getByteFrequencyData(dataArray);
            visualizerCtx.fillStyle = '#0a0a0a';
            visualizerCtx.fillRect(0, 0, visualizerCanvas.width, visualizerCanvas.height);
            const barWidth = (visualizerCanvas.width / bufferLength) * 2.5;
            let x = 0;
            for (let i = 0; i < bufferLength; i++) {
                const barHeight = dataArray[i] / 2;
                visualizerCtx.fillStyle = `rgb(${barHeight + 100}, 50, 50)`;
                visualizerCtx.fillRect(x, visualizerCanvas.height - barHeight, barWidth, barHeight);
                x += barWidth + 1;
            }
        }

        async function sendVoiceToServer(audioBlob) {
            const statusText = document.getElementById('voice-status-text');
            statusText.textContent = 'Отправка на сервер...';
            const formData = new FormData();
            formData.append('audio', audioBlob, 'voice.wav');
            formData.append('audio_output_mode_param', audioOutputMode);
            try {
                const response = await fetch('/voice', { method: 'POST', body: formData });
                const data = await response.json();
                if (data.status === 'ok') {
                    addMessage('user', '🎤 ' + data.user_text);
                    const emotionClass = 'emotion-' + (data.emotion || 'calm');
                    addMessage('robot', data.response + `<span class="emotion-badge ${emotionClass}">${data.emotion}</span>`);
                    if (audioOutputMode === 'local' && data.audio_base64) {
                        playAudio(data.audio_base64);
                    }
                    if (data.ai_modes) {
                        aiModes = data.ai_modes;
                        updateAIModeUI();
                    }
                    statusText.textContent = 'Готово! Нажмите для новой записи';
                } else {
                    addMessage('robot', 'Ошибка: ' + (data.message || 'Неизвестная'));
                    statusText.textContent = 'Ошибка: ' + data.message;
                }
            } catch (e) {
                log('❌ Ошибка сети: ' + e);
                statusText.textContent = 'Ошибка сети';
            }
        }

        // ===== CHAT =====
        async function sendChat() {
            const input = document.getElementById('chat-input');
            const text = input.value.trim();
            if (!text) return;
            input.value = '';
            addMessage('user', text);
            if (audioOutputMode === 'robot') {
                sendCmd({type: 'text', text: text});
            } else {
                await sendLocal(text);
            }
        }

        async function sendLocal(text) {
            try {
                const formData = new FormData();
                formData.append('text', text);
                const response = await fetch('/speak', { method: 'POST', body: formData });
                const data = await response.json();
                if (data.status === 'ok') {
                    const emotionClass = 'emotion-' + (data.emotion || 'calm');
                    addMessage('robot', data.response + `<span class="emotion-badge ${emotionClass}">${data.emotion}</span>`);
                    if (data.audio_base64) playAudio(data.audio_base64);
                    if (data.ai_modes) {
                        aiModes = data.ai_modes;
                        updateAIModeUI();
                    }
                } else {
                    addMessage('robot', 'Ошибка: ' + (data.message || 'Неизвестная'));
                }
            } catch (e) {
                log('❌ Ошибка сети: ' + e);
            }
        }

        function playAudio(base64Audio) {
            const audio = document.getElementById('audio-player');
            audio.src = 'data:audio/wav;base64,' + base64Audio;
            audio.play().catch(e => log('❌ Audio play error: ' + e.message));
        }

        function addMessage(sender, text) {
            const chat = document.getElementById('chat-history');
            const div = document.createElement('div');
            div.className = 'message ' + sender;
            div.innerHTML = sender === 'user' ? '<b>👤 Вы:</b> ' + text : '<b>🦉 Сорен:</b> ' + text;
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
        }

        function sendCmd(cmd) {
            ws.send(JSON.stringify(cmd));
            log('→ ' + JSON.stringify(cmd));
        }

        function log(msg) {
            const el = document.getElementById('log');
            el.innerHTML += `[${new Date().toLocaleTimeString()}] ${msg}<br>`;
            el.scrollTop = el.scrollHeight;
        }

        // ===== SERVOS =====
        function createServoGrid() {
            const grid = document.getElementById('servo-grid');
            for (let i = 0; i < 18; i++) {
                const div = document.createElement('div');
                div.className = 'servo';
                div.innerHTML = `<b>S${i}</b><br><span id="val-${i}">90°</span><br><input type="range" id="servo-${i}" min="0" max="180" value="90" oninput="document.getElementById('val-${i}').textContent=this.value+'°'">`;
                grid.appendChild(div);
            }
        }
        function setAllServos() {
            const angles = [];
            for (let i = 0; i < 18; i++) angles.push(parseInt(document.getElementById(`servo-${i}`).value));
            sendCmd({type: 'servo_multi', angles: angles});
        }
        function resetServos() {
            for (let i = 0; i < 18; i++) {
                document.getElementById(`servo-${i}`).value = 90;
                document.getElementById(`val-${i}`).textContent = '90°';
            }
            sendCmd({type: 'servo_multi', angles: new Array(18).fill(90)});
        }
        function updateServoDisplay(angles) {
            for (let i = 0; i < angles.length; i++) {
                const slider = document.getElementById(`servo-${i}`);
                const val = document.getElementById(`val-${i}`);
                if (slider && val) { slider.value = angles[i]; val.textContent = angles[i] + '°'; }
            }
        }

        createServoGrid();
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)