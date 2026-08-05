"""Главный мозг робота — оркестратор с эмоциональным движком и памятью"""
import asyncio
import logging
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, List
from modules.stt import STTEngine
from modules.tts import TTSEngine
from modules.llm import LLMEngine
from modules.vision import VisionEngine
from modules.audio_buffer import AudioBuffer
from modules.servo_controller import ServoController
from modules.memory import MemoryManager
from config.settings import (
    CHARACTER_DIR,
    FAST_MODE,
    FACE_TRACKING_ENABLED,
    DIALOG_ACTIVE_TIMEOUT_SEC,
    PROVISIONAL_DIALOG_TIMEOUT_SEC,
    FACE_PAN_SERVO,
    FACE_TILT_SERVO,
    FACE_PAN_GAIN,
    FACE_TILT_GAIN,
    HEAD_SMOOTHING_ALPHA,
    FACE_DEADZONE_X,
    FACE_DEADZONE_Y,
    EXIT_EASE_ALPHA,
    EXIT_EASE_EPSILON,
)

logger = logging.getLogger(__name__)


class EmotionEngine:
    """Эмоциональный движок Сорена — позы и LED из JSON"""

    def __init__(self, emotions_path: Path):
        self.emotions: Dict = {}
        self.current_emotion = "calm"
        self._load(emotions_path)

    def _load(self, path: Path):
        if not path.exists():
            logger.warning(f"Файл эмоций не найден: {path}")
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.emotions = data.get("emotions", {})
            logger.info(f"Эмоции загружены: {list(self.emotions.keys())}")
        except Exception as e:
            logger.error(f"Ошибка загрузки эмоций: {e}")

    def get_pose(self, emotion: str) -> Optional[Dict[str, int]]:
        if emotion in self.emotions:
            return self.emotions[emotion].get("servo_pose")
        return None

    def get_eye_led(self, emotion: str) -> str:
        if emotion in self.emotions:
            return self.emotions[emotion].get("eye_led", "soft_white_low")
        return "soft_white_low"

    def get_servo_angles(self, emotion: str) -> List[int]:
        pose = self.get_pose(emotion)
        if not pose:
            return [90] * 18

        angles = []
        for i in range(18):
            key = f"S{i}"
            angles.append(pose.get(key, 90))
        return angles


class RobotBrain:
    """Основной контроллер — соединяет все модули с памятью"""

    def __init__(self):
        logger.info("=== Инициализация RobotBrain (Сорен v3.5 + Vector RAG) ===")

        # Память ДО остальных модулей
        self.memory = MemoryManager()

        self.stt = STTEngine()
        self.tts = TTSEngine()
        self.llm = LLMEngine(memory_manager=self.memory)
        self.vision = VisionEngine()
        self.audio_buffer = AudioBuffer()
        self.servos = ServoController()

        emotions_path = CHARACTER_DIR / "Soren_emotions.json"
        self.emotion_engine = EmotionEngine(emotions_path)

        self.vision_context = ""
        self.is_processing = False
        self.current_emotion = "calm"

        # === Фоновые потоки для тяжёлых синхронных операций ===
        # Не даём CV (YOLO/DNN/FaceMesh) и STT/LLM/TTS блокировать asyncio event loop —
        # иначе на время одного тяжёлого вызова встают ВСЕ подключения (панель, /speak, /voice,
        # другие устройства), а не только текущее. Два ОТДЕЛЬНЫХ пула (а не общий executor
        # asyncio), чтобы долгая генерация ответа LLM не задерживала обработку видео —
        # слежение за лицом должно продолжаться, даже пока робот "думает" над ответом.
        # max_workers=1 в каждом — этого достаточно (устройство одно), и это же служит
        # бесплатной защитой от случайного параллельного запуска двух тяжёлых кадров разом.
        self._cv_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cv-worker")
        self._speech_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="speech-worker")

        # === Состояние "активного диалога" (для слежения за лицом) ===
        # Робот следит за лицом только пока идёт общение: сразу после
        # распознанной речи/текстовой команды и ещё DIALOG_ACTIVE_TIMEOUT_SEC
        # секунд после последнего обмена репликами (чтобы не дёргать голову
        # между отдельными фразами диалога).
        # === Состояние "активного диалога" (для слежения за лицом) ===
        # Два независимых таймера вместо одного, с разным уровнем доверия к сигналу:
        #  - last_interaction_ts ("подтверждённый"): продлевается, когда речь РЕАЛЬНО
        #    распознана / пришла текстовая команда — держит окно DIALOG_ACTIVE_TIMEOUT_SEC,
        #    а после ответа ещё и на время самого озвучивания (см. _process_speech).
        #  - provisional_interaction_ts ("предварительный"): продлевается уже на VAD
        #    speech_start/speech, ДО распознавания — короткое окно PROVISIONAL_DIALOG_TIMEOUT_SEC,
        #    чтобы голова начинала реагировать сразу, но быстро отпускала, если это
        #    оказался просто шум/кашель, а не обращение к роботу.
        # Диалог считается активным, если жив хотя бы один из двух таймеров.
        self.last_interaction_ts = 0.0
        self.provisional_interaction_ts = 0.0

        # === Сглаживание слежения за лицом + "мёртвая зона" ===
        self._smoothed_face_offset = [0.0, 0.0]   # (x, y), между кадрами
        self._last_target_track_id = None         # для детекции смены цели (снап в центр)
        self._was_dialog_active = False            # для детекции старта диалога (снап в центр)

        # Последнее известное состояние лица — используется при отрисовке кадров ПАНЕЛИ
        # (VIDP), которые приходят отдельно от кадров детекции (VIDE) и сами по себе
        # детекцию не запускают (см. update_panel_frame / render_panel_jpeg).
        self._last_face_detected = False
        self._last_face_bbox = None
        self._last_faces_count = 0

        logger.info("=== RobotBrain (Сорен v3.5 + Vector RAG) готов ===")

    def mark_dialog_active(self):
        """Подтверждённое взаимодействие (распознанная речь / текстовая команда) —
        продлевает основное окно слежения за лицом"""
        self.last_interaction_ts = time.time()

    def mark_dialog_provisional(self):
        """Предварительный сигнал (VAD засёк голос, ещё не распознано) —
        продлевает короткое окно, чтобы голова среагировала раньше"""
        self.provisional_interaction_ts = time.time()

    def is_dialog_active(self) -> bool:
        """Активен ли сейчас диалог (для решения — следить за лицом или нет)"""
        if self.is_processing:
            return True

        now = time.time()

        confirmed_active = (
            self.last_interaction_ts > 0 and
            (now - self.last_interaction_ts) < DIALOG_ACTIVE_TIMEOUT_SEC
        )
        provisional_active = (
            self.provisional_interaction_ts > 0 and
            (now - self.provisional_interaction_ts) < PROVISIONAL_DIALOG_TIMEOUT_SEC
        )
        return confirmed_active or provisional_active

    def set_stt_mode(self, mode: str):
        self.stt.set_mode(mode)

    def set_tts_mode(self, mode: str):
        self.tts.set_mode(mode)

    def set_llm_mode(self, mode: str):
        self.llm.set_mode(mode)

    def get_modes(self) -> Dict[str, str]:
        return {
            "stt": self.stt.get_mode(),
            "tts": self.tts.get_mode(),
            "llm": self.llm.get_mode()
        }

    async def process_audio_chunk(self, pcm_bytes: bytes) -> Optional[dict]:
        status, audio = self.audio_buffer.process_chunk(pcm_bytes)

        if status == "speech_start":
            # Начало новой фразы — включаем режим "слежу по губам, кто говорит"
            # (используется в vision.process_video_frame для выбора цели слежения)
            # и сразу продлеваем предварительное окно диалога — голова реагирует
            # ДО того, как фраза будет распознана.
            self.vision.notify_speech_start()
            self.mark_dialog_provisional()
        elif status == "speech":
            # Человек продолжает говорить — держим предварительное окно свежим,
            # чтобы оно не истекло на середине длинной фразы.
            self.mark_dialog_provisional()
        elif status == "complete" and audio:
            # Фраза закончилась — прекращаем перевыбор цели по губам до следующей фразы
            self.vision.notify_speech_end()
            return await self._process_speech(audio)

        return None

    async def _process_speech(self, audio_bytes: bytes) -> dict:
        self.is_processing = True
        try:
            loop = asyncio.get_event_loop()
            # STT + LLM + TTS — тяжёлые синхронные вызовы (LLM может думать секундами) —
            # выполняем в отдельном потоке, чтобы не блокировать event loop целиком.
            result = await loop.run_in_executor(self._speech_executor, self._process_speech_sync, audio_bytes)

            # asyncio.create_task требует запущенный event loop в текущем потоке,
            # поэтому анимацию запускаем здесь, а не внутри воркер-потока.
            action = result.get("action")
            if action:
                asyncio.create_task(self.servos.play_animation(action))

            return result
        finally:
            self.is_processing = False

    def _process_speech_sync(self, audio_bytes: bytes) -> dict:
        """Синхронное тело обработки речи (STT+LLM+TTS) — выполняется в фоновом потоке"""
        logger.info("Распознавание речи...")
        stt_result = self.stt.transcribe(audio_bytes)

        if not stt_result["success"]:
            logger.warning("Речь не распознана")
            return self._build_empty_response()

        raw_text = stt_result["text"]
        user_text = stt_result.get("corrected_text", raw_text) or raw_text

        if user_text != raw_text:
            logger.info(f"🎯 Fuzzy: используем исправленный текст: '{user_text}'")

        logger.info(f"Пользователь: {user_text}")
        self.mark_dialog_active()

        # Собираем полный контекст памяти
        memory_context = self._build_memory_context(user_text)

        logger.info("Генерация ответа Сорена...")
        llm_result = self.llm.generate(user_text, self.vision_context, memory_context)
        response_text = llm_result["text"]
        action = llm_result.get("action")
        emotion = llm_result.get("emotion", "calm")
        self.current_emotion = emotion

        logger.info(f"Сорен: {response_text}")
        logger.info(f"Эмоция: {emotion}")
        if action:
            logger.info(f"Действие: {action}")

        servo_angles = self.emotion_engine.get_servo_angles(emotion)
        eye_led = self.emotion_engine.get_eye_led(emotion)

        logger.info("Синтез речи...")
        tts_audio = self.tts.synthesize(response_text)

        if tts_audio:
            # Продлеваем подтверждённое окно диалога на время самого озвучивания ответа —
            # оба TTS-движка отдают PCM 16-bit mono 48kHz (см. modules/tts.py), поэтому
            # длительность оцениваем как байты / (48000 * 2). Так голова не "отключится"
            # на середине длинного ответа Сорена.
            estimated_playback_sec = len(tts_audio) / (48000 * 2)
            self.last_interaction_ts = time.time() + estimated_playback_sec

        if not action:
            # Анимацию (если есть) запустит async-обёртка _process_speech — ей нужен event loop
            self.servos.set_all_servos(servo_angles)

        return {
            "text": user_text,
            "raw_text": raw_text,
            "response": response_text,
            "audio": tts_audio,
            "action": action,
            "emotion": emotion,
            "servo_angles": servo_angles,
            "eye_led": eye_led
        }

    def _build_memory_context(self, user_text: str) -> str:
        """Собирает контекст для LLM из памяти"""
        # Краткосрочная память — последние реплики текущей сессии (RAM, не хранилище)
        stm = self.memory.short_term.get_context()

        if FAST_MODE:
            # Ничего лишнего: ни RAG-воспоминаний, ни эмоционального профиля,
            # ни приветствий по времени — только сама история текущего диалога.
            return f"Последние реплики:\n{stm}" if stm else ""

        parts = []
        if stm:
            parts.append(f"Последние реплики:\n{stm}")

        # Долгосрочная память — релевантные воспоминания
        ltm = self.memory.get_relevant_memories(user_text)
        if ltm:
            parts.append(f"Релевантные воспоминания:\n{ltm}")

        # Эмоциональный профиль
        dom = max(self.memory.profile.dominant_emotions, key=self.memory.profile.dominant_emotions.get)
        parts.append(f"Доминантная эмоция пользователя: {dom}")

        # Приветствие по времени
        greeting = self.memory.get_day_greeting()
        if greeting and not stm:
            parts.append(f"Приветствие: {greeting}")

        return "\n\n".join(parts)

    def _handle_text_command_sync(self, user_text: str) -> dict:
        """Синхронное тело обработки текстовой команды (LLM+TTS) — выполняется в фоновом потоке"""
        # Собираем контекст памяти
        memory_context = self._build_memory_context(user_text)

        llm_result = self.llm.generate(user_text, self.vision_context, memory_context)
        emotion = llm_result.get("emotion", "calm")
        servo_angles = self.emotion_engine.get_servo_angles(emotion)
        eye_led = self.emotion_engine.get_eye_led(emotion)
        tts_audio = self.tts.synthesize(llm_result["text"])

        if not llm_result.get("action"):
            # Анимацию (если есть) запустит handle_command — ей нужен event loop
            self.servos.set_all_servos(servo_angles)

        return {
            "response": llm_result["text"],
            "audio": tts_audio,
            "action": llm_result.get("action"),
            "emotion": emotion,
            "servo_angles": servo_angles,
            "eye_led": eye_led
        }

    def _build_empty_response(self) -> dict:
        return {
            "text": "",
            "raw_text": "",
            "response": "",
            "audio": b"",
            "action": None,
            "emotion": "calm",
            "servo_angles": [90] * 18,
            "eye_led": "soft_white_low"
        }

    async def process_video_frame(self, frame_bytes: bytes) -> dict:
        loop = asyncio.get_event_loop()
        # CV-обработка (YOLO/DNN-детектор лица/FaceMesh/Pose) — тяжёлая и синхронная,
        # выполняем в отдельном потоке, чтобы не блокировать event loop на время кадра.
        return await loop.run_in_executor(self._cv_executor, self._process_video_frame_sync, frame_bytes)

    def _process_video_frame_sync(self, frame_bytes: bytes) -> dict:
        """Синхронное тело обработки видеокадра — выполняется в фоновом потоке"""
        vision_result = self.vision.process_frame(frame_bytes)
        self.vision_context = vision_result.get("description", "")

        face_detected = vision_result.get("face_detected", False)
        target_track_id = vision_result.get("target_track_id")
        raw_offset = self.vision.get_face_offset(640, 480)

        # Слежение за лицом включается ТОЛЬКО во время активного диалога
        dialog_active = FACE_TRACKING_ENABLED and self.is_dialog_active()

        # Углы рук/плеч из позы тела применяются независимо от диалога,
        # а голову (pan/tilt) двигаем только когда реально следим за лицом —
        # иначе оставляем её в текущем положении, не дёргая робота.
        pose_angles = self.vision.get_servo_angles_from_pose()
        servo_angles = self.servos.get_current_angles()
        for i in range(16):
            servo_angles[i] = pose_angles[i]

        if dialog_active and face_detected:
            # "Снап" в центр нового лица — без сглаживания и без мёртвой зоны —
            # если диалог только что начался ИЛИ переключились на другого говорящего.
            # Физически голова всё равно доедет плавно за счёт интерполяции на ESP32.
            is_reorient = (not self._was_dialog_active) or (target_track_id != self._last_target_track_id)

            if is_reorient:
                self._smoothed_face_offset = [raw_offset[0], raw_offset[1]]
            else:
                in_deadzone = (
                    abs(raw_offset[0]) < FACE_DEADZONE_X and
                    abs(raw_offset[1]) < FACE_DEADZONE_Y
                )
                if not in_deadzone:
                    # Экспоненциальное сглаживание — гасит дрожание из-за шума детектора
                    self._smoothed_face_offset[0] += HEAD_SMOOTHING_ALPHA * (raw_offset[0] - self._smoothed_face_offset[0])
                    self._smoothed_face_offset[1] += HEAD_SMOOTHING_ALPHA * (raw_offset[1] - self._smoothed_face_offset[1])
                # если в мёртвой зоне — сглаженное состояние не трогаем, голова держит текущее положение

            servo_angles[FACE_PAN_SERVO] = int(90 - self._smoothed_face_offset[0] * FACE_PAN_GAIN)
            servo_angles[FACE_TILT_SERVO] = int(90 + self._smoothed_face_offset[1] * FACE_TILT_GAIN)

        elif dialog_active and not face_detected:
            # Диалог всё ещё идёт, лицо на мгновение потерялось (пара кадров) —
            # просто держим текущее положение головы, ждём, пока лицо снова найдётся.
            # Сглаженное состояние не трогаем, чтобы не начинать "выход" раньше времени.
            pass

        else:
            # Диалог закончился (или ещё не начинался) — плавно "отпускаем" голову
            # обратно в центр, а не оставляем её резко висеть в последней позиции.
            still_offset = (
                abs(self._smoothed_face_offset[0]) > EXIT_EASE_EPSILON or
                abs(self._smoothed_face_offset[1]) > EXIT_EASE_EPSILON
            )
            if still_offset:
                self._smoothed_face_offset[0] += EXIT_EASE_ALPHA * (0.0 - self._smoothed_face_offset[0])
                self._smoothed_face_offset[1] += EXIT_EASE_ALPHA * (0.0 - self._smoothed_face_offset[1])
                servo_angles[FACE_PAN_SERVO] = int(90 - self._smoothed_face_offset[0] * FACE_PAN_GAIN)
                servo_angles[FACE_TILT_SERVO] = int(90 + self._smoothed_face_offset[1] * FACE_TILT_GAIN)
            # иначе голова уже практически по центру — больше ничего не шлём,
            # чтобы не досылать бесконечные микро-поправки к идеальному нулю
        # (следующий раз, когда диалог начнётся заново, is_reorient сработает
        # и голова сразу прицелится в центр лица, прервав "оседание" при необходимости)

        self._was_dialog_active = dialog_active
        self._last_target_track_id = target_track_id
        self._last_face_detected = face_detected
        self._last_face_bbox = vision_result.get("face_bbox")
        self._last_faces_count = vision_result.get("faces_count", 0)

        return {
            "servo_angles": servo_angles,
            "face_offset": raw_offset,
            "description": self.vision_context,
            "objects": vision_result.get("objects", []),
            "face_detected": face_detected,
            "face_bbox": vision_result.get("face_bbox"),
            "faces_count": vision_result.get("faces_count", 0),
            "target_track_id": target_track_id,
            "dialog_active": dialog_active,
        }

    def get_panel_annotation_status(self) -> dict:
        """Текущее состояние лица/диалога для отрисовки в панели мониторинга —
        обновляется на каждом обработанном кадре детекции (VIDE)"""
        return {
            "face_detected": self._last_face_detected,
            "face_bbox": self._last_face_bbox,
            "faces_count": self._last_faces_count,
            "dialog_active": self._was_dialog_active,
        }

    async def render_panel_jpeg(self, quality: int) -> Optional[bytes]:
        """Рендерит текущий кадр для панели (с оверлеем лица) в фоновом CV-потоке"""
        loop = asyncio.get_event_loop()
        status = self.get_panel_annotation_status()
        tracking_active = bool(status["dialog_active"] and status["face_detected"])
        return await loop.run_in_executor(
            self._cv_executor, self.vision.get_annotated_jpeg, tracking_active, quality
        )

    async def update_panel_frame(self, frame_bytes: bytes) -> bool:
        """Принимает кадр повышенного качества ТОЛЬКО для панели (тег VIDP от прошивки,
        см. пункт про двойное разрешение) — выполняется в фоновом CV-потоке"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._cv_executor, self.vision.update_panel_frame, frame_bytes)

    async def handle_command(self, command: dict) -> dict:
        cmd_type = command.get("type")

        if cmd_type == "servo":
            self.servos.set_servo(command["id"], command["angle"])
            return {"status": "ok", "servo": command["id"], "angle": command["angle"]}

        elif cmd_type == "servo_multi":
            self.servos.set_all_servos(command["angles"])
            return {"status": "ok", "angles": command["angles"]}

        elif cmd_type == "animation":
            asyncio.create_task(self.servos.play_animation(command["name"]))
            return {"status": "ok", "animation": command["name"]}

        elif cmd_type == "text":
            self.mark_dialog_active()
            try:
                from modules.fuzzy_matcher import correct_speech_text
                raw_text = command["text"]
                corrected_text = correct_speech_text(raw_text)
                if corrected_text != raw_text:
                    logger.info(f"🎯 Fuzzy (text): '{raw_text}' → '{corrected_text}'")
                user_text = corrected_text
            except ImportError:
                user_text = command["text"]

            loop = asyncio.get_event_loop()
            # LLM.generate + TTS.synthesize — те же тяжёлые блокирующие вызовы, что и в
            # голосовом пути (_process_speech), тем же приёмом уводим их в фоновый поток.
            result = await loop.run_in_executor(self._speech_executor, self._handle_text_command_sync, user_text)

            if result.get("action"):
                asyncio.create_task(self.servos.play_animation(result["action"]))

            return {
                "status": "ok",
                "text": user_text,
                "raw_text": command.get("text", ""),
                "response": result["response"],
                "audio": result["audio"].hex() if result["audio"] else "",
                "action": result.get("action"),
                "emotion": result["emotion"],
                "servo_angles": result["servo_angles"],
                "eye_led": result["eye_led"]
            }

        elif cmd_type == "get_status":
            return {
                "status": "ok",
                "servo_angles": self.servos.get_current_angles(),
                "processing": self.is_processing,
                "vision_context": self.vision_context,
                "current_emotion": self.current_emotion,
                "modes": self.get_modes(),
                "memory": self.memory.short_term.get_summary() if self.memory else {},
                "ltm_enabled": self.memory.long_term.enabled if self.memory else False
            }

        elif cmd_type == "clear_history":
            self.llm.clear_history()
            if self.memory:
                self.memory.clear()
            self.current_emotion = "calm"
            return {"status": "ok", "message": "История и память очищены"}

        elif cmd_type == "set_mode":
            module = command.get("module")
            mode = command.get("mode")
            if module == "stt":
                self.set_stt_mode(mode)
            elif module == "tts":
                self.set_tts_mode(mode)
            elif module == "llm":
                self.set_llm_mode(mode)
            return {"status": "ok", "module": module, "mode": mode, "modes": self.get_modes()}

        else:
            return {"status": "error", "message": f"Неизвестная команда: {cmd_type}"}

    def shutdown(self):
        logger.info("Завершение работы RobotBrain...")
        if self.memory:
            self.memory.save_profile()
        self.vision.release()
        self._cv_executor.shutdown(wait=False)
        self._speech_executor.shutdown(wait=False)