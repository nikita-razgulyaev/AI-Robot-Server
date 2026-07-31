"""Конфигурация сервера робота Сорена — v3.5 (YAML + память + fine-tuning + Vector RAG)"""
import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent

# === Загрузка YAML конфига ===
CONFIG_PATH = BASE_DIR / "config.yaml"
_config = {}
if CONFIG_PATH.exists():
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        _config = yaml.safe_load(f)

# === Character / Knowledge System ===
CHARACTER_DIR = BASE_DIR / "character"
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))

# === VECTOR RAG ===
QDRANT_PATH = _config.get("qdrant", {}).get("path", "./qdrant_storage")
RAG_ENCODER_MODEL = _config.get("qdrant", {}).get("encoder_model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# === ПАМЯТЬ ===
MEMORY_DIR = BASE_DIR / "memory"
MEMORY_DIR.mkdir(parents=True, exist_ok=True)
SHORT_TERM_MEMORY_MAX = _config.get("memory", {}).get("short_term_max", 10)
LONG_TERM_MEMORY_ENABLED = _config.get("memory", {}).get("long_term_enabled", True)
QDRANT_HOST = _config.get("memory", {}).get("qdrant_host", "localhost")
QDRANT_PORT = _config.get("memory", {}).get("qdrant_port", 6333)
QDRANT_COLLECTIONS = _config.get("memory", {}).get("collections", {
    "emotional_moments": "soren_emotional_moments",
    "dialogue_turns": "soren_dialogue_turns",
    "facts": "soren_facts",
    "rag_chunks": "soren_rag_chunks",
})

# === FINE-TUNING / LoRA ===
LORA_ENABLED = _config.get("lora", {}).get("enabled", False)
LORA_PATH = BASE_DIR / _config.get("lora", {}).get("path", "models/soren_lora_v1.gguf")
LORA_SCALE = float(_config.get("lora", {}).get("scale", 1.0))
LORA_RANK = int(_config.get("lora", {}).get("rank", 16))
LORA_ALPHA = int(_config.get("lora", {}).get("alpha", 32))
DATASET_DIR = BASE_DIR / "datasets"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# Server
SERVER_HOST = _config.get("server", {}).get("host", "0.0.0.0")
SERVER_PORT = _config.get("server", {}).get("port", 8765)

# Paths
MODELS_DIR = BASE_DIR / "models"
AUDIO_CACHE_DIR = BASE_DIR / "audio_cache"
AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# WiFi (из .env)
WIFI_SSID = os.getenv("WIFI_SSID", "")
WIFI_PASSWORD = os.getenv("WIFI_PASSWORD", "")

# ========== STT ==========
STT_MODE = _config.get("stt", {}).get("mode", "local")
WHISPER_MODEL_SIZE = _config.get("stt", {}).get("whisper_model", "small")
WHISPER_DEVICE = _config.get("stt", {}).get("device", "cpu")
WHISPER_COMPUTE_TYPE = _config.get("stt", {}).get("compute_type", "int8")

# ========== TTS ==========
TTS_MODE = _config.get("tts", {}).get("mode", "local")
SILERO_SPEAKER = _config.get("tts", {}).get("silero_speaker", "aidar")
SILERO_PITCH_SEMITONES = float(_config.get("tts", {}).get("silero_pitch_semitones", 1.5))
SILERO_RATE = float(_config.get("tts", {}).get("silero_rate", 1.05))
FREETTS_VOICE = _config.get("tts", {}).get("edge_voice", "ru-RU-DmitryNeural")
FREETTS_SPEED = float(_config.get("tts", {}).get("edge_speed", 1.0))
FREETTS_PITCH = _config.get("tts", {}).get("edge_pitch", "+15Hz")

# ========== LLM ==========
LLM_MODE = _config.get("llm", {}).get("mode", "local")
LLM_MODEL_PATH = MODELS_DIR / "qwen2.5-7b-instruct-Q4_K_M.gguf"
LLM_N_CTX = int(_config.get("llm", {}).get("n_ctx", 4096))
LLM_N_THREADS = int(_config.get("llm", {}).get("n_threads", 4))
LLM_TEMPERATURE = float(_config.get("llm", {}).get("temperature", 0.6))
LLM_REPEAT_PENALTY = float(_config.get("llm", {}).get("repeat_penalty", 1.15))
GITHUB_MODELS_KEY = os.getenv("GITHUB_MODELS_KEY", _config.get("llm", {}).get("cloud_api_key", ""))
GITHUB_MODELS_NAME = _config.get("llm", {}).get("cloud_model", "gpt-4o-mini")

# Vision
YOLO_MODEL = MODELS_DIR / "yolov8n.pt"
ENABLE_POSE_TRACKING = _config.get("vision", {}).get("pose_tracking", True)

# Audio
SAMPLE_RATE = int(_config.get("audio", {}).get("sample_rate", 16000))
CHUNK_DURATION_MS = int(_config.get("audio", {}).get("chunk_duration_ms", 30))
VAD_AGGRESSIVENESS = int(_config.get("audio", {}).get("vad_aggressiveness", 2))
SILENCE_TIMEOUT_MS = int(_config.get("audio", {}).get("silence_timeout_ms", 1500))

# Servo config
SERVO_CFG = _config.get("servos", {})
SERVO_CONFIG = {
    "pca9685_channels": SERVO_CFG.get("pca9685_channels", 16),
    "pca9685_address": SERVO_CFG.get("pca9685_address", 0x40),
    "pca9685_freq": SERVO_CFG.get("pca9685_freq", 50),
    "extra_servos_pins": SERVO_CFG.get("extra_pins", [17, 18]),
    "min_angle": SERVO_CFG.get("min_angle", 0),
    "max_angle": SERVO_CFG.get("max_angle", 180),
}

# Анимации
ANIMATIONS = {
    "wave": [
        {"time": 0, "servos": [90]*18},
        {"time": 200, "servos": [90, 45, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90]},
        {"time": 400, "servos": [90, 135, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90]},
        {"time": 600, "servos": [90, 45, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90]},
        {"time": 800, "servos": [90]*18},
    ],
    "nod": [
        {"time": 0, "servos": [90]*18},
        {"time": 300, "servos": [90, 90, 60, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90]},
        {"time": 600, "servos": [90, 90, 120, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90]},
        {"time": 900, "servos": [90]*18},
    ],
    "shake_head": [
        {"time": 0, "servos": [90]*18},
        {"time": 200, "servos": [90, 90, 90, 60, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90]},
        {"time": 400, "servos": [90, 90, 90, 120, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90]},
        {"time": 600, "servos": [90, 90, 90, 60, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90]},
        {"time": 800, "servos": [90]*18},
    ],
    "idle": [
        {"time": 0, "servos": [90]*18},
    ],
}