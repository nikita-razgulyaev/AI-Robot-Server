# 🦉 Robot AI Server - Сорен

Локальный сервер ИИ для роботизированной совы Сорен на ESP32-S3 с голосовым управлением, компьютерным зрением и управлением сервоприводами.

## Архитектура

```
┌─────────────────────────────────────────────┐
│              НОУТБУК (Сервер)                │
│  ┌─────────┐ ┌─────────┐ ┌─────────────────┐ │
│  │ Whisper │ │ Silero  │ │  llama.cpp      │ │
│  │  (STT)  │ │  (TTS)  │ │  (Qwen 7B)      │ │
│  └────┬────┘ └────┬────┘ └────────┬──────┘ │
│       └─────────────┴────────────────┘       │
│              WebSocket Server                 │
│                   ↑↓ Wi-Fi                    │
└─────────────────────────────────────────────┘
                   ↑↓
┌─────────────────────────────────────────────┐
│           ESP32-S3 N16R8                    │
│  ┌────────┐ ┌────────┐ ┌────────────────┐  │
│  │INMP441 │ │MAX98357│ │   OV2640        │  │
│  │(микр.) │ │(динам.)│ │   (камера)      │  │
│  └────────┘ └────────┘ └────────────────┘  │
│  ┌────────────────────────────────────────┐  │
│  │ PCA9685 + 16 SG90 + 2 SG90 (GPIO)      │  │
│  │ LED-глаза (WS2812 / OLED)              │  │
│  └────────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

## Система знаний Сорена (3 уровня)

```
character/
├── Soren.txt              # [УРОВЕНЬ 1] System Prompt (личность, правила, few-shot)
├── Soren_rag_chunks.jsonl # [УРОВЕНЬ 2] RAG-чанки (биография, отношения, мир, философия)
└── Soren_emotions.json    # [УРОВЕНЬ 3] Эмоции → позы сервоприводов + LED
```

### Уровень 1: System Prompt (~1200 токенов)
- Личность Сорена (амбарная сова, Главный Страж)
- Правила речи (архаика, метафоры, паузы)
- Запреты (сленг, эмодзи, "я ИИ")
- 20 ключевых триггеров (Клудд, Эзилриб, Гильфи, Пеллиппер...)
- 3 few-shot примера

### Уровень 2: RAG (55 чанков)
- **Биография** (49 чанков): детство, Сант-Эголиус, путешествие, обучение, битвы, финал
- **Отношения** (20 чанков): Клудд, Гильфи, Эзилриб, Пеллиппер, Отулисса...
- **Мир** (7 чанков): Га'Хул, Сант-Эголиус, Чистые, крупинки, Клювы...
- **Философия** (7 чанков): серебряная душа, надежда, сила, любовь...
- **Техника** (2 чанка): аналогии для сервоприводов

Поиск: keyword matching по тегам + эвристики.

### Уровень 3: Эмоции (7 состояний)
- `calm` — спокойствие (поза 90°, глаза soft_white_low)
- `sad` — печаль (крылья опущены, глаза dim_blue_pulse)
- `angry` — гнев (крылья расправлены, глаза bright_orange_flicker)
- `loving` — нежность (крылья расслаблены, глаза warm_yellow_glow)
- `determined` — решимость (крылья в полётной позе, глаза steady_white_bright)
- `surprised` — удивление (голова вверх, глаза bright_white_flash)
- `tired` — усталость (крылья опущены, глаза dim_amber_slow)

## Установка

### 1. Требования
- **Python 3.10 - 3.11**
- **RAM**: 8GB минимум, 16GB+ рекомендуется
- **GPU**: опционально

### 2. Клонирование и зависимости
```bash
git clone <repo-url> robot_server
cd robot_server
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 3. Загрузка моделей
```bash
python download_models.py
```

| Модель | Размер | Описание |
|--------|--------|----------|
| **Whisper** | ~500MB | faster-whisper-small |
| **Silero TTS** | ~120MB | v4_ru.pt |
| **LLM (Qwen2.5)** | ~4.5GB | qwen2.5-7b-instruct-q4_k_m.gguf |
| **YOLO** | ~6MB | yolov8n.pt |

### 4. Настройка
```bash
cp .env.example .env
# Отредактируй .env под свою систему
```

### 5. Запуск
```bash
python websocket_server.py
```

Сервер будет доступен:
- WebSocket: `ws://localhost:8765/ws`
- HTTP API: `http://localhost:8765/status`
- Панель управления: `http://localhost:8765/panel`

## Протокол обмена с ESP32

### ESP32 → Сервер (JSON)
```json
{"type": "ping"}
{"type": "get_status"}
{"type": "servo", "id": 0, "angle": 90}
{"type": "servo_multi", "angles": [90,90,...,90]}
{"type": "animation", "name": "wave"}
{"type": "text", "text": "Привет, Сорен!"}
{"type": "clear_history"}
```

### ESP32 → Сервер (Binary)
```
[AUDI][PCM 16-bit mono 16kHz]  — аудио для распознавания
[VIDE][JPEG кадр]               — видео для анализа
```

### Сервер → ESP32 (JSON)
```json
{
  "type": "response",
  "user_text": "...",
  "robot_text": "...",
  "action": null,
  "emotion": "calm",
  "servo_angles": [90, 90, ...],
  "eye_led": "soft_white_low"
}
```

### Сервер → ESP32 (Binary)
```
[AUDI][PCM 16-bit mono 48000Hz]  — синтезированная речь
```

## Тестирование

```bash
# Тест без ESP32 (только текстовые команды)
python test_server.py

# Полное тестирование всех модулей
python test_all.py

# Проверка статуса
curl http://localhost:8765/status
```

## Структура проекта

```
robot_server/
├── websocket_server.py      # Главный WebSocket сервер
├── requirements.txt         # Python зависимости
├── .env.example             # Шаблон конфигурации
├── download_models.py       # Скрипт загрузки моделей
├── test_server.py           # Тестовый клиент
├── test_all.py              # Полное тестирование
├── README.md                # Документация
├── character/               # === СИСТЕМА ЗНАНИЙ СОРЕНА ===
│   ├── Soren.txt            # [УР 1] System prompt
│   ├── Soren_rag_chunks.jsonl # [УР 2] RAG чанки
│   └── Soren_emotions.json  # [УР 3] Эмоции + позы
├── config/
│   ├── __init__.py
│   └── settings.py          # Настройки
├── modules/
│   ├── __init__.py
│   ├── stt.py               # Speech-to-Text (Whisper)
│   ├── tts.py               # Text-to-Speech (Silero)
│   ├── llm.py               # LLM + RAG + Persona (Qwen)
│   ├── vision.py            # Computer Vision (YOLO + MediaPipe)
│   ├── audio_buffer.py      # Аудио буфер с VAD
│   ├── servo_controller.py  # Контроллер сервоприводов + позы
│   └── robot_brain.py       # Главный оркестратор + эмоции
├── models/                  # Модели ИИ
│   ├── silero_v4_ru.pt
│   └── *.gguf
├── audio_cache/             # Кэш TTS
└── esp32_firmware/
    ├── esp32_firmware.ino   # Прошивка ESP32-S3
    └── platformio.ini       # Конфигурация PlatformIO
```

## Кастомизация Сорена

### Добавить новый факт
1. Открой `character/Soren_rag_chunks.jsonl`
2. Добавь строку:
```json
{"id":"soren_custom_001","text":"Твой факт...","tags":["тег1","тег2"],"weight":"высокий","source":"твой_источник"}
```
3. Перезапусти сервер

### Изменить позу эмоции
1. Открой `character/Soren_emotions.json`
2. Измени `servo_pose` для нужной эмоции
3. Перезапусти сервер

### Изменить личность
1. Открой `character/Soren.txt`
2. Отредактируй system prompt
3. Перезапусти сервер

## Лицензия

MIT