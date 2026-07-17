# 🦉 Robot AI Server - Сорен v3.0

Локальный сервер ИИ для роботизированной совы Сорен на ESP32-S3 с голосовым управлением, компьютерным зрением и управлением сервоприводами.

## ✨ Новое в v3.0 — Переключение Local / Cloud

Каждый AI модуль (STT, LLM, TTS) теперь имеет **отдельную кнопку** переключения между локальным и облачным режимом!

| Модуль | Локально | Облачно |
|--------|----------|---------|
| **STT** 🎤 | faster-whisper (~500MB) | OpenAI Whisper API |
| **LLM** 🧠 | Qwen 7B GGUF (~4.5GB) | OpenAI GPT-4o-mini |
| **TTS** 🔊 | Silero (~120MB) | OpenAI TTS API |

## 🔑 Ключи API

Для работы **облачных режимов** нужен ключ OpenAI API:

1. Зарегистрируйся: https://platform.openai.com/signup
2. Получи ключ: https://platform.openai.com/api-keys
3. Добавь в `.env`:
```bash
OPENAI_API_KEY=sk-your-api-key-here
```

⚠️ **Без ключа облачные режимы работать НЕ будут!**

## 💰 Стоимость облачных API (примерно)

| Модуль | Стоимость | Пример |
|--------|-----------|--------|
| STT (Whisper) | $0.006 / минута | 10 мин разговора = $0.06 |
| LLM (GPT-4o-mini) | $0.15 / 1M токенов | Средний диалог ~$0.01 |
| TTS | $15 / 1M символов | Фраза ~$0.001 |

## 🚀 Быстрый старт

### 1. Установка
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 2. Настройка
```bash
cp .env.example .env
# Отредактируй .env — укажи OPENAI_API_KEY если хочешь облако
```

### 3. Загрузка моделей (для локальных режимов)
```bash
python download_models.py
```

### 4. Запуск
```bash
python websocket_server.py
```

Открой панель: http://localhost:8765/panel

## 🎛️ Панель управления

На панели теперь есть:
- **3 блока переключения AI** — STT, LLM, TTS (Local / Cloud)
- **2 переключателя аудио** — Ввод (микрофон) и Вывод (динамик)
- **Голосовой чат** — кнопка микрофона
- **Текстовый чат** — поле ввода
- **Анимации** — wave, nod, shake_head, idle
- **18 сервоприводов** — индивидуальное управление

## 📁 Структура проекта

```
robot_server/
├── websocket_server.py      # Главный WebSocket сервер + панель
├── requirements.txt         # Зависимости (+ requests для облака)
├── .env.example             # Шаблон конфигурации (с OpenAI ключом)
├── modules/
│   ├── stt.py               # STT: local (Whisper) + cloud (OpenAI)
│   ├── tts.py               # TTS: local (Silero) + cloud (OpenAI)
│   ├── llm.py               # LLM: local (llama.cpp) + cloud (OpenAI)
│   ├── robot_brain.py       # Оркестратор + переключение режимов
│   └── ...
└── character/
    ├── Soren.txt            # System prompt
    ├── Soren_rag_chunks.jsonl # RAG
    └── Soren_emotions.json  # Эмоции + позы
```

## ⚙️ Переменные .env

```bash
# Режимы AI (local / cloud)
STT_MODE=local
TTS_MODE=local
LLM_MODE=local

# OpenAI API (нужен для облачных режимов)
OPENAI_API_KEY=sk-...
OPENAI_LLM_MODEL=gpt-4o-mini
OPENAI_TTS_MODEL=tts-1
OPENAI_TTS_VOICE=alloy

# Локальные модели
WHISPER_MODEL_SIZE=small
LLM_MODEL_PATH=./models/qwen2.5-7b-instruct-q4_k_m.gguf
SILERO_SPEAKER=xenia
```

## 🔌 API Endpoints

| Endpoint | Метод | Описание |
|----------|-------|----------|
| `/` | GET | Статус сервера |
| `/status` | GET | Полный статус + режимы AI |
| `/audio_mode` | POST | Переключение аудио (input/output) |
| `/ai_mode` | POST | Переключение AI модуля (stt/tts/llm) |
| `/speak` | POST | Текст → LLM → TTS → аудио |
| `/voice` | POST | Аудио → STT → LLM → TTS → аудио |
| `/panel` | GET | HTML панель управления |
| `/ws` | WS | WebSocket для ESP32 |

## 🧪 Тестирование

```bash
# Тест без ESP32
python test_server.py

# Полное тестирование
python test_all.py
```

## Лицензия

MIT
