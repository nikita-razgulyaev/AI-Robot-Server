"""LLM модуль - Сорен через llama-cpp-python с RAG и persona"""
import logging
import json
import re
from pathlib import Path
from typing import List, Dict, Optional
from config.settings import (
    LLM_MODEL_PATH, LLM_N_CTX, LLM_N_THREADS, LLM_TEMPERATURE,
    CHARACTER_DIR, RAG_TOP_K
)

logger = logging.getLogger(__name__)


class RAGIndex:
    """Простой keyword-based RAG по чанкам"""

    def __init__(self, chunks_path: Path):
        self.chunks: List[Dict] = []
        self._load(chunks_path)

    def _load(self, path: Path):
        if not path.exists():
            logger.warning(f"RAG файл не найден: {path}")
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.chunks.append(json.loads(line))
            logger.info(f"RAG загружено: {len(self.chunks)} чанков")
        except Exception as e:
            logger.error(f"Ошибка загрузки RAG: {e}")

    def search(self, query: str, top_k: int = 3) -> List[str]:
        """Простой keyword matching по тегам и тексту"""
        query_lower = query.lower()
        # Извлекаем ключевые слова из запроса
        keywords = set(re.findall(r'[\w\-]+', query_lower))

        scored = []
        for chunk in self.chunks:
            score = 0
            text_lower = chunk["text"].lower()
            tags = [t.lower() for t in chunk.get("tags", [])]

            # Точное совпадение тегов
            for kw in keywords:
                if kw in tags:
                    score += 10
                if kw in text_lower:
                    score += 3

            # Специальные триггеры
            triggers = {
                "клудд": ["клудд", "брат", "металлический", "предательство"],
                "гильфи": ["гильфи", "друг", "подруга", "сычик"],
                "эзилриб": ["эзилриб", "учитель", "наставник"],
                "пеллиппер": ["пеллиппер", "любовь", "сердце"],
                "сант": ["сант-эголиус", "плен", "эголиус"],
                "древо": ["древо", "га'хул", "стражи"],
                "чистые": ["чистые", "крупинки", "враг"],
            }
            for trigger_word, related in triggers.items():
                if trigger_word in query_lower:
                    for rel in related:
                        if rel in text_lower or rel in tags:
                            score += 5

            if score > 0:
                scored.append((score, chunk))

        # Сортируем по score, берём top_k
        scored.sort(key=lambda x: x[0], reverse=True)
        return [chunk["text"] for _, chunk in scored[:top_k]]


class LLMEngine:
    """Движок языковой модели Сорена"""

    def __init__(self):
        self.model = None
        self.conversation_history = []
        self.system_prompt = ""
        self.rag = None
        self.emotion_keywords = {}

        self._load_system_prompt()
        self._load_rag()
        self._load_emotion_keywords()
        self._load_model()

    def _load_system_prompt(self):
        """Загружает system prompt из character/Soren.txt"""
        prompt_path = CHARACTER_DIR / "Soren.txt"
        if prompt_path.exists():
            with open(prompt_path, 'r', encoding='utf-8') as f:
                self.system_prompt = f.read()
            logger.info(f"System prompt загружен: {len(self.system_prompt)} chars")
        else:
            logger.warning(f"System prompt не найден: {prompt_path}")
            # Fallback
            self.system_prompt = "Ты — Сорен, амбарная сова, Главный Страж. Отвечай мудро и сдержанно."

    def _load_rag(self):
        """Загружает RAG индекс"""
        rag_path = CHARACTER_DIR / "Soren_rag_chunks.jsonl"
        self.rag = RAGIndex(rag_path)

    def _load_emotion_keywords(self):
        """Загружает ключевые слова для определения эмоций"""
        emotions_path = CHARACTER_DIR / "Soren_emotions.json"
        if emotions_path.exists():
            try:
                with open(emotions_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for emotion_name, emotion_data in data.get("emotions", {}).items():
                    self.emotion_keywords[emotion_name] = emotion_data.get("triggered_by", [])
                logger.info(f"Эмоции загружены: {list(self.emotion_keywords.keys())}")
            except Exception as e:
                logger.error(f"Ошибка загрузки эмоций: {e}")

    def _load_model(self):
        """Загружает GGUF модель"""
        logger.info(f"Загрузка LLM: {LLM_MODEL_PATH}")

        if not LLM_MODEL_PATH.exists():
            logger.error(f"Модель LLM не найдена: {LLM_MODEL_PATH}")
            return

        try:
            from llama_cpp import Llama

            self.model = Llama(
                model_path=str(LLM_MODEL_PATH),
                n_ctx=LLM_N_CTX,
                n_threads=LLM_N_THREADS,
                verbose=False
            )
            logger.info("LLM загружена")

        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            self.model = None

    def _detect_emotion(self, text: str) -> str:
        """Определяет эмоцию по тексту ответа Сорена"""
        text_lower = text.lower()

        # Проверяем триггеры
        emotion_scores = {}
        for emotion, triggers in self.emotion_keywords.items():
            score = 0
            for trigger in triggers:
                if trigger.lower() in text_lower:
                    score += 1
            if score > 0:
                emotion_scores[emotion] = score

        if emotion_scores:
            return max(emotion_scores, key=emotion_scores.get)

        # Эвристики по тексту
        if any(w in text_lower for w in ["клудд", "пепел", "потерял", "погиб", "тень", "скорбь"]):
            return "sad"
        if any(w in text_lower for w in ["огонь", "коготь", "буря", "гнев", "убить", "враг"]):
            return "angry"
        if any(w in text_lower for w in ["пеллиппер", "любовь", "тепло", "гнездо", "доверие", "сердце"]):
            return "loving"
        if any(w in text_lower for w in ["должен", "вперёд", "защищать", "битва", "миссия"]):
            return "determined"
        if any(w in text_lower for w in ["что...", "не может быть", "удивлён", "вспышка"]):
            return "surprised"
        if any(w in text_lower for w in ["устал", "отдохнуть", "пепел", "закат", "позже"]):
            return "tired"

        return "calm"

    def _build_prompt(self, user_message: str, vision_context: str = "") -> str:
        """Собирает полный prompt: system + RAG + history + user"""
        # 1. RAG поиск
        rag_chunks = self.rag.search(user_message, top_k=RAG_TOP_K) if self.rag else []
        rag_text = "\n".join(rag_chunks) if rag_chunks else ""

        # 2. Формируем messages для Qwen chat template
        messages = []

        # System prompt
        system_content = self.system_prompt
        if rag_text:
            system_content += f"\n\nРелевантные воспоминания для контекста:\n{rag_text}"
        if vision_context:
            system_content += f"\n\n[Ты видишь: {vision_context}]"

        messages.append({"role": "system", "content": system_content})

        # История (последние 5 пар сообщений)
        for msg in self.conversation_history[-10:]:
            messages.append(msg)

        # Текущий запрос
        messages.append({"role": "user", "content": user_message})

        return messages

    def generate(self, user_message: str, vision_context: str = "") -> dict:
        """
        Генерирует ответ от Сорена

        Returns:
            {"text": str, "action": str or None, "emotion": str}
        """
        if self.model is None:
            return {
                "text": "Модель LLM не загружена. Проверь логи и настройки.",
                "action": None,
                "emotion": "calm"
            }

        try:
            messages = self._build_prompt(user_message, vision_context)

            # Генерация через Qwen chat template
            output = self.model.create_chat_completion(
                messages=messages,
                temperature=LLM_TEMPERATURE,
                max_tokens=256,
                stop=["</s>", "Пользователь:", "User:"],
            )

            response_text = output["choices"][0]["message"]["content"].strip()

            # Очистка от артефактов
            response_text = re.sub(r'^(Сорен:|Assistant:|AI:)', '', response_text).strip()
            response_text = re.sub(r'\*.*?\*', '', response_text)  # убираем *курсив* если есть

            # Извлекаем действие (если LLM сам добавил)
            action = None
            action_match = re.search(r'\[ACTION:(\w+)\]', response_text)
            if action_match:
                action = action_match.group(1)
                response_text = re.sub(r'\[ACTION:\w+\]', '', response_text).strip()

            # Определяем эмоцию
            emotion = self._detect_emotion(response_text)

            # Сохраняем в историю
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": response_text})

            # Ограничиваем историю
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]

            return {
                "text": response_text,
                "action": action,
                "emotion": emotion
            }

        except Exception as e:
            logger.error(f"Ошибка LLM: {e}")
            return {
                "text": "*(пауза)* ... Прости, друг. Мысли улетели далеко, словно перо на ветру. Повтори, пожалуйста.",
                "action": None,
                "emotion": "calm"
            }

    def clear_history(self):
        """Очищает историю диалога"""
        self.conversation_history = []
        logger.info("История диалога очищена")