"""Speech-to-Text модуль - Whisper через faster-whisper + Fuzzy matching"""
import logging
import tempfile
import os
from pathlib import Path
from config.settings import WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE

logger = logging.getLogger(__name__)


class STTEngine:
    """Распознавание речи через faster-whisper с адаптивным fuzzy matching"""

    def __init__(self):
        self.model = None
        self._load_model()
        # Lazy import fuzzy matcher
        try:
            from modules.fuzzy_matcher import correct_speech_text
            self.correct_text = correct_speech_text
            self.fuzzy_available = True
            logger.info("Fuzzy matching загружен")
        except ImportError:
            self.correct_text = lambda x: x
            self.fuzzy_available = False
            logger.warning("Fuzzy matcher не найден, работаем без адаптивной коррекции")

    def _load_model(self):
        """Загружает Whisper модель"""
        logger.info(f"Загрузка Whisper: {WHISPER_MODEL_SIZE} ({WHISPER_DEVICE})")
        try:
            from faster_whisper import WhisperModel

            self.model = WhisperModel(
                WHISPER_MODEL_SIZE,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE
            )
            logger.info("Whisper загружен")
        except Exception as e:
            logger.error(f"Ошибка загрузки Whisper: {e}")
            self.model = None

    def transcribe(self, audio_bytes: bytes) -> dict:
        """
        Распознаёт речь из PCM байтов с fuzzy correction

        Args:
            audio_bytes: PCM 16-bit mono 16kHz

        Returns:
            {"success": bool, "text": str, "corrected_text": str, "error": str}
        """
        if self.model is None:
            return {"success": False, "text": "", "corrected_text": "", "error": "Модель не загружена"}

        try:
            # Сохраняем во временный WAV
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                import wave
                with wave.open(tmp.name, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(audio_bytes)
                tmp_path = tmp.name

            result = self.transcribe_from_file(tmp_path)
            os.unlink(tmp_path)
            return result

        except Exception as e:
            logger.error(f"Ошибка STT: {e}")
            return {"success": False, "text": "", "corrected_text": "", "error": str(e)}

    def transcribe_from_file(self, audio_path: str) -> dict:
        """
        Распознаёт речь из WAV файла с fuzzy correction

        Args:
            audio_path: путь к WAV файлу (16-bit mono 16kHz)

        Returns:
            {"success": bool, "text": str, "corrected_text": str, "error": str}
        """
        if self.model is None:
            return {"success": False, "text": "", "corrected_text": "", "error": "Модель не загружена"}

        try:
            segments, info = self.model.transcribe(audio_path, language="ru", beam_size=5)
            text = " ".join([segment.text for segment in segments]).strip()

            if text:
                # Применяем fuzzy correction
                corrected_text = self.correct_text(text) if self.fuzzy_available else text

                if corrected_text != text:
                    logger.info(f"🎤 STT: '{text}' → '{corrected_text}'")
                else:
                    logger.info(f"🎤 STT: '{text}'")

                return {
                    "success": True,
                    "text": text,
                    "corrected_text": corrected_text,
                    "error": None
                }
            else:
                return {"success": False, "text": "", "corrected_text": "", "error": "Речь не распознана"}

        except Exception as e:
            logger.error(f"Ошибка STT из файла: {e}")
            return {"success": False, "text": "", "corrected_text": "", "error": str(e)}
