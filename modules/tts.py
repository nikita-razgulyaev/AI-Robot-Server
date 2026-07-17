"""TTS модуль - синтез речи через Silero (PyTorch)"""
import io
import wave
import logging
import torch
import torchaudio
from pathlib import Path
from config.settings import AUDIO_CACHE_DIR, SAMPLE_RATE

logger = logging.getLogger(__name__)


class TTSEngine:
    """Движок синтеза речи на Silero"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Silero TTS device: {self.device}")

        self.model = None
        self.symbols = None
        self.sample_rate = 48000  # Silero использует 48kHz

        self._load_model()

    def _load_model(self):
        """Загружает Silero TTS модель"""
        try:
            # Silero русский голос
            model_url = 'https://models.silero.ai/models/tts/ru/v4_ru.pt'
            model_path = Path("./models/silero_v4_ru.pt")
            model_path.parent.mkdir(parents=True, exist_ok=True)

            if not model_path.exists():
                logger.info(f"Скачивание Silero TTS модели...")
                torch.hub.download_url_to_file(model_url, model_path)
                logger.info("Silero модель скачана")

            self.model = torch.package.PackageImporter(model_path).load_pickle(
                "tts_models", "model"
            )
            self.model.to(self.device)

            # Пробуем получить sample_rate
            if hasattr(self.model, 'sample_rate'):
                self.sample_rate = self.model.sample_rate

            logger.info(f"Silero TTS загружен, sample_rate={self.sample_rate}")

        except Exception as e:
            logger.error(f"Ошибка загрузки Silero: {e}")
            logger.info("Попробуй скачать вручную: https://models.silero.ai/models/tts/ru/v4_ru.pt")
            self.model = None

    def synthesize(self, text: str, speaker: str = 'xenia') -> bytes:
        """
        Синтезирует речь из текста

        Args:
            text: Текст для озвучки
            speaker: Голос (xenia, baya, kseniya, aidar)

        Returns:
            PCM аудио bytes (16-bit, mono, 48kHz)
        """
        try:
            if self.model is None:
                logger.error("Silero модель не загружена")
                return b""

            # Синтез
            audio = self.model.apply_tts(
                text=text,
                speaker=speaker,
                sample_rate=self.sample_rate
            )

            # Конвертируем в PCM 16-bit
            audio = audio.unsqueeze(0)  # [1, samples]

            # Нормализация и конвертация в int16
            audio_np = audio.squeeze().cpu().numpy()
            audio_np = audio_np / max(abs(audio_np)) * 32767
            audio_int16 = audio_np.astype('int16')

            return audio_int16.tobytes()

        except Exception as e:
            logger.error(f"Ошибка TTS: {e}")
            return b""

    def synthesize_to_wav(self, text: str, output_path: Path = None, speaker: str = 'xenia') -> Path:
        """Синтезирует в WAV файл"""
        if output_path is None:
            output_path = AUDIO_CACHE_DIR / f"tts_{hash(text)}.wav"

        pcm_data = self.synthesize(text, speaker)
        if not pcm_data:
            return None

        with wave.open(str(output_path), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm_data)

        return output_path

    def get_available_speakers(self) -> list:
        """Возвращает список доступных голосов"""
        return ['xenia', 'baya', 'kseniya', 'aidar']
