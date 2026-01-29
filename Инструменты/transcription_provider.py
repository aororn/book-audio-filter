"""
Абстракция интерфейса транскрибации v1.0

Позволяет подключать альтернативных провайдеров транскрибации
вместо Яндекс SpeechKit (например, Whisper, Google Speech-to-Text).

Использование:
    from transcription_provider import YandexProvider, get_provider

    # Яндекс (по умолчанию)
    provider = get_provider('yandex')
    result = provider.transcribe('audio.ogg')

    # Или через абстрактный интерфейс
    provider = get_provider('mock')  # для тестов
    result = provider.transcribe('audio.ogg')
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List


class TranscriptionResult:
    """Унифицированный результат транскрибации."""

    def __init__(
        self,
        text: str,
        words: List[Dict[str, Any]],
        raw_response: Optional[Dict] = None,
        provider: str = 'unknown',
    ):
        self.text = text
        self.words = words  # [{word: str, start_time: float, end_time: float}, ...]
        self.raw_response = raw_response or {}
        self.provider = provider

    @property
    def word_count(self) -> int:
        return len(self.words)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'words': self.words,
            'provider': self.provider,
            'word_count': self.word_count,
        }


class TranscriptionProvider(ABC):
    """Абстрактный интерфейс провайдера транскрибации."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Имя провайдера."""
        ...

    @abstractmethod
    def transcribe(
        self,
        audio_path: str,
        language: str = 'ru-RU',
        **kwargs,
    ) -> TranscriptionResult:
        """
        Транскрибирует аудиофайл.

        Args:
            audio_path: путь к аудиофайлу
            language: код языка
            **kwargs: дополнительные параметры провайдера

        Returns:
            TranscriptionResult с текстом и словами
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Проверяет доступность провайдера (ключи, зависимости)."""
        ...


class YandexProvider(TranscriptionProvider):
    """Провайдер Яндекс SpeechKit."""

    @property
    def name(self) -> str:
        return 'yandex'

    def __init__(self, api_key: Optional[str] = None, folder_id: Optional[str] = None):
        self._api_key = api_key
        self._folder_id = folder_id

    def is_available(self) -> bool:
        try:
            from transcribe import get_api_key
            return get_api_key() is not None
        except ImportError:
            return False

    def transcribe(
        self,
        audio_path: str,
        language: str = 'ru-RU',
        **kwargs,
    ) -> TranscriptionResult:
        from transcribe import transcribe as yandex_transcribe

        result = yandex_transcribe(
            audio_path,
            api_key=self._api_key,
            language=language,
            folder_id=self._folder_id,
            **kwargs,
        )

        # Преобразуем в унифицированный формат
        text_parts = []
        words = []

        for chunk in result.get('chunks', []):
            for alt in chunk.get('alternatives', []):
                text_parts.append(alt.get('text', ''))
                for w in alt.get('words', []):
                    words.append({
                        'word': w.get('word', ''),
                        'start_time': float(w.get('startTime', '0s').rstrip('s')),
                        'end_time': float(w.get('endTime', '0s').rstrip('s')),
                        'confidence': w.get('confidence', 1.0),
                    })

        return TranscriptionResult(
            text=' '.join(text_parts),
            words=words,
            raw_response=result,
            provider=self.name,
        )


class MockProvider(TranscriptionProvider):
    """Мок-провайдер для тестирования (без внешних API)."""

    @property
    def name(self) -> str:
        return 'mock'

    def __init__(self, mock_text: str = '', mock_words: Optional[List[Dict]] = None):
        self._text = mock_text
        self._words = mock_words or []

    def is_available(self) -> bool:
        return True

    def transcribe(
        self,
        audio_path: str,
        language: str = 'ru-RU',
        **kwargs,
    ) -> TranscriptionResult:
        return TranscriptionResult(
            text=self._text,
            words=self._words,
            raw_response={'mock': True},
            provider=self.name,
        )


# Реестр провайдеров
_PROVIDERS: Dict[str, type] = {
    'yandex': YandexProvider,
    'mock': MockProvider,
}


def register_provider(name: str, provider_class: type) -> None:
    """Регистрирует нового провайдера."""
    if not issubclass(provider_class, TranscriptionProvider):
        raise TypeError(f"{provider_class} должен наследовать TranscriptionProvider")
    _PROVIDERS[name] = provider_class


def get_provider(name: str = 'yandex', **kwargs) -> TranscriptionProvider:
    """Возвращает экземпляр провайдера по имени."""
    if name not in _PROVIDERS:
        available = ', '.join(_PROVIDERS.keys())
        raise ValueError(f"Провайдер '{name}' не найден. Доступные: {available}")
    return _PROVIDERS[name](**kwargs)


def list_providers() -> List[str]:
    """Возвращает список доступных провайдеров."""
    return list(_PROVIDERS.keys())
