#!/usr/bin/env python3
"""
Тесты для config.py v5.0

Покрывает:
- Пути (PROJECT_DIR, DICTIONARIES_DIR, TOOLS_DIR, RESULTS_DIR и т.д.)
- FileNaming (конвенция именования, поиск файлов)
- SmartCompareConfig, GoldenFilterConfig, YandexCloudConfig
- Утилиты (format_duration, check_file_exists, cleanup_deprecated_files)
- LogConfig, setup_logging, get_logger

Запуск:
    pytest Тесты/test_config.py -v
"""

import sys
import tempfile
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'Инструменты'))

import pytest
from config import (
    # Пути
    PROJECT_DIR, DICTIONARIES_DIR, TOOLS_DIR, RESULTS_DIR, TEMP_DIR,
    TESTS_DIR, READER_DIR, TRANSCRIPTIONS_DIR, ORIGINAL_DIR,
    CHAPTERS_DIR, AUDIO_DIR,
    NAMES_DICT, PROTECTED_WORDS, READER_ERRORS, CONFIG_JSON,
    # Классы
    FileNaming, SmartCompareConfig, GoldenFilterConfig, YandexCloudConfig,
    LogConfig,
    # Утилиты
    format_duration, check_file_exists, ensure_dirs_exist,
    get_chapter_output_dir, cleanup_deprecated_files,
    # Логирование
    setup_logging, get_logger, cleanup_old_logs,
    # Версия
    VERSION, VERSION_DATE,
)


# =============================================================================
# ТЕСТЫ ВЕРСИИ
# =============================================================================

class TestVersion:
    def test_version(self):
        assert VERSION == '5.0.0'

    def test_version_date(self):
        assert VERSION_DATE == '2026-01-25'


# =============================================================================
# ТЕСТЫ ПУТЕЙ
# =============================================================================

class TestPaths:
    """Тесты корректности путей."""

    def test_project_dir_exists(self):
        assert PROJECT_DIR.exists()

    def test_tools_dir_exists(self):
        assert TOOLS_DIR.exists()

    def test_dictionaries_dir_is_path(self):
        assert isinstance(DICTIONARIES_DIR, Path)

    def test_results_dir_is_path(self):
        assert isinstance(RESULTS_DIR, Path)

    def test_temp_dir_is_path(self):
        assert isinstance(TEMP_DIR, Path)

    def test_tests_dir_is_path(self):
        assert isinstance(TESTS_DIR, Path)

    def test_names_dict_is_path(self):
        assert isinstance(NAMES_DICT, Path)
        assert NAMES_DICT.name == 'Словарь_имён_персонажей.txt'

    def test_protected_words_is_path(self):
        assert isinstance(PROTECTED_WORDS, Path)

    def test_reader_errors_is_path(self):
        assert isinstance(READER_ERRORS, Path)

    def test_paths_consistent(self):
        """Все основные папки — подпапки PROJECT_DIR."""
        assert str(DICTIONARIES_DIR).startswith(str(PROJECT_DIR))
        assert str(TOOLS_DIR).startswith(str(PROJECT_DIR))
        assert str(RESULTS_DIR).startswith(str(PROJECT_DIR))
        assert str(TEMP_DIR).startswith(str(PROJECT_DIR))


# =============================================================================
# ТЕСТЫ FileNaming
# =============================================================================

class TestFileNaming:
    """Тесты конвенции именования файлов."""

    def test_stages_non_empty(self):
        assert len(FileNaming.STAGES) > 0

    def test_known_stages(self):
        for stage in ('transcript', 'compared', 'filtered', 'docx'):
            assert stage in FileNaming.STAGES

    def test_build_filename(self):
        name = FileNaming.build_filename('01', 'transcript')
        assert name == '01_transcript.json'

    def test_build_filename_filtered(self):
        name = FileNaming.build_filename('03', 'filtered')
        assert name == '03_filtered.json'

    def test_build_filename_unknown_stage(self):
        with pytest.raises(ValueError):
            FileNaming.build_filename('01', 'nonexistent')

    def test_get_chapter_id_numeric(self):
        chapter_id = FileNaming.get_chapter_id(Path('01_transcript.json'))
        assert chapter_id == '01'

    def test_get_chapter_id_chapter_format(self):
        chapter_id = FileNaming.get_chapter_id(Path('Глава_1_transcript.json'))
        assert 'Глава' in chapter_id

    def test_get_output_path(self, tmp_path):
        path = FileNaming.get_output_path('01', 'compared', tmp_path)
        assert path == tmp_path / '01_compared.json'

    def test_is_deprecated(self):
        assert FileNaming.is_deprecated(Path('01_final.json')) is True
        assert FileNaming.is_deprecated(Path('01_filtered.json')) is False

    def test_normalize_filename(self):
        assert FileNaming.normalize_filename('file name.json') == 'file_name.json'
        assert FileNaming.normalize_filename('a__b.txt') == 'a_b.txt'

    def test_get_transcription_dir(self):
        trans_dir = FileNaming.get_transcription_dir('01')
        assert isinstance(trans_dir, Path)
        assert 'Глава1' in str(trans_dir)

    def test_get_transcription_path(self):
        path = FileNaming.get_transcription_path('01')
        assert path.name == '01_transcript.json'


# =============================================================================
# ТЕСТЫ КОНФИГУРАЦИИ АЛГОРИТМОВ
# =============================================================================

class TestAlgorithmConfigs:
    """Тесты настроек алгоритмов."""

    def test_smart_compare_threshold(self):
        assert 0 < SmartCompareConfig.THRESHOLD <= 1

    def test_golden_filter_levenshtein(self):
        assert isinstance(GoldenFilterConfig.LEVENSHTEIN_THRESHOLD, int)
        assert GoldenFilterConfig.LEVENSHTEIN_THRESHOLD > 0

    def test_golden_filter_flags(self):
        assert isinstance(GoldenFilterConfig.USE_LEMMATIZATION, bool)
        assert isinstance(GoldenFilterConfig.USE_HOMOPHONES, bool)
        assert isinstance(GoldenFilterConfig.USE_PHONETIC, bool)

    def test_yandex_cloud_endpoints(self):
        assert YandexCloudConfig.S3_ENDPOINT.startswith('https://')
        assert YandexCloudConfig.STT_SHORT_URL.startswith('https://')
        assert YandexCloudConfig.STT_LONG_URL.startswith('https://')

    def test_yandex_cloud_retry_config(self):
        assert YandexCloudConfig.MAX_RETRIES > 0
        assert YandexCloudConfig.BASE_DELAY > 0
        assert YandexCloudConfig.MAX_DELAY > YandexCloudConfig.BASE_DELAY

    def test_yandex_cloud_audio_encodings(self):
        encodings = YandexCloudConfig.AUDIO_ENCODINGS
        assert '.ogg' in encodings
        assert '.mp3' in encodings
        assert encodings['.ogg'] == 'OGG_OPUS'

    def test_get_audio_encoding(self):
        assert YandexCloudConfig.get_audio_encoding('test.ogg') == 'OGG_OPUS'
        assert YandexCloudConfig.get_audio_encoding('test.mp3') == 'MP3'
        assert YandexCloudConfig.get_audio_encoding('test.wav') == 'LINEAR16_PCM'

    def test_get_api_key_returns_optional(self):
        """get_api_key возвращает строку или None."""
        result = YandexCloudConfig.get_api_key()
        assert result is None or isinstance(result, str)

    def test_get_folder_id_returns_optional(self):
        result = YandexCloudConfig.get_folder_id()
        assert result is None or isinstance(result, str)


# =============================================================================
# ТЕСТЫ УТИЛИТ
# =============================================================================

class TestUtilities:
    """Тесты утилитарных функций."""

    def test_format_duration_seconds(self):
        assert format_duration(30) == "0м 30с"

    def test_format_duration_minutes(self):
        result = format_duration(125)
        assert "2м" in result

    def test_format_duration_hours(self):
        result = format_duration(3725)
        assert "1ч" in result

    def test_format_duration_negative(self):
        assert format_duration(-1) == "неизвестно"

    def test_check_file_exists_no_file(self, tmp_path):
        path = tmp_path / 'nonexistent.txt'
        assert check_file_exists(path) is True

    def test_check_file_exists_skip(self, tmp_path):
        path = tmp_path / 'existing.txt'
        path.write_text('test')
        assert check_file_exists(path, action='skip') is False

    def test_check_file_exists_overwrite(self, tmp_path):
        path = tmp_path / 'existing.txt'
        path.write_text('test')
        assert check_file_exists(path, action='overwrite') is True

    def test_get_chapter_output_dir(self):
        output_dir = get_chapter_output_dir('01')
        assert isinstance(output_dir, Path)
        assert output_dir.exists()

    def test_cleanup_deprecated_files_dry_run(self, tmp_path):
        # Создаём deprecated файл
        (tmp_path / '01_final.json').write_text('{}')
        (tmp_path / '01_filtered.json').write_text('{}')

        deprecated = cleanup_deprecated_files(tmp_path, dry_run=True)
        assert len(deprecated) == 1
        # Файл не удалён (dry_run)
        assert (tmp_path / '01_final.json').exists()


# =============================================================================
# ТЕСТЫ ЛОГИРОВАНИЯ
# =============================================================================

class TestLogging:
    """Тесты системы логирования."""

    def test_log_config_defaults(self):
        assert LogConfig.DEFAULT_LEVEL == 'INFO'
        assert LogConfig.FILE_LEVEL == 'DEBUG'
        assert LogConfig.MAX_LOG_FILES > 0

    def test_get_logger(self):
        logger = get_logger('test_module')
        assert isinstance(logger, logging.Logger)
        assert 'test_module' in logger.name

    def test_get_logger_default(self):
        logger = get_logger()
        assert isinstance(logger, logging.Logger)

    def test_setup_logging(self, tmp_path):
        log_file = tmp_path / 'test.log'
        logger = setup_logging(
            level='DEBUG',
            log_file=str(log_file),
            module_name='test_setup',
            console=False,
            session_id='test_session'
        )
        assert isinstance(logger, logging.Logger)
        logger.info("Test message")
        # Сбрасываем флаг инициализации для следующих тестов
        LogConfig._initialized = False

    def test_cleanup_old_logs(self, tmp_path):
        # Создаём несколько файлов логов
        for i in range(5):
            (tmp_path / f'pipeline_2026_{i}.log').write_text(f'log {i}')

        deleted = cleanup_old_logs(tmp_path, keep_count=3)
        assert deleted == 2
        remaining = list(tmp_path.glob('pipeline_*.log'))
        assert len(remaining) == 3


# =============================================================================
# ЗАПУСК
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
