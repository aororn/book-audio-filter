#!/usr/bin/env python3
"""
Unit-тесты для smart_compare.py

Покрывает основные функции сравнения транскрипции с оригиналом:
- Фонетика (to_phonetic, phonetic_similarity)
- Нормализация (normalize_word, normalize_text)
- Структуры данных (Word, Error)
- Контекст (get_context, get_context_with_marker)
- Пост-обработка (fix_misaligned_errors)
- Детектор транспозиций

Запуск:
    pytest Тесты/test_smart_compare.py -v
"""

import sys
from pathlib import Path

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent.parent / 'Инструменты'))

import pytest
from smart_compare import (
    to_phonetic,
    phonetic_similarity,
    normalize_word,
    normalize_text,
    Word,
    Error,
    get_context,
    get_context_with_marker,
    get_context_from_transcript,
    fix_misaligned_errors,
    detect_transpositions_in_opcodes,
)


# =============================================================================
# ТЕСТЫ ФОНЕТИКИ
# =============================================================================

class TestToPhonetic:
    """Тесты фонетического преобразования"""

    def test_basic_word(self):
        result = to_phonetic('слово')
        assert isinstance(result, str)
        assert len(result) > 0

    def test_removes_soft_sign(self):
        """Мягкий знак убирается"""
        result = to_phonetic('мать')
        assert 'ь' not in result

    def test_removes_hard_sign(self):
        """Твёрдый знак убирается"""
        result = to_phonetic('объект')
        assert 'ъ' not in result

    def test_yo_to_e(self):
        """ё → е в фонетике"""
        r1 = to_phonetic('ёлка')
        r2 = to_phonetic('елка')
        assert r1 == r2

    def test_voiced_unvoiced_pairs(self):
        """Звонкие/глухие согласные сливаются"""
        # б и п → одна группа
        r1 = to_phonetic('б')
        r2 = to_phonetic('п')
        assert r1 == r2

        # д и т → одна группа
        r1 = to_phonetic('д')
        r2 = to_phonetic('т')
        assert r1 == r2

    def test_vowel_reduction(self):
        """Безударные гласные сливаются"""
        # а и о → одна группа
        r1 = to_phonetic('а')
        r2 = to_phonetic('о')
        assert r1 == r2

    def test_removes_duplicates(self):
        """Удвоенные согласные → одна"""
        r1 = to_phonetic('массаж')
        r2 = to_phonetic('масаж')
        # После фонетического преобразования оба дают одинаковый результат
        assert r1 == r2

    def test_empty_string(self):
        result = to_phonetic('')
        assert result == ''

    def test_lowercase(self):
        r1 = to_phonetic('Слово')
        r2 = to_phonetic('слово')
        assert r1 == r2


class TestPhoneticSimilarity:
    """Тесты фонетической схожести"""

    def test_identical_words(self):
        sim = phonetic_similarity('слово', 'слово')
        assert sim == 100.0

    def test_similar_words(self):
        """Фонетически похожие слова"""
        sim = phonetic_similarity('его', 'ево')
        assert sim > 50.0  # Умеренная-высокая схожесть

    def test_different_words(self):
        """Совершенно разные слова"""
        sim = phonetic_similarity('дом', 'книга')
        assert sim < 50.0  # Низкая схожесть

    def test_empty_strings(self):
        sim = phonetic_similarity('', '')
        assert sim == 100.0

    def test_symmetry(self):
        """Схожесть симметрична"""
        s1 = phonetic_similarity('кот', 'код')
        s2 = phonetic_similarity('код', 'кот')
        assert s1 == s2


# =============================================================================
# ТЕСТЫ НОРМАЛИЗАЦИИ
# =============================================================================

class TestNormalizeWord:
    """Тесты нормализации слова"""

    def test_lowercase(self):
        assert normalize_word('СЛОВО') == 'слово'

    def test_strip(self):
        assert normalize_word('  слово  ') == 'слово'

    def test_yo_replacement(self):
        assert normalize_word('ёлка') == 'елка'

    def test_punctuation_removed(self):
        assert normalize_word('слово,') == 'слово'
        assert normalize_word('(слово)') == 'слово'
        assert normalize_word('слово!') == 'слово'

    def test_empty_string(self):
        assert normalize_word('') == ''

    def test_combined(self):
        assert normalize_word('  ЁЖИК!  ') == 'ежик'


class TestNormalizeText:
    """Тесты нормализации текста в список слов"""

    def test_basic(self):
        result = normalize_text('Привет мир')
        assert result == ['привет', 'мир']

    def test_punctuation(self):
        result = normalize_text('Привет, мир!')
        assert result == ['привет', 'мир']

    def test_yo(self):
        result = normalize_text('Ёлка растёт')
        assert 'елка' in result
        assert 'растет' in result

    def test_empty(self):
        result = normalize_text('')
        assert result == []

    def test_only_punctuation(self):
        result = normalize_text('... !!! ???')
        assert result == []

    def test_multiple_spaces(self):
        result = normalize_text('слово   другое')
        assert result == ['слово', 'другое']


# =============================================================================
# ТЕСТЫ СТРУКТУР ДАННЫХ
# =============================================================================

class TestWord:
    """Тесты структуры Word"""

    def test_creation(self):
        w = Word(text='слово', normalized='слово', position=0)
        assert w.text == 'слово'
        assert w.normalized == 'слово'
        assert w.position == 0
        assert w.time_start == 0.0
        assert w.time_end == 0.0

    def test_with_time(self):
        w = Word(text='слово', normalized='слово', position=5,
                 time_start=10.5, time_end=11.0)
        assert w.time_start == 10.5
        assert w.time_end == 11.0


class TestError:
    """Тесты структуры Error"""

    def test_substitution(self):
        e = Error(type='substitution', time=10.0,
                  original='дом', transcript='том')
        assert e.type == 'substitution'
        assert e.original == 'дом'
        assert e.transcript == 'том'

    def test_deletion(self):
        e = Error(type='deletion', time=5.0, original='слово')
        assert e.type == 'deletion'
        assert e.transcript == ''

    def test_insertion(self):
        e = Error(type='insertion', time=15.0, transcript='лишнее')
        assert e.type == 'insertion'
        assert e.original == ''

    def test_defaults(self):
        e = Error(type='substitution', time=0.0)
        assert e.is_yandex_error is False
        assert e.similarity == 0.0
        assert e.context == ''
        assert e.marker_pos == -1


# =============================================================================
# ТЕСТЫ КОНТЕКСТА
# =============================================================================

class TestGetContext:
    """Тесты получения контекста"""

    def test_basic_context(self):
        words = [
            Word(text='первое', normalized='первое', position=0),
            Word(text='второе', normalized='второе', position=1),
            Word(text='третье', normalized='третье', position=2),
            Word(text='четвёртое', normalized='четвертое', position=3),
            Word(text='пятое', normalized='пятое', position=4),
        ]
        context, marker_pos = get_context(words, 2, window=2)
        assert 'третье' in context
        assert 'первое' in context
        assert isinstance(marker_pos, int)
        assert marker_pos >= 0

    def test_context_at_start(self):
        """Контекст в начале — нет слов до"""
        words = [
            Word(text='первое', normalized='первое', position=0),
            Word(text='второе', normalized='второе', position=1),
        ]
        context, marker_pos = get_context(words, 0, window=2)
        assert 'первое' in context
        assert marker_pos == 0

    def test_context_at_end(self):
        """Контекст в конце — нет слов после"""
        words = [
            Word(text='первое', normalized='первое', position=0),
            Word(text='последнее', normalized='последнее', position=1),
        ]
        context, marker_pos = get_context(words, 1, window=2)
        assert 'последнее' in context

    def test_empty_words_list(self):
        """Пустой список слов"""
        context, marker_pos = get_context([], 0, window=2)
        assert context == ''


class TestGetContextWithMarker:
    """Тесты контекста с маркером (для insertions)"""

    def test_basic(self):
        words = [
            Word(text='до', normalized='до', position=0),
            Word(text='после', normalized='после', position=1),
        ]
        context, marker_pos = get_context_with_marker(words, 1, window=2)
        assert isinstance(context, str)
        assert isinstance(marker_pos, int)

    def test_at_beginning(self):
        words = [
            Word(text='слово', normalized='слово', position=0),
        ]
        context, marker_pos = get_context_with_marker(words, 0, window=2)
        assert 'слово' in context


class TestGetContextFromTranscript:
    """Тесты контекста из транскрипции"""

    def test_basic(self):
        words = [
            Word(text='один', normalized='один', position=0),
            Word(text='два', normalized='два', position=1),
            Word(text='три', normalized='три', position=2),
        ]
        context = get_context_from_transcript(words, 1, window=1)
        assert 'два' in context


# =============================================================================
# ТЕСТЫ ПОСТ-ОБРАБОТКИ
# =============================================================================

class TestFixMisalignedErrors:
    """Тесты исправления неправильных сопоставлений"""

    def test_empty_list(self):
        result = fix_misaligned_errors([])
        assert result == []

    def test_no_misalignment(self):
        """Нет проблем — ошибки остаются как есть"""
        errors = [
            Error(type='substitution', time=10.0,
                  original='дом', transcript='том',
                  similarity=0.8),
        ]
        result = fix_misaligned_errors(errors)
        assert len(result) == 1
        assert result[0].type == 'substitution'

    def test_preserves_high_similarity(self):
        """Высокая схожесть — не трогаем"""
        errors = [
            Error(type='substitution', time=10.0,
                  original='кот', transcript='код',
                  similarity=0.67),
        ]
        result = fix_misaligned_errors(errors)
        assert len(result) == 1


# =============================================================================
# ТЕСТЫ ДЕТЕКТОРА ТРАНСПОЗИЦИЙ
# =============================================================================

class TestDetectTranspositions:
    """Тесты обнаружения транспозиций (перестановок слов)"""

    def test_empty_opcodes(self):
        """Пустой список opcodes"""
        errors, processed = detect_transpositions_in_opcodes(
            [], [], [], [], []
        )
        assert errors == []
        assert processed == set()

    def test_no_transposition_in_equal(self):
        """Только equal — нет транспозиций"""
        opcodes = [('equal', 0, 3, 0, 3)]
        orig = ['а', 'б', 'в']
        trans = ['а', 'б', 'в']
        orig_words = [Word(text=w, normalized=w, position=i) for i, w in enumerate(orig)]
        trans_words = [Word(text=w, normalized=w, position=i, time_start=float(i)) for i, w in enumerate(trans)]
        errors, processed = detect_transpositions_in_opcodes(
            opcodes, orig, trans, orig_words, trans_words
        )
        assert errors == []
        assert processed == set()

    def test_insert_equal_delete_pattern(self):
        """Паттерн INSERT-EQUAL-DELETE — транспозиция"""
        # Оригинал: а б, транскрипция: б а
        # SequenceMatcher даёт: insert(б), equal(а), delete(б)
        opcodes = [
            ('insert', 0, 0, 0, 1),   # вставлено trans[0:1] = 'б'
            ('equal', 0, 1, 1, 2),     # совпадает orig[0:1] = trans[1:2] = 'а'
            ('delete', 1, 2, 2, 2),    # удалено orig[1:2] = 'б'
        ]
        orig = ['а', 'б']
        trans = ['б', 'а']
        orig_words = [Word(text=w, normalized=w, position=i) for i, w in enumerate(orig)]
        trans_words = [Word(text=w, normalized=w, position=i, time_start=float(i)) for i, w in enumerate(trans)]

        errors, processed = detect_transpositions_in_opcodes(
            opcodes, orig, trans, orig_words, trans_words
        )
        assert len(errors) == 1
        assert errors[0].type == 'transposition'
        assert 'а' in errors[0].original
        assert 'б' in errors[0].original
        assert len(processed) == 3  # все 3 opcodes обработаны


# =============================================================================
# ТЕСТЫ ГРАНИЧНЫХ СЛУЧАЕВ
# =============================================================================

class TestSmartCompareEdgeCases:
    """Тесты граничных случаев"""

    def test_normalize_word_only_punctuation(self):
        """Слово из одной пунктуации"""
        result = normalize_word('...')
        assert result == ''

    def test_normalize_text_single_word(self):
        """Один слово"""
        result = normalize_text('слово')
        assert result == ['слово']

    def test_phonetic_single_char(self):
        """Один символ"""
        result = to_phonetic('а')
        assert isinstance(result, str)

    def test_phonetic_non_cyrillic(self):
        """Не-кириллические символы"""
        result = to_phonetic('hello')
        assert isinstance(result, str)
        assert len(result) > 0

    def test_word_with_original_text(self):
        """Word с полем original_text"""
        w = Word(text='слово', normalized='слово', position=0,
                 original_text='Слово,')
        assert w.original_text == 'Слово,'



# =============================================================================
# ТЕСТЫ ПАРСИНГА ТРАНСКРИПЦИИ ЯНДЕКСА
# =============================================================================

class TestParseYandexTranscription:
    """Тесты парсинга JSON транскрипции от Яндекса"""

    def test_basic_transcription(self, tmp_path):
        """Парсинг базовой транскрипции"""
        from smart_compare import parse_yandex_transcription

        data = {
            "chunks": [
                {
                    "alternatives": [
                        {
                            "words": [
                                {"word": "привет", "startTime": "1.0s", "endTime": "1.5s"},
                                {"word": "мир", "startTime": "1.6s", "endTime": "2.0s"},
                            ],
                            "confidence": 0.95
                        }
                    ]
                }
            ]
        }
        json_file = tmp_path / "test_transcript.json"
        import json
        json_file.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')

        words = parse_yandex_transcription(str(json_file))
        assert len(words) == 2
        assert words[0].text == 'привет'
        assert words[0].normalized == 'привет'
        assert words[0].time_start == 1.0
        assert words[0].time_end == 1.5
        assert words[1].text == 'мир'
        assert words[1].position == 1

    def test_phantom_skip(self, tmp_path):
        """Пропуск первых N секунд (метаданные/заставка)"""
        from smart_compare import parse_yandex_transcription

        data = {
            "chunks": [
                {
                    "alternatives": [
                        {
                            "words": [
                                {"word": "заставка", "startTime": "0.5s", "endTime": "1.0s"},
                                {"word": "глава", "startTime": "2.0s", "endTime": "2.5s"},
                                {"word": "текст", "startTime": "5.5s", "endTime": "6.0s"},
                            ]
                        }
                    ]
                }
            ]
        }
        json_file = tmp_path / "test_phantom.json"
        import json
        json_file.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')

        words = parse_yandex_transcription(str(json_file), phantom_seconds=5.0)
        assert len(words) == 1
        assert words[0].text == 'текст'

    def test_empty_chunks(self, tmp_path):
        """Пустой JSON"""
        from smart_compare import parse_yandex_transcription

        data = {"chunks": []}
        json_file = tmp_path / "empty.json"
        import json
        json_file.write_text(json.dumps(data), encoding='utf-8')

        words = parse_yandex_transcription(str(json_file))
        assert words == []

    def test_result_format(self, tmp_path):
        """Альтернативный формат с ключом 'result' вместо 'chunks'"""
        from smart_compare import parse_yandex_transcription

        data = {
            "result": [
                {
                    "alternatives": [
                        {
                            "words": [
                                {"word": "слово", "startTime": "0.0s", "endTime": "0.5s"},
                            ]
                        }
                    ]
                }
            ]
        }
        json_file = tmp_path / "result_format.json"
        import json
        json_file.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')

        words = parse_yandex_transcription(str(json_file))
        assert len(words) == 1
        assert words[0].text == 'слово'

    def test_numeric_time_format(self, tmp_path):
        """Время в числовом формате (не строка)"""
        from smart_compare import parse_yandex_transcription

        data = {
            "chunks": [
                {
                    "alternatives": [
                        {
                            "words": [
                                {"word": "тест", "startTime": 1.5, "endTime": 2.0},
                            ]
                        }
                    ]
                }
            ]
        }
        json_file = tmp_path / "numeric_time.json"
        import json
        json_file.write_text(json.dumps(data), encoding='utf-8')

        words = parse_yandex_transcription(str(json_file))
        assert len(words) == 1
        assert words[0].time_start == 1.5

    def test_empty_words_skipped(self, tmp_path):
        """Пустые слова пропускаются"""
        from smart_compare import parse_yandex_transcription

        data = {
            "chunks": [
                {
                    "alternatives": [
                        {
                            "words": [
                                {"word": "", "startTime": "0s", "endTime": "0.5s"},
                                {"word": "реальное", "startTime": "0.5s", "endTime": "1.0s"},
                            ]
                        }
                    ]
                }
            ]
        }
        json_file = tmp_path / "empty_words.json"
        import json
        json_file.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')

        words = parse_yandex_transcription(str(json_file))
        assert len(words) == 1
        assert words[0].text == 'реальное'

    def test_no_alternatives(self, tmp_path):
        """Chunk без alternatives"""
        from smart_compare import parse_yandex_transcription

        data = {"chunks": [{"alternatives": []}]}
        json_file = tmp_path / "no_alt.json"
        import json
        json_file.write_text(json.dumps(data), encoding='utf-8')

        words = parse_yandex_transcription(str(json_file))
        assert words == []


# =============================================================================
# ТЕСТЫ ПАРСИНГА ОРИГИНАЛЬНОГО ТЕКСТА
# =============================================================================

class TestParseOriginalText:
    """Тесты парсинга оригинального текста"""

    def test_txt_file(self, tmp_path):
        """Парсинг TXT файла"""
        from smart_compare import parse_original_text

        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Привет мир! Это тест.", encoding='utf-8')

        words = parse_original_text(str(txt_file))
        assert len(words) >= 4
        assert words[0].normalized == 'привет'
        assert words[1].normalized == 'мир'

    def test_preserves_original_text(self, tmp_path):
        """Сохраняет оригинальный текст с пунктуацией"""
        from smart_compare import parse_original_text

        txt_file = tmp_path / "original.txt"
        txt_file.write_text("Слово, другое.", encoding='utf-8')

        words = parse_original_text(str(txt_file))
        # original_text должен содержать пунктуацию
        originals = [w.original_text for w in words]
        assert any(',' in o for o in originals if o)

    def test_positions_sequential(self, tmp_path):
        """Позиции слов последовательные"""
        from smart_compare import parse_original_text

        txt_file = tmp_path / "pos.txt"
        txt_file.write_text("один два три четыре пять", encoding='utf-8')

        words = parse_original_text(str(txt_file))
        for i, w in enumerate(words):
            assert w.position == i

    def test_empty_file(self, tmp_path):
        """Пустой файл"""
        from smart_compare import parse_original_text

        txt_file = tmp_path / "empty.txt"
        txt_file.write_text("", encoding='utf-8')

        words = parse_original_text(str(txt_file))
        assert words == []

    def test_hyphenated_words(self, tmp_path):
        """Слова с дефисом разбиваются"""
        from smart_compare import parse_original_text

        txt_file = tmp_path / "hyphen.txt"
        txt_file.write_text("что-то по-хорошему", encoding='utf-8')

        words = parse_original_text(str(txt_file))
        normalized = [w.normalized for w in words]
        assert 'что' in normalized
        assert 'то' in normalized


# =============================================================================
# ТЕСТЫ АВТООПРЕДЕЛЕНИЯ ФАНТОМА
# =============================================================================

class TestDetectPhantomSeconds:
    """Тесты автоопределения длительности метаданных"""

    def test_no_phantom_needed(self, tmp_path):
        """Текст начинается сразу — phantom = 0"""
        from smart_compare import detect_phantom_seconds

        # Транскрипция начинается с тех же слов, что и оригинал
        trans_data = {
            "chunks": [
                {
                    "alternatives": [
                        {
                            "words": [
                                {"word": "глава", "startTime": "0.1s"},
                                {"word": "начинается", "startTime": "0.3s"},
                                {"word": "с", "startTime": "0.5s"},
                                {"word": "этих", "startTime": "0.7s"},
                                {"word": "слов", "startTime": "0.9s"},
                                {"word": "вот", "startTime": "1.1s"},
                            ]
                        }
                    ]
                }
            ]
        }
        trans_file = tmp_path / "trans.json"
        import json
        trans_file.write_text(json.dumps(trans_data, ensure_ascii=False), encoding='utf-8')

        orig_file = tmp_path / "orig.txt"
        orig_file.write_text("Глава начинается с этих слов вот так и далее продолжается текст", encoding='utf-8')

        result = detect_phantom_seconds(str(trans_file), str(orig_file))
        assert result == 0.0  # Текст начинается сразу (время < 0.5)

    def test_phantom_detected(self, tmp_path):
        """Метаданные в начале — phantom > 0"""
        from smart_compare import detect_phantom_seconds

        trans_data = {
            "chunks": [
                {
                    "alternatives": [
                        {
                            "words": [
                                {"word": "михаил", "startTime": "0.5s"},
                                {"word": "игнатов", "startTime": "1.5s"},
                                {"word": "книга", "startTime": "2.5s"},
                                # Настоящий текст начинается с 5 сек
                                {"word": "глава", "startTime": "5.0s"},
                                {"word": "начинается", "startTime": "5.5s"},
                                {"word": "с", "startTime": "5.8s"},
                                {"word": "этих", "startTime": "6.0s"},
                                {"word": "слов", "startTime": "6.2s"},
                                {"word": "вот", "startTime": "6.5s"},
                            ]
                        }
                    ]
                }
            ]
        }
        trans_file = tmp_path / "trans.json"
        import json
        trans_file.write_text(json.dumps(trans_data, ensure_ascii=False), encoding='utf-8')

        orig_file = tmp_path / "orig.txt"
        orig_file.write_text("Глава начинается с этих слов вот так и далее продолжается текст это очень длинный текст", encoding='utf-8')

        result = detect_phantom_seconds(str(trans_file), str(orig_file))
        assert result >= 5.0  # Должен обнаружить начало реального текста

    def test_empty_transcription(self, tmp_path):
        """Пустая транскрипция — phantom = 0"""
        from smart_compare import detect_phantom_seconds

        trans_data = {"chunks": []}
        trans_file = tmp_path / "empty_trans.json"
        import json
        trans_file.write_text(json.dumps(trans_data), encoding='utf-8')

        orig_file = tmp_path / "orig.txt"
        orig_file.write_text("Текст оригинала", encoding='utf-8')

        result = detect_phantom_seconds(str(trans_file), str(orig_file))
        assert result == 0.0

    def test_short_original(self, tmp_path):
        """Слишком короткий оригинал — phantom = 0"""
        from smart_compare import detect_phantom_seconds

        trans_data = {
            "chunks": [
                {
                    "alternatives": [
                        {
                            "words": [
                                {"word": "слово", "startTime": "1.0s"},
                            ]
                        }
                    ]
                }
            ]
        }
        trans_file = tmp_path / "trans.json"
        import json
        trans_file.write_text(json.dumps(trans_data, ensure_ascii=False), encoding='utf-8')

        orig_file = tmp_path / "short.txt"
        orig_file.write_text("два три", encoding='utf-8')

        result = detect_phantom_seconds(str(trans_file), str(orig_file))
        assert result == 0.0


# =============================================================================
# ТЕСТЫ get_lemma
# =============================================================================

class TestGetLemma:
    """Тесты обёртки лемматизации"""

    def test_get_lemma_returns_string(self):
        from smart_compare import get_lemma
        result = get_lemma('слово')
        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_lemma_normalizes(self):
        from smart_compare import get_lemma
        r1 = get_lemma('СЛОВО')
        r2 = get_lemma('слово')
        assert r1 == r2


# =============================================================================
# РАСШИРЕННЫЕ ТЕСТЫ ТРАНСПОЗИЦИЙ
# =============================================================================

class TestDetectTranspositionsExtended:
    """Расширенные тесты обнаружения транспозиций"""

    def test_delete_equal_insert_pattern(self):
        """Паттерн DELETE-EQUAL-INSERT — транспозиция"""
        # Оригинал: x y, Транскрипция: y x
        opcodes = [
            ('delete', 0, 1, 0, 0),    # удалено orig[0:1] = 'x'
            ('equal', 1, 2, 0, 1),      # совпадает orig[1:2] = trans[0:1] = 'y'
            ('insert', 2, 2, 1, 2),     # вставлено trans[1:2] = 'x'
        ]
        orig = ['x', 'y']
        trans = ['y', 'x']
        orig_words = [Word(text=w, normalized=w, position=i) for i, w in enumerate(orig)]
        trans_words = [Word(text=w, normalized=w, position=i, time_start=float(i)) for i, w in enumerate(trans)]

        errors, processed = detect_transpositions_in_opcodes(
            opcodes, orig, trans, orig_words, trans_words
        )
        assert len(errors) == 1
        assert errors[0].type == 'transposition'
        assert len(processed) == 3

    def test_no_transposition_different_words(self):
        """Вставка + удаление разных слов — НЕ транспозиция"""
        opcodes = [
            ('insert', 0, 0, 0, 1),    # вставлено 'a'
            ('equal', 0, 1, 1, 2),      # совпадает 'b'
            ('delete', 1, 2, 2, 2),     # удалено 'c'
        ]
        orig = ['b', 'c']
        trans = ['a', 'b']
        orig_words = [Word(text=w, normalized=w, position=i) for i, w in enumerate(orig)]
        trans_words = [Word(text=w, normalized=w, position=i, time_start=float(i)) for i, w in enumerate(trans)]

        errors, processed = detect_transpositions_in_opcodes(
            opcodes, orig, trans, orig_words, trans_words
        )
        assert len(errors) == 0  # Разные слова — не транспозиция

    def test_multiple_word_sequences_not_transposed(self):
        """Вставка/удаление нескольких слов — НЕ транспозиция (только по одному слову)"""
        opcodes = [
            ('insert', 0, 0, 0, 2),    # вставлено trans[0:2] (2 слова)
            ('equal', 0, 1, 2, 3),
            ('delete', 1, 3, 3, 3),     # удалено orig[1:3] (2 слова)
        ]
        orig = ['m', 'a', 'b']
        trans = ['a', 'b', 'm']
        orig_words = [Word(text=w, normalized=w, position=i) for i, w in enumerate(orig)]
        trans_words = [Word(text=w, normalized=w, position=i, time_start=float(i)) for i, w in enumerate(trans)]

        errors, processed = detect_transpositions_in_opcodes(
            opcodes, orig, trans, orig_words, trans_words
        )
        assert len(errors) == 0  # Транспозиция работает только для одного слова


# =============================================================================
# РАСШИРЕННЫЕ ТЕСТЫ fix_misaligned_errors
# =============================================================================

class TestFixMisalignedExtended:
    """Расширенные тесты пост-обработки неправильных сопоставлений"""

    def test_fixes_low_similarity_with_nearby_deletion(self):
        """Низкая схожесть substitution + рядом deletion = пересопоставление"""
        errors = [
            Error(type='substitution', time=10.0,
                  original='и', transcript='рагидон',
                  similarity=0.1, context='контекст', marker_pos=5),
            Error(type='deletion', time=10.5,
                  original='рагедон',
                  context='контекст удаления', marker_pos=10),
        ]
        result = fix_misaligned_errors(errors)
        # Должно было пересопоставить: substitution рагидон→рагедон + deletion и
        types = [e.type for e in result]
        assert 'substitution' in types
        assert 'deletion' in types

    def test_preserves_unrelated_errors(self):
        """Несвязанные ошибки не затрагиваются"""
        errors = [
            Error(type='substitution', time=10.0,
                  original='дом', transcript='том', similarity=0.67),
            Error(type='deletion', time=50.0,
                  original='слово'),
            Error(type='insertion', time=80.0,
                  transcript='лишнее'),
        ]
        result = fix_misaligned_errors(errors)
        assert len(result) == 3

    def test_single_error(self):
        """Одна ошибка — без изменений"""
        errors = [
            Error(type='deletion', time=5.0, original='слово'),
        ]
        result = fix_misaligned_errors(errors)
        assert len(result) == 1
        assert result[0].type == 'deletion'


# =============================================================================
# ТЕСТЫ КОНТЕКСТА: РАСШИРЕННЫЕ
# =============================================================================

class TestContextExtended:
    """Расширенные тесты контекстных функций"""

    def test_context_uses_original_text(self):
        """Контекст использует original_text с пунктуацией"""
        words = [
            Word(text='привет', normalized='привет', position=0, original_text='Привет,'),
            Word(text='мир', normalized='мир', position=1, original_text='мир!'),
        ]
        context, _ = get_context(words, 0, window=2)
        assert 'Привет,' in context

    def test_context_from_transcript_window(self):
        """Контекст из транскрипции с разным размером окна"""
        words = [
            Word(text=f'w{i}', normalized=f'w{i}', position=i)
            for i in range(10)
        ]
        context_small = get_context_from_transcript(words, 5, window=1)
        context_large = get_context_from_transcript(words, 5, window=5)
        assert len(context_large) > len(context_small)

    def test_context_with_marker_between_words(self):
        """Маркер вставки между словами"""
        words = [
            Word(text='до', normalized='до', position=0, original_text='до'),
            Word(text='после', normalized='после', position=1, original_text='после'),
            Word(text='конец', normalized='конец', position=2, original_text='конец'),
        ]
        context, marker_pos = get_context_with_marker(words, 1, window=5)
        # Маркер должен быть после "до" и перед "после"
        assert marker_pos > 0
        assert 'до' in context
        assert 'после' in context


# =============================================================================
# ИНТЕГРАЦИОННЫЙ ТЕСТ smart_compare
# =============================================================================

class TestSmartCompareIntegration:
    """Интеграционные тесты полного пайплайна сравнения"""

    def test_full_comparison(self, tmp_path):
        """Полное сравнение: транскрипция vs оригинал"""
        from smart_compare import smart_compare
        import json

        # Создаём транскрипцию с несколькими ошибками
        trans_data = {
            "chunks": [
                {
                    "alternatives": [
                        {
                            "words": [
                                {"word": "привет", "startTime": "0.0s", "endTime": "0.5s"},
                                {"word": "дорогой", "startTime": "0.5s", "endTime": "1.0s"},
                                {"word": "мой", "startTime": "1.0s", "endTime": "1.3s"},
                                # "друг" заменён на "враг" — substitution
                                {"word": "враг", "startTime": "1.3s", "endTime": "1.8s"},
                            ]
                        }
                    ]
                }
            ]
        }

        trans_file = tmp_path / "01_transcript.json"
        trans_file.write_text(json.dumps(trans_data, ensure_ascii=False), encoding='utf-8')

        orig_file = tmp_path / "original.txt"
        orig_file.write_text("привет дорогой мой друг", encoding='utf-8')

        output_file = tmp_path / "01_compared.json"

        result = smart_compare(
            str(trans_file),
            str(orig_file),
            output_path=str(output_file),
            phantom_seconds=0,
            force=True
        )

        assert 'stats' in result
        assert 'errors' in result
        assert result['stats']['original_words'] == 4
        assert result['stats']['transcript_words'] == 4
        assert result['stats']['similarity'] > 0.5
        assert output_file.exists()

    def test_identical_texts(self, tmp_path):
        """Идентичные тексты — нет ошибок"""
        from smart_compare import smart_compare
        import json

        trans_data = {
            "chunks": [
                {
                    "alternatives": [
                        {
                            "words": [
                                {"word": "привет", "startTime": "0.0s", "endTime": "0.5s"},
                                {"word": "мир", "startTime": "0.5s", "endTime": "1.0s"},
                            ]
                        }
                    ]
                }
            ]
        }

        trans_file = tmp_path / "01_transcript.json"
        trans_file.write_text(json.dumps(trans_data, ensure_ascii=False), encoding='utf-8')

        orig_file = tmp_path / "original.txt"
        orig_file.write_text("привет мир", encoding='utf-8')

        output_file = tmp_path / "01_compared.json"

        result = smart_compare(
            str(trans_file),
            str(orig_file),
            output_path=str(output_file),
            phantom_seconds=0,
            force=True
        )

        assert result['stats']['total_differences'] == 0
        assert len(result['errors']) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
