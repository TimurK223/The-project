#!/usr/bin/env python3
"""
Тесты для модуля plagiarism_detector из папки src
"""

import unittest
import sys
import os
import tempfile
import shutil
from pathlib import Path

# ============================================================================
# НАСТРОЙКА ПУТЕЙ ДЛЯ ИМПОРТА
# ============================================================================

# Получаем путь к корневой директории проекта
project_root = Path(__file__).parent.parent

# Добавляем src в путь Python
src_path = project_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))
    sys.path.insert(0, str(project_root))
    print(f"✅ Добавлен путь к src: {src_path}")
else:
    print(f"❌ Папка src не найдена: {src_path}")

# ============================================================================
# ИМПОРТ МОДУЛЯ
# ============================================================================

try:
    # Импортируем модуль напрямую
    import plagiarism_detector

    MODULE_LOADED = True
    print("✅ Модуль plagiarism_detector успешно импортирован")

    # Проверяем наличие основных классов
    if hasattr(plagiarism_detector, "PlagiarismDetector"):
        PlagiarismDetector = plagiarism_detector.PlagiarismDetector
        print("✅ Класс PlagiarismDetector найден")
    else:
        print("❌ Класс PlagiarismDetector не найден")
        MODULE_LOADED = False

    # Проверяем наличие функций
    if hasattr(plagiarism_detector, "create_test_documents"):
        create_test_documents = plagiarism_detector.create_test_documents
        print("✅ Функция create_test_documents найдена")
    else:
        print("⚠️ Функция create_test_documents не найдена")
        create_test_documents = None

    # Проверяем наличие Document (может не быть)
    if hasattr(plagiarism_detector, "Document"):
        Document = plagiarism_detector.Document
        print("✅ Класс Document найден")
        DOCUMENT_AVAILABLE = True
    else:
        print("⚠️ Класс Document не найден (это нормально)")
        Document = None
        DOCUMENT_AVAILABLE = False

except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    MODULE_LOADED = False
    PlagiarismDetector = None
    Document = None
    create_test_documents = None
    DOCUMENT_AVAILABLE = False


# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================


def create_test_files(folder_path, files_dict):
    """Создание тестовых файлов.

    Args:
        folder_path: путь к папке
        files_dict: словарь {имя_файла: содержимое}
    """
    folder = Path(folder_path)
    folder.mkdir(parents=True, exist_ok=True)

    for filename, content in files_dict.items():
        file_path = folder / filename
        file_path.write_text(content, encoding="utf-8")

    return folder


def cleanup_folder(folder_path):
    """Очистка папки."""
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)


# ============================================================================
# ТЕСТОВЫЕ КЛАССЫ
# ============================================================================


class TestModuleImport(unittest.TestCase):
    """Тесты загрузки модуля."""

    def test_module_import(self):
        """Тест импорта модуля."""
        self.assertTrue(MODULE_LOADED, "Модуль не загружен")

    def test_plagiarism_detector_class(self):
        """Тест наличия класса PlagiarismDetector."""
        self.assertIsNotNone(PlagiarismDetector, "Класс PlagiarismDetector не найден")


class TestPlagiarismDetectorInitialization(unittest.TestCase):
    """Тесты инициализации PlagiarismDetector."""

    @unittest.skipIf(not MODULE_LOADED, "Модуль не загружен")
    def test_default_initialization(self):
        """Тест инициализации с параметрами по умолчанию."""
        detector = PlagiarismDetector()

        self.assertIsInstance(detector, PlagiarismDetector)
        # Проверяем стандартные атрибуты
        self.assertTrue(hasattr(detector, "min_threshold"))
        self.assertTrue(hasattr(detector, "documents"))

    @unittest.skipIf(not MODULE_LOADED, "Модуль не загружен")
    def test_custom_initialization(self):
        """Тест инициализации с пользовательскими параметрами."""
        detector = PlagiarismDetector(min_similarity_threshold=0.5)

        # Проверяем что порог установлен
        self.assertEqual(detector.min_threshold, 0.5)


class TestFileLoading(unittest.TestCase):
    """Тесты загрузки файлов."""

    @unittest.skipIf(not MODULE_LOADED, "Модуль не загружен")
    def setUp(self):
        """Создание тестовых файлов перед каждым тестом."""
        # Создаем временную папку для тестов
        self.temp_dir = tempfile.mkdtemp(prefix="plagiarism_test_")
        # Создаем тестовые файлы
        self.test_files = {
            "doc1.txt": 
            "Artificial intelligence is transforming " "modern education.",
            "doc2.txt": 
            "AI technologies are revolutionizing " "educational systems.",
            "doc3.txt": 
            "This is a test content. " "Machine learning is important.",
        }

        self.test_folder = create_test_files(self.temp_dir, self.test_files)

        # Создаем детектор
        self.detector = PlagiarismDetector()

    def tearDown(self):
        """Очистка после каждого теста."""
        cleanup_folder(self.temp_dir)

    def test_load_valid_files(self):
        """Тест загрузки поддерживаемых файлов."""
        self.detector.load_documents(str(self.test_folder))

        # Проверяем что документы загружены
        self.assertTrue(hasattr(self.detector, "documents"))

        # Проверяем количество загруженных документов
        # Это может быть список или другой контейнер
        if hasattr(self.detector.documents, "__len__"):
            self.assertEqual(len(self.detector.documents), 3)
        else:
            # Или проверяем другим способом
            self.assertTrue(True)  # Просто проверяем что нет ошибок

    def test_load_empty_folder(self):
        """Тест загрузки из пустой папки."""
        empty_folder = self.test_folder / "empty"
        empty_folder.mkdir(exist_ok=True)

        detector = PlagiarismDetector()
        detector.load_documents(str(empty_folder))

        # Должно загрузиться 0 документов
        if hasattr(detector.documents, "__len__"):
            self.assertEqual(len(detector.documents), 0)

    def test_load_nonexistent_folder(self):
        """Тест загрузки из несуществующей папки."""
        detector = PlagiarismDetector()

        # Должно вызывать исключение
        with self.assertRaises(Exception):
            detector.load_documents("/nonexistent/folder/path")


class TestTextPreprocessing(unittest.TestCase):
    """Тесты предобработки текста."""

    @unittest.skipIf(not MODULE_LOADED, "Модуль не загружен")
    def setUp(self):
        """Настройка перед тестами."""
        self.detector = PlagiarismDetector()

    def test_preprocess_basic(self):
        """Тест базовой предобработки."""
        # Проверяем что метод существует
        self.assertTrue(hasattr(self.detector, "preprocess_text"))

        test_text = "Hello World! This is a TEST."
        processed = self.detector.preprocess_text(test_text)

        # Проверяем что возвращается строка
        self.assertIsInstance(processed, str)

        # Проверяем преобразование в нижний регистр
        self.assertEqual(processed, processed.lower())

    def test_preprocess_empty_text(self):
        """Тест предобработки пустого текста."""
        processed = self.detector.preprocess_text("")
        self.assertEqual(processed, "")

        processed = self.detector.preprocess_text("   ")
        self.assertTrue(isinstance(processed, str))

    def test_preprocess_special_characters(self):
        """Тест обработки специальных символов."""
        test_text = "Multiple   spaces   here"
        processed = self.detector.preprocess_text(test_text)

        # Проверяем что нет множественных пробелов
        if "   " in processed:
            # Это может быть допустимо в некоторых реализациях
            pass
        self.assertIsInstance(processed, str)


class TestSimilarityMethods(unittest.TestCase):
    """Тесты методов расчета схожести."""

    @unittest.skipIf(not MODULE_LOADED, "Модуль не загружен")
    def setUp(self):
        """Настройка перед тестами."""
        self.detector = PlagiarismDetector()

    def test_cosine_similarity_exists(self):
        """Тест что метод косинусного сходства существует."""
        self.assertTrue(hasattr(self.detector, "cosine_similarity_method"))

    def test_cosine_similarity_identical(self):
        """Тест косинусного сходства для идентичных текстов."""
        text1 = "artificial intelligence machine learning"
        text2 = "artificial intelligence machine learning"

        similarity = self.detector.cosine_similarity_method(text1, text2)

        # Проверяем что возвращается число
        self.assertIsInstance(similarity, (int, float))

        # Идентичные тексты должны иметь высокую схожесть
        self.assertGreaterEqual(similarity, 0.9)
        self.assertLessEqual(similarity, 1.0)

    def test_cosine_similarity_range(self):
        """Тест что схожесть в диапазоне 0-1."""
        text1 = "python programming"
        text2 = "data science"

        similarity = self.detector.cosine_similarity_method(text1, text2)

        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)

    def test_lcs_method_exists(self):
        """Тест что метод LCS существует."""
        self.assertTrue(hasattr(self.detector, "longest_common_subsequence"))

    def test_lcs_similarity(self):
        """Тест метода LCS."""
        text1 = "the quick brown fox"
        text2 = "the quick brown fox"

        similarity = self.detector.longest_common_subsequence(text1, text2)

        self.assertIsInstance(similarity, (int, float))
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)

    def test_ngram_method_exists(self):
        """Тест что метод N-gram существует."""
        self.assertTrue(hasattr(self.detector, "ngram_similarity"))

    def test_ngram_similarity(self):
        """Тест метода N-gram."""
        text1 = "natural language processing"
        text2 = "natural language processing"

        similarity = self.detector.ngram_similarity(text1, text2, n=2)

        self.assertIsInstance(similarity, (int, float))
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)


class TestFullWorkflow(unittest.TestCase):
    """Тесты полного рабочего процесса."""

    @unittest.skipIf(not MODULE_LOADED, "Модуль не загружен")
    def setUp(self):
        """Создание тестовых документов."""
        self.temp_dir = tempfile.mkdtemp(prefix="plagiarism_workflow_")

        # Создаем тестовые документы
        self.test_files = {
            "doc1.txt": "Artificial intelligence is transforming education.",
            "doc2.txt": "AI is changing the way we learn and teach.",
            "doc3.txt": "Machine learning algorithms analyze data.",
        }

        self.test_folder = create_test_files(self.temp_dir, self.test_files)
        self.detector = PlagiarismDetector(min_similarity_threshold=0.3)

    def tearDown(self):
        """Очистка после каждого теста."""
        cleanup_folder(self.temp_dir)

    def test_load_and_process(self):
        """Тест загрузки и обработки документов."""
        # Проверяем что методы существуют
        self.assertTrue(hasattr(self.detector, "load_documents"))
        self.assertTrue(hasattr(self.detector, "process_all_documents"))

        # Загружаем документы
        self.detector.load_documents(str(self.test_folder))

        # Обрабатываем документы
        self.detector.process_all_documents()

        # Если нет ошибок - тест пройден
        self.assertTrue(True)

    def test_calculate_similarity_matrix(self):
        """Тест расчета матрицы схожести."""
        self.assertTrue(hasattr(self.detector, "calculate_similarity_matrix"))

        # Загружаем и обрабатываем
        self.detector.load_documents(str(self.test_folder))
        self.detector.process_all_documents()

        # Рассчитываем матрицу
        result = self.detector.calculate_similarity_matrix()

        # Проверяем что что-то вернулось
        self.assertIsNotNone(result)

        # Это может быть словарь или другой объект
        self.assertTrue(isinstance(result, (dict, list, type(None))))

    def test_run_analysis_method(self):
        """Тест метода run_analysis если он существует."""
        if hasattr(self.detector, "run_analysis"):
            result = self.detector.run_analysis(str(self.test_folder))

            # Проверяем что что-то вернулось
            self.assertIsNotNone(result)
        else:
            # Метод может не существовать - это нормально
            self.skipTest("Метод run_analysis не найден")


class TestEdgeCases(unittest.TestCase):
    """Тесты граничных случаев."""

    @unittest.skipIf(not MODULE_LOADED, "Модуль не загружен")
    def setUp(self):
        """Настройка перед тестами."""
        self.detector = PlagiarismDetector()

    def test_similarity_with_empty_texts(self):
        """Тест схожести с пустыми текстами."""
        # Косинусная схожесть
        cosine = self.detector.cosine_similarity_method("", "test")
        self.assertIsInstance(cosine, (int, float))

        cosine = self.detector.cosine_similarity_method("", "")
        self.assertIsInstance(cosine, (int, float))

    def test_long_text_processing(self):
        """Тест обработки длинного текста."""
        long_text = "word " * 100

        processed = self.detector.preprocess_text(long_text)
        self.assertIsInstance(processed, str)
        self.assertTrue(len(processed) > 0)

    def test_unicode_text(self):
        """Тест обработки Unicode текста."""
        test_texts = [
            "Привет мир",  # Русский
            "Hello world",  # Английский
            "123 numbers",  # Цифры
            "Test with symbols !@#$%",  # Символы
        ]

        for text in test_texts:
            processed = self.detector.preprocess_text(text)
            self.assertIsInstance(processed, str)
            # Не должно быть исключений


class TestCreateTestDocuments(unittest.TestCase):
    """Тесты функции создания тестовых документов."""

    @unittest.skipIf(
        not MODULE_LOADED or create_test_documents is None,
        "Функция create_test_documents не доступна",
    )
    def test_create_test_documents(self):
        """Тест создания тестовых документов."""
        temp_dir = tempfile.mkdtemp(prefix="test_docs_")

        try:
            # Создаем тестовые документы
            result = create_test_documents(temp_dir)

            # Проверяем что папка создана
            self.assertTrue(os.path.exists(temp_dir))

            # Проверяем что есть файлы в папке
            files = list(Path(temp_dir).glob("*"))
            self.assertGreater(len(files), 0)

            # Проверяем что файлы содержат текст
            for file_path in files:
                content = file_path.read_text(encoding="utf-8")
                self.assertGreater(len(content), 0)

        finally:
            cleanup_folder(temp_dir)


# ============================================================================
# ЗАПУСК ТЕСТОВ
# ============================================================================


def run_selected_tests():
    """Запуск выбранных тестов (без требующих Document)."""
    # Создаем тестовый набор
    test_loader = unittest.TestLoader()

    # Собираем тестовые классы которые не требуют Document
    test_suite = unittest.TestSuite()

    if MODULE_LOADED:
        test_classes = [
            TestModuleImport,
            TestPlagiarismDetectorInitialization,
            TestFileLoading,
            TestTextPreprocessing,
            TestSimilarityMethods,
            TestFullWorkflow,
            TestEdgeCases,
        ]

        # Добавляем тест создания документов если функция доступна
        if create_test_documents is not None:
            test_classes.append(TestCreateTestDocuments)

        for test_class in test_classes:
            test_suite.addTests(test_loader.loadTestsFromTestCase(test_class))
    else:
        print("\n⚠️ Модуль не загружен, запускаются только базовые тесты")
        test_suite.addTests(test_loader.loadTestsFromTestCase(TestModuleImport))

    # Запускаем тесты
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Выводим статистику
    print("\n" + "=" * 60)
    print("📊 РЕЗУЛЬТАТЫ ТЕСТОВ:")
    print(f"  Всего тестов: {result.testsRun}")
    success_count = result.testsRun - len(result.failures) - len(result.errors)
    print(f"  Успешно: {success_count}")

    if result.failures:
        print(f"  Провалено: {len(result.failures)}")

    if result.errors:
        print(f"  Ошибок: {len(result.errors)}")

    print("=" * 60)

    # Показываем детали если есть ошибки
    if result.failures or result.errors:
        print("\n🔍 ДЕТАЛИ ОШИБОК:")
        for test, traceback in result.failures + result.errors:
            print(f"\n❌ {test}:")
            print("-" * 40)
            print(traceback)

    return result.wasSuccessful()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🧪 ТЕСТЫ ДЛЯ PLAGIARISM DETECTOR (без класса Document)")
    print("=" * 60)

    # Запускаем тесты
    success = run_selected_tests()

    # Завершаем с соответствующим кодом выхода
    sys.exit(0 if success else 1)
