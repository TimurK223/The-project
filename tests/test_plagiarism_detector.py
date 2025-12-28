#!/usr/bin/env python3
"""
Основные тесты для системы обнаружения плагиата

Содержит модульные и интеграционные тесты для проверки:
1. Основного функционала PlagiarismDetector
2. Режима работы с отдельными файлами
3. Утилит
4. Интеграции компонентов
"""

import unittest
import tempfile
import shutil
import os
from pathlib import Path
import json

# Импорт тестируемых модулей
try:
    from src import PlagiarismDetector, setup_environment, compare_specific_files
    from src.plagiarism_detector import create_test_documents, create_test_documents_english
    from src.single_file_mode import compare_folder_with_reference
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: Could not import main modules: {e}")
    MODULES_AVAILABLE = False

# Импорт тестовых утилит
from tests import (
    create_test_file,
    create_sample_texts,
    compare_results,
    assert_similarity_range,
    cleanup_test_files,
    TEST_DATA_DIR,
    TEST_OUTPUT_DIR
)

class TestBasicFunctionality(unittest.TestCase):
    """Тесты базового функционала"""
    
    @classmethod
    def setUpClass(cls):
        """Настройка перед всеми тестами класса"""
        if not MODULES_AVAILABLE:
            raise unittest.SkipTest("Main modules not available")
        
        # Создание тестовых данных
        cls.test_data = create_sample_texts()
        
        # Создание тестовой папки с документами
        cls.test_folder = TEST_DATA_DIR / "basic_test"
        cls.test_folder.mkdir(exist_ok=True)
        
        # Создание тестовых файлов
        for i, (name, content) in enumerate(cls.test_data.items()):
            file_path = cls.test_folder / f"{name}.txt"
            file_path.write_text(content, encoding='utf-8')
    
    @classmethod
    def tearDownClass(cls):
        """Очистка после всех тестов класса"""
        # Удаляем тестовую папку
        if cls.test_folder.exists():
            shutil.rmtree(cls.test_folder)
    
    def setUp(self):
        """Настройка перед каждым тестом"""
        self.detector = PlagiarismDetector(min_similarity_threshold=0.3)
    
    def test_detector_initialization(self):
        """Тест инициализации детектора"""
        self.assertIsNotNone(self.detector)
        self.assertEqual(self.detector.min_threshold, 0.3)
        self.assertIsInstance(self.detector.documents, list)
    
    def test_load_documents(self):
        """Тест загрузки документов"""
        self.detector.load_documents(str(self.test_folder))
        
        self.assertEqual(len(self.detector.documents), len(self.test_data))
        
        # Проверяем, что все файлы загружены
        loaded_filenames = [doc.filename for doc in self.detector.documents]
        expected_filenames = [f"{name}.txt" for name in self.test_data.keys()]
        
        for expected in expected_filenames:
            self.assertIn(expected, loaded_filenames)
    
    def test_text_preprocessing(self):
        """Тест предварительной обработки текста"""
        test_text = "Hello World! This is a TEST with numbers 123."
        processed = self.detector.preprocess_text(test_text)
        
        # Проверяем, что текст в нижнем регистре
        self.assertEqual(processed, processed.lower())
        
        # Проверяем, что нет специальных символов
        self.assertNotIn('!', processed)
        self.assertNotIn('123', processed)
    
    def test_empty_document_handling(self):
        """Тест обработки пустых документов"""
        # Создаем временный пустой файл
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("")
            empty_file = f.name
        
        try:
            # Пробуем загрузить пустой файл
            temp_dir = tempfile.mkdtemp()
            shutil.move(empty_file, os.path.join(temp_dir, "empty.txt"))
            
            detector = PlagiarismDetector()
            detector.load_documents(temp_dir)
            
            # Должен быть загружен, но с пустым содержимым
            self.assertEqual(len(detector.documents), 1)
            
        finally:
            # Очистка
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

class TestSimilarityMethods(unittest.TestCase):
    """Тесты методов расчета схожести"""
    
    def test_cosine_similarity(self):
        """Тест косинусного сходства"""
        if not MODULES_AVAILABLE:
            self.skipTest("Main modules not available")
        
        detector = PlagiarismDetector()
        
        # Идентичные тексты
        text1 = "artificial intelligence is changing the world"
        text2 = "artificial intelligence is changing the world"
        similarity = detector.cosine_similarity_method(text1, text2)
        self.assertAlmostEqual(similarity, 1.0, places=2)
        
        # Совершенно разные тексты
        text3 = "machine learning algorithms"
        text4 = "quantum physics experiments"
        similarity = detector.cosine_similarity_method(text3, text4)
        self.assertLess(similarity, 0.3)
        
        # Частично похожие тексты
        text5 = "deep learning neural networks for image recognition"
        text6 = "neural networks and deep learning algorithms"
        similarity = detector.cosine_similarity_method(text5, text6)
        self.assertGreater(similarity, 0.5)
        self.assertLess(similarity, 1.0)
    
    def test_lcs_similarity(self):
        """Тест метода LCS"""
        if not MODULES_AVAILABLE:
            self.skipTest("Main modules not available")
        
        detector = PlagiarismDetector()
        
        # Идентичные последовательности
        text1 = "the quick brown fox jumps over the lazy dog"
        text2 = "the quick brown fox jumps over the lazy dog"
        similarity = detector.longest_common_subsequence(text1, text2)
        self.assertAlmostEqual(similarity, 1.0, places=2)
        
        # Частичное совпадение
        text3 = "artificial intelligence and machine learning"
        text4 = "machine learning and artificial intelligence"
        similarity = detector.longest_common_subsequence(text3, text4)
        self.assertGreater(similarity, 0.5)
        
        # Разные тексты
        text5 = "hello world python programming"
        text6 = "data science artificial intelligence"
        similarity = detector.longest_common_subsequence(text5, text6)
        self.assertEqual(similarity, 0.0)
    
    def test_ngram_similarity(self):
        """Тест метода N-gram"""
        if not MODULES_AVAILABLE:
            self.skipTest("Main modules not available")
        
        detector = PlagiarismDetector()
        
        # Идентичные тексты
        text1 = "natural language processing is interesting"
        text2 = "natural language processing is interesting"
        similarity = detector.ngram_similarity(text1, text2, n=2)
        self.assertAlmostEqual(similarity, 1.0, places=2)
        
        # С разным n
        text3 = "machine learning deep learning"
        text4 = "deep learning machine learning"
        
        similarity_n2 = detector.ngram_similarity(text3, text4, n=2)
        similarity_n3 = detector.ngram_similarity(text3, text4, n=3)
        
        self.assertNotEqual(similarity_n2, similarity_n3)
        
        # Проверка граничных случаев
        similarity_empty = detector.ngram_similarity("", "test", n=2)
        self.assertEqual(similarity_empty, 0.0)

class TestSingleFileMode(unittest.TestCase):
    """Тесты режима работы с отдельными файлами"""
    
    def setUp(self):
        """Настройка перед каждым тестом"""
        if not MODULES_AVAILABLE:
            self.skipTest("Main modules not available")
        
        # Создание тестовых файлов
        self.test_files = []
        
        # Файл 1: Оригинальный текст
        self.file1 = TEST_DATA_DIR / "original.txt"
        self.file1.write_text(
            "Artificial intelligence is transforming modern education "
            "through personalized learning systems.",
            encoding='utf-8'
        )
        self.test_files.append(str(self.file1))
        
        # Файл 2: Перефразированный текст (плагиат)
        self.file2 = TEST_DATA_DIR / "paraphrased.txt"
        self.file2.write_text(
            "AI technologies are revolutionizing education by enabling "
            "personalized learning approaches.",
            encoding='utf-8'
        )
        self.test_files.append(str(self.file2))
        
        # Файл 3: Совершенно другой текст
        self.file3 = TEST_DATA_DIR / "different.txt"
        self.file3.write_text(
            "Quantum computing uses quantum bits to perform calculations "
            "much faster than classical computers.",
            encoding='utf-8'
        )
        self.test_files.append(str(self.file3))
    
    def tearDown(self):
        """Очистка после каждого теста"""
        # Файлы удаляются в cleanup
        pass
    
    def test_compare_specific_files(self):
        """Тест сравнения конкретных файлов"""
        results = compare_specific_files(self.test_files)
        
        self.assertIsNotNone(results)
        self.assertIn('combined_matrix', str(results))
        
        # Проверяем, что результаты содержат нужные данные
        if isinstance(results, dict):
            self.assertIn('filenames', results)
            self.assertEqual(len(results.get('filenames', [])), 3)
    
    def test_compare_folder_with_reference(self):
        """Тест сравнения папки с эталоном"""
        # Создаем папку с документами
        test_folder = TEST_DATA_DIR / "folder_test"
        test_folder.mkdir(exist_ok=True)
        
        # Копируем файлы в папку (кроме оригинального, который будет эталоном)
        shutil.copy2(str(self.file2), str(test_folder / "student1.txt"))
        shutil.copy2(str(self.file3), str(test_folder / "student2.txt"))
        
        try:
            results = compare_folder_with_reference(
                reference_file=str(self.file1),
                folder_path=str(test_folder)
            )
            
            self.assertIsNotNone(results)
            
            # Должны быть 3 документа (эталон + 2 студенческих)
            if isinstance(results, dict) and 'filenames' in results:
                self.assertEqual(len(results['filenames']), 3)
                
        finally:
            # Очистка
            if test_folder.exists():
                shutil.rmtree(test_folder)
    
    def test_single_file_mode_invalid_input(self):
        """Тест обработки неверного ввода"""
        # Несуществующий файл
        results = compare_specific_files(["nonexistent.txt", "another_fake.txt"])
        self.assertIsNone(results)
        
        # Только один файл
        results = compare_specific_files([str(self.file1)])
        self.assertIsNone(results)
        
        # Пустой список
        results = compare_specific_files([])
        self.assertIsNone(results)

class TestIntegration(unittest.TestCase):
    """Интеграционные тесты"""
    
    def test_full_workflow(self):
        """Тест полного рабочего процесса"""
        if not MODULES_AVAILABLE:
            self.skipTest("Main modules not available")
        
        # Создаем тестовые документы
        test_folder = TEST_DATA_DIR / "workflow_test"
        create_test_documents(str(test_folder))
        
        try:
            # Создаем детектор
            detector = PlagiarismDetector(
                min_similarity_threshold=0.3,
                language='russian'
            )
            
            # Загружаем документы
            detector.load_documents(str(test_folder))
            self.assertGreater(len(detector.documents), 0)
            
            # Обрабатываем
            detector.process_all_documents()
            
            # Проверяем обработку
            for doc in detector.documents:
                self.assertTrue(doc.processed_content)
            
            # Рассчитываем схожесть
            results = detector.calculate_similarity_matrix()
            
            self.assertIsNotNone(results)
            self.assertIn('combined', results)
            self.assertIn('filenames', results)
            
            # Генерируем отчет
            detector.generate_report(results)
            
            # Визуализация (проверяем, что не падает)
            try:
                detector.visualize_results(results)
                visualization_exists = (TEST_OUTPUT_DIR / "similarity_matrix.png").exists() or \
                                      Path("similarity_matrix.png").exists()
                self.assertTrue(visualization_exists)
            except Exception as e:
                print(f"Visualization warning: {e}")
                # Визуализация может падать из-за отсутствия графических библиотек
            
        finally:
            # Очистка
            if test_folder.exists():
                shutil.rmtree(test_folder)

class TestEdgeCases(unittest.TestCase):
    """Тесты граничных случаев"""
    
    def test_large_files(self):
        """Тест обработки больших файлов"""
        if not MODULES_AVAILABLE:
            self.skipTest("Main modules not available")
        
        # Создаем большой файл
        large_file = TEST_DATA_DIR / "large.txt"
        
        # Генерируем большой текст (около 1MB)
        large_text = "Sample sentence. " * 50000
        
        with open(large_file, 'w', encoding='utf-8') as f:
            f.write(large_text)
        
        try:
            detector = PlagiarismDetector()
            
            # Должен обработать без ошибок
            processed = detector.preprocess_text(large_text)
            self.assertIsInstance(processed, str)
            self.assertLess(len(processed), len(large_text))  # После обработки короче
            
        finally:
            if large_file.exists():
                large_file.unlink()
    
    def test_special_characters(self):
        """Тест обработки специальных символов"""
        if not MODULES_AVAILABLE:
            self.skipTest("Main modules not available")
        
        detector = PlagiarismDetector()
        
        test_cases = [
            ("Hello © World ®", "hello world"),
            ("Text with emoji 😀 👍", "text with emoji"),
            ("HTML entities &lt;div&gt;", "html entities lt div gt"),
            ("Multiple   spaces   here", "multiple spaces here"),
            ("Line\nbreaks\nhere", "line breaks here"),
            ("Mixed CASE TeXt", "mixed case text"),
        ]
        
        for input_text, expected_clean in test_cases:
            processed = detector.preprocess_text(input_text)
            # Проверяем, что обработанный текст содержит ожидаемые слова
            for word in expected_clean.split():
                if word:  # Пропускаем пустые строки
                    self.assertIn(word, processed)
    
    def test_different_languages(self):
        """Тест обработки разных языков"""
        if not MODULES_AVAILABLE:
            self.skipTest("Main modules not available")
        
        # Русский текст
        detector_ru = PlagiarismDetector(language='russian')
        text_ru = "Искусственный интеллект меняет мир"
        processed_ru = detector_ru.preprocess_text(text_ru)
        
        self.assertIsInstance(processed_ru, str)
        self.assertEqual(processed_ru, processed_ru.lower())
        
        # Английский текст
        detector_en = PlagiarismDetector(language='english')
        text_en = "Artificial Intelligence is changing the world"
        processed_en = detector_en.preprocess_text(text_en)
        
        self.assertIsInstance(processed_en, str)
        self.assertEqual(processed_en, processed_en.lower())
        
        # Автоопределение (должно работать для обоих)
        detector_auto = PlagiarismDetector(language='auto')
        processed_auto_ru = detector_auto.preprocess_text(text_ru)
        processed_auto_en = detector_auto.preprocess_text(text_en)
        
        self.assertTrue(processed_auto_ru)
        self.assertTrue(processed_auto_en)

class TestPerformance(unittest.TestCase):
    """Тесты производительности"""
    
    def test_similarity_calculation_speed(self):
        """Тест скорости расчета схожести"""
        if not MODULES_AVAILABLE:
            self.skipTest("Main modules not available")
        
        import time
        
        detector = PlagiarismDetector()
        
        # Создаем тестовые тексты
        text1 = " ".join(["word"] * 100)  # 100 слов
        text2 = " ".join(["word"] * 100)  # Идентичный текст
        
        # Измеряем время выполнения
        start_time = time.time()
        
        # Выполняем несколько раз для статистики
        for _ in range(10):
            similarity = detector.cosine_similarity_method(text1, text2)
            self.assertAlmostEqual(similarity, 1.0, places=2)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Проверяем, что выполнение не слишком долгое
        # (10 операций должны выполняться менее чем за 5 секунд)
        self.assertLess(execution_time, 5.0)
        
        print(f"\n⏱ Similarity calculation time: {execution_time:.3f} seconds for 10 operations")

def run_tests():
    """Запуск всех тестов"""
    # Настройка тестового окружения
    from tests import setup_test_environment
    setup_test_environment()
    
    # Запуск тестов
    loader = unittest.TestLoader()
    
    # Собираем все тесты
    test_suite = unittest.TestSuite()
    
    test_suite.addTests(loader.loadTestsFromTestCase(TestBasicFunctionality))
    test_suite.addTests(loader.loadTestsFromTestCase(TestSimilarityMethods))
    test_suite.addTests(loader.loadTestsFromTestCase(TestSingleFileMode))
    test_suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    test_suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    test_suite.addTests(loader.loadTestsFromTestCase(TestPerformance))
    
    # Запускаем тесты
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Очистка
    from tests import teardown_test_environment
    teardown_test_environment()
    
    return result.wasSuccessful()

if __name__ == '__main__':
    # Запуск тестов при прямом вызове
    success = run_tests()
    
    # Возвращаем код выхода
    import sys
    sys.exit(0 if success else 1)
