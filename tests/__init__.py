
"""
Тесты для системы обнаружения плагиата Educational Plagiarism Detector

Содержит модульные тесты для проверки корректности работы основных компонентов системы.
"""

__version__ = "1.0.0"
__author__ = "Test Suite"
__license__ = "MIT"

import os
import sys
from pathlib import Path

# Добавляем путь к корневому каталогу проекта
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Настройка окружения для тестов
TEST_DATA_DIR = project_root / "tests" / "test_data"
TEST_OUTPUT_DIR = project_root / "tests" / "test_output"

# Создаем директории для тестовых данных
TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Глобальные переменные для тестов
TEST_CONFIG = {
    'similarity_threshold': 0.3,
    'language': 'english',
    'test_mode': True,
    'debug': False
}

# Импорт тестовых утилит
from .test_utils import (
    create_test_file,
    create_sample_texts,
    compare_results,
    assert_similarity_range,
    cleanup_test_files
)

# Импорт тестовых классов
from .test_plagiarism_detector import TestPlagiarismDetector
from .test_single_file_mode import TestSingleFileMode
from .test_utils_module import TestUtilsModule
from .test_integration import TestIntegration

# Экспорт публичного API тестов
__all__ = [
    # Константы
    'TEST_DATA_DIR',
    'TEST_OUTPUT_DIR',
    'TEST_CONFIG',
    
    # Утилиты
    'create_test_file',
    'create_sample_texts', 
    'compare_results',
    'assert_similarity_range',
    'cleanup_test_files',
    
    # Тестовые классы
    'TestPlagiarismDetector',
    'TestSingleFileMode',
    'TestUtilsModule',
    'TestIntegration',
    
    # Функции
    'run_all_tests',
    'setup_test_environment',
    'teardown_test_environment',
]

def setup_test_environment():
    """
    Настройка тестового окружения
    
    Returns:
        dict: Статус настройки
    """
    print("⚙️ Setting up test environment...")
    
    # Очистка предыдущих тестовых данных
    if TEST_OUTPUT_DIR.exists():
        import shutil
        shutil.rmtree(TEST_OUTPUT_DIR)
    
    # Создание директорий
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Создание тестовых файлов
    from .test_utils import create_sample_test_files
    create_sample_test_files()
    
    status = {
        'test_data_dir': str(TEST_DATA_DIR),
        'test_output_dir': str(TEST_OUTPUT_DIR),
        'directories_created': True,
        'test_files_created': True
    }
    
    print("✅ Test environment setup completed")
    return status

def teardown_test_environment():
    """
    Очистка тестового окружения
    """
    print("🧹 Cleaning up test environment...")
    
    # Удаляем тестовые выходные данные (но сохраняем тестовые файлы)
    if TEST_OUTPUT_DIR.exists():
        import shutil
        shutil.rmtree(TEST_OUTPUT_DIR)
    
    # Воссоздаем пустую директорию
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("✅ Test environment cleaned up")

def run_all_tests():
    """
    Запуск всех тестов
    
    Returns:
        bool: True если все тесты прошли успешно
    """
    import unittest
    
    # Настройка окружения
    setup_test_environment()
    
    try:
        # Загрузка всех тестов
        loader = unittest.TestLoader()
        
        # Поиск всех тестов
        test_suite = loader.discover(
            start_dir=str(Path(__file__).parent),
            pattern='test_*.py'
        )
        
        # Запуск тестов
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(test_suite)
        
        # Очистка окружения
        teardown_test_environment()
        
        return result.wasSuccessful()
        
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        teardown_test_environment()
        return False

# Автоматическая настройка при импорте
if os.getenv('AUTO_SETUP_TESTS', '0') == '1':
    setup_test_environment()

# Информация при импорте
if __name__ != "__main__":
    print(f"🧪 Test suite v{__version__} loaded")
    print(f"📁 Test data directory: {TEST_DATA_DIR}")
    print(f"📁 Test output directory: {TEST_OUTPUT_DIR}")
    print("💡 Use run_all_tests() to execute all tests")
