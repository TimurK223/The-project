"""
Educational Plagiarism Detector - Source Package

Основной пакет системы обнаружения плагиата в студенческих работах.
Предоставляет функционал для анализа текстов на схожесть с использованием
нескольких методов сравнения.

Основные компоненты:
- plagiarism_detector: Основной модуль детектора плагиата
- single_file_mode: Модуль для работы с отдельными файлами
- utils: Вспомогательные утилиты
"""

__version__ = "2.1.0"
__author__ = "Educational Plagiarism Detector Team"
__license__ = "MIT"
__copyright__ = "Copyright 2024 Educational Plagiarism Detector"

import os
import sys
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Any

# ============================================================================
# НАСТРОЙКА ПУТЕЙ И СРЕДЫ
# ============================================================================

# Добавляем корневой каталог в путь Python
_ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT_DIR))

# Подавление предупреждений
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ============================================================================
# АВТОМАТИЧЕСКИЙ ИМПОРТ МОДУЛЕЙ
# ============================================================================

def _import_all_modules():
    """Динамический импорт всех доступных модулей в src"""
    import importlib
    import pkgutil
    
    package_dir = Path(__file__).parent
    
    # Список модулей, которые должны быть импортированы явно
    core_modules = ['plagiarism_detector', 'single_file_mode', 'utils']
    
    for module_name in core_modules:
        try:
            module = importlib.import_module(f'.{module_name}', __package__)
            
            # Экспортируем все публичные атрибуты модуля
            for attr_name in dir(module):
                if not attr_name.startswith('_'):
                    globals()[attr_name] = getattr(module, attr_name)
                    
        except ImportError as e:
            print(f"⚠️ Warning: Could not import module '{module_name}': {e}")

# ============================================================================
# ЯВНЫЙ ИМПОРТ ОСНОВНЫХ КОМПОНЕНТОВ
# ============================================================================

# Основные классы из plagiarism_detector
try:
    from .plagiarism_detector import (
        PlagiarismDetector,
        Document,
        create_test_documents,
        create_test_documents_english
    )
except ImportError as e:
    print(f"⚠️ Warning: Could not import from plagiarism_detector: {e}")
    PlagiarismDetector = None
    Document = None
    create_test_documents = None
    create_test_documents_english = None

# Функции из single_file_mode
try:
    from .single_file_mode import (
        compare_specific_files,
        compare_folder_with_reference,
        run_interactive_mode
    )
except ImportError:
    compare_specific_files = None
    compare_folder_with_reference = None
    run_interactive_mode = None

# Утилиты
try:
    from .utils import (
        setup_logger,
        get_logger,
        FileHandler,
        TextPreprocessor,
        SimilarityCalculator,
        Visualizer
    )
except ImportError:
    setup_logger = None
    get_logger = None
    FileHandler = None
    TextPreprocessor = None
    SimilarityCalculator = None
    Visualizer = None

# ============================================================================
# ГЛОБАЛЬНЫЕ КОНСТАНТЫ И НАСТРОЙКИ
# ============================================================================

# Константы
SUPPORTED_FORMATS = ['.txt', '.pdf', '.doc', '.docx']
DEFAULT_THRESHOLD = 0.3
DEFAULT_LANGUAGE = 'auto'
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

# Методы анализа
ANALYSIS_METHODS = ['cosine', 'lcs', 'ngram']
METHOD_WEIGHTS = {'cosine': 0.4, 'lcs': 0.3, 'ngram': 0.3}

# ============================================================================
# ПУБЛИЧНОЕ API
# ============================================================================

__all__ = [
    # Основные классы
    'PlagiarismDetector',
    'Document',
    
    # Функции из основных модулей
    'create_test_documents',
    'create_test_documents_english',
    'compare_specific_files',
    'compare_folder_with_reference',
    'run_interactive_mode',
    
    # Утилиты
    'setup_logger',
    'get_logger',
    'FileHandler',
    'TextPreprocessor',
    'SimilarityCalculator',
    'Visualizer',
    
    # Функции пакета
    'get_version',
    'get_supported_formats',
    'quick_analyze',
    'analyze_folder',
    'compare_two_files',
    'batch_analyze',
    'setup_environment',
    'check_dependencies',
]

# ============================================================================
# СЛУЖЕБНЫЕ ФУНКЦИИ
# ============================================================================

def _ensure_directories():
    """Создание необходимых директорий проекта"""
    directories = ['data', 'results', 'logs', 'uploads', 'processed']
    
    for dir_name in directories:
        dir_path = _ROOT_DIR / dir_name
        dir_path.mkdir(exist_ok=True)
        print(f"✓ Directory ensured: {dir_path}")

def _setup_nltk_data():
    """Настройка и загрузка данных NLTK"""
    try:
        import nltk
        
        required_data = ['punkt', 'wordnet', 'stopwords', 'punkt_tab']
        for data_package in required_data:
            try:
                nltk.data.find(data_package)
                print(f"✓ NLTK data already available: {data_package}")
            except LookupError:
                print(f"📥 Downloading NLTK data: {data_package}")
                nltk.download(data_package, quiet=True)
                
    except ImportError:
        print("⚠️ NLTK not available, some text processing features will be limited")
        return False
    
    return True

def _check_dependencies():
    """Проверка доступности зависимостей"""
    dependencies = {
        'numpy': 'Математические вычисления и матричные операции',
        'pandas': 'Обработка и анализ данных',
        'scikit-learn': 'Машинное обучение и TF-IDF',
        'nltk': 'Обработка естественного языка',
        'matplotlib': 'Визуализация результатов',
        'seaborn': 'Расширенная визуализация',
    }
    
    missing = []
    available = []
    
    for dep, description in dependencies.items():
        try:
            __import__(dep)
            available.append(f"✓ {dep}: {description}")
        except ImportError:
            missing.append(f"✗ {dep}: {description}")
    
    return available, missing

# ============================================================================
# ПУБЛИЧНЫЕ ФУНКЦИИ ПАКЕТА
# ============================================================================

def get_version() -> str:
    """
    Получить версию пакета
    
    Returns:
        str: Версия пакета
    """
    return __version__

def get_supported_formats() -> List[str]:
    """
    Получить список поддерживаемых форматов файлов
    
    Returns:
        List[str]: Список расширений файлов
    """
    return SUPPORTED_FORMATS.copy()

def setup_environment() -> Dict[str, Any]:
    """
    Настройка окружения для работы пакета
    
    Returns:
        Dict[str, Any]: Статус настройки
    """
    print("⚙️ Setting up environment...")
    
    # Создание директорий
    _ensure_directories()
    
    # Настройка NLTK
    nltk_status = _setup_nltk_data()
    
    # Проверка зависимостей
    available, missing = _check_dependencies()
    
    status = {
        'directories_created': True,
        'nltk_available': nltk_status,
        'dependencies_available': len(available),
        'dependencies_missing': len(missing),
        'available_deps': available,
        'missing_deps': missing
    }
    
    print("✅ Environment setup completed")
    return status

def check_dependencies() -> Dict[str, List[str]]:
    """
    Проверка доступности зависимостей
    
    Returns:
        Dict[str, List[str]]: Словарь с доступными и отсутствующими зависимостями
    """
    available, missing = _check_dependencies()
    
    print("\n" + "="*60)
    print("DEPENDENCIES CHECK")
    print("="*60)
    
    for item in available:
        print(item)
    
    for item in missing:
        print(item)
    
    if missing:
        print(f"\n⚠️ Missing {len(missing)} dependencies")
        print("Install with: pip install " + " ".join([dep.split(':')[0].replace('✗ ', '') for dep in missing]))
    
    return {
        'available': [dep.replace('✓ ', '') for dep in available],
        'missing': [dep.replace('✗ ', '') for dep in missing]
    }

def quick_analyze(folder_path: str, **kwargs) -> Optional[Dict]:
    """
    Быстрый анализ документов в папке
    
    Args:
        folder_path: Путь к папке с документами
        **kwargs: Дополнительные параметры для PlagiarismDetector
    
    Returns:
        Optional[Dict]: Результаты анализа или None при ошибке
    """
    if not PlagiarismDetector:
        print("❌ PlagiarismDetector not available")
        return None
    
    try:
        print(f"🔍 Quick analysis of: {folder_path}")
        
        detector = PlagiarismDetector(**kwargs)
        results = detector.run_analysis(folder_path)
        
        return results
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return None

def analyze_folder(folder_path: str, 
                   threshold: float = DEFAULT_THRESHOLD,
                   language: str = DEFAULT_LANGUAGE,
                   output_dir: Optional[str] = None) -> Optional[Dict]:
    """
    Анализ всех документов в указанной папке
    
    Args:
        folder_path: Путь к папке с документами
        threshold: Порог схожести (0.0-1.0)
        language: Язык документов
        output_dir: Директория для сохранения результатов
    
    Returns:
        Optional[Dict]: Результаты анализа
    """
    if not PlagiarismDetector:
        return None
    
    try:
        detector = PlagiarismDetector(
            min_similarity_threshold=threshold,
            language=language
        )
        
        results = detector.run_analysis(folder_path)
        
        if output_dir and results:
            output_path = Path(output_dir) / "analysis_results.json"
            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"💾 Results saved to: {output_path}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def compare_two_files(file1: str, file2: str, 
                      method: str = 'combined') -> Optional[Dict]:
    """
    Сравнение двух конкретных файлов
    
    Args:
        file1: Путь к первому файлу
        file2: Путь ко второму файлу
        method: Метод сравнения ('cosine', 'lcs', 'ngram', 'combined')
    
    Returns:
        Optional[Dict]: Результаты сравнения
    """
    if not compare_specific_files:
        print("❌ compare_specific_files function not available")
        return None
    
    try:
        # Создаем временную папку с двумя файлами
        import tempfile
        import shutil
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Копируем файлы во временную папку
            for file_path in [file1, file2]:
                if os.path.exists(file_path):
                    shutil.copy2(file_path, os.path.join(temp_dir, os.path.basename(file_path)))
            
            # Сравниваем
            from .single_file_mode import compare_specific_files
            results = compare_specific_files([file1, file2])
            
            return results
            
    except Exception as e:
        print(f"❌ Error comparing files: {e}")
        return None

def batch_analyze(folders: List[str], 
                  output_csv: Optional[str] = None) -> pd.DataFrame:
    """
    Пакетный анализ нескольких папок
    
    Args:
        folders: Список путей к папкам с документами
        output_csv: Путь для сохранения CSV отчета
    
    Returns:
        pd.DataFrame: Сводная таблица результатов
    """
    try:
        import pandas as pd
    except ImportError:
        print("❌ pandas not available for batch analysis")
        return None
    
    results = []
    
    for folder in folders:
        print(f"\n📁 Analyzing: {folder}")
        
        folder_results = analyze_folder(folder)
        
        if folder_results and 'summary' in folder_results:
            summary = {
                'folder': folder,
                'total_documents': folder_results['summary'].get('total_documents', 0),
                'potential_cases': folder_results['summary'].get('potential_plagiarism_cases', 0),
                'max_similarity': folder_results['summary'].get('max_similarity', 0),
                'avg_similarity': folder_results['summary'].get('avg_similarity', 0),
            }
            results.append(summary)
    
    if results:
        df = pd.DataFrame(results)
        
        if output_csv:
            df.to_csv(output_csv, index=False, encoding='utf-8')
            print(f"💾 Batch results saved to: {output_csv}")
        
        return df
    
    return pd.DataFrame()

def get_available_methods() -> List[str]:
    """
    Получить список доступных методов анализа
    
    Returns:
        List[str]: Список методов
    """
    return ANALYSIS_METHODS.copy()

# ============================================================================
# ИНИЦИАЛИЗАЦИЯ ПАКЕТА
# ============================================================================

# Автоматический импорт модулей
_import_all_modules()

# Настройка окружения при импорте пакета
_AUTO_SETUP = os.getenv('PLAGIARISM_AUTO_SETUP', '1')
if _AUTO_SETUP == '1':
    try:
        # Только создаем директории, без загрузки NLTK
        _ensure_directories()
    except:
        pass

# ============================================================================
# ИНФОРМАЦИЯ ПРИ ИМПОРТЕ
# ============================================================================

if __name__ != "__main__":
    print(f"📚 Educational Plagiarism Detector v{__version__} loaded")
    print(f"📁 Supported formats: {', '.join(SUPPORTED_FORMATS)}")
    print(f"🔧 Available methods: {', '.join(ANALYSIS_METHODS)}")
    print("💡 Use setup_environment() to configure the package")
    print("="*60)
