#!/usr/bin/env python3
"""
Мониторинг папки uploads/ с автоматическим запуском анализа при загрузке файлов
"""

import os
import sys
import time
import json
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Добавляем путь к проекту
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from src import PlagiarismDetector, setup_environment, get_supported_formats

    MODULE_LOADED = True
except ImportError:
    MODULE_LOADED = False
    print("❌ Модуль src не найден. Убедитесь, что вы в корне проекта.")

# Настройки
UPLOADS_DIR = project_root / "uploads"
RESULTS_DIR = project_root / "results"
PROCESSED_DIR = project_root / "processed"
LOG_FILE = project_root / "monitor.log"

# Создаем необходимые директории
UPLOADS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class UploadHandler(FileSystemEventHandler):
    """Обработчик событий файловой системы для папки uploads/"""

    def __init__(self, detector: Optional[PlagiarismDetector] = None):
        super().__init__()
        self.detector = detector
        self.processing_files = set()
        self.delay = 5  # Задержка перед анализом (секунды)

    def on_created(self, event):
        """Обработка события создания файла"""
        if event.is_directory:
            return

        file_path = Path(event.src_path)

        # Проверяем расширение файла
        supported_formats = get_supported_formats() if MODULE_LOADED else [".txt"]
        if file_path.suffix.lower() not in supported_formats:
            logger.info(f"Файл {file_path.name} имеет неподдерживаемый формат")
            return

        logger.info(f"Обнаружен новый файл: {file_path.name}")

        # Добавляем файл в обработку
        self.processing_files.add(str(file_path))

        # Запускаем анализ через задержку
        time.sleep(self.delay)

        # Проверяем, что файл все еще существует
        if file_path.exists():
            self.process_file(file_path)
        else:
            logger.warning(f"Файл {file_path.name} был удален перед анализом")

    def on_modified(self, event):
        """Обработка события изменения файла (если файл загружается частями)"""
        if event.is_directory:
            return

        file_path = Path(event.src_path)

        # Проверяем расширение файла
        supported_formats = get_supported_formats() if MODULE_LOADED else [".txt"]
        if file_path.suffix.lower() not in supported_formats:
            return

        # Если файл уже в обработке, игнорируем
        if str(file_path) in self.processing_files:
            return

        # Проверяем размер файла
        if file_path.stat().st_size > 100 * 1024 * 1024:  # 100MB
            logger.warning(
                f"Файл {file_path.name} слишком большой (>{file_path.stat().st_size/1024/1024:.1f}MB)"
            )
            return

        logger.info(f"Файл изменен: {file_path.name}")

        # Добавляем задержку для полной загрузки
        time.sleep(2)

        if file_path.exists():
            self.processing_files.add(str(file_path))
            self.process_file(file_path)

    def process_file(self, file_path: Path):
        """Обработка файла - запуск анализа"""
        try:
            logger.info(f"Начинаем анализ файла: {file_path.name}")

            if not self.detector:
                logger.error("Детектор не инициализирован")
                return

            # Создаем временную папку для анализа
            temp_dir = UPLOADS_DIR / "temp_analysis"
            temp_dir.mkdir(exist_ok=True)

            # Копируем файл во временную папку
            temp_file = temp_dir / file_path.name
            shutil.copy2(file_path, temp_file)

            # Запускаем анализ
            results = self.detector.run_analysis(str(temp_dir))

            # Сохраняем результаты
            self.save_results(file_path.name, results)

            # Перемещаем обработанный файл
            self.move_processed_file(file_path)

            # Очищаем временную папку
            shutil.rmtree(temp_dir)

            logger.info(f"Анализ файла {file_path.name} завершен успешно")

        except Exception as e:
            logger.error(f"Ошибка при обработке файла {file_path.name}: {e}")

        finally:
            # Удаляем файл из списка обработки
            if str(file_path) in self.processing_files:
                self.processing_files.remove(str(file_path))

    def save_results(self, filename: str, results: Dict):
        """Сохранение результатов анализа в JSON с timestamp"""
        try:
            # Создаем имя файла с timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_filename = f"analysis_{filename}_{timestamp}.json"
            result_path = RESULTS_DIR / result_filename

            # Добавляем метаданные
            results_with_metadata = {
                "metadata": {
                    "original_file": filename,
                    "analysis_timestamp": datetime.now().isoformat(),
                    "analysis_duration": results.get("analysis_duration", 0),
                    "system_version": "1.0.0",
                },
                "analysis": results,
            }

            # Сохраняем в JSON
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(
                    results_with_metadata, f, ensure_ascii=False, indent=2, default=str
                )

            logger.info(f"Результаты сохранены в: {result_path}")

            # Также создаем краткий отчет
            self.create_summary_report(filename, results)

        except Exception as e:
            logger.error(f"Ошибка при сохранении результатов: {e}")

    def create_summary_report(self, filename: str, results: Dict):
        """Создание краткого отчета в CSV"""
        try:
            csv_path = RESULTS_DIR / f"summary_{datetime.now().strftime('%Y%m%d')}.csv"

            # Если файл уже существует, читаем его
            if csv_path.exists():
                import pandas as pd

                df = pd.read_csv(csv_path)
            else:
                # Создаем новый DataFrame
                import pandas as pd

                df = pd.DataFrame(
                    columns=[
                        "timestamp",
                        "filename",
                        "total_documents",
                        "potential_cases",
                        "max_similarity",
                        "avg_similarity",
                    ]
                )

            # Добавляем новую запись
            new_row = {
                "timestamp": datetime.now().isoformat(),
                "filename": filename,
                "total_documents": results.get("summary", {}).get("total_documents", 0),
                "potential_cases": results.get("summary", {}).get(
                    "potential_plagiarism_cases", 0
                ),
                "max_similarity": results.get("summary", {}).get("max_similarity", 0),
                "avg_similarity": results.get("summary", {}).get("avg_similarity", 0),
            }

            # Добавляем в DataFrame и сохраняем
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(csv_path, index=False, encoding="utf-8")

        except Exception as e:
            logger.warning(f"Не удалось создать CSV отчет: {e}")

    def move_processed_file(self, file_path: Path):
        """Перемещение обработанного файла в папку processed"""
        try:
            # Создаем подпапку с датой
            date_folder = PROCESSED_DIR / datetime.now().strftime("%Y-%m-%d")
            date_folder.mkdir(exist_ok=True)

            # Новое имя файла с timestamp
            timestamp = datetime.now().strftime("%H%M%S")
            new_filename = f"{file_path.stem}_{timestamp}{file_path.suffix}"
            new_path = date_folder / new_filename

            # Перемещаем файл
            shutil.move(file_path, new_path)
            logger.info(f"Файл перемещен в: {new_path}")

        except Exception as e:
            logger.error(f"Ошибка при перемещении файла: {e}")


def check_existing_files(uploads_dir: Path) -> List[Path]:
    """Проверка существующих файлов в папке uploads"""
    existing_files = []
    for file_path in uploads_dir.glob("*"):
        if file_path.is_file():
            supported_formats = get_supported_formats() if MODULE_LOADED else [".txt"]
            if file_path.suffix.lower() in supported_formats:
                existing_files.append(file_path)

    return existing_files


def initialize_detector() -> Optional[PlagiarismDetector]:
    """Инициализация детектора плагиата"""
    try:
        # Настройка окружения
        setup_environment()

        # Создание детектора
        detector = PlagiarismDetector(min_similarity_threshold=0.3, language="auto")

        logger.info("Детектор плагиата инициализирован")
        return detector

    except Exception as e:
        logger.error(f"Ошибка при инициализации детектора: {e}")
        return None


def main():
    """Основная функция мониторинга"""
    print("\n" + "=" * 60)
    print("📁 МОНИТОРИНГ ПАПКИ UPLOADS/")
    print("=" * 60)

    if not MODULE_LOADED:
        print("❌ Модуль src не загружен. Завершение работы.")
        return

    # Инициализация детектора
    detector = initialize_detector()
    if not detector:
        print("❌ Не удалось инициализировать детектор")
        return

    # Проверяем существующие файлы
    print(f"📂 Папка для загрузок: {UPLOADS_DIR}")
    existing_files = check_existing_files(UPLOADS_DIR)

    if existing_files:
        print(f"📋 Найдено существующих файлов: {len(existing_files)}")
        for file_path in existing_files:
            print(f"  - {file_path.name}")

        # Предлагаем проанализировать существующие файлы
        response = input("\n📊 Проанализировать существующие файлы? (y/n): ")
        if response.lower() == "y":
            for file_path in existing_files:
                handler = UploadHandler(detector)
                handler.process_file(file_path)

    # Создаем обработчик событий
    event_handler = UploadHandler(detector)

    # Создаем наблюдатель
    observer = Observer()
    observer.schedule(event_handler, str(UPLOADS_DIR), recursive=False)

    print("\n🚀 Мониторинг запущен...")
    print(f"📁 Папка: {UPLOADS_DIR}")
    print("📊 Автоматический анализ при загрузке файлов")
    print("💾 Результаты сохраняются в папку results/")
    print("🛑 Для остановки нажмите Ctrl+C\n")

    try:
        # Запускаем мониторинг
        observer.start()

        # Бесконечный цикл
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\n🛑 Остановка мониторинга...")

    finally:
        observer.stop()
        observer.join()
        print("✅ Мониторинг остановлен")


if __name__ == "__main__":
    main()
