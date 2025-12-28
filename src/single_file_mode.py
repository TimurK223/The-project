#!/usr/bin/env python3
"""
Загрузка отдельных файлов для сравнения
"""

import os
import tempfile
from pathlib import Path
from plagiarism_detector import PlagiarismDetector


def compare_specific_files():
    """Сравнение конкретных файлов"""

    print("🔍 Сравнение отдельных файлов")
    print("-" * 40)

    files = []

    while True:
        file_path = input(
            f"Введите путь к файлу {len(files)+1} (или Enter для завершения): "
        ).strip()

        if not file_path:
            break

        if os.path.exists(file_path):
            files.append(file_path)
            print(f"✓ Добавлен: {os.path.basename(file_path)}")
        else:
            print(f"✗ Файл не найден: {file_path}")

    if len(files) < 2:
        print("❌ Нужно как минимум 2 файла для сравнения")
        return

    # Создаем временную папку с файлами
    with tempfile.TemporaryDirectory() as temp_dir:
        for file_path in files:
            # Копируем файл во временную папку
            import shutil

            dest_path = os.path.join(temp_dir, os.path.basename(file_path))
            shutil.copy2(file_path, dest_path)

        # Запускаем анализ
        detector = PlagiarismDetector(min_similarity_threshold=0.3)
        results = detector.run_analysis(temp_dir)

    return results


def compare_folder_with_reference():
    """Сравнение папки файлов с эталонным документом"""

    print("📊 Сравнение с эталонным документом")
    print("-" * 40)

    # Эталонный документ
    reference = input("Введите путь к эталонному документу: ").strip()
    if not os.path.exists(reference):
        print("❌ Эталонный документ не найден")
        return

    # Папка с документами для проверки
    folder = input("Введите путь к папке с документами для проверки: ").strip()
    if not os.path.exists(folder):
        print("❌ Папка не найдена")
        return

    # Создаем временную папку
    with tempfile.TemporaryDirectory() as temp_dir:
        # Копируем эталонный документ
        import shutil

        ref_name = os.path.basename(reference)
        shutil.copy2(reference, os.path.join(temp_dir, f"REFERENCE_{ref_name}"))

        # Копируем все документы из папки
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path) and filename.lower().endswith((".txt", ".pdf")):
                shutil.copy2(file_path, os.path.join(temp_dir, filename))

        # Запускаем анализ
        detector = PlagiarismDetector(min_similarity_threshold=0.3)
        results = detector.run_analysis(temp_dir)

    return results


if __name__ == "__main__":
    print("Выберите режим:")
    print("1. Сравнить несколько конкретных файлов")
    print("2. Сравнить документы в папке с эталоном")

    choice = input("Введите номер: ").strip()

    if choice == "1":
        compare_specific_files()
    elif choice == "2":
        compare_folder_with_reference()
    else:
        print("❌ Неверный выбор")
