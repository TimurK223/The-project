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

