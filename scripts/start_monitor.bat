@echo off
echo 🚀 Запуск монитора папки uploads/
echo =================================

REM Проверяем Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python не найден
    pause
    exit /b 1
)

REM Проверяем зависимости
echo 📦 Проверка зависимостей...
pip install -r requirements_monitor.txt

REM Создаем папки если их нет
if not exist "uploads" mkdir uploads
if not exist "results" mkdir results
if not exist "processed" mkdir processed

REM Запускаем монитор
echo 📁 Мониторинг папки: %cd%\uploads
echo 📊 Результаты будут сохраняться в: %cd%\results
echo ⏳ Автоматический анализ при загрузке файлов...
echo 🛑 Для остановки закройте окно
echo.

REM Запускаем монитор
python monitor_uploads.py

pause
