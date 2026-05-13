# Требования
На ПК должны быть предустановлены:
- Python 3.10+
- git
- pip
# Последовательность действий
1. Склонировать проект
```bash
git clone https://github.com/AndreyKatsupeev/EITSynthAI.git
cd EITSynthAI
```
2. Скачать веса модели

Скачайте веса модели по ссылке из основного README и поместите их в директорию weights/:
```bash
mkdir -p weights
```
3. Создать виртуальное окружение
```bash
python3 -m venv venv
source venv/bin/activate
```
5. Установить зависимости

Все необходимые библиотеки находятся в файле:

kt_service/requirements.txt

Установка:
```bash
pip install -r kt_service/requirements.txt
```
5. Запуск отдельных скриптов

После установки зависимостей можно запускать отдельные Python-файлы напрямую.

Пример:
```bash
python path/to/file.py
```
или из директории kt_service:
```bash
cd kt_service
python script_name.py
```
6. Импорт и использование функций проекта

Функции проекта можно использовать как обычные Python-модули.

Пример:
```bash
from module_name import function_name

result = function_name(...)
print(result)
```

Перед запуском убедитесь, что:

веса модели находятся в папке weights/
пути к данным указаны корректно
виртуальное окружение активировано
