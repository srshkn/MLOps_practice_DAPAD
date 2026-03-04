set -euo pipefail

# Определяем директории
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
PYTHON_VERSION="3.12"

echo "🔧 Проверка виртуального окружения (Python ${PYTHON_VERSION})..." >&2

install_python() {
    echo "⚠️ python${PYTHON_VERSION} не найден. Пытаюсь установить..." >&2

    if command -v apt-get >/dev/null 2>&1; then
        echo "📦 Обновление индексов пакетов (apt-get)..." >&2
        sudo apt-get update -y
        echo "📦 Установка python${PYTHON_VERSION} и python${PYTHON_VERSION}-venv..." >&2
        sudo apt-get install -y "python${PYTHON_VERSION}" "python${PYTHON_VERSION}-venv"
    else
        echo "❌ Не удалось автоматически установить python${PYTHON_VERSION}: не найден поддерживаемый пакетный менеджер." >&2
        echo "   Установи Python ${PYTHON_VERSION} вручную и запусти скрипт ещё раз." >&2
        exit 1
    fi
}

create_venv() {
    local python_bin
    python_bin="$(command -v "python${PYTHON_VERSION}" || true)"
    if [ -z "${python_bin}" ]; then
        install_python
        python_bin="$(command -v "python${PYTHON_VERSION}" || true)"
        if [ -z "${python_bin}" ]; then
            echo "❌ После установки python${PYTHON_VERSION} всё ещё не найден в PATH." >&2
            exit 1
        fi
    fi

    echo "📦 Создание виртуального окружения в ${VENV_DIR}..." >&2
    "${python_bin}" -m venv "${VENV_DIR}"
}

# 1. Проверяем/создаём venv с нужной версией Python
if [ -d "${VENV_DIR}" ]; then
    VENV_PYTHON="${VENV_DIR}/bin/python"
    if [ ! -x "${VENV_PYTHON}" ]; then
        echo "⚠️ В venv нет исполняемого python. Пересоздаю окружение..." >&2
        rm -rf "${VENV_DIR}"
        create_venv
    else
        CURRENT_VERSION="$("${VENV_PYTHON}" -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')"
        if [ "${CURRENT_VERSION}" != "${PYTHON_VERSION}" ]; then
            echo "⚠️ Найдено окружение с Python ${CURRENT_VERSION}, требуется ${PYTHON_VERSION}. Пересоздаю..." >&2
            rm -rf "${VENV_DIR}"
            create_venv
        else
            echo "✅ Найдено подходящее окружение с Python ${CURRENT_VERSION}." >&2
        fi
    fi
else
    create_venv
fi

echo "🔄 Активация окружения..." >&2
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

echo "⬆️ Обновление pip..." >&2
python -m pip install --upgrade pip

# 2. Установка/доустановка зависимостей из корневого requirements.txt
REQ_FILE="${PROJECT_ROOT}/requirements.txt"
if [ -f "${REQ_FILE}" ]; then
    echo "📦 Проверка и установка зависимостей из ${REQ_FILE}..." >&2
    pip install -r "${REQ_FILE}"
else
    echo "⚠️ Файл ${REQ_FILE} не найден. Пропускаю установку зависимостей." >&2
fi

# 3. Запуск пайплайна в lab1
cd "${SCRIPT_DIR}"

echo "🔍 Проверка наличия скриптов в директории lab1..." >&2
required_scripts=("data_creation.py" "data_preprocessing.py" "model_preparation.py" "model_testing.py")
for script in "${required_scripts[@]}"; do
    if [ ! -f "${script}" ]; then
        echo "❌ Ошибка: ${script} не найден в директории ${SCRIPT_DIR}!" >&2
        exit 1
    fi
done
echo "✅ Все необходимые скрипты присутствуют." >&2

echo "📦 Этап 1: Генерация наборов данных (train/test)..." >&2
python data_creation.py > /dev/null
echo "✅ Данные созданы." >&2

echo "🛠️ Этап 2: Предобработка данных (масштабирование)..." >&2
python data_preprocessing.py > /dev/null
echo "✅ Предобработка завершена." >&2

echo "🤖 Этап 3: Обучение модели на тренировочных данных..." >&2
python model_preparation.py > /dev/null
echo "✅ Модель обучена и сохранена." >&2

echo "📊 Этап 4: Тестирование модели..." >&2
python model_testing.py