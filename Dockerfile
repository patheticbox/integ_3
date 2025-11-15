# STAGE 1: ТРЕНУВАННЯ МОДЕЛІ
FROM python:3.10-slim AS trainer

# Встановлюємо системні бібліотеки
RUN apt-get update && apt-get install -y \
    wget \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копіюємо requirements і встановлюємо
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копіюємо скрипт тренування
COPY train.py .

# ТРЕНУЄМО МОДЕЛЬ (це відбувається під час збірки образу!)
RUN python train.py

# STAGE 2: RUNTIME (ТІЛЬКИ ІНФЕРЕНС)
FROM python:3.10-slim AS runtime

# Мінімальні системні бібліотеки
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копіюємо requirements і встановлюємо
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копіюємо Flask API
COPY flask_api.py .

# ВАЖЛИВО: Копіюємо навчену модель з першого stage
COPY --from=trainer /app/speech_command_model.pth .

# Порт Flask
EXPOSE 5000

# Запуск API (без тренування!)
CMD ["python", "flask_api.py"]