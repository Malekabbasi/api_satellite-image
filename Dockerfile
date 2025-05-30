FROM python:3.11-slim

WORKDIR /app

# Mise à jour des sources et installation des dépendances système
RUN apt-get update && apt-get install -y \
    # Dépendances existantes
    gdal-bin \
    libgdal-dev \
    python3-dev \
    build-essential \
    # Nouvelles dépendances pour OpenCV (VERSIONS MINIMALES)
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    # Dépendances pour scikit-image
    libffi-dev \
    # Nettoyage pour réduire la taille
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# Copier requirements en premier (pour cache Docker)
COPY requirements.txt .

# Installation des packages Python avec gestion d'erreur
RUN pip install --upgrade pip --no-cache-dir && \
    pip install --no-cache-dir --timeout 1000 -r requirements.txt

# Copier le reste des fichiers
COPY . .

# CORRECTION PERMISSIONS (gardé comme votre version)
RUN mkdir -p /tmp/matplotlib /tmp/cache && chmod 777 /tmp/matplotlib /tmp/cache

# Variables d'environnement (gardées comme votre version)
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV HOME=/tmp
ENV XDG_CACHE_HOME=/tmp/cache
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]