#!/usr/bin/env python3
"""Configuration management system for satellite API"""

import os
from typing import Optional
from dataclasses import dataclass
from pathlib import Path

@dataclass
class SentinelHubConfig:
    """Configuration pour SentinelHub"""
    instance_id: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> 'SentinelHubConfig':
        """Charger la configuration depuis les variables d'environnement"""
        return cls(
            instance_id=os.getenv('SENTINELHUB_INSTANCE_ID'),
            client_id=os.getenv('SENTINELHUB_CLIENT_ID'),
            client_secret=os.getenv('SENTINELHUB_CLIENT_SECRET')
        )
    
    def is_configured(self) -> bool:
        """Vérifier si la configuration est complète"""
        return all([self.instance_id, self.client_id, self.client_secret])

@dataclass
class RedisConfig:
    """Configuration pour Redis"""
    url: str = "redis://localhost:6379"
    max_connections: int = 10
    socket_timeout: int = 5
    
    @classmethod
    def from_env(cls) -> 'RedisConfig':
        """Charger la configuration depuis les variables d'environnement"""
        return cls(
            url=os.getenv('REDIS_URL', 'redis://localhost:6379'),
            max_connections=int(os.getenv('REDIS_MAX_CONNECTIONS', '10')),
            socket_timeout=int(os.getenv('REDIS_SOCKET_TIMEOUT', '5'))
        )

@dataclass
class APIConfig:
    """Configuration générale de l'API"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    log_level: str = "INFO"
    cors_origins: list = None
    rate_limit_per_minute: int = 10
    
    @classmethod
    def from_env(cls) -> 'APIConfig':
        """Charger la configuration depuis les variables d'environnement"""
        cors_origins = os.getenv('CORS_ORIGINS', '*').split(',')
        if cors_origins == ['*']:
            cors_origins = ["*"]
        
        return cls(
            host=os.getenv('HOST', '0.0.0.0'),
            port=int(os.getenv('PORT', '8000')),
            debug=os.getenv('DEBUG', 'false').lower() == 'true',
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
            cors_origins=cors_origins,
            rate_limit_per_minute=int(os.getenv('RATE_LIMIT_PER_MINUTE', '10'))
        )

@dataclass
class ProcessingConfig:
    """Configuration pour le traitement d'images"""
    max_image_size: int = 10 * 1024 * 1024  # 10MB
    allowed_file_types: list = None
    default_enhancement_method: str = "adaptive"
    grid_size_default: int = 10
    grid_size_max: int = 50
    
    def __post_init__(self):
        if self.allowed_file_types is None:
            self.allowed_file_types = ['.geojson', '.json']
    
    @classmethod
    def from_env(cls) -> 'ProcessingConfig':
        """Charger la configuration depuis les variables d'environnement"""
        return cls(
            max_image_size=int(os.getenv('MAX_IMAGE_SIZE', str(10 * 1024 * 1024))),
            default_enhancement_method=os.getenv('DEFAULT_ENHANCEMENT_METHOD', 'adaptive'),
            grid_size_default=int(os.getenv('GRID_SIZE_DEFAULT', '10')),
            grid_size_max=int(os.getenv('GRID_SIZE_MAX', '50'))
        )

class AppConfig:
    """Configuration principale de l'application"""
    
    def __init__(self):
        self.sentinelhub = SentinelHubConfig.from_env()
        self.redis = RedisConfig.from_env()
        self.api = APIConfig.from_env()
        self.processing = ProcessingConfig.from_env()
    
    def validate(self) -> list:
        """Valider la configuration et retourner les erreurs"""
        errors = []
        
        if not self.sentinelhub.is_configured():
            errors.append("Configuration SentinelHub incomplète")
        
        if self.api.port < 1 or self.api.port > 65535:
            errors.append("Port API invalide")
        
        if self.processing.max_image_size <= 0:
            errors.append("Taille maximale d'image invalide")
        
        return errors
    
    def to_dict(self) -> dict:
        """Convertir la configuration en dictionnaire"""
        return {
            "sentinelhub": {
                "configured": self.sentinelhub.is_configured(),
                "instance_id_set": bool(self.sentinelhub.instance_id)
            },
            "redis": {
                "url": self.redis.url,
                "max_connections": self.redis.max_connections
            },
            "api": {
                "host": self.api.host,
                "port": self.api.port,
                "debug": self.api.debug,
                "rate_limit": self.api.rate_limit_per_minute
            },
            "processing": {
                "max_image_size": self.processing.max_image_size,
                "default_enhancement": self.processing.default_enhancement_method,
                "grid_size_limits": {
                    "default": self.processing.grid_size_default,
                    "max": self.processing.grid_size_max
                }
            }
        }

app_config = AppConfig()

def get_config() -> AppConfig:
    """Obtenir l'instance de configuration"""
    return app_config

def load_env_file(env_file: str = ".env") -> None:
    """Charger un fichier .env"""
    env_path = Path(env_file)
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
