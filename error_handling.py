#!/usr/bin/env python3
"""Enhanced error handling and logging for satellite API"""

import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
import traceback

logger = logging.getLogger(__name__)

class CorrelationIDMiddleware:
    """Middleware pour ajouter des IDs de corrélation aux requêtes"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            correlation_id = str(uuid.uuid4())
            scope["correlation_id"] = correlation_id
            
            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    headers = dict(message.get("headers", []))
                    headers[b"x-correlation-id"] = correlation_id.encode()
                    message["headers"] = list(headers.items())
                await send(message)
            
            await self.app(scope, receive, send_wrapper)
        else:
            await self.app(scope, receive, send)

def get_correlation_id(request: Request) -> str:
    """Obtenir l'ID de corrélation de la requête"""
    return getattr(request.scope, "correlation_id", str(uuid.uuid4()))

class StructuredError:
    """Classe pour les erreurs structurées"""
    
    def __init__(
        self,
        error_code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        status_code: int = 500
    ):
        self.error_code = error_code
        self.message = message
        self.details = details or {}
        self.status_code = status_code
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self, correlation_id: str) -> Dict[str, Any]:
        """Convertir en dictionnaire pour la réponse JSON"""
        return {
            "error": {
                "code": self.error_code,
                "message": self.message,
                "details": self.details,
                "timestamp": self.timestamp,
                "correlation_id": correlation_id
            }
        }

def create_error_response(
    request: Request,
    error_code: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    status_code: int = 500
) -> JSONResponse:
    """Créer une réponse d'erreur structurée"""
    correlation_id = get_correlation_id(request)
    
    structured_error = StructuredError(
        error_code=error_code,
        message=message,
        details=details,
        status_code=status_code
    )
    
    logger.error(
        f"API Error [{correlation_id}]: {error_code} - {message}",
        extra={
            "correlation_id": correlation_id,
            "error_code": error_code,
            "details": details,
            "status_code": status_code
        }
    )
    
    return JSONResponse(
        status_code=status_code,
        content=structured_error.to_dict(correlation_id)
    )

def handle_validation_error(request: Request, exc: Exception) -> JSONResponse:
    """Gérer les erreurs de validation"""
    return create_error_response(
        request=request,
        error_code="VALIDATION_ERROR",
        message="Données d'entrée invalides",
        details={"validation_errors": str(exc)},
        status_code=422
    )

def handle_geojson_error(request: Request, exc: Exception) -> JSONResponse:
    """Gérer les erreurs de format GeoJSON"""
    return create_error_response(
        request=request,
        error_code="GEOJSON_FORMAT_ERROR",
        message="Format GeoJSON invalide",
        details={"error": str(exc)},
        status_code=400
    )

def handle_sentinelhub_error(request: Request, exc: Exception) -> JSONResponse:
    """Gérer les erreurs SentinelHub"""
    return create_error_response(
        request=request,
        error_code="SENTINELHUB_ERROR",
        message="Erreur lors de l'accès aux données satellite",
        details={"error": str(exc)},
        status_code=503
    )

def handle_processing_error(request: Request, exc: Exception) -> JSONResponse:
    """Gérer les erreurs de traitement d'image"""
    return create_error_response(
        request=request,
        error_code="PROCESSING_ERROR",
        message="Erreur lors du traitement de l'image",
        details={"error": str(exc)},
        status_code=500
    )

def handle_generic_error(request: Request, exc: Exception) -> JSONResponse:
    """Gérer les erreurs génériques"""
    correlation_id = get_correlation_id(request)
    
    logger.error(
        f"Unhandled error [{correlation_id}]: {str(exc)}",
        extra={
            "correlation_id": correlation_id,
            "traceback": traceback.format_exc()
        }
    )
    
    return create_error_response(
        request=request,
        error_code="INTERNAL_ERROR",
        message="Erreur interne du serveur",
        details={"error": "Une erreur inattendue s'est produite"},
        status_code=500
    )

def setup_enhanced_logging():
    """Configurer le logging amélioré"""
    
    log_format = (
        "%(asctime)s - %(name)s - %(levelname)s - "
        "[%(correlation_id)s] - %(message)s"
    )
    
    class CorrelationFormatter(logging.Formatter):
        def format(self, record):
            if not hasattr(record, 'correlation_id'):
                record.correlation_id = 'N/A'
            return super().format(record)
    
    logger = logging.getLogger("satellite_api")
    logger.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(CorrelationFormatter(log_format))
    
    try:
        file_handler = logging.FileHandler("satellite_api.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(CorrelationFormatter(log_format))
        logger.addHandler(file_handler)
    except Exception:
        pass  # Ignorer si impossible de créer le fichier
    
    logger.addHandler(console_handler)
    
    return logger

enhanced_logger = setup_enhanced_logging()
