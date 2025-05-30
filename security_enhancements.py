#!/usr/bin/env python3
"""Security enhancements for the satellite API"""

import os
import json
try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False
from typing import Dict, Any
from fastapi import HTTPException, UploadFile
from models.validation import GeoJSONValidation, IndicesRequest
from dotenv import load_dotenv

load_dotenv()

def validate_geojson_file(file_content: bytes) -> bool:
    """Validation stricte du fichier GeoJSON avec vérification du type MIME"""
    try:
        if MAGIC_AVAILABLE:
            mime_type = magic.from_buffer(file_content, mime=True)
            if mime_type not in ['application/json', 'text/plain', 'application/geo+json']:
                return False
        
        if len(file_content) > 10 * 1024 * 1024:
            return False
            
        json_str = file_content.decode('utf-8')
        GeoJSONValidation.from_json_string(json_str)
        return True
    except Exception:
        return False

async def validate_upload_file(geojson_file: UploadFile) -> Dict[str, Any]:
    """Valider un fichier uploadé avec sécurité renforcée"""
    if not geojson_file.filename.endswith(('.json', '.geojson')):
        raise HTTPException(status_code=400, detail="Le fichier doit être un fichier .json ou .geojson")
    
    geojson_content = await geojson_file.read()
    
    if not validate_geojson_file(geojson_content):
        raise HTTPException(status_code=400, detail="Fichier GeoJSON invalide ou trop volumineux")
    
    geojson_validation = GeoJSONValidation.from_json_string(geojson_content.decode('utf-8'))
    return geojson_validation.data.dict()

def validate_indices_request(indices: str, enhancement_method: str) -> IndicesRequest:
    """Valider les paramètres de requête avec Pydantic"""
    return IndicesRequest(indices=indices.split(","), enhancement_method=enhancement_method)

def get_secure_sentinelhub_config():
    """Configuration sécurisée de SentinelHub avec variables d'environnement"""
    from sentinelhub import SHConfig
    
    config = SHConfig()
    config.instance_id = os.getenv("SENTINELHUB_INSTANCE_ID")
    config.sh_client_id = os.getenv("SENTINELHUB_CLIENT_ID") 
    config.sh_client_secret = os.getenv("SENTINELHUB_CLIENT_SECRET")
    
    if not all([config.instance_id, config.sh_client_id, config.sh_client_secret]):
        raise ValueError("Identifiants SentinelHub non configurés dans les variables d'environnement")
    
    return config
