#!/usr/bin/env python3
"""Performance optimizations for the satellite API"""

import redis
import numpy as np
import asyncio
import hashlib
import json
import logging
from typing import Optional, Dict, Any, List
from functools import wraps
import pickle
import os

logger = logging.getLogger(__name__)

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
CACHE_TTL = int(os.getenv("CACHE_TTL", 3600))  # 1 hour default

try:
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=False)
    redis_client.ping()
    REDIS_AVAILABLE = True
    logger.info("Redis cache disponible")
except Exception as e:
    REDIS_AVAILABLE = False
    logger.warning(f"Redis non disponible: {e}")

def cache_key_generator(geometry_coords: List, date_range: Optional[tuple] = None) -> str:
    """Générer une clé de cache unique pour les requêtes SentinelHub"""
    key_data = {
        "geometry": geometry_coords,
        "date_range": date_range
    }
    key_string = json.dumps(key_data, sort_keys=True)
    return f"sentinelhub:{hashlib.md5(key_string.encode()).hexdigest()}"

def cache_sentinelhub_request(ttl: int = CACHE_TTL):
    """Décorateur pour mettre en cache les requêtes SentinelHub"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not REDIS_AVAILABLE:
                return await func(*args, **kwargs)
            
            cache_key = cache_key_generator(args[0].geometry if hasattr(args[0], 'geometry') else str(args))
            
            try:
                cached_result = redis_client.get(cache_key)
                if cached_result:
                    logger.info(f"Cache hit pour {cache_key}")
                    return pickle.loads(cached_result)
                
                result = await func(*args, **kwargs)
                redis_client.setex(cache_key, ttl, pickle.dumps(result))
                logger.info(f"Résultat mis en cache pour {cache_key}")
                return result
                
            except Exception as e:
                logger.error(f"Erreur de cache: {e}")
                return await func(*args, **kwargs)
        
        return wrapper
    return decorator

def vectorized_vegetation_indices(image: np.ndarray, indices: List[str]) -> Dict[str, np.ndarray]:
    """Calcul vectorisé optimisé des indices de végétation"""
    if image.shape[-1] < 4:
        raise ValueError("L'image doit avoir au moins 4 bandes (B, G, R, NIR)")
    
    blue = image[:, :, 0].astype(np.float32)
    green = image[:, :, 1].astype(np.float32)
    red = image[:, :, 2].astype(np.float32)
    nir = image[:, :, 3].astype(np.float32)
    
    valid_mask = ~(np.isnan(blue) | np.isnan(green) | np.isnan(red) | np.isnan(nir))
    
    results = {}
    
    if "ndvi" in indices:
        denominator = nir + red
        ndvi = np.full_like(nir, np.nan)
        valid_denom = valid_mask & (denominator != 0)
        ndvi[valid_denom] = (nir[valid_denom] - red[valid_denom]) / denominator[valid_denom]
        results["ndvi"] = np.clip(ndvi, -1, 1)
    
    if "ndwi" in indices:
        denominator = nir + green
        ndwi = np.full_like(nir, np.nan)
        valid_denom = valid_mask & (denominator != 0)
        ndwi[valid_denom] = (green[valid_denom] - nir[valid_denom]) / denominator[valid_denom]
        results["ndwi"] = np.clip(ndwi, -1, 1)
    
    if "savi" in indices:
        L = 0.5  # Facteur de correction du sol
        denominator = nir + red + L
        savi = np.full_like(nir, np.nan)
        valid_denom = valid_mask & (denominator != 0)
        savi[valid_denom] = ((nir[valid_denom] - red[valid_denom]) * (1 + L)) / denominator[valid_denom]
        results["savi"] = np.clip(savi, -1, 1)
    
    if "evi" in indices:
        C1, C2, L = 6.0, 7.5, 1.0
        denominator = nir + C1 * red - C2 * blue + L
        evi = np.full_like(nir, np.nan)
        valid_denom = valid_mask & (denominator != 0)
        evi[valid_denom] = 2.5 * (nir[valid_denom] - red[valid_denom]) / denominator[valid_denom]
        results["evi"] = np.clip(evi, -1, 1)
    
    return results

async def async_time_series_processing(geometry, config, date_ranges: List[tuple], indices: List[str]) -> List[Dict]:
    """Traitement asynchrone des séries temporelles"""
    async def process_single_date(date_range):
        try:
            from services.satellite_service import get_single_image
            
            image = await asyncio.to_thread(get_single_image, geometry, config, date_range)
            indices_data = vectorized_vegetation_indices(image, indices)
            
            return {
                "date_range": date_range,
                "indices": {idx: float(np.nanmean(data)) for idx, data in indices_data.items()},
                "image_shape": image.shape
            }
        except Exception as e:
            logger.error(f"Erreur pour la période {date_range}: {e}")
            return {
                "date_range": date_range,
                "error": str(e)
            }
    
    semaphore = asyncio.Semaphore(3)  # Max 3 requêtes simultanées
    
    async def limited_process(date_range):
        async with semaphore:
            return await process_single_date(date_range)
    
    tasks = [limited_process(date_range) for date_range in date_ranges]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    return [r for r in results if not isinstance(r, Exception)]

def memory_efficient_image_processing(image: np.ndarray, chunk_size: int = 1024) -> np.ndarray:
    """Traitement d'image optimisé en mémoire pour les grandes images"""
    if image.size < chunk_size * chunk_size * image.shape[-1]:
        return image
    
    height, width = image.shape[:2]
    processed_chunks = []
    
    for i in range(0, height, chunk_size):
        row_chunks = []
        for j in range(0, width, chunk_size):
            chunk = image[i:i+chunk_size, j:j+chunk_size]
            processed_chunk = np.clip(chunk.astype(np.float32) / 10000.0, 0, 1)
            row_chunks.append(processed_chunk)
        processed_chunks.append(np.concatenate(row_chunks, axis=1))
    
    return np.concatenate(processed_chunks, axis=0)

def clear_cache(pattern: str = "sentinelhub:*"):
    """Nettoyer le cache Redis"""
    if not REDIS_AVAILABLE:
        return False
    
    try:
        keys = redis_client.keys(pattern)
        if keys:
            redis_client.delete(*keys)
            logger.info(f"Cache nettoyé: {len(keys)} clés supprimées")
        return True
    except Exception as e:
        logger.error(f"Erreur lors du nettoyage du cache: {e}")
        return False

def get_cache_stats() -> Dict[str, Any]:
    """Obtenir les statistiques du cache"""
    if not REDIS_AVAILABLE:
        return {"available": False}
    
    try:
        info = redis_client.info()
        keys_count = len(redis_client.keys("sentinelhub:*"))
        
        return {
            "available": True,
            "keys_count": keys_count,
            "memory_usage": info.get("used_memory_human", "N/A"),
            "hits": info.get("keyspace_hits", 0),
            "misses": info.get("keyspace_misses", 0)
        }
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des stats: {e}")
        return {"available": False, "error": str(e)}
