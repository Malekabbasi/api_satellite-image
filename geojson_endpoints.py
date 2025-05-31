#!/usr/bin/env python3
"""GeoJSON endpoints for Flutter map integration"""

import json
import numpy as np
from typing import Dict, List, Any, Optional
from fastapi import HTTPException
from shapely.geometry import Polygon, Point, mapping
from shapely.ops import transform
import geopandas as gpd
from pyproj import Transformer
import logging

logger = logging.getLogger(__name__)

def create_vegetation_mask_polygons(
    index_array: np.ndarray, 
    bounds: tuple, 
    threshold_low: float = 0.2, 
    threshold_high: float = 0.6,
    index_name: str = "ndvi"
) -> List[Dict[str, Any]]:
    """Créer des polygones GeoJSON pour les masques de végétation"""
    
    height, width = index_array.shape
    lon_min, lat_min, lon_max, lat_max = bounds
    
    lon_res = (lon_max - lon_min) / width
    lat_res = (lat_max - lat_min) / height
    
    features = []
    
    classifications = [
        {"name": "low_vegetation", "min": -1.0, "max": threshold_low, "color": "#ffeb3b"},
        {"name": "medium_vegetation", "min": threshold_low, "max": threshold_high, "color": "#8bc34a"},
        {"name": "high_vegetation", "min": threshold_high, "max": 1.0, "color": "#2e7d32"}
    ]
    
    for classification in classifications:
        mask = (index_array >= classification["min"]) & (index_array < classification["max"])
        
        if np.any(mask):
            contours = find_contours_simplified(mask, lon_min, lat_min, lon_res, lat_res)
            
            for contour in contours:
                if len(contour) >= 4:  # Minimum pour un polygone valide
                    feature = {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [contour]
                        },
                        "properties": {
                            "vegetation_level": classification["name"],
                            "index_type": index_name,
                            "color": classification["color"],
                            "min_value": float(classification["min"]),
                            "max_value": float(classification["max"]),
                            "mean_value": float(np.mean(index_array[mask])),
                            "pixel_count": int(np.sum(mask))
                        }
                    }
                    features.append(feature)
    
    return features

def find_contours_simplified(mask: np.ndarray, lon_min: float, lat_min: float, 
                           lon_res: float, lat_res: float) -> List[List[List[float]]]:
    """Trouver les contours simplifiés d'un masque binaire"""
    contours = []
    
    height, width = mask.shape
    
    visited = np.zeros_like(mask, dtype=bool)
    
    for i in range(height):
        for j in range(width):
            if mask[i, j] and not visited[i, j]:
                min_i, max_i = i, i
                min_j, max_j = j, j
                
                while max_j + 1 < width and mask[i, max_j + 1]:
                    max_j += 1
                
                while max_i + 1 < height and all(mask[max_i + 1, jj] for jj in range(min_j, max_j + 1)):
                    max_i += 1
                
                visited[min_i:max_i+1, min_j:max_j+1] = True
                
                lon1 = lon_min + min_j * lon_res
                lat1 = lat_min + (height - max_i - 1) * lat_res
                lon2 = lon_min + (max_j + 1) * lon_res
                lat2 = lat_min + (height - min_i) * lat_res
                
                contour = [
                    [lon1, lat1],
                    [lon2, lat1],
                    [lon2, lat2],
                    [lon1, lat2],
                    [lon1, lat1]  # Fermer le polygone
                ]
                contours.append(contour)
    
    return contours

def create_indices_geojson_features(
    indices_data: Dict[str, np.ndarray],
    bounds: tuple,
    grid_size: int = 10
) -> List[Dict[str, Any]]:
    """Créer des features GeoJSON avec valeurs d'indices sur une grille"""
    
    lon_min, lat_min, lon_max, lat_max = bounds
    
    lon_step = (lon_max - lon_min) / grid_size
    lat_step = (lat_max - lat_min) / grid_size
    
    features = []
    
    for i in range(grid_size):
        for j in range(grid_size):
            lon = lon_min + j * lon_step + lon_step / 2
            lat = lat_min + i * lat_step + lat_step / 2
            
            height, width = list(indices_data.values())[0].shape
            pixel_i = int((lat_max - lat) / (lat_max - lat_min) * height)
            pixel_j = int((lon - lon_min) / (lon_max - lon_min) * width)
            
            if 0 <= pixel_i < height and 0 <= pixel_j < width:
                properties = {
                    "grid_id": f"{i}_{j}",
                    "longitude": lon,
                    "latitude": lat
                }
                
                for index_name, index_array in indices_data.items():
                    value = index_array[pixel_i, pixel_j]
                    if not np.isnan(value):
                        properties[f"{index_name}_value"] = float(value)
                        properties[f"{index_name}_class"] = classify_index_value(value, index_name)
                
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [lon, lat]
                    },
                    "properties": properties
                }
                features.append(feature)
    
    return features

def classify_index_value(value: float, index_name: str) -> str:
    """Classifier une valeur d'indice en catégorie"""
    if index_name.lower() == "ndvi":
        if value < 0.2:
            return "no_vegetation"
        elif value < 0.4:
            return "sparse_vegetation"
        elif value < 0.6:
            return "moderate_vegetation"
        else:
            return "dense_vegetation"
    elif index_name.lower() == "ndwi":
        if value < -0.3:
            return "dry"
        elif value < 0.0:
            return "moderate_moisture"
        else:
            return "high_moisture"
    elif index_name.lower() in ["savi", "evi"]:
        if value < 0.1:
            return "no_vegetation"
        elif value < 0.3:
            return "sparse_vegetation"
        elif value < 0.5:
            return "moderate_vegetation"
        else:
            return "dense_vegetation"
    else:
        return "unknown"

def get_image_bounds(geometry_coords: List) -> tuple:
    """Extraire les limites géographiques d'une géométrie"""
    try:
        all_coords = []
        
        def extract_coords(coords):
            if isinstance(coords[0], (int, float)):
                all_coords.append(coords)
            else:
                for coord in coords:
                    extract_coords(coord)
        
        extract_coords(geometry_coords)
        
        if not all_coords:
            raise ValueError("Aucune coordonnée trouvée")
        
        lons = [coord[0] for coord in all_coords]
        lats = [coord[1] for coord in all_coords]
        
        return (min(lons), min(lats), max(lons), max(lats))
        
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction des limites: {e}")
        return (0.0, 0.0, 1.0, 1.0)

def validate_geojson_response(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Valider et formater la réponse GeoJSON"""
    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "feature_count": len(features),
            "coordinate_system": "WGS84",
            "generated_at": "2025-05-30T03:50:00Z"
        }
    }
