"""Pydantic models for data validation"""
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import json

class GeometryType(str, Enum):
    POINT = "Point"
    LINESTRING = "LineString"
    POLYGON = "Polygon"
    MULTIPOINT = "MultiPoint"
    MULTILINESTRING = "MultiLineString"
    MULTIPOLYGON = "MultiPolygon"
    GEOMETRYCOLLECTION = "GeometryCollection"

class Coordinates(BaseModel):
    """Validation pour les coordonnées GeoJSON"""
    coordinates: Union[
        List[float],  # Point
        List[List[float]],  # LineString, MultiPoint
        List[List[List[float]]],  # Polygon, MultiLineString
        List[List[List[List[float]]]]  # MultiPolygon
    ]

class Geometry(BaseModel):
    """Validation pour la géométrie GeoJSON"""
    type: GeometryType
    coordinates: Union[
        List[float],
        List[List[float]],
        List[List[List[float]]],
        List[List[List[List[float]]]]
    ]
    
    @validator('coordinates')
    def validate_coordinates(cls, v, values):
        geom_type = values.get('type')
        if not geom_type:
            return v
            
        if geom_type == GeometryType.POINT:
            if not isinstance(v, list) or len(v) < 2:
                raise ValueError("Point coordinates must be [longitude, latitude]")
        elif geom_type == GeometryType.POLYGON:
            if not isinstance(v, list) or len(v) == 0:
                raise ValueError("Polygon coordinates must be a list of linear rings")
            for ring in v:
                if not isinstance(ring, list) or len(ring) < 4:
                    raise ValueError("Polygon ring must have at least 4 coordinates")
                if ring[0] != ring[-1]:
                    raise ValueError("Polygon ring must be closed (first and last coordinates must be the same)")
        
        return v

class Feature(BaseModel):
    """Validation pour une feature GeoJSON"""
    type: str = Field(..., regex="^Feature$")
    geometry: Geometry
    properties: Optional[Dict[str, Any]] = {}
    
    @validator('properties')
    def validate_properties(cls, v):
        if v is None:
            return {}
        return v

class FeatureCollection(BaseModel):
    """Validation pour une collection de features GeoJSON"""
    type: str = Field(..., regex="^FeatureCollection$")
    features: List[Feature]
    
    @validator('features')
    def validate_features(cls, v):
        if not v:
            raise ValueError("FeatureCollection must contain at least one feature")
        if len(v) > 100:  # Limite de sécurité
            raise ValueError("FeatureCollection cannot contain more than 100 features")
        return v

class GeoJSONValidation(BaseModel):
    """Validation principale pour les données GeoJSON"""
    data: Union[Feature, FeatureCollection]
    
    @classmethod
    def from_json_string(cls, json_str: str):
        """Créer une instance à partir d'une chaîne JSON"""
        try:
            data = json.loads(json_str)
            return cls(data=data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
        except Exception as e:
            raise ValueError(f"Invalid GeoJSON structure: {e}")

class IndicesRequest(BaseModel):
    """Validation pour les requêtes de calcul d'indices"""
    indices: List[str] = Field(..., min_items=1, max_items=10)
    enhancement_method: Optional[str] = Field(default="adaptive", regex="^(adaptive|super_resolution|edge_preserving|bilateral|segmentation_based|gaussian)$")
    
    @validator('indices')
    def validate_indices(cls, v):
        valid_indices = {'ndvi', 'ndwi', 'savi', 'evi'}
        for index in v:
            if index.lower() not in valid_indices:
                raise ValueError(f"Invalid index: {index}. Valid indices are: {', '.join(valid_indices)}")
        return [index.lower() for index in v]

class TimeSeriesRequest(BaseModel):
    """Validation pour les requêtes de séries temporelles"""
    start_date: str = Field(..., regex=r"^\d{4}-\d{2}-\d{2}$")
    end_date: str = Field(..., regex=r"^\d{4}-\d{2}-\d{2}$")
    interval_days: int = Field(default=10, ge=1, le=30)
    indices: List[str] = Field(..., min_items=1, max_items=5)
    
    @validator('indices')
    def validate_indices(cls, v):
        valid_indices = {'ndvi', 'ndwi', 'savi', 'evi'}
        for index in v:
            if index.lower() not in valid_indices:
                raise ValueError(f"Invalid index: {index}. Valid indices are: {', '.join(valid_indices)}")
        return [index.lower() for index in v]
    
    @validator('end_date')
    def validate_date_range(cls, v, values):
        start_date = values.get('start_date')
        if start_date and v <= start_date:
            raise ValueError("end_date must be after start_date")
        return v
