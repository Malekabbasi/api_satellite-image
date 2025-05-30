import os
import io
import json
import base64
import logging
import uvicorn
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict, Any
import matplotlib
matplotlib.use('Agg')
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import geopandas as gpd
import numpy as np
from sentinelhub import SHConfig, Geometry, CRS, SentinelHubRequest, DataCollection, MimeType
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
import warnings
warnings.filterwarnings('ignore')

from scipy.ndimage import gaussian_filter, median_filter, uniform_filter
from scipy.interpolate import griddata
from matplotlib.patches import Rectangle
from services.indices_service import calculate_vegetation_indices_improved, calculate_masked_average
from services.satellite_service import get_sentinelhub_config, get_time_series_images, get_single_image
from services.visualization_service import generate_enhanced_visualization, create_timelapse_animation, create_time_series_plot, s2_to_rgb
from services.enhancement_service import advanced_pixel_enhancement


try:
    import cv2
    CV2_AVAILABLE = True
    logging.info("OpenCV disponible pour l'amélioration avancée")
except ImportError:
    CV2_AVAILABLE = False
    logging.warning("OpenCV non disponible, certaines améliorations seront limitées")

try:
    import skimage
    from skimage import filters, restoration, morphology
    from skimage.segmentation import slic
    from skimage.transform import resize
    SKIMAGE_AVAILABLE = True
    logging.info("scikit-image disponible pour l'amélioration avancée")
except ImportError:
    SKIMAGE_AVAILABLE = False
    logging.warning("scikit-image non disponible, certaines améliorations seront limitées")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="API Indices de Végétation Satellite - Version Améliorée avec Séries Temporelles",
    description="Affichage ultra-clair des indices de végétation avec amélioration des pixels et animations temporelles",
    version="3.0.0"
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def validate_geojson(geojson_data: dict) -> bool:
    """Validation basique du GeoJSON"""
    try:
        if geojson_data.get("type") not in ["Feature", "FeatureCollection"]:
            return False
        if "geometry" not in geojson_data and "features" not in geojson_data:
            return False
        return True
    except Exception:
        return False

def create_mask_from_geometry(geometry: dict, image_shape: tuple, 
                            bounds: Optional[tuple] = None) -> np.ndarray:
    """Créer un masque à partir de la géométrie GeoJSON"""
    try:
        mask = np.ones(image_shape[:2], dtype=bool)
        return mask
    except Exception as e:
        logger.error(f"Erreur lors de la création du masque: {e}")
        return np.ones(image_shape[:2], dtype=bool)

def transparent_cmap(cmap_name: str, alpha: float = 0.7):
    """Créer une colormap avec transparence"""
    cmap = plt.cm.get_cmap(cmap_name)
    colors = cmap(np.arange(cmap.N))
    colors[:, -1] = alpha
    return mcolors.ListedColormap(colors)

def simple_vegetation_segmentation(ndvi: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    """Segmentation simple de la végétation basée sur NDVI"""
    return ndvi > threshold

@app.get("/health")
async def health_check():
    """Vérification de l'état de l'API"""
    try:
        config = get_sentinelhub_config()
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "sentinelhub_configured": bool(config.instance_id and config.sh_client_id),
            "opencv_available": CV2_AVAILABLE,
            "skimage_available": SKIMAGE_AVAILABLE,
            "version": "3.0.0"
        }
    except Exception as e:
        logger.error(f"Erreur lors de la vérification de santé: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur de configuration: {str(e)}")

@app.get("/")
async def api_info():
    """Informations sur l'API"""
    return {
        "name": "API Indices de Végétation Satellite",
        "version": "3.0.0",
        "description": "Calcul et visualisation d'indices de végétation avec amélioration des pixels",
        "endpoints": {
            "/calculate-indices": "Calcul des indices de végétation",
            "/time-series": "Analyse de séries temporelles",
            "/enhancement-methods": "Liste des méthodes d'amélioration disponibles",
            "/health": "État de l'API"
        },
        "supported_indices": ["NDVI", "NDWI", "SAVI", "EVI"],
        "enhancement_methods": ["adaptive", "super_resolution", "edge_preserving", "bilateral", "segmentation_based", "gaussian"]
    }

@app.post("/calculate-indices")
@limiter.limit("10/minute")
async def calculate_indices(
    request: Request,
    geojson_file: UploadFile = File(...),
    indices: str = Query("ndvi,ndwi", description="Indices à calculer (séparés par des virgules)"),
    enhancement_method: str = Query("adaptive", description="Méthode d'amélioration des pixels")
):
    """Calculer les indices de végétation pour une zone donnée"""
    try:
        geojson_content = await geojson_file.read()
        geojson_data = json.loads(geojson_content.decode('utf-8'))
        
        if not validate_geojson(geojson_data):
            raise HTTPException(status_code=400, detail="Format GeoJSON invalide")
        
        config = get_sentinelhub_config()
        
        if geojson_data["type"] == "Feature":
            geometry_coords = geojson_data["geometry"]["coordinates"]
        else:
            geometry_coords = geojson_data["features"][0]["geometry"]["coordinates"]
        
        geometry = Geometry(geometry_coords, CRS.WGS84)
        
        image = get_single_image(geometry, config)
        
        indices_list = [idx.strip().lower() for idx in indices.split(",")]
        results = {}
        
        for index_name in indices_list:
            if index_name in ["ndvi", "ndwi", "savi", "evi"]:
                index_array = calculate_vegetation_indices_improved(image, index_name)
                
                if enhancement_method != "none":
                    index_array = advanced_pixel_enhancement(index_array, enhancement_method)
                
                mask = create_mask_from_geometry(geojson_data["geometry"] if geojson_data["type"] == "Feature" 
                                               else geojson_data["features"][0]["geometry"], 
                                               image.shape, None)
                
                masked_avg = calculate_masked_average(index_array, mask)
                
                visualization = generate_enhanced_visualization(
                    index_array, index_name, mask, enhancement_method
                )
                
                results[index_name] = {
                    "average_value": float(masked_avg),
                    "min_value": float(np.nanmin(index_array)),
                    "max_value": float(np.nanmax(index_array)),
                    "std_value": float(np.nanstd(index_array)),
                    "visualization": visualization,
                    "enhancement_method": enhancement_method
                }
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "indices": results,
            "metadata": {
                "image_shape": image.shape,
                "enhancement_available": CV2_AVAILABLE or SKIMAGE_AVAILABLE
            }
        }
        
    except Exception as e:
        logger.error(f"Erreur lors du calcul des indices: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors du calcul: {str(e)}")

@app.get("/enhancement-methods")
async def get_enhancement_methods():
    """Obtenir la liste des méthodes d'amélioration disponibles"""
    methods = {
        "adaptive": {
            "name": "Amélioration Adaptive",
            "description": "Amélioration basée sur les caractéristiques locales de l'image",
            "available": True
        },
        "super_resolution": {
            "name": "Super-résolution",
            "description": "Augmentation de la résolution avec interpolation avancée",
            "available": SKIMAGE_AVAILABLE
        },
        "edge_preserving": {
            "name": "Préservation des contours",
            "description": "Lissage tout en préservant les contours importants",
            "available": CV2_AVAILABLE
        },
        "bilateral": {
            "name": "Filtrage bilatéral",
            "description": "Réduction du bruit tout en préservant les détails",
            "available": CV2_AVAILABLE or SKIMAGE_AVAILABLE
        },
        "segmentation_based": {
            "name": "Basé sur la segmentation",
            "description": "Amélioration basée sur la segmentation de l'image",
            "available": SKIMAGE_AVAILABLE
        },
        "gaussian": {
            "name": "Lissage gaussien",
            "description": "Lissage simple avec filtre gaussien",
            "available": True
        }
    }
    
    return {
        "methods": methods,
        "opencv_available": CV2_AVAILABLE,
        "skimage_available": SKIMAGE_AVAILABLE,
        "recommended": "adaptive" if not (CV2_AVAILABLE or SKIMAGE_AVAILABLE) else "super_resolution"
    }

@app.get("/health/detailed")
async def detailed_health_check():
    """Vérification détaillée de l'état du système"""
    try:
        config = get_sentinelhub_config()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "sentinelhub": {
                    "configured": bool(config.instance_id and config.sh_client_id),
                    "instance_id_set": bool(config.instance_id),
                    "credentials_set": bool(config.sh_client_id and config.sh_client_secret)
                },
                "image_processing": {
                    "opencv_available": CV2_AVAILABLE,
                    "skimage_available": SKIMAGE_AVAILABLE,
                    "matplotlib_backend": matplotlib.get_backend()
                },
                "api": {
                    "version": "3.0.0",
                    "rate_limiting": True,
                    "cors_enabled": True
                }
            }
        }
    except Exception as e:
        logger.error(f"Erreur lors de la vérification détaillée: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur système: {str(e)}")

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"Démarrage de l'API sur {host}:{port}")
    logger.info(f"OpenCV disponible: {CV2_AVAILABLE}")
    logger.info(f"scikit-image disponible: {SKIMAGE_AVAILABLE}")
    
    uvicorn.run(
        app, 
        host=host, 
        port=port,
        log_level="info",
        access_log=True
    )
