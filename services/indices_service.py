"""Service for vegetation indices calculation"""
import numpy as np
import logging
from typing import Dict

logger = logging.getLogger(__name__)

def calculate_vegetation_indices_improved(bands_img: np.ndarray) -> Dict[str, np.ndarray]:
    """Calcul amélioré des indices de végétation avec gestion des erreurs"""
    logger.info(f"Calcul des indices pour une image de taille: {bands_img.shape}")
    
    blue = bands_img[:, :, 0].astype('float64')
    green = bands_img[:, :, 1].astype('float64') 
    red = bands_img[:, :, 2].astype('float64')
    nir = bands_img[:, :, 3].astype('float64')
    
    if np.max(bands_img) > 1.5:
        logger.info("Normalisation des bandes detectée nécessaire")
        blue = blue / 10000.0
        green = green / 10000.0
        red = red / 10000.0
        nir = nir / 10000.0
    
    indices = {}
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = (nir - red) / (nir + red)
        ndvi = np.clip(ndvi, -1, 1)
        ndvi = np.where(np.isfinite(ndvi), ndvi, np.nan)
        indices['ndvi'] = ndvi
        
        ndwi = (nir - green) / (nir + green)
        ndwi = np.clip(ndwi, -1, 1)
        ndwi = np.where(np.isfinite(ndwi), ndwi, np.nan)
        indices['ndwi'] = ndwi
        
        savi = ((nir - red) / (nir + red + 0.5)) * 1.5
        savi = np.clip(savi, -1, 1)
        savi = np.where(np.isfinite(savi), savi, np.nan)
        indices['savi'] = savi
        
        evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)
        evi = np.clip(evi, -1, 1)
        evi = np.where(np.isfinite(evi), evi, np.nan)
        indices['evi'] = evi
    
    logger.info(f"Indices calculés: {list(indices.keys())}")
    return indices

def calculate_masked_average(array: np.ndarray, mask: np.ndarray) -> float:
    """Calcule la moyenne d'un array en appliquant un masque"""
    masked_array = array[mask]
    valid_values = masked_array[np.isfinite(masked_array)]
    return float(np.mean(valid_values)) if len(valid_values) > 0 else np.nan
