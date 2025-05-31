"""Service for pixel enhancement methods"""
import numpy as np
import logging
from scipy.ndimage import generic_filter, gaussian_filter
from scipy.interpolate import griddata
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV non disponible, certaines améliorations seront limitées")

try:
    from skimage import filters, restoration, morphology
    from skimage.segmentation import slic
    from skimage.transform import resize
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    logger.warning("scikit-image non disponible, certaines améliorations seront limitées")

def advanced_pixel_enhancement(array: np.ndarray, mask: np.ndarray, enhancement_type: str = 'adaptive') -> np.ndarray:
    """
    Amélioration avancée des pixels avec plusieurs techniques
    """
    enhanced_array = np.where(mask, array, np.nan)
    
    if enhancement_type == 'adaptive':
        enhanced_array = adaptive_smoothing(enhanced_array)
    elif enhancement_type == 'super_resolution':
        enhanced_array = super_resolution_upscale(enhanced_array)
    elif enhancement_type == 'edge_preserving' and CV2_AVAILABLE:
        enhanced_array = edge_preserving_filter(enhanced_array)
    elif enhancement_type == 'bilateral':
        enhanced_array = bilateral_filter_2d(enhanced_array)
    elif enhancement_type == 'segmentation_based' and SKIMAGE_AVAILABLE:
        enhanced_array = segmentation_based_enhancement(enhanced_array)
    else:
        enhanced_array = enhanced_gaussian_filter(enhanced_array)
    
    return enhanced_array

def adaptive_smoothing(array: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Lissage adaptatif basé sur la variance locale"""
    
    def adaptive_mean(values):
        valid_values = values[~np.isnan(values)]
        if len(valid_values) < 3:
            return np.nanmean(values)
        
        variance = np.var(valid_values)
        
        if variance < 0.01:  # Zone homogène
            return np.nanmean(values)
        else:  # Zone avec détails
            return np.nanmedian(values)
    
    enhanced = generic_filter(array, adaptive_mean, size=window_size, mode='constant', cval=np.nan)
    return enhanced

def super_resolution_upscale(array: np.ndarray, scale_factor: int = 2) -> np.ndarray:
    """Super-résolution par interpolation bicubique avancée"""
    h, w = array.shape
    
    y_orig, x_orig = np.mgrid[0:h, 0:w]
    
    h_new, w_new = h * scale_factor, w * scale_factor
    y_new = np.linspace(0, h-1, h_new)
    x_new = np.linspace(0, w-1, w_new)
    x_new_grid, y_new_grid = np.meshgrid(x_new, y_new)
    
    valid_mask = ~np.isnan(array)
    
    if np.sum(valid_mask) < 4:
        return array
    
    try:
        points = np.column_stack((y_orig[valid_mask], x_orig[valid_mask]))
        values = array[valid_mask]
        
        enhanced = griddata(points, values, (y_new_grid, x_new_grid), method='cubic', fill_value=np.nan)
        
        nan_mask = np.isnan(enhanced)
        if np.any(nan_mask):
            enhanced_linear = griddata(points, values, (y_new_grid, x_new_grid), method='linear', fill_value=np.nan)
            enhanced[nan_mask] = enhanced_linear[nan_mask]
        
        if SKIMAGE_AVAILABLE:
            enhanced_downsampled = resize(enhanced, (h, w), order=3, mode='constant', cval=0, anti_aliasing=True)
        else:
            enhanced_downsampled = gaussian_filter(array, sigma=0.8)
        
        return enhanced_downsampled
        
    except Exception as e:
        logger.warning(f"Super-résolution échouée: {e}")
        return gaussian_filter(array, sigma=0.8)

def edge_preserving_filter(array: np.ndarray) -> np.ndarray:
    """Filtre préservant les contours avec OpenCV"""
    if not CV2_AVAILABLE:
        logger.warning("OpenCV non disponible, utilisation du lissage gaussien")
        return enhanced_gaussian_filter(array)
    
    try:
        array_norm = np.nan_to_num(array, nan=0.0)
        
        array_uint8 = ((array_norm - np.nanmin(array_norm)) / 
                      (np.nanmax(array_norm) - np.nanmin(array_norm)) * 255).astype(np.uint8)
        
        enhanced_uint8 = cv2.edgePreservingFilter(array_uint8, flags=1, sigma_s=50, sigma_r=0.4)
        
        enhanced = enhanced_uint8.astype(np.float64) / 255.0
        enhanced = enhanced * (np.nanmax(array_norm) - np.nanmin(array_norm)) + np.nanmin(array_norm)
        
        enhanced = np.where(np.isnan(array), np.nan, enhanced)
        
        return enhanced
        
    except Exception as e:
        logger.warning(f"Filtre edge-preserving échoué: {e}")
        return enhanced_gaussian_filter(array)

def bilateral_filter_2d(array: np.ndarray) -> np.ndarray:
    """Filtre bilatéral 2D pour réduction du bruit"""
    try:
        median_val = np.nanmedian(array)
        array_filled = np.where(np.isnan(array), median_val, array)
        
        enhanced = gaussian_filter(array_filled, sigma=1.0)
        
        enhanced = np.where(np.isnan(array), np.nan, enhanced)
        
        return enhanced
        
    except Exception as e:
        logger.warning(f"Filtre bilatéral échoué: {e}")
        return array

def segmentation_based_enhancement(array: np.ndarray) -> np.ndarray:
    """Amélioration basée sur la segmentation SLIC"""
    if not SKIMAGE_AVAILABLE:
        logger.warning("scikit-image non disponible, utilisation du lissage gaussien")
        return enhanced_gaussian_filter(array)
    
    try:
        array_norm = np.nan_to_num(array, nan=0.0)
        
        array_min, array_max = np.nanmin(array_norm), np.nanmax(array_norm)
        if array_max > array_min:
            array_normalized = (array_norm - array_min) / (array_max - array_min)
        else:
            return array
        
        image_3d = np.stack([array_normalized] * 3, axis=-1)
        
        segments = slic(image_3d, n_segments=100, compactness=10, sigma=1)
        
        enhanced = array_normalized.copy()
        for segment_id in np.unique(segments):
            mask = segments == segment_id
            segment_values = array_normalized[mask]
            
            if len(segment_values) > 0:
                segment_mean = np.mean(segment_values)
                enhanced[mask] = segment_mean
        
        enhanced = enhanced * (array_max - array_min) + array_min
        
        enhanced = np.where(np.isnan(array), np.nan, enhanced)
        
        return enhanced
        
    except Exception as e:
        logger.warning(f"Segmentation SLIC échouée: {e}")
        return enhanced_gaussian_filter(array)

def enhanced_gaussian_filter(array: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Filtre gaussien amélioré avec gestion des NaN"""
    try:
        valid_mask = ~np.isnan(array)
        
        if np.sum(valid_mask) == 0:
            return array
        
        median_val = np.nanmedian(array)
        array_filled = np.where(valid_mask, array, median_val)
        
        enhanced = gaussian_filter(array_filled, sigma=sigma)
        
        enhanced = np.where(valid_mask, enhanced, np.nan)
        
        return enhanced
        
    except Exception as e:
        logger.warning(f"Filtre gaussien échoué: {e}")
        return array
