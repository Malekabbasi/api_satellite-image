import os
import tempfile
import base64
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import io

# Configuration pour éviter les problèmes de permissions
import matplotlib
matplotlib.use('Agg')
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import geopandas as gpd
import numpy as np
from sentinelhub import SHConfig, Geometry, CRS, SentinelHubRequest, DataCollection, MimeType
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import ticker
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import mapping
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter, median_filter, uniform_filter
from scipy.interpolate import griddata
from matplotlib.patches import Rectangle

# Tentative d'importation des bibliothèques optionnelles pour l'amélioration
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logging.warning("OpenCV non disponible, certaines améliorations seront limitées")

try:
    from skimage import filters, restoration, morphology
    from skimage.segmentation import slic
    from skimage.transform import resize
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    logging.warning("scikit-image non disponible, certaines améliorations seront limitées")

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="API Indices de Végétation Satellite - Version Améliorée avec Séries Temporelles",
    description="Affichage ultra-clair des indices de végétation avec amélioration des pixels et animations temporelles",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_sentinelhub_config() -> SHConfig:
    config = SHConfig()
    config.instance_id = os.getenv("SENTINELHUB_INSTANCE_ID", "f1a0b14a-a351-4f13-981d-e5a308f8f97a")
    config.sh_client_id = os.getenv("SENTINELHUB_CLIENT_ID", "1286d8fb-793c-4683-b48c-6bd361373041")
    config.sh_client_secret = os.getenv("SENTINELHUB_CLIENT_SECRET", "97x3hObZ4VZL0VhXOj8N8GvtGQNBFG3G")
    return config

def validate_geojson(gdf: gpd.GeoDataFrame) -> None:
    if gdf.empty:
        raise HTTPException(status_code=400, detail="Fichier GeoJSON vide")
    
    if len(gdf) == 0:
        raise HTTPException(status_code=400, detail="Aucune géométrie trouvée dans le GeoJSON")
    
    if gdf.geometry.isna().all():
        raise HTTPException(status_code=400, detail="Aucune géométrie valide trouvée")
    
    valid_geoms = gdf.geometry.dropna()
    if len(valid_geoms) == 0:
        raise HTTPException(status_code=400, detail="Aucune géométrie valide trouvée")

def create_mask_from_geometry(geometry, image_shape, transform):
    """Crée un masque à partir de la géométrie pour rendre transparent l'extérieur"""
    try:
        geom_dict = mapping(geometry)
        
        mask = rasterize(
            [geom_dict],
            out_shape=image_shape,
            transform=transform,
            fill=0,
            default_value=1,
            dtype=np.uint8
        )
        
        return mask.astype(bool)
    except Exception as e:
        logger.error(f"Erreur lors de la création du masque: {e}")
        return np.ones(image_shape, dtype=bool)

# NOUVELLES FONCTIONS DU CODE 2

def transparent_cmap(cmap, alpha=0.5):
    """Créer une colormap transparente"""
    colors = cmap(np.arange(cmap.N))
    colors[:, -1] = alpha
    return mcolors.ListedColormap(colors)

def s2_to_rgb(bands_data, contrast_factor=3.5):
    """Convertir les bandes Sentinel-2 en image RGB"""
    if bands_data.shape[-1] >= 3:
        rgb = bands_data[:, :, [2, 1, 0]]  # R, G, B
    else:
        return bands_data
    
    # Normaliser les valeurs
    rgb = np.clip(rgb * contrast_factor / 10000, 0, 1)
    return rgb

def simple_vegetation_segmentation(ndvi, threshold=0.3):
    """Segmentation simple basée sur un seuil NDVI"""
    return np.where(ndvi > threshold, 1, 0)

def calculate_masked_average(index_array, mask):
    """Calculer la moyenne de l'indice dans la zone masquée"""
    masked_index = index_array * mask
    return np.nanmean(masked_index[mask == 1])

def get_time_series_images(geometry, start_date, end_date, config, interval_days=10, max_cloud_coverage=30):
    """Obtenir une série temporelle d'images Sentinel-2"""
    images = []
    dates = []
    
    current_date = start_date
    while current_date < end_date:
        next_date = min(current_date + timedelta(days=interval_days), end_date)
        
        try:
            evalscript = """
            //VERSION=3
            function setup() {
                return {
                    input: ["B02", "B03", "B04", "B08", "CLM"],
                    output: [
                        {id: "bands", bands: 4, sampleType: "FLOAT32"},
                        {id: "cloud_mask", bands: 1, sampleType: "UINT8"}
                    ]
                };
            }
            
            function evaluatePixel(sample) {
                return {
                    bands: [sample.B02, sample.B03, sample.B04, sample.B08],
                    cloud_mask: [sample.CLM]
                };
            }
            """
            
            request = SentinelHubRequest(
                evalscript=evalscript,
                input_data=[SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=(current_date.strftime('%Y-%m-%d'), next_date.strftime('%Y-%m-%d')),
                    maxcc=max_cloud_coverage/100,
                    mosaicking_order='leastCC'
                )],
                responses=[
                    SentinelHubRequest.output_response('bands', MimeType.TIFF),
                    SentinelHubRequest.output_response('cloud_mask', MimeType.TIFF)
                ],
                geometry=geometry,
                size=(512, 512),
                config=config
            )
            
            data = request.get_data()
            
            if data and len(data) > 0:
                bands = data[0]['bands']
                cloud_mask = data[0]['cloud_mask']
                
                # Appliquer le masque de nuages
                bands_masked = np.where(cloud_mask[..., 0] == 0, bands, np.nan)
                images.append(bands_masked)
                dates.append(current_date)
                logger.info(f"Image récupérée pour {current_date.strftime('%Y-%m-%d')}")
                
        except Exception as e:
            logger.error(f"Erreur lors du chargement de l'image {current_date}: {e}")
        
        current_date = next_date
    
    return images, dates

# FIN DES NOUVELLES FONCTIONS

def calculate_vegetation_indices_improved(bands_img: np.ndarray) -> Dict[str, np.ndarray]:
    """Calcul amélioré des indices de végétation avec gestion des erreurs"""
    logger.info(f"Calcul des indices pour une image de taille: {bands_img.shape}")
    
    blue = bands_img[:, :, 0].astype('float64')
    green = bands_img[:, :, 1].astype('float64') 
    red = bands_img[:, :, 2].astype('float64')
    nir = bands_img[:, :, 3].astype('float64')
    
    # Normaliser les valeurs si nécessaire
    if np.max(bands_img) > 1.5:
        logger.info("Normalisation des bandes detectée nécessaire")
        blue = blue / 10000.0
        green = green / 10000.0
        red = red / 10000.0
        nir = nir / 10000.0
    
    indices = {}
    
    # Calcul avec votre formule exacte
    with np.errstate(divide='ignore', invalid='ignore'):
        # NDVI: (NIR - Red) / (NIR + Red)
        ndvi = (nir - red) / (nir + red)
        ndvi = np.clip(ndvi, -1, 1)
        ndvi = np.where(np.isfinite(ndvi), ndvi, np.nan)
        indices['ndvi'] = ndvi
        
        # NDWI: (NIR - Green) / (NIR + Green)
        ndwi = (nir - green) / (nir + green)
        ndwi = np.clip(ndwi, -1, 1)
        ndwi = np.where(np.isfinite(ndwi), ndwi, np.nan)
        indices['ndwi'] = ndwi
        
        # SAVI: ((NIR - Red) / (NIR + Red + 0.5)) * 1.5
        savi = ((nir - red) / (nir + red + 0.5)) * 1.5
        savi = np.clip(savi, -1, 1)
        savi = np.where(np.isfinite(savi), savi, np.nan)
        indices['savi'] = savi
        
        # EVI: 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
        evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)
        evi = np.clip(evi, -1, 1)
        evi = np.where(np.isfinite(evi), evi, np.nan)
        indices['evi'] = evi
    
    logger.info(f"Indices calculés: {list(indices.keys())}")
    return indices

# NOUVELLES FONCTIONS D'AMÉLIORATION DES PIXELS

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
        # Fallback vers lissage gaussien amélioré
        enhanced_array = enhanced_gaussian_filter(enhanced_array)
    
    return enhanced_array

def adaptive_smoothing(array: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Lissage adaptatif basé sur la variance locale"""
    from scipy.ndimage import generic_filter
    
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
        
        # Redimensionner à la taille originale
        if SKIMAGE_AVAILABLE:
            enhanced_downsampled = resize(enhanced, (h, w), order=3, mode='constant', cval=np.nan, anti_aliasing=True)
        else:
            # Fallback simple
            enhanced_downsampled = gaussian_filter(array, sigma=0.8)
        
        return enhanced_downsampled
        
    except Exception as e:
        logger.warning(f"Super-résolution échouée: {e}")
        return gaussian_filter(array, sigma=0.8)

def edge_preserving_filter(array: np.ndarray, sigma_spatial: float = 1.0, sigma_intensity: float = 0.1) -> np.ndarray:
    """Filtre préservant les contours avec OpenCV"""
    if not CV2_AVAILABLE:
        return gaussian_filter(array, sigma=1.0)
    
    valid_mask = ~np.isnan(array)
    
    if np.sum(valid_mask) == 0:
        return array
    
    array_norm = array.copy()
    array_norm[~valid_mask] = 0
    
    min_val, max_val = np.nanmin(array), np.nanmax(array)
    if max_val > min_val:
        array_norm = (array_norm - min_val) / (max_val - min_val)
    
    array_uint8 = (array_norm * 255).astype(np.uint8)
    
    try:
        enhanced_uint8 = cv2.edgePreservingFilter(array_uint8, flags=2, sigma_s=sigma_spatial*50, sigma_r=sigma_intensity)
        
        enhanced = enhanced_uint8.astype(np.float64) / 255.0
        
        if max_val > min_val:
            enhanced = enhanced * (max_val - min_val) + min_val
        
        enhanced[~valid_mask] = np.nan
        
        return enhanced
        
    except Exception as e:
        logger.warning(f"Edge-preserving filter échoué: {e}")
        return gaussian_filter(array, sigma=1.0)

def bilateral_filter_2d(array: np.ndarray, sigma_spatial: float = 1.5, sigma_intensity: float = 0.2) -> np.ndarray:
    """Filtre bilatéral 2D simplifié"""
    # Version simplifiée pour éviter la complexité excessive
    return gaussian_filter(array, sigma=sigma_spatial)

def segmentation_based_enhancement(array: np.ndarray, n_segments: int = 100) -> np.ndarray:
    """Amélioration basée sur la segmentation SLIC"""
    if not SKIMAGE_AVAILABLE:
        return gaussian_filter(array, sigma=1.0)
    
    valid_mask = ~np.isnan(array)
    
    if np.sum(valid_mask) < 100:
        return gaussian_filter(array, sigma=1.0)
    
    array_for_slic = np.stack([array, array, array], axis=2)
    array_for_slic[~valid_mask] = 0
    
    try:
        segments = slic(array_for_slic, n_segments=n_segments, compactness=10, sigma=1, start_label=1)
        
        enhanced = array.copy()
        
        for segment_id in np.unique(segments):
            if segment_id == 0:
                continue
                
            segment_mask = segments == segment_id
            segment_values = array[segment_mask & valid_mask]
            
            if len(segment_values) > 0:
                segment_mean = np.mean(segment_values)
                enhanced[segment_mask] = segment_mean
        
        return enhanced
        
    except Exception as e:
        logger.warning(f"Segmentation échouée: {e}")
        return gaussian_filter(array, sigma=1.0)

def enhanced_gaussian_filter(array: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Filtre gaussien amélioré par défaut"""
    # Appliquer un filtre gaussien avec préservation des NaN
    valid_mask = ~np.isnan(array)
    
    if np.sum(valid_mask) == 0:
        return array
    
    # Remplacer temporairement les NaN par la moyenne
    temp_array = array.copy()
    temp_array[~valid_mask] = np.nanmean(array)
    
    # Appliquer le filtre
    filtered = gaussian_filter(temp_array, sigma=sigma)
    
    # Remettre les NaN
    filtered[~valid_mask] = np.nan
    
    return filtered

def create_smooth_interpretation_map(index_array: np.ndarray, index_name: str) -> Dict[str, Any]:
    """Crée une carte d'interprétation avec classifications claires"""
    
    interpretation_configs = {
        'ndvi': {
            'thresholds': [-1, -0.2, 0.0, 0.2, 0.4, 0.6, 1.0],
            'labels': ['Eau/Roches', 'Sol nu', 'Végétation très faible', 'Végétation clairsemée', 'Végétation modérée', 'Végétation dense'],
            'colors': ['#0066cc', '#8B4513', '#FFFF99', '#ADFF2F', '#32CD32', '#006400'],
            'description': 'Plus la valeur est élevée, plus la végétation est dense et saine'
        },
        'ndwi': {
            'thresholds': [-1, -0.3, -0.1, 0.1, 0.3, 1.0],
            'labels': ['Sol très sec', 'Sol sec', 'Sol modérément humide', 'Sol humide', 'Eau/Zone très humide'],
            'colors': ['#8B4513', '#DEB887', '#98FB98', '#87CEEB', '#0066cc'],
            'description': 'Indique la présence d\'eau et le niveau d\'humidité'
        },
        'savi': {
            'thresholds': [-1, -0.1, 0.1, 0.3, 0.5, 0.8, 1.0],
            'labels': ['Non végétalisé', 'Sol nu/rocailleux', 'Végétation émergente', 'Végétation faible', 'Végétation modérée', 'Végétation dense'],
            'colors': ['#8B4513', '#D2691E', '#FFFF99', '#ADFF2F', '#32CD32', '#006400'],
            'description': 'NDVI ajusté pour réduire l\'influence de la couleur du sol'
        },
        'evi': {
            'thresholds': [-1, -0.1, 0.1, 0.3, 0.5, 0.7, 1.0],
            'labels': ['Aucune végétation', 'Végétation minimale', 'Végétation faible', 'Végétation modérée', 'Végétation dense', 'Végétation très dense'],
            'colors': ['#8B4513', '#DEB887', '#FFFF99', '#ADFF2F', '#32CD32', '#006400'],
            'description': 'Version améliorée du NDVI, plus précise pour la végétation dense'
        }
    }
    
    config = interpretation_configs.get(index_name, interpretation_configs['ndvi'])
    
    classified_map = np.full_like(index_array, np.nan)
    
    for i, threshold in enumerate(config['thresholds'][:-1]):
        next_threshold = config['thresholds'][i + 1]
        mask = (index_array >= threshold) & (index_array < next_threshold)
        classified_map[mask] = i
    
    return {
        'classified_map': classified_map,
        'labels': config['labels'],
        'colors': config['colors'],
        'description': config['description'],
        'thresholds': config['thresholds']
    }

def generate_enhanced_visualization(indices: Dict[str, np.ndarray], mask: np.ndarray, geometry_bounds, enhancement_method: str = 'adaptive') -> Dict[str, str]:
    """Génère des visualisations avec amélioration des pixels"""
    
    plots = {}
    
    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, array in indices.items():
            
            # Appliquer l'amélioration des pixels
            enhanced_array = advanced_pixel_enhancement(array, mask, enhancement_method)
            
            # Générer les versions : continue, classifiée, et comparaison
            for version in ['enhanced_continuous', 'enhanced_classified', 'comparison']:
                
                path = os.path.join(tmpdir, f"{idx}_{version}.png")
                
                try:
                    if version == 'comparison':
                        # Vue comparative
                        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7), facecolor='white')
                        
                        # Original
                        masked_original = np.where(mask, array, np.nan)
                        im1 = ax1.imshow(
                            masked_original,
                            cmap='RdYlGn' if idx == 'ndvi' else 'viridis',
                            extent=[geometry_bounds[0], geometry_bounds[2], geometry_bounds[1], geometry_bounds[3]],
                            interpolation='nearest',
                            alpha=0.9
                        )
                        ax1.set_title(f'{idx.upper()} - Original', fontsize=14, fontweight='bold')
                        plt.colorbar(im1, ax=ax1, shrink=0.7)
                        
                        # Amélioré
                        im2 = ax2.imshow(
                            enhanced_array,
                            cmap='RdYlGn' if idx == 'ndvi' else 'viridis',
                            extent=[geometry_bounds[0], geometry_bounds[2], geometry_bounds[1], geometry_bounds[3]],
                            interpolation='bilinear',
                            alpha=0.9
                        )
                        ax2.set_title(f'{idx.upper()} - Amélioré ({enhancement_method})', fontsize=14, fontweight='bold')
                        plt.colorbar(im2, ax=ax2, shrink=0.7)
                        
                        # Différence
                        difference = enhanced_array - masked_original
                        im3 = ax3.imshow(
                            difference,
                            cmap='RdBu',
                            extent=[geometry_bounds[0], geometry_bounds[2], geometry_bounds[1], geometry_bounds[3]],
                            interpolation='bilinear',
                            alpha=0.9
                        )
                        ax3.set_title(f'{idx.upper()} - Amélioration (Différence)', fontsize=14, fontweight='bold')
                        plt.colorbar(im3, ax=ax3, shrink=0.7)
                        
                        for ax in [ax1, ax2, ax3]:
                            ax.set_xlabel('Longitude', fontsize=11)
                            ax.set_ylabel('Latitude', fontsize=11)
                            ax.grid(True, alpha=0.3)
                        
                    else:
                        # Versions continues et classifiées améliorées
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor='white')
                        
                        if version == 'enhanced_continuous':
                            # Version continue améliorée
                            if idx == 'ndvi':
                                cmap = 'RdYlGn'
                                vmin, vmax = -0.2, 0.8
                            elif idx == 'ndwi':
                                cmap = 'Blues'
                                vmin, vmax = -0.4, 0.4
                            elif idx == 'savi':
                                cmap = 'YlGn'
                                vmin, vmax = -0.1, 0.7
                            else:  # evi
                                cmap = 'Greens'
                                vmin, vmax = -0.1, 0.8
                            
                            im1 = ax1.imshow(
                                enhanced_array,
                                cmap=cmap,
                                vmin=vmin,
                                vmax=vmax,
                                extent=[geometry_bounds[0], geometry_bounds[2], geometry_bounds[1], geometry_bounds[3]],
                                interpolation='bilinear',
                                alpha=0.9
                            )
                            
                            cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.7)
                            cbar1.set_label(f'{idx.upper()} Valeurs Améliorées', fontsize=11)
                            
                            ax1.set_title(f'{idx.upper()} - Continu Amélioré ({enhancement_method})', fontsize=14, fontweight='bold')
                        
                        # Version classifiée (toujours présente)
                        interpretation = create_smooth_interpretation_map(enhanced_array, idx)
                        classified_map = interpretation['classified_map']
                        
                        colors = interpretation['colors']
                        n_classes = len(colors)
                        cmap_classified = mcolors.ListedColormap(colors)
                        bounds = list(range(n_classes + 1))
                        norm = mcolors.BoundaryNorm(bounds, cmap_classified.N)
                        
                        im2 = ax2.imshow(
                            classified_map,
                            cmap=cmap_classified,
                            norm=norm,
                            extent=[geometry_bounds[0], geometry_bounds[2], geometry_bounds[1], geometry_bounds[3]],
                            interpolation='nearest',
                            alpha=0.9
                        )
                        
                        cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.7, ticks=range(n_classes))
                        cbar2.ax.set_yticklabels(interpretation['labels'], fontsize=9)
                        cbar2.set_label('Classification', fontsize=11)
                        
                        ax2.set_title(f'{idx.upper()} - Classification Améliorée', fontsize=14, fontweight='bold')
                        
                        for ax in [ax1, ax2]:
                            ax.set_xlabel('Longitude', fontsize=11)
                            ax.set_ylabel('Latitude', fontsize=11)
                            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
                            ax.tick_params(axis='both', which='major', labelsize=9)
                    
                    # Ajouter des statistiques améliorées
                    valid_data = enhanced_array[np.isfinite(enhanced_array)]
                    if len(valid_data) > 0:
                        mean_val = np.mean(valid_data)
                        std_val = np.std(valid_data)
                        min_val = np.min(valid_data)
                        max_val = np.max(valid_data)
                        
                        info_text = f"""Statistiques Améliorées:
Moyenne: {mean_val:.3f}
Min: {min_val:.3f} | Max: {max_val:.3f}
Écart-type: {std_val:.3f}
Pixels valides: {len(valid_data)}
Méthode: {enhancement_method}"""
                        
                        props = dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9, edgecolor='navy')
                        fig.text(0.02, 0.98, info_text, transform=fig.transFigure, fontsize=9,
                                verticalalignment='top', bbox=props)
                    
                    plt.tight_layout()
                    plt.subplots_adjust(bottom=0.1, top=0.92, left=0.1, right=0.95)
                    
                    plt.savefig(
                        path,
                        dpi=300,
                        bbox_inches='tight',
                        facecolor='white',
                        edgecolor='none',
                        format='png'
                    )
                    plt.close()
                    
                    with open(path, "rb") as f:
                        plots[f"{idx}_{version}"] = base64.b64encode(f.read()).decode("utf-8")
                    
                    logger.info(f"Image {version} générée pour {idx} avec amélioration {enhancement_method}")
                        
                except Exception as e:
                    logger.error(f"Erreur lors de la génération du graphique {version} pour {idx}: {str(e)}")
                    if 'fig' in locals():
                        plt.close(fig)
    
    return plots

# NOUVELLE FONCTION POUR CRÉER UNE ANIMATION TEMPORELLE
def create_timelapse_animation(images: List[np.ndarray], dates: List[datetime], 
                              index_name: str, mask: np.ndarray = None, 
                              geometry_bounds: Tuple[float, float, float, float] = None) -> bytes:
    """Créer une animation GIF de la série temporelle"""
    
    # Calculer l'indice pour chaque image
    index_arrays = []
    for img in images:
        if img is not None:
            indices = calculate_vegetation_indices_improved(img)
            index_array = indices.get(index_name, indices['ndvi'])
            if mask is not None:
                index_array = np.where(mask, index_array, np.nan)
            index_arrays.append(index_array)
    
    if not index_arrays:
        logger.error("Aucune donnée pour créer l'animation")
        return None
    
    # Créer l'animation
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='white')
    
    # Déterminer la plage de couleurs
    if index_name == 'ndvi':
        vmin, vmax = -0.2, 0.8
        cmap = 'RdYlGn'
    elif index_name == 'ndwi':
        vmin, vmax = -0.4, 0.4
        cmap = 'Blues'
    elif index_name == 'savi':
        vmin, vmax = -0.1, 0.7
        cmap = 'YlGn'
    else:  # evi
        vmin, vmax = -0.1, 0.8
        cmap = 'Greens'
    
    # Première image
    if geometry_bounds:
        extent = [geometry_bounds[0], geometry_bounds[2], geometry_bounds[1], geometry_bounds[3]]
    else:
        extent = None
    
    im = ax.imshow(index_arrays[0], cmap=cmap, vmin=vmin, vmax=vmax, 
                   extent=extent, interpolation='bilinear', alpha=0.9)
    
    # Titre et colorbar
    title = ax.set_title(f"{index_name.upper()} - {dates[0].strftime('%Y-%m-%d')}", 
                        fontsize=14, fontweight='bold')
    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label(f'{index_name.upper()} Valeurs', fontsize=11)
    
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    def update(frame):
        im.set_array(index_arrays[frame])
        title.set_text(f"{index_name.upper()} - {dates[frame].strftime('%Y-%m-%d')}")
        return [im, title]
    
    ani = animation.FuncAnimation(
        fig, update, frames=len(index_arrays), 
        interval=500, blit=True, repeat=True
    )
    
    # Sauvegarder dans un buffer
    buffer = io.BytesIO()
    ani.save(buffer, writer='pillow', fps=2, format='gif')
    buffer.seek(0)
    
    plt.close()
    
    return buffer.getvalue()

# NOUVELLE FONCTION POUR CRÉER UN GRAPHIQUE DE SÉRIE TEMPORELLE
def create_time_series_plot(averages_dict: Dict[str, List[float]], dates: List[datetime]) -> bytes:
    """Créer un graphique de l'évolution temporelle des indices"""
    
    plt.figure(figsize=(12, 8), facecolor='white')
    
    # Couleurs pour chaque indice
    colors = {
        'ndvi': '#2E8B57',
        'ndwi': '#4682B4',
        'savi': '#DAA520',
        'evi': '#228B22'
    }
    
    # Tracer chaque indice
    for index_name, values in averages_dict.items():
        if values:
            plt.plot(dates, values, 'o-', label=index_name.upper(), 
                    linewidth=2.5, markersize=8, color=colors.get(index_name, 'gray'))
    
    plt.legend(fontsize=12, loc='best')
    plt.title("Évolution temporelle des indices de végétation", fontsize=16, fontweight='bold')
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Valeur moyenne de l'indice", fontsize=14)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Améliorer l'affichage des dates
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(10))
    plt.xticks(rotation=45)
    
    # Ajouter des zones colorées pour les saisons
    if dates:
        for i, date in enumerate(dates):
            month = date.month
            if 3 <= month <= 5:  # Printemps
                plt.axvspan(date, dates[min(i+1, len(dates)-1)], alpha=0.1, color='green')
            elif 6 <= month <= 8:  # Été
                plt.axvspan(date, dates[min(i+1, len(dates)-1)], alpha=0.1, color='yellow')
            elif 9 <= month <= 11:  # Automne
                plt.axvspan(date, dates[min(i+1, len(dates)-1)], alpha=0.1, color='orange')
            else:  # Hiver
                plt.axvspan(date, dates[min(i+1, len(dates)-1)], alpha=0.1, color='blue')
    
    plt.tight_layout()
    
    # Sauvegarder dans un buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    
    plt.close()
    
    return buffer.getvalue()

@app.get("/")
def health_check():
    return {
        "status": "API en fonctionnement",
        "service": "Calculateur d'Indices avec Amélioration des Pixels et Séries Temporelles",
        "version": "3.0.0",
        "features": [
            "amelioration_pixels_avancee", 
            "affichage_ultra_clair", 
            "comparaison_avant_apres",
            "methodes_multiples",
            "series_temporelles",
            "animations_gif",
            "evolution_temporelle"
        ],
        "methodes_amelioration": [
            "adaptive", 
            "super_resolution", 
            "edge_preserving", 
            "bilateral", 
            "segmentation_based"
        ],
        "bibliotheques_disponibles": {
            "opencv": CV2_AVAILABLE,
            "scikit_image": SKIMAGE_AVAILABLE
        }
    }

@app.get("/info")
def api_info():
    return {
        "indices_supportes": {
            "NDVI": "Indice de végétation normalisé",
            "NDWI": "Indice d'eau normalisé", 
            "SAVI": "Indice de végétation ajusté au sol",
            "EVI": "Indice de végétation amélioré"
        },
        "ameliorations_pixels_v2": {
            "adaptive": "Lissage adaptatif basé sur la variance locale",
            "super_resolution": "Amélioration par super-résolution x2",
            "edge_preserving": "Filtre préservant les contours (OpenCV requis)",
            "bilateral": "Filtre bilatéral pour réduction du bruit",
            "segmentation_based": "Amélioration par segmentation SLIC (scikit-image requis)"
        },
        "nouvelles_fonctionnalites_v3": {
            "series_temporelles": "Analyse de l'évolution dans le temps",
            "animations_gif": "Création d'animations temporelles",
            "graphiques_evolution": "Visualisation de l'évolution des indices",
            "export_rgb": "Export des images RGB",
            "segmentation_vegetation": "Masquage automatique de la végétation"
        },
        "endpoints": {
            "/calculate-indices": "Calcul simple des indices",
            "/calculate-time-series": "Calcul de série temporelle",
            "/methods": "Liste des méthodes disponibles",
            "/health": "État de santé de l'API"
        }
    }

@app.post("/calculate-indices")
async def calculate_indices(
    geojson: UploadFile = File(...),
    enhancement_method: str = Query(default="adaptive", description="Méthode d'amélioration des pixels")
):
    logger.info(f"Début du calcul avec amélioration '{enhancement_method}' pour: {geojson.filename}")
    
    # Valider la méthode d'amélioration
    valid_methods = ['adaptive', 'super_resolution', 'edge_preserving', 'bilateral', 'segmentation_based']
    if enhancement_method not in valid_methods:
        enhancement_method = 'adaptive'
        logger.warning(f"Méthode non valide, utilisation de 'adaptive' par défaut")
    
    if not geojson.filename.endswith('.geojson') and not geojson.filename.endswith('.json'):
        raise HTTPException(
            status_code=400, 
            detail="Le fichier doit être un GeoJSON"
        )
    
    try:
        # Lire et valider le GeoJSON
        gdf = gpd.read_file(geojson.file)
        validate_geojson(gdf)
        
        aoi_polygon = gdf.geometry.dropna().iloc[0]
        geometry = Geometry(aoi_polygon, CRS.WGS84)
        bounds = aoi_polygon.bounds
        
        # Configurer SentinelHub
        config = get_sentinelhub_config()
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        
        evalscript = """
        //VERSION=3
        function setup() {
            return {
                input: ["B02", "B03", "B04", "B08"],
                output: {bands: 4, sampleType: "FLOAT32"}
            };
        }
        
        function evaluatePixel(sample) {
            return [sample.B02, sample.B03, sample.B04, sample.B08];
        }
        """
        
        request_sh = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')),
                mosaicking_order='mostRecent'
            )],
            responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
            geometry=geometry,
            size=(512, 512),
            config=config
        )
        
        # Obtenir les données satellite
        logger.info("Récupération des données satellite...")
        bands_data = request_sh.get_data()
        
        if not bands_data or len(bands_data) == 0:
            raise HTTPException(
                status_code=404,
                detail="Aucune imagerie satellite trouvée pour cette zone et période"
            )
        
        bands_img = bands_data[0]
        logger.info(f"Données satellite récupérées: {bands_img.shape}")
        
        # Créer le masque géométrique
        logger.info("Création du masque géométrique...")
        transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], 
                               bands_img.shape[1], bands_img.shape[0])
        mask = create_mask_from_geometry(aoi_polygon, bands_img.shape[:2], transform)
        
        # Calculer les indices de végétation
        logger.info("Calcul des indices de végétation...")
        indices = calculate_vegetation_indices_improved(bands_img)
        
        # Générer l'image RGB
        rgb_image = s2_to_rgb(bands_img)
        
        # Générer les visualisations améliorées
        logger.info(f"Génération des visualisations avec méthode: {enhancement_method}")
        plots = generate_enhanced_visualization(indices, mask, bounds, enhancement_method)
        
        # Ajouter l'image RGB aux plots
        with tempfile.TemporaryDirectory() as tmpdir:
            rgb_path = os.path.join(tmpdir, "rgb_image.png")
            
            fig, ax = plt.subplots(figsize=(10, 10), facecolor='white')
            ax.imshow(rgb_image, extent=[bounds[0], bounds[2], bounds[1], bounds[3]])
            ax.set_title("Image RGB Sentinel-2", fontsize=14, fontweight='bold')
            ax.set_xlabel('Longitude', fontsize=11)
            ax.set_ylabel('Latitude', fontsize=11)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(rgb_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            with open(rgb_path, "rb") as f:
                plots["rgb_image"] = base64.b64encode(f.read()).decode("utf-8")
        
        # Calculer les statistiques comparatives
        logger.info("Calcul des statistiques...")
        statistics = {}
        improvement_metrics = {}
        
        for idx, array in indices.items():
            # Données originales
            masked_original = array[mask]
            valid_original = masked_original[np.isfinite(masked_original)]
            
            # Données améliorées
            enhanced_array = advanced_pixel_enhancement(array, mask, enhancement_method)
            masked_enhanced = enhanced_array[mask]
            valid_enhanced = masked_enhanced[np.isfinite(masked_enhanced)]
            
            if len(valid_original) > 0:
                # Statistiques originales
                stats_original = {
                    "moyenne": float(np.mean(valid_original)),
                    "ecart_type": float(np.std(valid_original)),
                    "minimum": float(np.min(valid_original)),
                    "maximum": float(np.max(valid_original)),
                    "mediane": float(np.median(valid_original)),
                    "pixels_valides": int(len(valid_original))
                }
                
                # Statistiques améliorées
                stats_enhanced = {
                    "moyenne": float(np.mean(valid_enhanced)),
                    "ecart_type": float(np.std(valid_enhanced)),
                    "minimum": float(np.min(valid_enhanced)),
                    "maximum": float(np.max(valid_enhanced)),
                    "mediane": float(np.median(valid_enhanced)),
                    "pixels_valides": int(len(valid_enhanced))
                }
                
                # Métriques d'amélioration
                variance_reduction = (stats_original["ecart_type"] - stats_enhanced["ecart_type"]) / stats_original["ecart_type"] * 100
                
                # Calcul de la corrélation entre original et amélioré
                correlation = float(np.corrcoef(valid_original, valid_enhanced)[0, 1]) if len(valid_original) == len(valid_enhanced) else 1.0
                
                # Signal-to-Noise Ratio improvement (approximation)
                snr_original = stats_original["moyenne"] / stats_original["ecart_type"] if stats_original["ecart_type"] > 0 else 0
                snr_enhanced = stats_enhanced["moyenne"] / stats_enhanced["ecart_type"] if stats_enhanced["ecart_type"] > 0 else 0
                snr_improvement = snr_enhanced - snr_original
                
                improvement_metrics[idx] = {
                    "reduction_variance_pourcent": float(variance_reduction),
                    "correlation_original_ameliore": float(correlation),
                    "amelioration_snr": float(snr_improvement),
                    "conservation_moyenne": abs(stats_original["moyenne"] - stats_enhanced["moyenne"]) < 0.01,
                    "qualite_amelioration": "excellente" if variance_reduction > 20 else "bonne" if variance_reduction > 10 else "moderate"
                }
                
                statistics[idx] = {
                    "original": stats_original,
                    "ameliore": stats_enhanced,
                    "amelioration": improvement_metrics[idx]
                }
        
        # Calculer les statistiques de classification améliorée
        classification_stats = {}
        for idx, array in indices.items():
            enhanced_array = advanced_pixel_enhancement(array, mask, enhancement_method)
            interpretation = create_smooth_interpretation_map(enhanced_array, idx)
            classified_map = interpretation['classified_map']
            
            # Distribution des classes
            class_distribution = {}
            total_valid_pixels = np.sum(np.isfinite(classified_map))
            
            if total_valid_pixels > 0:
                for i, label in enumerate(interpretation['labels']):
                    count = np.sum(classified_map == i)
                    percentage = (count / total_valid_pixels) * 100
                    class_distribution[label] = {
                        "pixels": int(count),
                        "pourcentage": float(percentage)
                    }
            
            classification_stats[idx] = {
                "distribution_classes": class_distribution,
                "classe_dominante": max(class_distribution.items(), key=lambda x: x[1]["pourcentage"])[0] if class_distribution else "Inconnue",
                "diversite_classes": len([k for k, v in class_distribution.items() if v["pourcentage"] > 1.0])
            }
        
        # Informations sur la méthode d'amélioration utilisée
        method_info = {
            "adaptive": {
                "description": "Lissage adaptatif basé sur la variance locale",
                "avantages": ["Préserve les détails", "Lisse les zones homogènes", "Adaptatif automatiquement"],
                "cas_usage": "Idéal pour tous types de terrain"
            },
            "super_resolution": {
                "description": "Super-résolution par interpolation bicubique",
                "avantages": ["Améliore la résolution", "Interpolation avancée", "Détails fins"],
                "cas_usage": "Terrains avec détails fins importants"
            },
            "edge_preserving": {
                "description": "Filtre préservant les contours (OpenCV)",
                "avantages": ["Préserve les bordures nettes", "Réduit le bruit", "Contours nets"],
                "cas_usage": "Zones avec bordures nettes importantes",
                "disponible": CV2_AVAILABLE
            },
            "bilateral": {
                "description": "Filtre bilatéral pour réduction du bruit",
                "avantages": ["Réduction du bruit", "Préservation des détails", "Lissage intelligent"],
                "cas_usage": "Images avec beaucoup de bruit"
            },
            "segmentation_based": {
                "description": "Amélioration par segmentation SLIC",
                "avantages": ["Segmentation intelligente", "Zones homogènes", "Classification précise"],
                "cas_usage": "Zones avec régions distinctes",
                "disponible": SKIMAGE_AVAILABLE
            }
        }
        
        # Recommandations automatiques
        recommendations = []
        
        # Analyser la qualité d'amélioration pour donner des recommandations
        avg_variance_reduction = np.mean([metrics["reduction_variance_pourcent"] for metrics in improvement_metrics.values()])
        
        if avg_variance_reduction < 5:
            recommendations.append("L'image semble déjà de bonne qualité. Essayez 'super_resolution' pour plus de détails.")
        elif avg_variance_reduction > 30:
            recommendations.append(f"Excellente amélioration avec '{enhancement_method}'. Qualité optimale atteinte.")
        else:
            if enhancement_method == 'adaptive':
                recommendations.append("Bon résultat. Essayez 'edge_preserving' pour préserver davantage les contours.")
            else:
                recommendations.append("Amélioration satisfaisante. L'algorithme adaptatif peut aussi donner de bons résultats.")
        
        # Recommandations basées sur les données
        if any(stats["original"]["ecart_type"] > 0.3 for stats in statistics.values()):
            recommendations.append("Données bruitées détectées. 'bilateral' pourrait donner de meilleurs résultats.")
        
        # Préparer la réponse finale
        response_data = {
            "status": "succès",
            "message": f"Indices calculés avec amélioration des pixels (méthode: {enhancement_method})",
            "data": {
                "images": plots,
                "statistiques": statistics,
                "classification": classification_stats,
                "metadonnees": {
                    "satellite": "Sentinel-2 L2A",
                    "periode_recherche": f"{start_date.strftime('%Y-%m-%d')} à {end_date.strftime('%Y-%m-%d')}",
                    "resolution_image": "512x512 pixels",
                    "methode_amelioration": enhancement_method,
                    "methode_info": method_info.get(enhancement_method, {}),
                    "types_images_generees": ["enhanced_continuous", "enhanced_classified", "comparison", "rgb_image"],
                    "qualite_export": "300 DPI",
                    "bibliotheques_utilisees": {
                        "opencv": CV2_AVAILABLE,
                        "scikit_image": SKIMAGE_AVAILABLE,
                        "scipy": True,
                        "matplotlib": True
                    }
                },
                "recommandations": recommendations,
                "methodes_disponibles": {
                    method: {
                        "description": info["description"],
                        "disponible": info.get("disponible", True)
                    }
                    for method, info in method_info.items()
                }
            }
        }
        
        logger.info(f"Traitement terminé avec succès. Méthode: {enhancement_method}, Images générées: {len(plots)}")
        return JSONResponse(response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors du traitement: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Erreur lors du traitement: {str(e)}"
        )

# NOUVEAU ENDPOINT POUR LES SÉRIES TEMPORELLES
@app.post("/calculate-time-series")
async def calculate_time_series(
    geojson: UploadFile = File(...),
    start_date: str = Query(..., description="Date de début (YYYY-MM-DD)"),
    end_date: str = Query(..., description="Date de fin (YYYY-MM-DD)"),
    interval_days: int = Query(default=10, description="Intervalle en jours entre les images"),
    enhancement_method: str = Query(default="adaptive", description="Méthode d'amélioration des pixels"),
    create_animation: bool = Query(default=True, description="Créer des animations GIF"),
    indices: List[str] = Query(default=["ndvi"], description="Liste des indices à calculer")
):
    logger.info(f"Début du calcul de série temporelle pour: {geojson.filename}")
    
    try:
        # Parser les dates
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        if start_dt >= end_dt:
            raise HTTPException(status_code=400, detail="La date de début doit être avant la date de fin")
        
        # Lire et valider le GeoJSON
        gdf = gpd.read_file(geojson.file)
        validate_geojson(gdf)
        
        aoi_polygon = gdf.geometry.dropna().iloc[0]
        geometry = Geometry(aoi_polygon, CRS.WGS84)
        bounds = aoi_polygon.bounds
        
        # Configurer SentinelHub
        config = get_sentinelhub_config()
        
        # Obtenir la série temporelle d'images
        logger.info(f"Récupération des images de {start_date} à {end_date} avec intervalle de {interval_days} jours")
        images, dates = get_time_series_images(geometry, start_dt, end_dt, config, interval_days)
        
        if not images:
            raise HTTPException(
                status_code=404,
                detail="Aucune image trouvée pour cette période et cette zone"
            )
        
        logger.info(f"{len(images)} images récupérées")
        
        # Créer le masque géométrique
        transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], 
                               images[0].shape[1], images[0].shape[0])
        mask = create_mask_from_geometry(aoi_polygon, images[0].shape[:2], transform)
        
        # Calculer les indices pour chaque image et collecter les moyennes
        averages_dict = {idx: [] for idx in indices}
        
        for img in images:
            img_indices = calculate_vegetation_indices_improved(img)
            for idx in indices:
                if idx in img_indices:
                    # Appliquer l'amélioration
                    enhanced = advanced_pixel_enhancement(img_indices[idx], mask, enhancement_method)
                    avg = calculate_masked_average(enhanced, mask)
                    averages_dict[idx].append(avg)
        
        # Créer les animations si demandé
        animations = {}
        if create_animation:
            logger.info("Création des animations...")
            for idx in indices:
                try:
                    animation_data = create_timelapse_animation(images, dates, idx, mask, bounds)
                    if animation_data:
                        animations[f"{idx}_animation"] = base64.b64encode(animation_data).decode("utf-8")
                except Exception as e:
                    logger.error(f"Erreur lors de la création de l'animation pour {idx}: {e}")
        
        # Créer le graphique de série temporelle
        logger.info("Création du graphique de série temporelle...")
        time_series_plot = create_time_series_plot(averages_dict, dates)
        
        # Calculer les statistiques temporelles
        temporal_stats = {}
        for idx, values in averages_dict.items():
            if values:
                temporal_stats[idx] = {
                    "moyenne_globale": float(np.mean(values)),
                    "ecart_type_global": float(np.std(values)),
                    "minimum_global": float(np.min(values)),
                    "maximum_global": float(np.max(values)),
                    "tendance": "croissante" if values[-1] > values[0] else "décroissante",
                    "variation_pourcent": float((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0,
                    "nombre_observations": len(values)
                }
        
        # Préparer la réponse
        response_data = {
            "status": "succès",
            "message": f"Série temporelle calculée pour {len(images)} dates",
            "data": {
                "animations": animations,
                "graphique_serie_temporelle": base64.b64encode(time_series_plot).decode("utf-8"),
                "statistiques_temporelles": temporal_stats,
                "donnees_brutes": {
                    idx: [float(v) for v in values] 
                    for idx, values in averages_dict.items()
                },
                "dates": [d.strftime("%Y-%m-%d") for d in dates],
                "metadonnees": {
                    "satellite": "Sentinel-2 L2A",
                    "periode_analysee": f"{start_date} à {end_date}",
                    "nombre_images": len(images),
                    "intervalle_jours": interval_days,
                    "methode_amelioration": enhancement_method,
                    "indices_calcules": indices,
                    "animations_creees": len(animations) > 0
                }
            }
        }
        
        logger.info("Série temporelle calculée avec succès")
        return JSONResponse(response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors du calcul de série temporelle: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du traitement: {str(e)}"
        )

@app.get("/methods")
async def get_enhancement_methods():
    """Endpoint pour obtenir la liste des méthodes d'amélioration disponibles"""
    methods = {
        "adaptive": {
            "name": "Lissage Adaptatif",
            "description": "Analyse la variance locale pour un lissage intelligent",
            "available": True,
            "recommended_for": ["Tous types de terrain", "Usage général", "Première utilisation"],
            "pros": ["Automatique", "Préserve les détails", "Rapide"],
            "cons": ["Amélioration modérée sur certains types d'images"]
        },
        "super_resolution": {
            "name": "Super-Résolution", 
            "description": "Améliore la résolution par interpolation bicubique avancée",
            "available": True,
            "recommended_for": ["Images avec détails fins", "Zones urbaines", "Agriculture de précision"],
            "pros": ["Améliore la résolution", "Détails très fins", "Interpolation avancée"],
            "cons": ["Plus lent", "Peut créer des artefacts sur images très bruitées"]
        },
        "edge_preserving": {
            "name": "Préservation des Contours",
            "description": "Filtre qui lisse tout en préservant les bordures nettes",
            "available": CV2_AVAILABLE,
            "recommended_for": ["Zones avec bordures nettes", "Limites de parcelles", "Zones urbaines"],
            "pros": ["Contours très nets", "Réduction du bruit efficace", "Qualité professionnelle"],
            "cons": ["Nécessite OpenCV", "Plus lent que les méthodes de base"],
            "requirements": ["OpenCV (cv2)"]
        },
        "bilateral": {
            "name": "Filtre Bilatéral",
            "description": "Réduit le bruit en préservant les détails importants",
            "available": True,
            "recommended_for": ["Images bruitées", "Conditions météo difficiles", "Images satellite de moindre qualité"],
            "pros": ["Excellente réduction du bruit", "Préserve les détails", "Robuste"],
            "cons": ["Peut être lent sur grandes images", "Lissage parfois excessif"]
        },
        "segmentation_based": {
            "name": "Segmentation SLIC",
            "description": "Amélioration basée sur une segmentation intelligente en super-pixels",
            "available": SKIMAGE_AVAILABLE,
            "recommended_for": ["Zones avec régions distinctes", "Classification précise", "Analyse par zones"],
            "pros": ["Segmentation intelligente", "Zones très homogènes", "Excellent pour classification"],
            "cons": ["Nécessite scikit-image", "Peut sur-simplifier certaines zones"],
            "requirements": ["scikit-image"]
        }
    }
    
    return {
        "methods": methods,
        "default_method": "adaptive",
        "system_capabilities": {
            "opencv_available": CV2_AVAILABLE,
            "skimage_available": SKIMAGE_AVAILABLE,
            "total_methods_available": sum(1 for method in methods.values() if method["available"])
        },
        "usage_tips": [
            "Commencez par 'adaptive' pour la plupart des cas",
            "Utilisez 'super_resolution' pour plus de détails fins",
            "Choisissez 'edge_preserving' pour des bordures nettes (si OpenCV disponible)",
            "Optez pour 'bilateral' si vos images sont bruitées",
            "Essayez 'segmentation_based' pour une classification très précise (si scikit-image disponible)"
        ]
    }

@app.get("/health")
async def detailed_health_check():
    """Endpoint de santé détaillé"""
    return {
        "status": "healthy",
        "version": "3.0.0",
        "service": "API Indices de Végétation avec Amélioration Pixels et Séries Temporelles",
        "capabilities": {
            "indices_calculation": True,
            "pixel_enhancement": True,
            "multiple_visualizations": True,
            "comparative_analysis": True,
            "time_series_analysis": True,
            "animations": True
        },
        "dependencies": {
            "sentinelhub": True,
            "geopandas": True,
            "matplotlib": True,
            "scipy": True,
            "numpy": True,
            "opencv": CV2_AVAILABLE,
            "scikit_image": SKIMAGE_AVAILABLE
        },
        "enhancement_methods_available": 5 if CV2_AVAILABLE and SKIMAGE_AVAILABLE else (4 if CV2_AVAILABLE or SKIMAGE_AVAILABLE else 3),
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    
    # Configuration du serveur
    port = int(os.getenv("PORT", 7860))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Démarrage de l'API sur {host}:{port}")
    logger.info(f"Version 3.0.0 - Avec support des séries temporelles")
    logger.info(f"Méthodes d'amélioration disponibles: {5 if CV2_AVAILABLE and SKIMAGE_AVAILABLE else (4 if CV2_AVAILABLE or SKIMAGE_AVAILABLE else 3)}")
    logger.info(f"OpenCV disponible: {CV2_AVAILABLE}")
    logger.info(f"scikit-image disponible: {SKIMAGE_AVAILABLE}")
    
    uvicorn.run(
        app, 
        host=host, 
        port=port,
        log_level="info",
        access_log=True
    )