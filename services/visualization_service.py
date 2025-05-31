"""Service for visualization generation and image processing"""
import os
import tempfile
import base64
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import io
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def s2_to_rgb(bands_img: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """Convertit les bandes Sentinel-2 en image RGB"""
    red = bands_img[:, :, 2]
    green = bands_img[:, :, 1] 
    blue = bands_img[:, :, 0]
    
    rgb = np.stack([red, green, blue], axis=2)
    rgb = np.clip(rgb * 3.5, 0, 1)
    rgb = np.power(rgb, 1/gamma)
    
    return rgb

def generate_enhanced_visualization(indices: Dict[str, np.ndarray], mask: np.ndarray, 
                                  geometry_bounds: Tuple[float, float, float, float], 
                                  enhancement_method: str = 'adaptive') -> Dict[str, str]:
    """Génère des visualisations avec amélioration des pixels"""
    from .enhancement_service import advanced_pixel_enhancement
    
    plots = {}
    
    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, array in indices.items():
            
            enhanced_array = advanced_pixel_enhancement(array, mask, enhancement_method)
            
            for version in ['enhanced_continuous', 'enhanced_classified', 'comparison']:
                
                path = os.path.join(tmpdir, f"{idx}_{version}.png")
                
                try:
                    if version == 'comparison':
                        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7), facecolor='white')
                        
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
                        
                        im2 = ax2.imshow(
                            enhanced_array,
                            cmap='RdYlGn' if idx == 'ndvi' else 'viridis',
                            extent=[geometry_bounds[0], geometry_bounds[2], geometry_bounds[1], geometry_bounds[3]],
                            interpolation='bilinear',
                            alpha=0.9
                        )
                        ax2.set_title(f'{idx.upper()} - Amélioré ({enhancement_method})', fontsize=14, fontweight='bold')
                        plt.colorbar(im2, ax=ax2, shrink=0.7)
                        
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
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor='white')
                        
                        if version == 'enhanced_continuous':
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
                                vmin, vmax = -0.1, 0.6
                            
                            im1 = ax1.imshow(
                                enhanced_array,
                                cmap=cmap,
                                vmin=vmin, vmax=vmax,
                                extent=[geometry_bounds[0], geometry_bounds[2], geometry_bounds[1], geometry_bounds[3]],
                                interpolation='bilinear',
                                alpha=0.9
                            )
                            ax1.set_title(f'{idx.upper()} - Continue Améliorée', fontsize=14, fontweight='bold')
                            plt.colorbar(im1, ax=ax1, shrink=0.8)
                        
                        else:  # enhanced_classified
                            classified = _classify_index(enhanced_array, idx)
                            
                            im2 = ax2.imshow(
                                classified,
                                cmap='RdYlGn' if idx == 'ndvi' else 'viridis',
                                extent=[geometry_bounds[0], geometry_bounds[2], geometry_bounds[1], geometry_bounds[3]],
                                interpolation='nearest',
                                alpha=0.9
                            )
                            ax2.set_title(f'{idx.upper()} - Classifiée Améliorée', fontsize=14, fontweight='bold')
                            plt.colorbar(im2, ax=ax2, shrink=0.8)
                        
                        for ax in [ax1, ax2]:
                            ax.set_xlabel('Longitude', fontsize=11)
                            ax.set_ylabel('Latitude', fontsize=11)
                            ax.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
                    plt.close()
                    
                    with open(path, "rb") as f:
                        plots[f"{idx}_{version}"] = base64.b64encode(f.read()).decode("utf-8")
                        
                except Exception as e:
                    logger.error(f"Erreur lors de la génération de {idx}_{version}: {e}")
                    continue
    
    return plots

def _classify_index(array: np.ndarray, index_name: str) -> np.ndarray:
    """Classifie un indice de végétation en catégories"""
    classified = np.full_like(array, np.nan)
    
    if index_name == 'ndvi':
        classified = np.where(array < 0.2, 1,  # Sol nu/eau
                    np.where(array < 0.4, 2,  # Végétation faible
                    np.where(array < 0.6, 3,  # Végétation modérée
                             4)))  # Végétation dense
    elif index_name == 'ndwi':
        classified = np.where(array < -0.1, 1,  # Sec
                    np.where(array < 0.1, 2,   # Humidité faible
                    np.where(array < 0.3, 3,   # Humidité modérée
                             4)))  # Eau/très humide
    elif index_name in ['savi', 'evi']:
        classified = np.where(array < 0.15, 1,
                    np.where(array < 0.3, 2,
                    np.where(array < 0.5, 3,
                             4)))
    
    return classified

def create_timelapse_animation(images: List[np.ndarray], dates: List[datetime], 
                              index_name: str, mask: Optional[np.ndarray] = None, 
                              geometry_bounds: Optional[Tuple[float, float, float, float]] = None) -> Optional[bytes]:
    """Créer une animation GIF de la série temporelle"""
    from .indices_service import calculate_vegetation_indices_improved
    
    index_arrays = []
    for img in images:
        try:
            indices = calculate_vegetation_indices_improved(img)
            if index_name in indices:
                index_array = indices[index_name]
                if mask is not None:
                    index_array = np.where(mask, index_array, np.nan)
                index_arrays.append(index_array)
        except Exception as e:
            logger.error(f"Erreur lors du calcul de l'indice pour une image: {e}")
            continue
    
    if not index_arrays:
        logger.error("Aucune donnée pour créer l'animation")
        return None
    
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='white')
    
    if index_name == 'ndvi':
        cmap = 'RdYlGn'
        vmin, vmax = -0.2, 0.8
    elif index_name == 'ndwi':
        cmap = 'Blues'
        vmin, vmax = -0.4, 0.4
    elif index_name == 'savi':
        cmap = 'YlGn'
        vmin, vmax = -0.1, 0.7
    else:  # evi
        cmap = 'Greens'
        vmin, vmax = -0.1, 0.6
    
    if geometry_bounds:
        extent = [geometry_bounds[0], geometry_bounds[2], geometry_bounds[1], geometry_bounds[3]]
    else:
        extent = None
    
    im = ax.imshow(index_arrays[0], cmap=cmap, vmin=vmin, vmax=vmax, 
                   extent=extent, interpolation='bilinear', alpha=0.9)
    
    title = ax.set_title(f"{index_name.upper()} - {dates[0].strftime('%Y-%m-%d')}", 
                        fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    def update(frame):
        im.set_array(index_arrays[frame])
        title.set_text(f"{index_name.upper()} - {dates[frame].strftime('%Y-%m-%d')}")
        return [im, title]
    
    ani = animation.FuncAnimation(fig, update, frames=len(index_arrays), 
                                 interval=500, blit=False, repeat=True)
    
    buffer = io.BytesIO()
    ani.save(buffer, writer='pillow', fps=2)
    buffer.seek(0)
    
    plt.close()
    return buffer.getvalue()

def create_time_series_plot(averages_dict: Dict[str, List[float]], dates: List[datetime]) -> bytes:
    """Créer un graphique de série temporelle"""
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
    
    colors = {
        'ndvi': '#2E8B57',
        'ndwi': '#4169E1', 
        'savi': '#32CD32',
        'evi': '#228B22'
    }
    
    for index_name, values in averages_dict.items():
        if values:
            date_nums = [d.timestamp() for d in dates]
            ax.plot(date_nums, values, 'o-', label=index_name.upper(), 
                    linewidth=2.5, markersize=8, color=colors.get(index_name, 'gray'))
    
    ax.legend(fontsize=12, loc='best')
    ax.set_title('Évolution Temporelle des Indices de Végétation', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Valeur de l\'Indice', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    plt.close()
    
    return buffer.getvalue()
