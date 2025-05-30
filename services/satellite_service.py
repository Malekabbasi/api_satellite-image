"""Service for SentinelHub satellite data interactions"""
import os
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple
from sentinelhub import SHConfig, SentinelHubRequest, DataCollection, MimeType, Geometry, CRS

logger = logging.getLogger(__name__)

def get_sentinelhub_config() -> SHConfig:
    """Configure SentinelHub with environment variables"""
    config = SHConfig()
    config.instance_id = os.getenv("SENTINELHUB_INSTANCE_ID")
    config.sh_client_id = os.getenv("SENTINELHUB_CLIENT_ID") 
    config.sh_client_secret = os.getenv("SENTINELHUB_CLIENT_SECRET")
    
    if not all([config.instance_id, config.sh_client_id, config.sh_client_secret]):
        raise ValueError("SentinelHub credentials not properly configured in environment variables")
    
    return config

def get_time_series_images(geometry: Geometry, start_date: datetime, end_date: datetime, 
                          config: SHConfig, interval_days: int = 10) -> Tuple[List[np.ndarray], List[datetime]]:
    """Récupère une série temporelle d'images satellite"""
    logger.info(f"Récupération des images de {start_date} à {end_date}")
    
    images = []
    dates = []
    
    current_date = start_date
    
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
    
    while current_date <= end_date:
        next_date = current_date + timedelta(days=interval_days)
        
        try:
            request = SentinelHubRequest(
                evalscript=evalscript,
                input_data=[SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=(current_date.strftime('%Y-%m-%d'), 
                                 min(next_date, end_date).strftime('%Y-%m-%d')),
                    mosaicking_order='mostRecent'
                )],
                responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
                geometry=geometry,
                size=(512, 512),
                config=config
            )
            
            data = request.get_data()
            
            if data and len(data) > 0:
                images.append(data[0])
                dates.append(current_date)
                logger.info(f"Image récupérée pour {current_date.strftime('%Y-%m-%d')}")
            else:
                logger.warning(f"Aucune image trouvée pour {current_date.strftime('%Y-%m-%d')}")
                
        except Exception as e:
            logger.error(f"Erreur lors de la récupération pour {current_date}: {e}")
        
        current_date = next_date
    
    logger.info(f"Total: {len(images)} images récupérées")
    return images, dates

def get_single_image(geometry: Geometry, config: SHConfig, days_back: int = 180) -> np.ndarray:
    """Récupère une seule image satellite récente"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
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
    
    request = SentinelHubRequest(
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
    
    data = request.get_data()
    
    if not data or len(data) == 0:
        raise ValueError("Aucune imagerie satellite trouvée pour cette zone et période")
    
    return data[0]
