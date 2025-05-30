#!/usr/bin/env python3
"""Test script to verify GeoJSON endpoints work correctly"""

import sys
sys.path.append('.')

try:
    from geojson_endpoints import (
        create_vegetation_mask_polygons, 
        create_indices_geojson_features,
        get_image_bounds, 
        validate_geojson_response,
        classify_index_value
    )
    import numpy as np
    print('✅ GeoJSON endpoints modules import successfully')
    
    test_coords = [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]]
    bounds = get_image_bounds(test_coords)
    print(f'✅ Image bounds extraction works: {bounds}')
    
    test_ndvi = np.random.rand(50, 50) * 2 - 1  # NDVI range -1 to 1
    mask_features = create_vegetation_mask_polygons(test_ndvi, bounds, 0.2, 0.6, "ndvi")
    print(f'✅ Vegetation mask creation works: {len(mask_features)} features')
    
    indices_data = {
        "ndvi": test_ndvi,
        "ndwi": np.random.rand(50, 50) * 2 - 1
    }
    point_features = create_indices_geojson_features(indices_data, bounds, 5)
    print(f'✅ Indices GeoJSON features work: {len(point_features)} points')
    
    ndvi_class = classify_index_value(0.5, "ndvi")
    print(f'✅ Index classification works: NDVI 0.5 = {ndvi_class}')
    
    response = validate_geojson_response(point_features)
    print(f'✅ GeoJSON response validation works: {response["type"]}')
    
    print('✅ All GeoJSON endpoint tests passed')
    
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
except Exception as e:
    print(f'❌ Test error: {e}')
    sys.exit(1)
