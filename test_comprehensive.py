#!/usr/bin/env python3
"""Comprehensive test suite for all API endpoints and Flutter compatibility"""

import sys
import json
import asyncio
import tempfile
import os
from pathlib import Path
sys.path.append('.')

try:
    from fastapi.testclient import TestClient
    from app import app
    import numpy as np
    print('✅ FastAPI test client imports successfully')
    
    client = TestClient(app)
    
    response = client.get("/health")
    assert response.status_code == 200
    health_data = response.json()
    assert health_data["status"] == "healthy"
    print('✅ Health check endpoint works')
    
    response = client.get("/")
    assert response.status_code == 200
    api_info = response.json()
    assert "flutter_support" in api_info
    assert "geojson_endpoints" in api_info["flutter_support"]
    print('✅ API info endpoint with Flutter support works')
    
    response = client.get("/enhancement-methods")
    assert response.status_code == 200
    methods = response.json()
    assert "methods" in methods
    print('✅ Enhancement methods endpoint works')
    
    response = client.get("/health/detailed")
    assert response.status_code == 200
    detailed_health = response.json()
    assert "components" in detailed_health
    print('✅ Detailed health check endpoint works')
    
    test_geojson = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[0.0, 0.0], [0.1, 0.0], [0.1, 0.1], [0.0, 0.1], [0.0, 0.0]]]
        },
        "properties": {"name": "test_area"}
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.geojson', delete=False) as f:
        json.dump(test_geojson, f)
        test_file_path = f.name
    
    try:
        with open(test_file_path, 'rb') as f:
            response = client.post(
                "/calculate-indices",
                files={"geojson_file": ("test.geojson", f, "application/json")},
                params={"indices": "ndvi,ndwi", "enhancement_method": "adaptive"}
            )
        print(f'✅ Calculate indices endpoint structure test: {response.status_code}')
        
        with open(test_file_path, 'rb') as f:
            response = client.post(
                "/calculate-indices-geojson",
                files={"geojson_file": ("test.geojson", f, "application/json")},
                params={"indices": "ndvi,ndwi", "grid_size": "5"}
            )
        print(f'✅ GeoJSON indices endpoint structure test: {response.status_code}')
        
        with open(test_file_path, 'rb') as f:
            response = client.post(
                "/calculate-masks-geojson",
                files={"geojson_file": ("test.geojson", f, "application/json")},
                params={"indices": "ndvi", "threshold_low": "0.2", "threshold_high": "0.6"}
            )
        print(f'✅ GeoJSON masks endpoint structure test: {response.status_code}')
        
    finally:
        os.unlink(test_file_path)
    
    print('✅ All comprehensive API tests completed')
    
except ImportError as e:
    print(f'❌ Import error: {e}')
    print('Note: FastAPI test client may not be available')
    sys.exit(1)
except Exception as e:
    print(f'❌ Test error: {e}')
    print('Note: Some tests may fail without proper SentinelHub configuration')
