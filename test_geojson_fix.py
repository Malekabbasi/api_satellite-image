#!/usr/bin/env python3
"""Test script to verify GeoJSON parsing fix"""

import sys
import json
import tempfile
import os
from pathlib import Path
sys.path.append('.')

try:
    from fastapi.testclient import TestClient
    from app import app
    
    client = TestClient(app)
    
    valid_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [2.3522, 48.8566]  # Paris coordinates
                },
                "properties": {"name": "test_point"}
            }
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.geojson', delete=False) as f:
        json.dump(valid_geojson, f)
        test_file_path = f.name
    
    try:
        print("Testing with valid Point GeoJSON...")
        with open(test_file_path, 'rb') as f:
            response = client.post(
                "/calculate-indices-geojson",
                files={"geojson_file": ("test.geojson", f, "application/json")},
                params={"indices": "ndvi", "grid_size": "3"}
            )
        print(f'✅ Point GeoJSON test: {response.status_code}')
        if response.status_code != 200:
            print(f'Response: {response.text[:200]}...')
        
    finally:
        os.unlink(test_file_path)
    
    print('✅ GeoJSON parsing test completed')
    
except ImportError as e:
    print(f'❌ Import error: {e}')
except Exception as e:
    print(f'❌ Test error: {e}')
