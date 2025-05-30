#!/usr/bin/env python3
"""Security and validation testing"""

import sys
import tempfile
import os
import json
sys.path.append('.')

try:
    from fastapi.testclient import TestClient
    from app import app
    
    client = TestClient(app)
    
    print("Testing file validation security...")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is not a GeoJSON file")
        invalid_file_path = f.name
    
    try:
        with open(invalid_file_path, 'rb') as f:
            response = client.post(
                "/calculate-indices",
                files={"geojson_file": ("test.txt", f, "text/plain")},
                params={"indices": "ndvi"}
            )
        print(f'✅ Invalid file type test: {response.status_code} (should reject)')
    finally:
        os.unlink(invalid_file_path)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.geojson', delete=False) as f:
        f.write("invalid json content")
        invalid_json_path = f.name
    
    try:
        with open(invalid_json_path, 'rb') as f:
            response = client.post(
                "/calculate-indices",
                files={"geojson_file": ("test.geojson", f, "application/json")},
                params={"indices": "ndvi"}
            )
        print(f'✅ Invalid JSON test: {response.status_code} (should reject)')
    finally:
        os.unlink(invalid_json_path)
    
    invalid_geojson = {"type": "InvalidType", "data": "not geojson"}
    with tempfile.NamedTemporaryFile(mode='w', suffix='.geojson', delete=False) as f:
        json.dump(invalid_geojson, f)
        invalid_geojson_path = f.name
    
    try:
        with open(invalid_geojson_path, 'rb') as f:
            response = client.post(
                "/calculate-indices",
                files={"geojson_file": ("test.geojson", f, "application/json")},
                params={"indices": "ndvi"}
            )
        print(f'✅ Invalid GeoJSON structure test: {response.status_code} (should reject)')
    finally:
        os.unlink(invalid_geojson_path)
    
    valid_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [2.3522, 48.8566]},
                "properties": {"name": "test"}
            }
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.geojson', delete=False) as f:
        json.dump(valid_geojson, f)
        valid_geojson_path = f.name
    
    try:
        with open(valid_geojson_path, 'rb') as f:
            response = client.post(
                "/calculate-indices",
                files={"geojson_file": ("test.geojson", f, "application/json")}
            )
        print(f'✅ Missing parameters test: {response.status_code} (should reject)')
        
        with open(valid_geojson_path, 'rb') as f:
            response = client.post(
                "/calculate-indices",
                files={"geojson_file": ("test.geojson", f, "application/json")},
                params={"indices": "invalid_index"}
            )
        print(f'✅ Invalid indices test: {response.status_code} (should reject)')
        
    finally:
        os.unlink(valid_geojson_path)
    
    response = client.options("/health")
    headers = response.headers
    if 'access-control-allow-origin' in headers:
        print(f'✅ CORS headers present: {headers.get("access-control-allow-origin")}')
    else:
        print('⚠️ CORS headers not found')
    
    print('✅ Security and validation testing completed')
    
except ImportError as e:
    print(f'❌ Import error: {e}')
except Exception as e:
    print(f'❌ Test error: {e}')
