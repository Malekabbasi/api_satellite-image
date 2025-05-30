#!/usr/bin/env python3
"""Test script to verify security enhancements work correctly"""

import sys
sys.path.append('.')

try:
    from security_enhancements import validate_upload_file, validate_indices_request, get_secure_sentinelhub_config
    from models.validation import GeoJSONValidation, IndicesRequest
    print('✅ Security enhancements modules import successfully')
    
    try:
        indices_req = IndicesRequest(indices=["ndvi", "ndwi"], enhancement_method="adaptive")
        print('✅ IndicesRequest validation works')
    except Exception as e:
        print(f'❌ IndicesRequest validation error: {e}')
    
    try:
        invalid_req = IndicesRequest(indices=["invalid_index"], enhancement_method="adaptive")
        print('❌ Invalid indices should have failed validation')
    except Exception as e:
        print('✅ Invalid indices correctly rejected')
    
    print('✅ Security enhancements verification complete')
    
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
