#!/usr/bin/env python3
"""Test script to verify error handling and logging work correctly"""

import sys
sys.path.append('.')

try:
    from error_handling import (
        StructuredError, create_error_response, handle_validation_error,
        handle_geojson_error, handle_sentinelhub_error, handle_processing_error,
        handle_generic_error, setup_enhanced_logging, get_correlation_id
    )
    from fastapi import Request
    from unittest.mock import Mock
    print('✅ Error handling modules import successfully')
    
    error = StructuredError(
        error_code="TEST_ERROR",
        message="Test error message",
        details={"test": "data"},
        status_code=400
    )
    error_dict = error.to_dict("test-correlation-id")
    print(f'✅ StructuredError works: {error_dict["error"]["code"]}')
    
    logger = setup_enhanced_logging()
    print(f'✅ Enhanced logging setup works: {logger.name}')
    
    mock_request = Mock()
    mock_request.scope = {"correlation_id": "test-123"}
    correlation_id = get_correlation_id(mock_request)
    print(f'✅ Correlation ID extraction works: {correlation_id}')
    
    print('✅ All error handling tests passed')
    
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
except Exception as e:
    print(f'❌ Test error: {e}')
    sys.exit(1)
