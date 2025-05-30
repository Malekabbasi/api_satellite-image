#!/usr/bin/env python3
"""Test script to verify performance optimizations work correctly"""

import sys
sys.path.append('.')

try:
    from performance_optimizations import (
        vectorized_vegetation_indices, 
        get_cache_stats,
        memory_efficient_image_processing
    )
    import numpy as np
    print('✅ Performance optimization modules import successfully')
    
    test_image = np.random.rand(100, 100, 4).astype(np.float32)
    indices = ["ndvi", "ndwi", "savi", "evi"]
    
    try:
        results = vectorized_vegetation_indices(test_image, indices)
        print(f'✅ Vectorized indices calculation works: {list(results.keys())}')
    except Exception as e:
        print(f'❌ Vectorized indices calculation error: {e}')
    
    try:
        stats = get_cache_stats()
        print(f'✅ Cache stats available: {stats.get("available", False)}')
    except Exception as e:
        print(f'❌ Cache stats error: {e}')
    
    print('✅ Performance optimizations verification complete')
    
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
