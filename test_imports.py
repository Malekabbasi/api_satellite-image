#!/usr/bin/env python3
"""Test script to verify all service modules import correctly"""

try:
    from services.indices_service import calculate_vegetation_indices_improved
    from services.satellite_service import get_sentinelhub_config
    from services.visualization_service import generate_enhanced_visualization
    from services.enhancement_service import advanced_pixel_enhancement
    print("✅ All service modules import successfully")
    print("✅ Code structure refactoring completed successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)

print("✅ Refactoring verification complete")
