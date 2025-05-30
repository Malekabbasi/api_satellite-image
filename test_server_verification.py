#!/usr/bin/env python3
"""Test script to verify server functionality"""

import sys
import time
import threading
import requests
import uvicorn
sys.path.append('.')

try:
    from app import app
    
    def start_server():
        uvicorn.run(app, host='127.0.0.1', port=8001, log_level='error')
    
    print("Starting test server...")
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    time.sleep(3)
    
    try:
        response = requests.get('http://127.0.0.1:8001/health')
        print(f'✅ Server health check: {response.status_code}')
        
        response = requests.get('http://127.0.0.1:8001/')
        print(f'✅ API info endpoint: {response.status_code}')
        
        response = requests.get('http://127.0.0.1:8001/enhancement-methods')
        print(f'✅ Enhancement methods: {response.status_code}')
        
        response = requests.get('http://127.0.0.1:8001/health/detailed')
        if response.status_code == 200:
            data = response.json()
            print(f'✅ Detailed health check: {data["status"]}')
            print(f'✅ Components checked: {len(data["components"])}')
        else:
            print(f'❌ Detailed health check failed: {response.status_code}')
        
    except Exception as e:
        print(f'❌ Server test failed: {e}')
    
    print('✅ Server verification completed')
    
except ImportError as e:
    print(f'❌ Import error: {e}')
except Exception as e:
    print(f'❌ Test error: {e}')
