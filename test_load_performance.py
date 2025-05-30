#!/usr/bin/env python3
"""Load and performance testing for the API"""

import sys
import time
import threading
import requests
import concurrent.futures
import statistics
sys.path.append('.')

try:
    import uvicorn
    from app import app
    
    def start_server():
        uvicorn.run(app, host='127.0.0.1', port=8002, log_level='error')
    
    print("Starting performance test server...")
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    time.sleep(3)
    
    def make_request(endpoint):
        """Make a single request and return response time"""
        start_time = time.time()
        try:
            response = requests.get(f'http://127.0.0.1:8002{endpoint}', timeout=10)
            end_time = time.time()
            return {
                'status_code': response.status_code,
                'response_time': end_time - start_time,
                'success': response.status_code == 200
            }
        except Exception as e:
            return {
                'status_code': 0,
                'response_time': 10.0,
                'success': False,
                'error': str(e)
            }
    
    endpoints = ['/health', '/', '/enhancement-methods', '/health/detailed']
    
    print("Testing individual endpoint performance...")
    for endpoint in endpoints:
        results = []
        for i in range(5):
            result = make_request(endpoint)
            results.append(result)
            time.sleep(0.1)
        
        response_times = [r['response_time'] for r in results if r['success']]
        success_rate = sum(1 for r in results if r['success']) / len(results)
        
        if response_times:
            avg_time = statistics.mean(response_times)
            print(f'✅ {endpoint}: {avg_time:.3f}s avg, {success_rate:.1%} success')
        else:
            print(f'❌ {endpoint}: No successful requests')
    
    print("\nTesting concurrent load (10 requests)...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request, '/health') for _ in range(10)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    successful_results = [r for r in results if r['success']]
    success_rate = len(successful_results) / len(results)
    
    if successful_results:
        avg_time = statistics.mean([r['response_time'] for r in successful_results])
        max_time = max([r['response_time'] for r in successful_results])
        print(f'✅ Concurrent test: {avg_time:.3f}s avg, {max_time:.3f}s max, {success_rate:.1%} success')
    else:
        print('❌ Concurrent test: No successful requests')
    
    print("\nTesting rate limiting...")
    rapid_results = []
    for i in range(15):  # More than the 10/minute limit
        result = make_request('/health')
        rapid_results.append(result)
        time.sleep(0.1)
    
    status_codes = [r['status_code'] for r in rapid_results]
    rate_limited = sum(1 for code in status_codes if code == 429)
    
    if rate_limited > 0:
        print(f'✅ Rate limiting working: {rate_limited} requests rate limited')
    else:
        print(f'⚠️ Rate limiting: {rate_limited} requests rate limited (may need adjustment)')
    
    print('✅ Performance and load testing completed')
    
except ImportError as e:
    print(f'❌ Import error: {e}')
except Exception as e:
    print(f'❌ Test error: {e}')
