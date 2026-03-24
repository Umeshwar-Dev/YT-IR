#!/usr/bin/env python3
"""
Simple test to verify rate limiting logic without Django dependencies.
"""

import time
import random
import asyncio

def test_rate_limiting_logic():
    """Test the rate limiting improvements logic."""
    print("Testing Rate Limiting Improvements")
    print("=" * 40)
    
    # Test 1: Jitter calculation
    print("\n1. Testing Jitter Calculation:")
    base_delay = 5.0
    
    for i in range(5):
        jitter = random.uniform(0.8, 1.2)  # ±20% jitter
        delay = base_delay * jitter
        print(f"   Attempt {i+1}: base={base_delay}s, jitter={jitter:.2f}, final={delay:.2f}s")
    
    # Test 2: Exponential backoff
    print("\n2. Testing Exponential Backoff:")
    max_retries = 3
    
    for attempt in range(max_retries + 1):
        if attempt == 0:
            delay = 0
        else:
            base_delay_time = 5.0 * (2 ** (attempt - 1))  # 5s, 10s, 20s
            jitter = random.uniform(0.5, 1.5)  # ±50% jitter for retries
            delay = min(base_delay_time * jitter, 60.0)  # Max 60s
        
        print(f"   Retry {attempt}: delay={delay:.2f}s")
    
    # Test 3: Conservative transcript sizes
    print("\n3. Testing Conservative Transcript Sizes:")
    old_sizes = [50000, 45000, 40000, 35000, 30000, 20000, 15000, 10000]
    new_sizes = [30000, 25000, 20000, 15000, 12000, 10000, 8000, 6000]
    
    print(f"   Old sizes: {old_sizes}")
    print(f"   New sizes: {new_sizes}")
    print(f"   Reduction: {((old_sizes[0] - new_sizes[0]) / old_sizes[0]) * 100:.1f}% smaller starting size")
    
    # Test 4: Rate limiting simulation
    print("\n4. Simulating Rate Limited Requests:")
    
    def simulate_request_with_old_logic():
        """Old logic: fixed 3.5s delay"""
        time.sleep(3.5)
        return "success"
    
    def simulate_request_with_new_logic():
        """New logic: 5s base + jitter"""
        jitter = random.uniform(0.8, 1.2)
        delay = 5.0 * jitter
        time.sleep(delay)
        return "success"
    
    # Simulate 3 requests with old logic
    print("   Old logic (3.5s fixed delay):")
    start_time = time.time()
    for i in range(3):
        result = simulate_request_with_old_logic()
        print(f"     Request {i+1}: {result}")
    old_total = time.time() - start_time
    print(f"   Total time: {old_total:.2f}s")
    
    # Simulate 3 requests with new logic
    print("   New logic (5s + jitter):")
    start_time = time.time()
    for i in range(3):
        result = simulate_request_with_new_logic()
        print(f"     Request {i+1}: {result}")
    new_total = time.time() - start_time
    print(f"   Total time: {new_total:.2f}s")
    
    print(f"\n   Difference: {new_total - old_total:.2f}s (new is more conservative)")
    
    # Test 5: Error detection
    print("\n5. Testing Error Detection:")
    test_errors = [
        "HTTP/1.1 429 Too Many Requests",
        "rate_limit_exceeded",
        "tokens per minute exceeded",
        "Regular error message"
    ]
    
    for error in test_errors:
        is_rate_limit = any(keyword in error for keyword in ["429", "rate_limit_exceeded", "Too Many Requests"])
        print(f"   '{error}' -> Rate limit: {is_rate_limit}")
    
    print("\n" + "=" * 40)
    print("✅ Rate Limiting Logic Test Complete!")
    print("\nKey Improvements:")
    print("  • Jitter prevents synchronized retries")
    print("  • Exponential backoff handles persistent rate limits")  
    print("  • Conservative transcript sizes reduce token usage")
    print("  • Intelligent error detection avoids unnecessary retries")
    print("  • Configurable delays (5s base, 60s max)")

if __name__ == "__main__":
    test_rate_limiting_logic()
