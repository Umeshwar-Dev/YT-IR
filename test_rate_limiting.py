#!/usr/bin/env python3
"""
Test script to verify rate limiting improvements.
This simulates the conditions that were causing 429 errors.
"""

import asyncio
import time
import random
from app.services.content_generator.notes_generator import NotesGenerator
from app.services.content_generator.mindmap_generator import MindmapGenerator

async def test_rate_limiting():
    """Test the improved rate limiting implementation."""
    print("Testing improved rate limiting implementation...")
    
    # Sample transcript that would previously cause rate limiting
    sample_transcript = " ".join(["This is a test transcript."] * 2000)  # ~40K chars
    
    # Test notes generator
    print("\n=== Testing Notes Generator ===")
    notes_gen = NotesGenerator()
    
    try:
        start_time = time.time()
        notes = notes_gen.generate_notes(sample_transcript, "Test Video")
        end_time = time.time()
        
        print(f"✅ Notes generation completed in {end_time - start_time:.2f}s")
        print(f"   Main topics: {len(notes.get('main_topics', []))}")
        print(f"   Key concepts: {len(notes.get('key_concepts', []))}")
        
    except Exception as e:
        print(f"❌ Notes generation failed: {e}")
    
    # Test mindmap generator  
    print("\n=== Testing Mindmap Generator ===")
    mindmap_gen = MindmapGenerator()
    
    try:
        start_time = time.time()
        mindmap = mindmap_gen.generate_mindmap(sample_transcript, "Test Video")
        end_time = time.time()
        
        print(f"✅ Mindmap generation completed in {end_time - start_time:.2f}s")
        print(f"   Branches: {len(mindmap.get('branches', []))}")
        
    except Exception as e:
        print(f"❌ Mindmap generation failed: {e}")
    
    # Test batch processing (the main source of rate limiting)
    print("\n=== Testing Batch Processing Simulation ===")
    
    # Simulate multiple rapid requests (like the original logs showed)
    async def simulate_batch_request(batch_num):
        print(f"Starting batch {batch_num}...")
        try:
            # Small transcript to simulate batch processing
            batch_transcript = " ".join(["Batch content."] * 500)
            notes = notes_gen.generate_notes(batch_transcript, f"Batch {batch_num}")
            print(f"✅ Batch {batch_num} completed")
            return notes
        except Exception as e:
            print(f"❌ Batch {batch_num} failed: {e}")
            return None
    
    # Run multiple batches concurrently to test rate limiting
    print("Running 5 concurrent batches...")
    start_time = time.time()
    
    tasks = [simulate_batch_request(i) for i in range(1, 6)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    end_time = time.time()
    successful = sum(1 for r in results if r is not None and not isinstance(r, Exception))
    
    print(f"\n=== Results ===")
    print(f"Total time: {end_time - start_time:.2f}s")
    print(f"Successful batches: {successful}/5")
    print(f"Rate limiting improvements: ✅ Working" if successful >= 4 else "❌ Still having issues")

if __name__ == "__main__":
    asyncio.run(test_rate_limiting())
