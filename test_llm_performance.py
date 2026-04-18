#!/usr/bin/env python
"""Standalone LLM performance testing script."""

import asyncio
import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'yt_navigator.settings')
django.setup()

from app.services.testing.llm_metrics import LLMMetricsCollector, DEFAULT_TEST_CASES


async def main():
    """Main test runner."""
    
    print("Starting LLM Performance Tests...")
    print("=" * 50)
    
    # Initialize metrics collector
    collector = LLMMetricsCollector()
    
    # Run test suite
    results = await collector.run_test_suite(DEFAULT_TEST_CASES)
    
    # Generate report
    report = collector.generate_report(results)
    
    # Save to file
    report_file = f"llm_performance_report_{results['total_tests']}tests.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    # Display results
    print(f"\nTest Results:")
    print(f"Total Tests: {results['total_tests']}")
    print(f"Passed: {results['passed_tests']}")
    print(f"Failed: {results['failed_tests']}")
    print(f"Success Rate: {results['success_rate']:.2%}")
    print(f"Average Response Time: {results['average_response_time']:.2f}s")
    
    if results['metrics']:
        avg_relevance = sum(m['relevance_score'] for m in results['metrics']) / len(results['metrics'])
        avg_completeness = sum(m['completeness_score'] for m in results['metrics']) / len(results['metrics'])
        avg_accuracy = sum(m['accuracy_score'] for m in results['metrics']) / len(results['metrics'])
        
        print(f"Average Relevance: {avg_relevance:.2f}")
        print(f"Average Completeness: {avg_completeness:.2f}")
        print(f"Average Accuracy: {avg_accuracy:.2f}")
    
    print(f"\nDetailed report saved to: {report_file}")
    
    # Save metrics
    metrics_file = f"llm_metrics_{results['total_tests']}tests.json"
    collector.save_metrics(metrics_file)
    print(f"Detailed metrics saved to: {metrics_file}")
    
    # Exit with appropriate code
    sys.exit(0 if results['failed_tests'] == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
