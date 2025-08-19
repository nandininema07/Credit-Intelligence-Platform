"""
Test runner for Stage 5 alerting workflows.
"""

import pytest
import sys
import os
import asyncio
from pathlib import Path

def main():
    """Run Stage 5 alerting workflows tests"""
    
    # Add the project root to Python path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    # Test configuration
    test_args = [
        '--verbose',
        '--tb=short',
        '--asyncio-mode=auto',
        '--cov=stage5_alerting_workflows',
        '--cov-report=html',
        '--cov-report=term-missing',
        '--junit-xml=test_results.xml'
    ]
    
    # Add test directory
    test_dir = Path(__file__).parent
    test_args.append(str(test_dir))
    
    print("=" * 60)
    print("Running Stage 5 Alerting Workflows Tests")
    print("=" * 60)
    
    # Run tests
    exit_code = pytest.main(test_args)
    
    if exit_code == 0:
        print("\n" + "=" * 60)
        print("✅ All tests passed successfully!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ Some tests failed. Check the output above.")
        print("=" * 60)
    
    return exit_code

if __name__ == '__main__':
    sys.exit(main())
