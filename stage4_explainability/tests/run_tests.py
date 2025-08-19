"""
Test runner for Stage 4 explainability module.
"""

import pytest
import sys
import os
import asyncio
from pathlib import Path

# Add the parent directory to the Python path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

def run_all_tests():
    """Run all tests for Stage 4 explainability"""
    
    test_files = [
        "test_xai_module.py",
        "test_chatbot_module.py", 
        "test_simulation_module.py",
        "test_integration.py"
    ]
    
    print("Running Stage 4 Explainability Tests...")
    print("=" * 50)
    
    # Run tests with verbose output
    args = [
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "--asyncio-mode=auto",  # Auto async mode
        "-x",  # Stop on first failure
    ]
    
    # Add test files
    for test_file in test_files:
        test_path = current_dir / test_file
        if test_path.exists():
            args.append(str(test_path))
    
    # Run pytest
    exit_code = pytest.main(args)
    
    if exit_code == 0:
        print("\n" + "=" * 50)
        print("✅ All tests passed successfully!")
    else:
        print("\n" + "=" * 50)
        print("❌ Some tests failed. Check output above.")
    
    return exit_code

def run_specific_module(module_name):
    """Run tests for a specific module"""
    
    module_map = {
        "xai": "test_xai_module.py",
        "chatbot": "test_chatbot_module.py",
        "simulation": "test_simulation_module.py",
        "integration": "test_integration.py"
    }
    
    if module_name not in module_map:
        print(f"Unknown module: {module_name}")
        print(f"Available modules: {', '.join(module_map.keys())}")
        return 1
    
    test_file = current_dir / module_map[module_name]
    
    if not test_file.exists():
        print(f"Test file not found: {test_file}")
        return 1
    
    print(f"Running tests for {module_name} module...")
    print("=" * 50)
    
    args = ["-v", "--tb=short", "--asyncio-mode=auto", str(test_file)]
    return pytest.main(args)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        module = sys.argv[1]
        exit_code = run_specific_module(module)
    else:
        exit_code = run_all_tests()
    
    sys.exit(exit_code)
