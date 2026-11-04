#!/usr/bin/env python3
"""
run_benchmark.py
────────────────────────────────────────────────────────────────────────────
A simple runner for compare_pt_vs_tess.py that checks for dependencies 
and ensures the environment is ready.
"""

import os
import sys
import subprocess

def check_dependencies():
    print("Checking dependencies...")
    try:
        import pytesseract
        import torch
        import PIL
        print("  [OK] All Python dependencies found.")
    except ImportError as e:
        print(f"  [ERROR] Missing dependency: {e}")
        print("  Please run: pip install pytesseract torch Pillow")
        return False
    return True

def check_files():
    print("Checking for required files and directories...")
    test_dir = "data/lao-ground-truth-test"
    if not os.path.exists(test_dir):
        print(f"  [WARNING] Directory '{test_dir}' not found.")
        print(f"  Creating it now... please add your *.png and *.gt.txt files there.")
        os.makedirs(test_dir, exist_ok=True)
    else:
        files = os.listdir(test_dir)
        gt_files = [f for f in files if f.endswith('.gt.txt')]
        if not gt_files:
            print(f"  [WARNING] No *.gt.txt files found in '{test_dir}'.")
            print("  The benchmark will fail if no test data is provided.")
        else:
            print(f"  [OK] Found {len(gt_files)} test pairs in '{test_dir}'.")
    return True

def run():
    if not check_dependencies():
        return
    
    check_files()
    
    print("\nStarting Benchmark: python3 compare_pt_vs_tess.py")
    print("-" * 50)
    
    try:
        # Run the main comparison script
        subprocess.run([sys.executable, "compare_pt_vs_tess.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Benchmark script failed with exit code {e.returncode}")
    except KeyboardInterrupt:
        print("\n[INFO] Benchmark interrupted by user.")

if __name__ == "__main__":
    run()
