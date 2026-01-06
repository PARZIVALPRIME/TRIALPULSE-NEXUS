"""
TRIALPULSE NEXUS 10X - Data Pipeline Initializer
Runs all required pipelines to generate processed data from raw files.
Used for Streamlit Cloud deployment and fresh installations.
"""

import sys
import time
import subprocess
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_data_exists() -> bool:
    """Check if processed data already exists."""
    analytics_dir = PROJECT_ROOT / "data" / "processed" / "analytics"
    required_files = [
        "patient_dqi_enhanced.parquet",
        "patient_clean_status.parquet",
        "patient_dblock_status.parquet",
        "site_benchmarks.parquet"
    ]
    
    if not analytics_dir.exists():
        return False
    
    for f in required_files:
        if not (analytics_dir / f).exists():
            return False
    
    return True


def run_pipeline(name: str, script_path: str) -> bool:
    """Run a single pipeline script."""
    print(f"\n{'='*60}")
    print(f"🔄 Running: {name}")
    print(f"{'='*60}")
    
    full_path = PROJECT_ROOT / script_path
    if not full_path.exists():
        print(f"❌ Script not found: {script_path}")
        return False
    
    start = time.time()
    try:
        # Run the script as a subprocess
        result = subprocess.run(
            [sys.executable, str(full_path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        elapsed = time.time() - start
        
        if result.returncode == 0:
            print(f"✅ {name} completed in {elapsed:.1f}s")
            return True
        else:
            print(f"❌ {name} failed (exit code {result.returncode})")
            if result.stderr:
                print(f"   Error: {result.stderr[:500]}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ {name} timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ {name} failed: {e}")
        return False


def initialize_data():
    """Run all pipelines to generate processed data."""
    print("\n" + "="*70)
    print("🚀 TRIALPULSE NEXUS 10X - DATA INITIALIZATION")
    print("="*70)
    
    if check_data_exists():
        print("✅ Processed data already exists. Skipping pipeline execution.")
        return True
    
    print("📊 Processed data not found. Running pipelines...")
    
    # Create required directories
    dirs = [
        "data/processed/analytics",
        "data/processed/cleaned",
        "data/processed/upr",
        "data/processed/metrics",
        "data/outputs",
        "logs",
        "models",
        "knowledge_base"
    ]
    for d in dirs:
        (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)
    
    # Pipeline sequence (these are the actual run_*.py scripts)
    pipelines = [
        ("1. Data Ingestion", "src/run_ingestion.py"),
        ("2. Data Cleaning", "src/run_cleaning.py"),
        ("3. UPR Builder", "src/run_upr_builder.py"),
        ("4. Metrics Engine", "src/run_metrics.py"),
        ("5. Enhanced DQI", "src/run_dqi_enhanced.py"),
        ("6. Clean Patient", "src/run_clean_patient.py"),
        ("7. DB Lock Ready", "src/run_dblock_ready.py"),
        ("8. Benchmark Engine", "src/run_benchmark.py"),
    ]
    
    total_start = time.time()
    success_count = 0
    
    for name, script in pipelines:
        if run_pipeline(name, script):
            success_count += 1
        else:
            print(f"⚠️ Pipeline {name} failed, continuing with next...")
    
    total_time = time.time() - total_start
    
    print("\n" + "="*70)
    print(f"📊 INITIALIZATION COMPLETE")
    print(f"   Pipelines: {success_count}/{len(pipelines)} successful")
    print(f"   Total time: {total_time:.1f}s")
    print("="*70 + "\n")
    
    return success_count == len(pipelines)


if __name__ == "__main__":
    success = initialize_data()
    sys.exit(0 if success else 1)