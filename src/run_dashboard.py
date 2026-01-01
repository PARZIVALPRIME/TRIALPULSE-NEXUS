"""
TRIALPULSE NEXUS 10X - Dashboard Runner
Phase 7.1: Entry point for running the Streamlit dashboard
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run the Streamlit dashboard."""
    
    # Get the path to the dashboard app
    dashboard_path = Path(__file__).parent.parent / "dashboard" / "app.py"
    
    if not dashboard_path.exists():
        print(f"❌ Dashboard app not found at: {dashboard_path}")
        sys.exit(1)
    
    print("🚀 Starting TRIALPULSE NEXUS 10X Dashboard...")
    print(f"📁 App location: {dashboard_path}")
    print("=" * 60)
    print()
    print("🌐 Dashboard will be available at: http://localhost:8501")
    print()
    print("📝 Demo Credentials:")
    print("   Study Lead: lead / demo123")
    print("   CRA: cra / demo123")
    print("   Data Manager: dm / demo123")
    print("   Safety Lead: safety / demo123")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    print()
    
    # Run Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(dashboard_path),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false",
            "--theme.primaryColor", "#3498db",
            "--theme.backgroundColor", "#ffffff",
            "--theme.secondaryBackgroundColor", "#f8f9fa",
            "--theme.textColor", "#2c3e50"
        ], check=True)
    except KeyboardInterrupt:
        print("\n\n👋 Dashboard stopped.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running dashboard: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()