"""
Diagnostic script to debug agent framework issues
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd


def diagnose_data_files():
    """Check what columns exist in the data files"""
    print("=" * 70)
    print("TRIALPULSE NEXUS - DATA DIAGNOSTICS")
    print("=" * 70)
    
    data_dir = PROJECT_ROOT / "data" / "processed"
    
    # Files to check
    files_to_check = [
        ("analytics/patient_issues.parquet", "Patient Issues"),
        ("analytics/patient_dqi_enhanced.parquet", "Patient DQI"),
        ("analytics/patient_clean_status.parquet", "Clean Status"),
        ("analytics/patient_dblock_status.parquet", "DB Lock Status"),
        ("analytics/patient_cascade_analysis.parquet", "Cascade Analysis"),
        ("analytics/patient_anomalies.parquet", "Anomalies"),
        ("analytics/site_benchmarks.parquet", "Site Benchmarks"),
        ("upr/unified_patient_record.parquet", "Unified Patient Record"),
    ]
    
    results = {}
    
    for filepath, name in files_to_check:
        full_path = data_dir / filepath
        print(f"\n{'─' * 70}")
        print(f"📁 {name}")
        print(f"   Path: {filepath}")
        print("─" * 70)
        
        if full_path.exists():
            try:
                df = pd.read_parquet(full_path)
                print(f"   ✅ Exists: {len(df)} rows × {len(df.columns)} columns")
                print(f"\n   Columns ({len(df.columns)}):")
                
                # Group columns for readability
                cols = sorted(df.columns.tolist())
                for i, col in enumerate(cols):
                    dtype = str(df[col].dtype)
                    sample = df[col].iloc[0] if len(df) > 0 else "N/A"
                    if isinstance(sample, (str,)) and len(str(sample)) > 30:
                        sample = str(sample)[:30] + "..."
                    print(f"      {i+1:3d}. {col:<40} ({dtype:<10}) = {sample}")
                
                results[name] = {
                    "exists": True,
                    "rows": len(df),
                    "columns": df.columns.tolist()
                }
                
                # Check for key columns needed by tools
                key_columns = ["patient_key", "study_id", "site_id", "issue_count", 
                               "priority_tier", "primary_issue", "dqi_score", "risk_level"]
                print(f"\n   Key Columns Check:")
                for col in key_columns:
                    status = "✅" if col in df.columns else "❌"
                    print(f"      {status} {col}")
                    
            except Exception as e:
                print(f"   ❌ Error reading: {e}")
                results[name] = {"exists": True, "error": str(e)}
        else:
            print(f"   ❌ File not found")
            results[name] = {"exists": False}
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for name, info in results.items():
        if info.get("exists") and "columns" in info:
            print(f"✅ {name}: {info['rows']} rows, {len(info['columns'])} cols")
        elif info.get("exists"):
            print(f"⚠️ {name}: Error - {info.get('error', 'Unknown')}")
        else:
            print(f"❌ {name}: Not found")
    
    return results


def diagnose_langgraph_state():
    """Check LangGraph state handling"""
    print("\n" + "=" * 70)
    print("LANGGRAPH STATE DIAGNOSTICS")
    print("=" * 70)
    
    try:
        from langgraph.graph import StateGraph, END
        print("✅ LangGraph imported successfully")
        
        # Check what StateGraph.stream returns
        from src.agents.state import AgentState, create_initial_state, TaskPriority
        
        state = create_initial_state("Test query", priority=TaskPriority.MEDIUM)
        print(f"\n   Initial state type: {type(state)}")
        print(f"   Is AgentState: {isinstance(state, AgentState)}")
        
        # Check state dict conversion
        state_dict = state.model_dump()
        print(f"   model_dump() type: {type(state_dict)}")
        print(f"   Keys: {list(state_dict.keys())[:10]}...")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def suggest_fixes():
    """Suggest fixes based on diagnostics"""
    print("\n" + "=" * 70)
    print("SUGGESTED FIXES")
    print("=" * 70)
    
    print("""
ISSUE 1: 'issue_count' column not found
─────────────────────────────────────────
The tools.py file expects 'issue_count' column but it may have a different name.

FIX: Update tools.py to use the correct column names from the actual data.

ISSUE 2: 'dict' object has no attribute 'task_status'  
─────────────────────────────────────────
LangGraph's stream() returns dict, not AgentState object.

FIX: Convert dict back to AgentState in orchestrator.run() method.

ISSUE 3: Multiple Ollama connections
─────────────────────────────────────────
Each agent creates a new LLM connection (6 connections).

FIX: Share a single LLM instance across all agents.
""")


def main():
    print("\n🔍 Running Diagnostics...\n")
    
    # 1. Check data files
    results = diagnose_data_files()
    
    # 2. Check LangGraph
    diagnose_langgraph_state()
    
    # 3. Suggest fixes
    suggest_fixes()
    
    print("\n" + "=" * 70)
    print("📋 COPY THE OUTPUT ABOVE AND SHARE IT")
    print("=" * 70)


if __name__ == "__main__":
    main()