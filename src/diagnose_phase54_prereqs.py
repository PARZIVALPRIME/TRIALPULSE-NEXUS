"""
DIAGNOSTIC AGENT PRE-CHECK
Run this to verify all dependencies for Phase 5.4
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

def check_dependencies():
    """Check all dependencies for Diagnostic Agent"""
    
    print("=" * 70)
    print("PHASE 5.4 DIAGNOSTIC AGENT - DEPENDENCY CHECK")
    print("=" * 70)
    
    results = {}
    
    # 1. Check data files
    print("\n📁 DATA FILES:")
    data_files = {
        "patient_issues": "data/processed/analytics/patient_issues.parquet",
        "patient_cascade": "data/processed/analytics/patient_cascade_analysis.parquet",
        "causal_hypotheses": "data/processed/analytics/causal_hypotheses/causal_hypotheses.json",
        "pattern_matches": "data/processed/analytics/pattern_library/pattern_matches.parquet",
        "site_benchmarks": "data/processed/analytics/site_benchmarks.parquet",
        "patient_anomalies": "data/processed/analytics/patient_anomalies.parquet",
    }
    
    for name, path in data_files.items():
        exists = Path(path).exists()
        status = "✅" if exists else "❌"
        print(f"   {status} {name}: {path}")
        results[f"data_{name}"] = exists
    
    # 2. Check existing engines
    print("\n🔧 EXISTING ENGINES:")
    engines = [
        ("CausalHypothesisEngine", "src.knowledge.causal_hypothesis_engine", "CausalHypothesisEngine"),
        ("PatternLibrary", "src.ml.pattern_library", "PatternLibrary"),
        ("IssueDetector", "src.ml.issue_detector", "IssueDetector"),
        ("ResolutionGenome", "src.ml.resolution_genome", "ResolutionGenome"),
    ]
    
    for name, module, class_name in engines:
        try:
            mod = __import__(module, fromlist=[class_name])
            cls = getattr(mod, class_name)
            print(f"   ✅ {name}: {module}.{class_name}")
            results[f"engine_{name}"] = True
        except Exception as e:
            print(f"   ❌ {name}: {e}")
            results[f"engine_{name}"] = False
    
    # 3. Check tools
    print("\n🛠️ AGENT TOOLS:")
    try:
        from src.agents.tools import ToolRegistry
        registry = ToolRegistry()
        tools = registry.list_tools()
        print(f"   ✅ ToolRegistry loaded: {len(tools)} tools")
        
        diagnostic_tools = ["search_patterns", "get_cascade_impact", "get_patient", "get_site_summary"]
        for tool in diagnostic_tools:
            if tool in tools:
                print(f"      ✅ {tool}")
            else:
                print(f"      ❌ {tool} NOT FOUND")
        results["tools"] = True
    except Exception as e:
        print(f"   ❌ ToolRegistry: {e}")
        results["tools"] = False
    
    # 4. Check LLM
    print("\n🤖 LLM WRAPPER:")
    try:
        from src.agents.llm_wrapper import get_llm
        llm = get_llm()
        health = llm.health_check()
        if health.get("ollama", {}).get("status") == "healthy":
            print(f"   ✅ Ollama: healthy")
            results["llm"] = True
        else:
            print(f"   ⚠️ Ollama: {health}")
            results["llm"] = False
    except Exception as e:
        print(f"   ❌ LLM: {e}")
        results["llm"] = False
    
    # 5. Check base agent
    print("\n🏗️ BASE AGENT:")
    try:
        from src.agents.base_agent import BaseAgent
        print(f"   ✅ BaseAgent imported")
        results["base_agent"] = True
    except Exception as e:
        print(f"   ❌ BaseAgent: {e}")
        results["base_agent"] = False
    
    # 6. Check state schema
    print("\n📊 STATE SCHEMA:")
    try:
        from src.agents.state import AgentState, Hypothesis, Message
        print(f"   ✅ AgentState, Hypothesis, Message imported")
        results["state"] = True
    except Exception as e:
        print(f"   ❌ State: {e}")
        results["state"] = False
    
    # 7. Load sample data for testing
    print("\n📈 SAMPLE DATA CHECK:")
    try:
        import pandas as pd
        
        # Load patient issues
        issues_df = pd.read_parquet("data/processed/analytics/patient_issues.parquet")
        patients_with_issues = issues_df[issues_df['has_issues'] == True] if 'has_issues' in issues_df.columns else issues_df[issues_df['issue_count'] > 0]
        print(f"   ✅ Patients with issues: {len(patients_with_issues)}")
        
        # Load cascade analysis
        cascade_df = pd.read_parquet("data/processed/analytics/patient_cascade_analysis.parquet")
        print(f"   ✅ Cascade analysis: {len(cascade_df)} patients")
        
        # Check issue columns
        issue_cols = [c for c in issues_df.columns if c.startswith('has_') or c.startswith('issue_')]
        print(f"   ✅ Issue columns: {len(issue_cols)}")
        
        results["sample_data"] = True
    except Exception as e:
        print(f"   ❌ Sample data: {e}")
        results["sample_data"] = False
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\n   Checks Passed: {passed}/{total}")
    
    if passed == total:
        print("\n   ✅ ALL CHECKS PASSED - Ready for Phase 5.4")
        print("\n   Run: python src/run_diagnostic_agent.py")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"\n   ❌ FAILED CHECKS: {failed}")
        print("\n   Please fix the issues above before proceeding.")
    
    return results


if __name__ == "__main__":
    check_dependencies()