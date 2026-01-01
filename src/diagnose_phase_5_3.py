#!/usr/bin/env python3
"""
DIAGNOSTIC: Phase 5.3 Pre-check
Verifies current agent framework state before SUPERVISOR enhancement
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_diagnostics():
    """Run all diagnostic checks"""
    print("=" * 70)
    print("TRIALPULSE NEXUS 10X - PHASE 5.3 PRE-CHECK DIAGNOSTIC")
    print("=" * 70)
    
    results = {}
    
    # Check 1: Verify existing agent files
    print("\n[1/6] Checking existing agent files...")
    agent_files = [
        "src/agents/__init__.py",
        "src/agents/state.py",
        "src/agents/tools.py",
        "src/agents/base_agent.py",
        "src/agents/llm_wrapper.py",
        "src/agents/orchestrator.py"
    ]
    
    for f in agent_files:
        path = project_root / f
        exists = path.exists()
        status = "✅" if exists else "❌"
        print(f"  {status} {f}")
        results[f] = exists
    
    # Check 2: Import AgentState
    print("\n[2/6] Importing AgentState...")
    try:
        from src.agents.state import AgentState, Message, Hypothesis, Recommendation
        print("  ✅ AgentState imported successfully")
        print(f"     Fields: {list(AgentState.__annotations__.keys())[:10]}...")
        results['agent_state'] = True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        results['agent_state'] = False
    
    # Check 3: Import ToolRegistry
    print("\n[3/6] Importing ToolRegistry...")
    try:
        from src.agents.tools import ToolRegistry
        registry = ToolRegistry()
        tools = registry.list_tools()
        print(f"  ✅ ToolRegistry imported: {len(tools)} tools available")
        print(f"     Tools: {tools[:5]}...")
        results['tool_registry'] = True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        results['tool_registry'] = False
    
    # Check 4: Import BaseAgent
    print("\n[4/6] Importing BaseAgent...")
    try:
        from src.agents.base_agent import BaseAgent
        print("  ✅ BaseAgent imported successfully")
        results['base_agent'] = True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        results['base_agent'] = False
    
    # Check 5: Check current orchestrator structure
    print("\n[5/6] Analyzing current orchestrator...")
    try:
        orchestrator_path = project_root / "src/agents/orchestrator.py"
        content = orchestrator_path.read_text(encoding='utf-8')
        
        # Check for existing agent classes
        agents_found = []
        for agent in ['SupervisorAgent', 'DiagnosticAgent', 'ForecasterAgent', 
                      'ResolverAgent', 'CommunicatorAgent', 'SynthesizerAgent']:
            if f'class {agent}' in content:
                agents_found.append(agent)
        
        print(f"  ✅ Found {len(agents_found)} agent classes:")
        for a in agents_found:
            print(f"     - {a}")
        
        # Check for routing logic
        has_routing = 'route_to_specialist' in content or 'next_agent' in content
        print(f"  {'✅' if has_routing else '⚠️'} Routing logic: {'Found' if has_routing else 'Not found'}")
        
        results['orchestrator'] = True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        results['orchestrator'] = False
    
    # Check 6: Verify LLM is accessible
    print("\n[6/6] Checking LLM availability...")
    try:
        import httpx
        response = httpx.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [m.get('name', 'unknown') for m in models]
            print(f"  ✅ Ollama running, models: {model_names}")
            results['llm'] = True
        else:
            print(f"  ⚠️ Ollama responded with status {response.status_code}")
            results['llm'] = False
    except Exception as e:
        print(f"  ⚠️ Ollama not accessible: {e}")
        print("     (Start with: ollama serve)")
        results['llm'] = False
    
    # Summary
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ ALL CHECKS PASSED - Ready for Phase 5.3")
    else:
        print("\n⚠️ SOME CHECKS FAILED - Review issues above")
        failed = [k for k, v in results.items() if not v]
        print(f"   Failed: {failed}")
    
    # Current SupervisorAgent analysis
    print("\n" + "-" * 70)
    print("CURRENT SUPERVISOR AGENT ANALYSIS")
    print("-" * 70)
    
    try:
        orchestrator_path = project_root / "src/agents/orchestrator.py"
        content = orchestrator_path.read_text(encoding='utf-8')
        
        # Find SupervisorAgent class
        if 'class SupervisorAgent' in content:
            # Extract the class (rough)
            start = content.find('class SupervisorAgent')
            end = content.find('\nclass ', start + 1)
            if end == -1:
                end = len(content)
            supervisor_code = content[start:end]
            
            # Analyze features
            features = {
                'Task Decomposition': 'decompos' in supervisor_code.lower() or 'subtask' in supervisor_code.lower(),
                'Agent Routing': 'route' in supervisor_code.lower() or 'next_agent' in supervisor_code.lower(),
                'Keyword Detection': 'keyword' in supervisor_code.lower() or 'why' in supervisor_code or 'when' in supervisor_code,
                'Conflict Resolution': 'conflict' in supervisor_code.lower() or 'priority' in supervisor_code.lower(),
                'Output Synthesis': 'synthes' in supervisor_code.lower() or 'combine' in supervisor_code.lower(),
            }
            
            print("\nCurrent SupervisorAgent features:")
            for feature, has_it in features.items():
                status = "✅" if has_it else "❌ NEEDS"
                print(f"  {status} {feature}")
            
            print("\n📋 PHASE 5.3 ENHANCEMENT PLAN:")
            print("  1. Enhanced task decomposition (complex → subtasks)")
            print("  2. Intelligent agent routing (not just keywords)")
            print("  3. Conflict resolution between agents")
            print("  4. Output synthesis and deduplication")
            print("  5. Parallel agent execution support")
            print("  6. Confidence-based routing")
            
    except Exception as e:
        print(f"  ❌ Could not analyze: {e}")
    
    return all(results.values())

if __name__ == "__main__":
    success = run_diagnostics()
    sys.exit(0 if success else 1)