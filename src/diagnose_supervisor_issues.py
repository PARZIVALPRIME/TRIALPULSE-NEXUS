#!/usr/bin/env python3
"""
DIAGNOSTIC: Phase 5.3 - Supervisor Issues Analysis
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.supervisor_enhanced import EnhancedSupervisorAgent, QueryIntent

def diagnose_issues():
    """Diagnose the supervisor issues found in testing"""
    
    print("=" * 70)
    print("SUPERVISOR AGENT - ISSUE DIAGNOSIS")
    print("=" * 70)
    
    supervisor = EnhancedSupervisorAgent()
    
    # Test the problematic query
    query = "Why are queries increasing, when will they be resolved, and draft an escalation email"
    
    print(f"\n📋 QUERY: {query}")
    print("-" * 70)
    
    # Step 1: Check intent detection
    print("\n[1] INTENT DETECTION:")
    intent_scores = supervisor._detect_intents(query.lower())
    
    for intent, score in sorted(intent_scores.items(), key=lambda x: x[1], reverse=True):
        status = "✅" if score > 0.3 else "⚠️"
        print(f"  {status} {intent.value}: {score:.2f}")
    
    # Check what keywords matched
    print("\n[2] KEYWORD MATCHES:")
    query_lower = query.lower()
    
    for intent, config in supervisor.INTENT_PATTERNS.items():
        matches = [kw for kw in config['keywords'] if kw in query_lower]
        if matches:
            print(f"  {intent.value}: {matches}")
    
    # Step 2: Check compound detection
    print("\n[3] COMPOUND DETECTION:")
    high_score_intents = [i for i, s in intent_scores.items() if s > 0.5]
    print(f"  Intents with score > 0.5: {[i.value for i in high_score_intents]}")
    print(f"  Should be COMPOUND: {len(high_score_intents) > 1}")
    
    # Step 3: Check routing decision
    print("\n[4] ROUTING DECISION:")
    routing = supervisor.analyze_query(query)
    print(f"  Primary Intent: {routing.primary_intent.value}")
    print(f"  Secondary Intents: {[i.value for i in routing.secondary_intents]}")
    print(f"  Complexity: {routing.complexity.value}")
    print(f"  Agent Sequence: {routing.agent_sequence}")
    
    # Step 4: Check subtask generation
    print("\n[5] SUBTASK GENERATION:")
    subtasks = supervisor.decompose_task(query, routing)
    print(f"  Total subtasks: {len(subtasks)}")
    for st in subtasks:
        print(f"    [{st.priority}] {st.assigned_agent}: {st.description[:50]}...")
    
    # Identify the issue
    print("\n" + "=" * 70)
    print("DIAGNOSIS RESULT")
    print("=" * 70)
    
    issues = []
    
    # Issue 1: Communication not detected
    if QueryIntent.COMMUNICATION not in intent_scores or intent_scores.get(QueryIntent.COMMUNICATION, 0) < 0.3:
        issues.append("❌ COMMUNICATION intent not detected (need 'draft' pattern)")
    
    # Issue 2: Not detected as compound
    if routing.primary_intent != QueryIntent.COMPOUND and len(high_score_intents) > 1:
        issues.append("❌ Should be COMPOUND but detected as " + routing.primary_intent.value)
    
    # Issue 3: Missing subtasks
    expected_agents = {'diagnostic', 'forecaster', 'resolver', 'communicator', 'synthesizer'}
    actual_agents = {st.assigned_agent for st in subtasks}
    missing = expected_agents - actual_agents
    if missing:
        issues.append(f"❌ Missing subtasks for: {missing}")
    
    # Issue 4: Decomposition logic
    if routing.complexity.value in ['complex', 'moderate'] and len(subtasks) < 4:
        issues.append(f"❌ Complex query has only {len(subtasks)} subtasks (expected 4+)")
    
    if issues:
        print("\n🔴 ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✅ NO ISSUES FOUND")
    
    # Root cause analysis
    print("\n📌 ROOT CAUSES:")
    print("  1. 'draft' keyword matches COMMUNICATION but pattern score may be low")
    print("  2. decompose_task() only adds subtasks based on primary_intent")
    print("  3. Need to check secondary_intents in decomposition logic")
    
    return issues

if __name__ == "__main__":
    diagnose_issues()