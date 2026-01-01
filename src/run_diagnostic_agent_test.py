# src/run_diagnostic_agent_test.py
"""
Test runner for Enhanced DIAGNOSTIC Agent v1.0
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import directly from the module to avoid __init__.py issues
from src.agents.diagnostic_enhanced import (
    EnhancedDiagnosticAgent,
    get_diagnostic_agent,
    InvestigationType,
    EvidenceStrength,
    Evidence,
    EvidenceChain,
    DiagnosticHypothesis,
    InvestigationResult
)


def print_separator(title: str = ""):
    print("\n" + "=" * 70)
    if title:
        print(f" {title}")
        print("=" * 70)


def test_site_investigation():
    """Test site-level investigation."""
    print_separator("TEST 1: Site Investigation")
    
    agent = get_diagnostic_agent()
    
    # Find a site with issues
    if agent.data_loader.patient_issues is not None:
        df = agent.data_loader.patient_issues
        # Get site with most issues
        site_issues = df.groupby('site_id')['total_issues'].sum().sort_values(ascending=False)
        test_site = site_issues.index[0] if len(site_issues) > 0 else "Site_1"
        
        print(f"\nInvestigating: {test_site}")
        print(f"Known issues at site: {int(site_issues.iloc[0]) if len(site_issues) > 0 else 'Unknown'}")
        
        start = time.time()
        result = agent.investigate_site(test_site)
        duration = time.time() - start
        
        print(f"\nDuration: {duration:.2f}s")
        print(f"Investigation ID: {result.investigation_id}")
        print(f"Data Sources: {result.data_sources_consulted}")
        print(f"Hypotheses Generated: {len(result.hypotheses)}")
        print(f"Patterns Matched: {len(result.patterns_matched)}")
        
        print("\n--- SUMMARY ---")
        print(result.summary)
        
        print("\n--- RECOMMENDATIONS ---")
        for i, rec in enumerate(result.recommendations, 1):
            print(f"{i}. {rec}")
        
        return True
    else:
        print("❌ Patient issues data not available")
        return False


def test_patient_investigation():
    """Test patient-level investigation."""
    print_separator("TEST 2: Patient Investigation")
    
    agent = get_diagnostic_agent()
    
    # Find a patient with issues
    if agent.data_loader.patient_issues is not None:
        df = agent.data_loader.patient_issues
        # Get patient with issues
        patients_with_issues = df[df['has_any_issue'] == True]
        
        if len(patients_with_issues) > 0:
            test_patient = patients_with_issues.iloc[0]['patient_key']
            
            print(f"\nInvestigating patient: {test_patient}")
            
            start = time.time()
            result = agent.investigate_patient(test_patient)
            duration = time.time() - start
            
            print(f"\nDuration: {duration:.2f}s")
            print(f"Investigation ID: {result.investigation_id}")
            print(f"Data Sources: {result.data_sources_consulted}")
            print(f"Hypotheses Generated: {len(result.hypotheses)}")
            
            print("\n--- SUMMARY ---")
            print(result.summary)
            
            if result.hypotheses:
                print("\n--- TOP HYPOTHESIS DETAILS ---")
                top_hyp = result.hypotheses[0]
                print(f"Title: {top_hyp.title}")
                print(f"Confidence: {top_hyp.confidence:.1%}")
                print(f"Evidence Count: {len(top_hyp.evidence_chain.evidence_list)}")
                print(f"Causal Pathway: {' → '.join(top_hyp.evidence_chain.causal_pathway)}")
                print(f"Confounders: {top_hyp.confounders}")
                print(f"Verification Steps: {top_hyp.verification_steps}")
            
            return True
        else:
            print("No patients with issues found")
            return False
    else:
        print("❌ Patient issues data not available")
        return False


def test_query_investigation():
    """Test natural language query investigation."""
    print_separator("TEST 3: Natural Language Query Investigation")
    
    agent = get_diagnostic_agent()
    
    test_queries = [
        "Why does Site_1 have so many open queries?",
        "What's causing the SDV backlog?",
        "Investigate signature gaps at the site level",
        "Why are SAE reconciliations pending?"
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        print("-" * 50)
        
        start = time.time()
        result = agent.investigate_query(query)
        duration = time.time() - start
        
        print(f"Type: {result.investigation_type.value}")
        print(f"Entity: {result.entity_id}")
        print(f"Hypotheses: {len(result.hypotheses)}")
        print(f"Duration: {duration:.2f}s")
        
        if result.hypotheses:
            top = result.hypotheses[0]
            print(f"Top Hypothesis: {top.title} ({top.confidence:.1%})")
    
    return True


def test_issue_type_investigation():
    """Test issue-type specific investigation."""
    print_separator("TEST 4: Issue Type Investigation")
    
    agent = get_diagnostic_agent()
    
    query = "Why do we have so many open queries across all sites?"
    
    print(f"\n📝 Query: {query}")
    
    start = time.time()
    result = agent.process(query)
    duration = time.time() - start
    
    print(f"\nDuration: {duration:.2f}s")
    print(f"Investigation Type: {result['investigation_type']}")
    print(f"Entity: {result['entity_id']}")
    print(f"Hypothesis Count: {result['hypothesis_count']}")
    
    print("\n--- SUMMARY ---")
    print(result['summary'])
    
    print("\n--- RECOMMENDATIONS ---")
    for i, rec in enumerate(result['recommendations'], 1):
        print(f"{i}. {rec}")
    
    return True


def test_evidence_chain_building():
    """Test evidence chain construction."""
    print_separator("TEST 5: Evidence Chain Building")
    
    # Create a chain
    chain = EvidenceChain(
        chain_id="CHN-TEST",
        hypothesis_id="HYP-TEST",
        causal_pathway=["High Workload", "Delayed Response", "Open Queries"]
    )
    
    # Add supporting evidence
    chain.add_evidence(Evidence(
        evidence_id="EVD-001",
        source="patient_issues",
        description="Site has 50 patients with open queries",
        strength=EvidenceStrength.STRONG,
        data_points={'count': 50},
        supports_hypothesis=True
    ))
    
    chain.add_evidence(Evidence(
        evidence_id="EVD-002",
        source="benchmarks",
        description="Site is in bottom 10% for query resolution",
        strength=EvidenceStrength.MODERATE,
        data_points={'percentile': 8},
        supports_hypothesis=True
    ))
    
    chain.add_evidence(Evidence(
        evidence_id="EVD-003",
        source="patterns",
        description="High query volume pattern matched",
        strength=EvidenceStrength.STRONG,
        data_points={'pattern_id': 'PAT-DQ-001'},
        supports_hypothesis=True
    ))
    
    # Add some counter-evidence
    chain.add_evidence(Evidence(
        evidence_id="EVD-004",
        source="site_data",
        description="Site has adequate staffing",
        strength=EvidenceStrength.WEAK,
        data_points={'staff_count': 5},
        supports_hypothesis=False  # Counter-evidence
    ))
    
    print(f"Chain ID: {chain.chain_id}")
    print(f"Evidence Count: {len(chain.evidence_list)}")
    print(f"Supporting: {len([e for e in chain.evidence_list if e.supports_hypothesis])}")
    print(f"Opposing: {len([e for e in chain.evidence_list if not e.supports_hypothesis])}")
    print(f"Overall Strength: {chain.overall_strength.value}")
    print(f"Confidence Score: {chain.confidence_score:.1%}")
    print(f"Confidence Interval: {chain.confidence_interval[0]:.1%} - {chain.confidence_interval[1]:.1%}")
    print(f"Causal Pathway: {' → '.join(chain.causal_pathway)}")
    
    return chain.confidence_score > 0.5


def test_data_loader():
    """Test that data loader can access all required files."""
    print_separator("TEST 6: Data Loader Verification")
    
    from src.agents.diagnostic_enhanced import DiagnosticDataLoader
    
    loader = DiagnosticDataLoader()
    
    results = {}
    
    # Check each data source
    print("\nChecking data sources:")
    
    if loader.patient_issues is not None:
        print(f"  ✅ patient_issues: {len(loader.patient_issues)} rows")
        results['patient_issues'] = True
    else:
        print(f"  ❌ patient_issues: NOT FOUND")
        results['patient_issues'] = False
    
    if loader.patient_cascade is not None:
        print(f"  ✅ patient_cascade: {len(loader.patient_cascade)} rows")
        results['patient_cascade'] = True
    else:
        print(f"  ❌ patient_cascade: NOT FOUND")
        results['patient_cascade'] = False
    
    if loader.site_benchmarks is not None:
        print(f"  ✅ site_benchmarks: {len(loader.site_benchmarks)} rows")
        results['site_benchmarks'] = True
    else:
        print(f"  ❌ site_benchmarks: NOT FOUND")
        results['site_benchmarks'] = False
    
    if loader.patient_anomalies is not None:
        print(f"  ✅ patient_anomalies: {len(loader.patient_anomalies)} rows")
        results['patient_anomalies'] = True
    else:
        print(f"  ❌ patient_anomalies: NOT FOUND")
        results['patient_anomalies'] = False
    
    if loader.pattern_matches is not None:
        print(f"  ✅ pattern_matches: {len(loader.pattern_matches)} rows")
        results['pattern_matches'] = True
    else:
        print(f"  ❌ pattern_matches: NOT FOUND")
        results['pattern_matches'] = False
    
    if loader.upr is not None:
        print(f"  ✅ upr: {len(loader.upr)} rows")
        results['upr'] = True
    else:
        print(f"  ❌ upr: NOT FOUND")
        results['upr'] = False
    
    passed = sum(results.values())
    total = len(results)
    print(f"\nData Sources: {passed}/{total} available")
    
    return passed >= 4  # At least 4 sources should be available


def run_all_tests():
    """Run all diagnostic agent tests."""
    print("\n" + "=" * 70)
    print(" ENHANCED DIAGNOSTIC AGENT v1.0 - TEST SUITE")
    print("=" * 70)
    
    results = {}
    
    # Test 0: Data Loader
    try:
        results['data_loader'] = test_data_loader()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results['data_loader'] = False
    
    # Test 1: Site Investigation
    try:
        results['site_investigation'] = test_site_investigation()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results['site_investigation'] = False
    
    # Test 2: Patient Investigation
    try:
        results['patient_investigation'] = test_patient_investigation()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results['patient_investigation'] = False
    
    # Test 3: Query Investigation
    try:
        results['query_investigation'] = test_query_investigation()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results['query_investigation'] = False
    
    # Test 4: Issue Type Investigation
    try:
        results['issue_type_investigation'] = test_issue_type_investigation()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results['issue_type_investigation'] = False
    
    # Test 5: Evidence Chain Building
    try:
        results['evidence_chain'] = test_evidence_chain_building()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results['evidence_chain'] = False
    
    # Summary
    print_separator("TEST RESULTS SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_flag in results.items():
        status = "✅ PASSED" if passed_flag else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)