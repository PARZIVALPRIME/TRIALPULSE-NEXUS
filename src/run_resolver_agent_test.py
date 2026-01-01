# src/run_resolver_agent_test.py
"""
Test runner for Enhanced RESOLVER Agent v1.0
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.resolver_enhanced import (
    EnhancedResolverAgent,
    get_resolver_agent,
    ResolutionGenome,
    CascadeAnalyzer,
    ActionPriority,
    ActionCategory,
    ResponsibleRole
)


def print_separator(title: str = ""):
    print("\n" + "=" * 70)
    if title:
        print(f" {title}")
        print("=" * 70)


def test_data_loader():
    """Test data loader."""
    print_separator("TEST 1: Data Loader Verification")
    
    from src.agents.resolver_enhanced import ResolverDataLoader
    
    loader = ResolverDataLoader()
    results = {}
    
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
    
    # Test site issues retrieval
    print("\nTesting site issue retrieval:")
    if loader.patient_issues is not None:
        sites = loader.patient_issues['site_id'].unique()
        test_site = sites[0] if len(sites) > 0 else "Site_1"
        
        site_issues = loader.get_site_issues(test_site)
        if site_issues:
            print(f"  ✅ {test_site}: {site_issues.get('total_issues', 0)} issues")
            results['site_retrieval'] = True
        else:
            print(f"  ❌ Site retrieval failed")
            results['site_retrieval'] = False
    
    passed = sum(results.values())
    total = len(results)
    print(f"\nData Sources: {passed}/{total} available")
    
    return passed >= 3


def test_resolution_genome():
    """Test resolution genome templates."""
    print_separator("TEST 2: Resolution Genome")
    
    genome = ResolutionGenome()
    
    print(f"\nTotal templates: {genome._count_templates()}")
    
    # Test template retrieval
    issue_types = ['sdv_incomplete', 'open_queries', 'signature_gaps', 'sae_dm_pending']
    
    print("\nTemplates by issue type:")
    for issue_type in issue_types:
        templates = genome.get_templates_for_issue(issue_type)
        print(f"  {issue_type}: {len(templates)} templates")
        
        if templates:
            best = genome.get_best_template(issue_type)
            print(f"    Best: {best.title} (success: {best.success_rate:.0%})")
    
    # Test search
    print("\nSearch results for 'signature':")
    results = genome.search_templates("signature")
    for template in results[:3]:
        print(f"  - {template.title} ({template.issue_type})")
    
    return genome._count_templates() > 20


def test_cascade_analyzer():
    """Test cascade impact analyzer."""
    print_separator("TEST 3: Cascade Analyzer")
    
    analyzer = CascadeAnalyzer()
    
    # Test cascade calculation
    test_cases = [
        ('missing_visits', 10, "Site_1"),
        ('open_queries', 50, "Site_2"),
        ('sae_dm_pending', 5, "Site_3")
    ]
    
    print("\nCascade Impact Analysis:")
    for issue_type, count, entity in test_cases:
        impact = analyzer.calculate_cascade_impact(issue_type, count, entity)
        
        print(f"\n{issue_type.upper()} ({count} issues at {entity}):")
        print(f"  Direct Score: {impact.direct_impact.get('direct_score', 0):.1f}")
        print(f"  Cascade Effects: {len(impact.cascade_effects)}")
        print(f"  Total Issues Resolved: {impact.total_issues_resolved}")
        print(f"  Patients Unblocked: {impact.total_patients_unblocked}")
        print(f"  DQI Improvement: +{impact.dqi_improvement:.1f}")
        print(f"  DB Lock Acceleration: {impact.db_lock_acceleration_days:.1f} days")
        print(f"  ROI Score: {impact.roi_score:.1f}")
    
    return True


def test_site_action_plan():
    """Test site-level action plan generation."""
    print_separator("TEST 4: Site Action Plan")
    
    agent = get_resolver_agent()
    
    # Find a site with issues
    if agent.data_loader.patient_issues is not None:
        df = agent.data_loader.patient_issues
        site_issues = df.groupby('site_id')['total_issues'].sum().sort_values(ascending=False)
        test_site = site_issues.index[0] if len(site_issues) > 0 else "Site_1"
        
        print(f"\nGenerating action plan for {test_site}:")
        print(f"Known issues: {int(site_issues.iloc[0]) if len(site_issues) > 0 else 'Unknown'}")
        
        start = time.time()
        plan = agent.create_action_plan_for_site(test_site)
        duration = time.time() - start
        
        print(f"\nDuration: {duration:.2f}s")
        print(f"Plan ID: {plan.plan_id}")
        print(f"Total Actions: {len(plan.actions)}")
        print(f"Quick Wins: {len(plan.quick_wins)}")
        print(f"Total Effort: {plan.total_effort_hours:.1f} hours")
        print(f"Timeline: ~{plan.timeline_days} days")
        
        # Show actions by priority
        print("\nActions by Priority:")
        for priority, actions in plan.get_actions_by_priority().items():
            print(f"  {priority.upper()}: {len(actions)}")
        
        # Show actions by role
        print("\nActions by Role:")
        for role, actions in plan.get_actions_by_role().items():
            print(f"  {role}: {len(actions)}")
        
        # Show top 3 actions
        print("\nTop 3 Actions:")
        for i, action in enumerate(plan.actions[:3], 1):
            print(f"\n{i}. {action.title}")
            print(f"   Priority: {action.priority.value} | Impact: {action.impact_score:.0f}")
            print(f"   Role: {action.responsible_role.value}")
            print(f"   Effort: {action.effort_hours:.1f} hrs")
            if action.steps:
                print(f"   Steps: {len(action.steps)}")
        
        return len(plan.actions) > 0
    
    return False


def test_portfolio_action_plan():
    """Test portfolio-level action plan."""
    print_separator("TEST 5: Portfolio Action Plan")
    
    agent = get_resolver_agent()
    
    print("\nGenerating portfolio action plan (top 15 actions):")
    
    start = time.time()
    plan = agent.create_portfolio_action_plan(top_n=15)
    duration = time.time() - start
    
    print(f"\nDuration: {duration:.2f}s")
    print(f"Total Actions: {len(plan.actions)}")
    print(f"Total Effort: {plan.total_effort_hours:.1f} hours")
    
    # Show distribution
    print("\nActions by Site (sample):")
    site_counts = {}
    for action in plan.actions:
        site = action.entity_id
        site_counts[site] = site_counts.get(site, 0) + 1
    
    for site, count in sorted(site_counts.items(), key=lambda x: -x[1])[:5]:
        print(f"  {site}: {count} actions")
    
    print("\nTop 5 Portfolio Actions:")
    for i, action in enumerate(plan.actions[:5], 1):
        print(f"{i}. [{action.entity_id}] {action.issue_type}")
        print(f"   Impact: {action.impact_score:.0f} | Effort: {action.effort_hours:.1f}h")
    
    return len(plan.actions) > 0


def test_quick_wins():
    """Test quick wins identification."""
    print_separator("TEST 6: Quick Wins Identification")
    
    agent = get_resolver_agent()
    
    print("\nIdentifying quick wins (max 2 hrs effort, impact >= 30):")
    
    start = time.time()
    quick_wins = agent.identify_quick_wins("portfolio", max_effort=2.0)
    duration = time.time() - start
    
    print(f"\nDuration: {duration:.2f}s")
    print(f"Quick Wins Found: {len(quick_wins)}")
    
    if quick_wins:
        total_effort = sum(a.effort_hours for a in quick_wins)
        total_impact = sum(a.impact_score for a in quick_wins)
        
        print(f"Total Effort: {total_effort:.1f} hours")
        print(f"Total Impact: {total_impact:.0f}")
        print(f"Average ROI: {total_impact / max(total_effort, 0.1):.1f}")
        
        print("\nTop 5 Quick Wins:")
        for i, action in enumerate(quick_wins[:5], 1):
            roi = action.impact_score / max(action.effort_hours, 0.1)
            print(f"{i}. {action.title}")
            print(f"   Effort: {action.effort_hours:.1f}h | Impact: {action.impact_score:.0f} | ROI: {roi:.1f}")
    
    return len(quick_wins) > 0


def test_natural_language_queries():
    """Test natural language query parsing."""
    print_separator("TEST 7: Natural Language Queries")
    
    agent = get_resolver_agent()
    
    queries = [
        "How do I resolve issues at Site_1?",
        "What are the quick wins for our portfolio?",
        "Show me the cascade impact for Site_3",
        "Create an action plan for the SDV backlog"
    ]
    
    for query in queries:
        print(f"\n📝 Query: {query}")
        print("-" * 50)
        
        start = time.time()
        result = agent.resolve_from_query(query)
        duration = time.time() - start
        
        print(f"Duration: {duration:.2f}s")
        print(f"Action Plans: {len(result.action_plans)}")
        print(f"Top Actions: {len(result.top_actions)}")
        print(f"Cascade Analysis: {len(result.cascade_analysis)}")
        
        if result.recommendations:
            print(f"Recommendations: {result.recommendations[0][:60]}...")
    
    return True


def test_action_creation():
    """Test action creation from templates."""
    print_separator("TEST 8: Action Creation")
    
    agent = get_resolver_agent()
    genome = ResolutionGenome()
    
    # Test creating actions from templates
    test_cases = [
        ('sdv_incomplete', 25),
        ('sae_dm_pending', 3),
        ('open_queries', 100)
    ]
    
    print("\nCreating actions from templates:")
    
    for issue_type, count in test_cases:
        template = genome.get_best_template(issue_type)
        if template:
            action = agent.create_action_from_template(
                template, "Site_TEST", "site", count
            )
            
            print(f"\n{issue_type} ({count} issues):")
            print(f"  Action: {action.title}")
            print(f"  Priority: {action.priority.value}")
            print(f"  Category: {action.category.value}")
            print(f"  Role: {action.responsible_role.value}")
            print(f"  Effort: {action.effort_hours:.1f} hours")
            print(f"  Impact: {action.impact_score:.0f}")
            print(f"  Success Rate: {action.success_rate:.0%}")
            print(f"  Due: {action.due_date.strftime('%Y-%m-%d')}")
            print(f"  Requires Approval: {action.requires_approval}")
    
    return True


def test_cascade_in_plan():
    """Test cascade impacts in action plans."""
    print_separator("TEST 9: Cascade Impacts in Plans")
    
    agent = get_resolver_agent()
    
    # Get a site with issues
    if agent.data_loader.patient_issues is not None:
        df = agent.data_loader.patient_issues
        sites = df['site_id'].unique()
        test_site = sites[0] if len(sites) > 0 else "Site_1"
        
        plan = agent.create_action_plan_for_site(test_site)
        
        print(f"\nCascade Analysis for {test_site}:")
        print(f"Total Cascade Impacts: {len(plan.cascade_impacts)}")
        
        if plan.cascade_impacts:
            # Sort by ROI
            sorted_cascades = sorted(plan.cascade_impacts, key=lambda c: -c.roi_score)
            
            print("\nTop Cascade Opportunities:")
            for i, cascade in enumerate(sorted_cascades[:3], 1):
                print(f"\n{i}. {cascade.source_issue}")
                print(f"   Direct: {cascade.direct_impact.get('issues_resolved', 0)} issues")
                print(f"   Downstream: {len(cascade.cascade_effects)} effect types")
                print(f"   Total Unblocked: {cascade.total_issues_resolved} issues")
                print(f"   ROI: {cascade.roi_score:.1f}")
            
            return len(plan.cascade_impacts) > 0
    
    return False


def run_all_tests():
    """Run all resolver agent tests."""
    print("\n" + "=" * 70)
    print(" ENHANCED RESOLVER AGENT v1.0 - TEST SUITE")
    print("=" * 70)
    
    results = {}
    
    # Test 1: Data Loader
    try:
        results['data_loader'] = test_data_loader()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results['data_loader'] = False
    
    # Test 2: Resolution Genome
    try:
        results['resolution_genome'] = test_resolution_genome()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results['resolution_genome'] = False
    
    # Test 3: Cascade Analyzer
    try:
        results['cascade_analyzer'] = test_cascade_analyzer()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results['cascade_analyzer'] = False
    
    # Test 4: Site Action Plan
    try:
        results['site_action_plan'] = test_site_action_plan()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results['site_action_plan'] = False
    
    # Test 5: Portfolio Action Plan
    try:
        results['portfolio_plan'] = test_portfolio_action_plan()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results['portfolio_plan'] = False
    
    # Test 6: Quick Wins
    try:
        results['quick_wins'] = test_quick_wins()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results['quick_wins'] = False
    
    # Test 7: Natural Language
    try:
        results['nl_queries'] = test_natural_language_queries()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results['nl_queries'] = False
    
    # Test 8: Action Creation
    try:
        results['action_creation'] = test_action_creation()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results['action_creation'] = False
    
    # Test 9: Cascade in Plans
    try:
        results['cascade_in_plans'] = test_cascade_in_plan()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results['cascade_in_plans'] = False
    
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