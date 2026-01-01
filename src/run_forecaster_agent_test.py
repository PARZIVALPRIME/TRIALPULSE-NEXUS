# src/run_forecaster_agent_test.py
"""
Test runner for Enhanced FORECASTER Agent v1.0
"""

import sys
import time
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.forecaster_enhanced import (
    EnhancedForecasterAgent,
    get_forecaster_agent,
    ForecastType,
    ConfidenceLevel,
    ScenarioType,
    MonteCarloSimulator,
    WhatIfSimulator
)


def print_separator(title: str = ""):
    print("\n" + "=" * 70)
    if title:
        print(f" {title}")
        print("=" * 70)


def test_data_loader():
    """Test that data loader can access required files."""
    print_separator("TEST 1: Data Loader Verification")
    
    from src.agents.forecaster_enhanced import ForecastDataLoader
    
    loader = ForecastDataLoader()
    
    results = {}
    
    print("\nChecking data sources:")
    
    if loader.patient_issues is not None:
        print(f"  ✅ patient_issues: {len(loader.patient_issues)} rows")
        results['patient_issues'] = True
    else:
        print(f"  ❌ patient_issues: NOT FOUND")
        results['patient_issues'] = False
    
    if loader.site_benchmarks is not None:
        print(f"  ✅ site_benchmarks: {len(loader.site_benchmarks)} rows")
        results['site_benchmarks'] = True
    else:
        print(f"  ❌ site_benchmarks: NOT FOUND")
        results['site_benchmarks'] = False
    
    if loader.upr is not None:
        print(f"  ✅ upr: {len(loader.upr)} rows")
        results['upr'] = True
    else:
        print(f"  ❌ upr: NOT FOUND")
        results['upr'] = False
    
    # Test metric aggregation
    print("\nTesting metric aggregation:")
    
    portfolio = loader.get_portfolio_metrics()
    if portfolio:
        print(f"  ✅ Portfolio metrics: {portfolio.get('total_patients', 0):,} patients")
        print(f"     Total issues: {portfolio.get('total_issues', 0):,}")
        results['portfolio_metrics'] = True
    else:
        print(f"  ❌ Portfolio metrics: FAILED")
        results['portfolio_metrics'] = False
    
    passed = sum(results.values())
    total = len(results)
    print(f"\nData Sources: {passed}/{total} available")
    
    return passed >= 3


def test_monte_carlo_simulator():
    """Test Monte Carlo simulation engine."""
    print_separator("TEST 2: Monte Carlo Simulator")
    
    mc = MonteCarloSimulator(n_simulations=1000)
    
    # Test issue resolution simulation
    print("\nSimulating issue resolution (100 issues, 5/day rate):")
    
    result = mc.simulate_resolution_time(
        current_issues=100,
        resolution_rate_mean=5,
        resolution_rate_std=1.5
    )
    
    print(f"  Point Estimate: {result.point_estimate:.1f} days")
    print(f"  95% CI: [{result.lower_bound_95:.1f}, {result.upper_bound_95:.1f}]")
    print(f"  80% CI: [{result.lower_bound_80:.1f}, {result.upper_bound_80:.1f}]")
    print(f"  Standard Error: {result.standard_error:.3f}")
    print(f"  Confidence Level: {result.confidence_level.value}")
    
    # Validate results
    assert result.point_estimate > 0, "Point estimate should be positive"
    assert result.lower_bound_95 < result.point_estimate < result.upper_bound_95, "CI should contain point estimate"
    
    # Test clean patient trajectory
    print("\nSimulating clean patient trajectory (50% → 95%):")
    
    uncertainty, prob_success = mc.simulate_clean_patient_trajectory(
        current_clean_rate=0.50,
        target_clean_rate=0.95,
        daily_improvement_mean=0.005,
        daily_improvement_std=0.002
    )
    
    print(f"  Days to 95%: {uncertainty.point_estimate:.1f}")
    print(f"  95% CI: [{uncertainty.lower_bound_95:.1f}, {uncertainty.upper_bound_95:.1f}]")
    print(f"  Probability of Success: {prob_success:.1%}")
    
    # Test DB Lock simulation
    print("\nSimulating DB Lock readiness:")
    
    dblock_result = mc.simulate_dblock_readiness(
        current_ready_rate=0.30,
        current_pending_rate=0.40,
        current_blocked_rate=0.30,
        daily_conversion_rate=0.05,
        conversion_std=0.02
    )
    
    print(f"  Days to 95% ready: {dblock_result['days_to_95_ready'].point_estimate:.1f}")
    print(f"  Probability within 90 days: {dblock_result['probability_within_90_days']:.1%}")
    print(f"  Expected final rate: {dblock_result['expected_final_rate']:.1%}")
    
    return True


def test_db_lock_forecast():
    """Test DB Lock forecasting."""
    print_separator("TEST 3: DB Lock Forecast")
    
    agent = get_forecaster_agent()
    
    # Portfolio forecast
    print("\nForecasting DB Lock for portfolio:")
    
    start = time.time()
    result = agent.forecast_db_lock("portfolio")
    duration = time.time() - start
    
    print(f"\nDuration: {duration:.2f}s")
    print(f"Result ID: {result.result_id}")
    print(f"Forecasts: {len(result.forecasts)}")
    
    if result.forecasts:
        fc = result.forecasts[0]
        print(f"\n--- FORECAST ---")
        print(f"Type: {fc.forecast_type.value}")
        print(f"Current Issues: {fc.current_value:.0f}")
        print(f"Days to Resolution: {fc.predicted_value:.0f}")
        print(f"95% CI: [{fc.uncertainty.lower_bound_95:.0f}, {fc.uncertainty.upper_bound_95:.0f}]")
        print(f"Confidence: {fc.uncertainty.confidence_level.value}")
        print(f"Expected Date: {fc.prediction_date.strftime('%Y-%m-%d')}")
    
    print(f"\n--- SUMMARY ---")
    print(result.summary)
    
    print(f"\n--- RECOMMENDATIONS ---")
    for i, rec in enumerate(result.recommendations, 1):
        print(f"{i}. {rec}")
    
    return len(result.forecasts) > 0


def test_db_lock_with_target():
    """Test DB Lock forecast with target date."""
    print_separator("TEST 4: DB Lock Forecast with Target Date")
    
    agent = get_forecaster_agent()
    
    # Set target 90 days from now
    target_date = datetime.now() + timedelta(days=90)
    
    print(f"\nTarget Date: {target_date.strftime('%Y-%m-%d')} (90 days from now)")
    
    result = agent.forecast_db_lock("portfolio", target_date=target_date)
    
    if result.milestones:
        ms = result.milestones[0]
        print(f"\n--- MILESTONE STATUS ---")
        print(f"Target: {ms.target_date.strftime('%Y-%m-%d')}")
        print(f"Predicted: {ms.predicted_date.strftime('%Y-%m-%d')}")
        print(f"Status: {ms.status.upper()}")
        print(f"Probability On-Time: {ms.probability_on_time:.1%}")
        print(f"Days Variance: {'+' if ms.days_variance > 0 else ''}{ms.days_variance}")
        
        if ms.blockers:
            print(f"\nBlockers:")
            for b in ms.blockers:
                print(f"  - {b}")
        
        if ms.accelerators:
            print(f"\nAccelerators:")
            for a in ms.accelerators:
                print(f"  - {a}")
    
    return len(result.milestones) > 0


def test_clean_patient_forecast():
    """Test clean patient forecasting."""
    print_separator("TEST 5: Clean Patient Forecast")
    
    agent = get_forecaster_agent()
    
    print("\nForecasting clean patient trajectory to 95%:")
    
    result = agent.forecast_clean_patient("portfolio", target_rate=0.95)
    
    print(f"\n--- SUMMARY ---")
    print(result.summary)
    
    if result.forecasts:
        fc = result.forecasts[0]
        print(f"\nCurrent Clean Rate: {fc.current_value:.1%}")
        print(f"Days to 95%: {fc.predicted_value:.0f}")
        print(f"95% CI: [{fc.uncertainty.lower_bound_95:.0f}, {fc.uncertainty.upper_bound_95:.0f}]")
    
    return len(result.forecasts) > 0


def test_issue_resolution_forecast():
    """Test issue-specific resolution forecast."""
    print_separator("TEST 6: Issue Resolution Forecast")
    
    agent = get_forecaster_agent()
    
    issue_types = ['sdv_incomplete', 'open_queries', 'signature_gaps']
    
    for issue_type in issue_types:
        print(f"\n--- {issue_type.upper()} ---")
        
        result = agent.forecast_issue_resolution(issue_type, "portfolio")
        
        if result.forecasts:
            fc = result.forecasts[0]
            print(f"Current Count: {fc.current_value:.0f}")
            print(f"Days to Resolve: {fc.predicted_value:.0f}")
            print(f"95% CI: [{fc.uncertainty.lower_bound_95:.0f}, {fc.uncertainty.upper_bound_95:.0f}]")
        else:
            print("No forecast generated")
    
    return True


def test_what_if_scenarios():
    """Test what-if scenario simulations."""
    print_separator("TEST 7: What-If Scenarios")
    
    agent = get_forecaster_agent()
    
    # Test adding CRA
    print("\n--- SCENARIO: Add 1 CRA ---")
    result = agent.run_what_if('add_resource', {'resource_type': 'cra', 'count': 1})
    
    if result.scenarios:
        sc = result.scenarios[0]
        print(f"Description: {sc.description}")
        print(f"Baseline: {sc.baseline_outcome.get('days_to_resolution', 'N/A')} days")
        print(f"Scenario: {sc.scenario_outcome.get('days_to_resolution', 'N/A')} days")
        print(f"Days Saved: {sc.impact.get('days_saved', 0)}")
        print(f"Probability of Success: {sc.probability_of_success:.0%}")
        print(f"Cost Estimate: ${sc.cost_estimate:,.0f}")
    
    # Test adding Data Manager
    print("\n--- SCENARIO: Add 2 Data Managers ---")
    result = agent.run_what_if('add_resource', {'resource_type': 'data_manager', 'count': 2})
    
    if result.scenarios:
        sc = result.scenarios[0]
        print(f"Days Saved: {sc.impact.get('days_saved', 0)}")
        print(f"Improvement: {sc.impact.get('percent_improvement', 0):.1f}%")
    
    # Test process change
    print("\n--- SCENARIO: Process Improvement (15%) ---")
    result = agent.run_what_if('process_change', {
        'process_name': 'query_resolution_workflow',
        'improvement': 0.15
    })
    
    if result.scenarios:
        sc = result.scenarios[0]
        print(f"Description: {sc.description}")
        print(f"Days Saved: {sc.impact.get('days_saved', 0)}")
        print(f"Recommendations:")
        for rec in sc.recommendations:
            print(f"  - {rec}")
    
    return True


def test_natural_language_queries():
    """Test natural language query parsing."""
    print_separator("TEST 8: Natural Language Queries")
    
    agent = get_forecaster_agent()
    
    queries = [
        "When will we be ready for database lock?",
        "How long to resolve SDV backlog?",
        "What if we add a CRA?",
        "When will we reach 95% clean patients?",
        "Forecast query resolution for Study_21"
    ]
    
    for query in queries:
        print(f"\n📝 Query: {query}")
        print("-" * 50)
        
        start = time.time()
        result = agent.forecast_from_query(query)
        duration = time.time() - start
        
        print(f"Duration: {duration:.2f}s")
        
        if result.forecasts:
            fc = result.forecasts[0]
            print(f"Type: {fc.forecast_type.value}")
            print(f"Prediction: {fc.predicted_value:.0f} days")
        elif result.scenarios:
            sc = result.scenarios[0]
            print(f"Scenario: {sc.scenario_type.value}")
            print(f"Impact: {sc.impact.get('days_saved', 0)} days saved")
        else:
            print("No forecast/scenario generated")
    
    return True


def test_study_level_forecast():
    """Test study-level forecasting."""
    print_separator("TEST 9: Study-Level Forecast")
    
    agent = get_forecaster_agent()
    
    # Get a study with data
    if agent.data_loader.patient_issues is not None:
        studies = agent.data_loader.patient_issues['study_id'].unique()
        test_study = studies[0] if len(studies) > 0 else "Study_21"
        
        print(f"\nForecasting DB Lock for {test_study}:")
        
        result = agent.forecast_db_lock(test_study)
        
        if result.forecasts:
            fc = result.forecasts[0]
            print(f"Entity: {fc.entity_id}")
            print(f"Entity Type: {fc.entity_type}")
            print(f"Days to Resolution: {fc.predicted_value:.0f}")
            print(f"95% CI: [{fc.uncertainty.lower_bound_95:.0f}, {fc.uncertainty.upper_bound_95:.0f}]")
        
        print(f"\n--- SUMMARY ---")
        print(result.summary[:500] + "..." if len(result.summary) > 500 else result.summary)
        
        return len(result.forecasts) > 0
    
    return False


def run_all_tests():
    """Run all forecaster agent tests."""
    print("\n" + "=" * 70)
    print(" ENHANCED FORECASTER AGENT v1.0 - TEST SUITE")
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
    
    # Test 2: Monte Carlo Simulator
    try:
        results['monte_carlo'] = test_monte_carlo_simulator()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results['monte_carlo'] = False
    
    # Test 3: DB Lock Forecast
    try:
        results['db_lock_forecast'] = test_db_lock_forecast()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results['db_lock_forecast'] = False
    
    # Test 4: DB Lock with Target
    try:
        results['db_lock_target'] = test_db_lock_with_target()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results['db_lock_target'] = False
    
    # Test 5: Clean Patient Forecast
    try:
        results['clean_patient'] = test_clean_patient_forecast()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results['clean_patient'] = False
    
    # Test 6: Issue Resolution
    try:
        results['issue_resolution'] = test_issue_resolution_forecast()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results['issue_resolution'] = False
    
    # Test 7: What-If Scenarios
    try:
        results['what_if'] = test_what_if_scenarios()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results['what_if'] = False
    
    # Test 8: Natural Language Queries
    try:
        results['nl_queries'] = test_natural_language_queries()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results['nl_queries'] = False
    
    # Test 9: Study-Level Forecast
    try:
        results['study_level'] = test_study_level_forecast()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results['study_level'] = False
    
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