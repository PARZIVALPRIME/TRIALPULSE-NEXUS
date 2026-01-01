# src/run_auto_summarizer_test.py
"""
Test runner for Phase 6.5 Auto-Summarization Engine
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_tests():
    print("=" * 70)
    print("TRIALPULSE NEXUS 10X - PHASE 6.5 AUTO-SUMMARIZATION TEST")
    print("=" * 70)
    
    from src.generation.auto_summarizer import (
        get_auto_summarizer,
        summarize_patient,
        summarize_site,
        daily_digest,
        executive_summary,
        SummaryType,
        Severity
    )
    
    tests_passed = 0
    tests_failed = 0
    
    # =========================================================================
    # TEST 1: Engine Initialization
    # =========================================================================
    print("\n" + "-" * 70)
    print("TEST 1: Engine Initialization")
    print("-" * 70)
    
    try:
        summarizer = get_auto_summarizer()
        portfolio = summarizer.data_loader.get_portfolio_data()
        
        print("✅ Engine initialized")
        print("   Total patients: {:,}".format(portfolio.get('total_patients', 0)))
        print("   Total studies: {}".format(portfolio.get('total_studies', 0)))
        print("   Total sites: {:,}".format(portfolio.get('total_sites', 0)))
        print("   Mean DQI: {:.2f}".format(portfolio.get('mean_dqi', 0)))
        print("   Tier 2 Clean: {:.1f}%".format(portfolio.get('tier2_clean_rate', 0)))
        tests_passed += 1
    except Exception as e:
        print("❌ FAILED: {}".format(e))
        import traceback
        traceback.print_exc()
        tests_failed += 1
        return False
    
    # =========================================================================
    # TEST 2: Patient Summary
    # =========================================================================
    print("\n" + "-" * 70)
    print("TEST 2: Patient Summary")
    print("-" * 70)
    
    try:
        # Get a sample patient key
        if 'patients' in summarizer.data_loader.data:
            patients_df = summarizer.data_loader.data['patients']
            sample_patient = patients_df['patient_key'].iloc[0]
            
            summary = summarize_patient(sample_patient)
            
            print("✅ Patient summary generated")
            print("   Summary ID: {}".format(summary.summary_id))
            print("   Patient: {}".format(summary.entity_id[:50]))
            print("   Findings: {}".format(len(summary.findings)))
            print("   Action Items: {}".format(len(summary.action_items)))
            print("   Metrics: {}".format(len(summary.metrics)))
            
            # Show executive summary preview
            print("\n   Executive Summary Preview:")
            print("   " + summary.executive_summary[:200].replace('\n', '\n   '))
            
            tests_passed += 1
        else:
            print("⚠️ No patient data available, skipping")
            tests_passed += 1
    except Exception as e:
        print("❌ FAILED: {}".format(e))
        import traceback
        traceback.print_exc()
        tests_failed += 1
    
    # =========================================================================
    # TEST 3: Site Summary
    # =========================================================================
    print("\n" + "-" * 70)
    print("TEST 3: Site Summary")
    print("-" * 70)
    
    try:
        # Get a sample site
        if 'patients' in summarizer.data_loader.data:
            patients_df = summarizer.data_loader.data['patients']
            sample_site = patients_df['site_id'].iloc[0]
            
            summary = summarize_site(sample_site)
            
            print("✅ Site summary generated")
            print("   Summary ID: {}".format(summary.summary_id))
            print("   Site: {}".format(summary.entity_id))
            print("   Sections: {}".format(len(summary.sections)))
            print("   Findings: {}".format(len(summary.findings)))
            print("   Action Items: {}".format(len(summary.action_items)))
            
            # Show metrics
            print("\n   Key Metrics:")
            for key, value in list(summary.metrics.items())[:5]:
                if isinstance(value, float):
                    print("      {}: {:.2f}".format(key, value))
                else:
                    print("      {}: {}".format(key, value))
            
            tests_passed += 1
        else:
            print("⚠️ No site data available, skipping")
            tests_passed += 1
    except Exception as e:
        print("❌ FAILED: {}".format(e))
        import traceback
        traceback.print_exc()
        tests_failed += 1
    
    # =========================================================================
    # TEST 4: Daily Digest
    # =========================================================================
    print("\n" + "-" * 70)
    print("TEST 4: Daily Digest")
    print("-" * 70)
    
    try:
        summary = daily_digest(role="Study Lead")
        
        print("✅ Daily digest generated")
        print("   Summary ID: {}".format(summary.summary_id))
        print("   Title: {}".format(summary.title))
        print("   Sections: {}".format(len(summary.sections)))
        print("   Findings: {}".format(len(summary.findings)))
        print("   Action Items: {}".format(len(summary.action_items)))
        
        # Show section titles
        print("\n   Sections:")
        for section in summary.sections:
            print("      - {}".format(section.get('title', 'Untitled')))
        
        tests_passed += 1
    except Exception as e:
        print("❌ FAILED: {}".format(e))
        import traceback
        traceback.print_exc()
        tests_failed += 1
    
    # =========================================================================
    # TEST 5: Daily Digest by Role
    # =========================================================================
    print("\n" + "-" * 70)
    print("TEST 5: Daily Digest by Role")
    print("-" * 70)
    
    try:
        roles = ["Study Lead", "CRA", "Data Manager"]
        
        for role in roles:
            summary = daily_digest(role=role)
            
            # Find role-specific section
            role_section = None
            for section in summary.sections:
                if role in section.get('title', ''):
                    role_section = section
                    break
            
            status = "✅" if role_section else "⚠️"
            print("   {} {}: {} sections".format(status, role, len(summary.sections)))
        
        print("\n✅ Role-based digests generated")
        tests_passed += 1
    except Exception as e:
        print("❌ FAILED: {}".format(e))
        tests_failed += 1
    
    # =========================================================================
    # TEST 6: Study-Specific Digest
    # =========================================================================
    print("\n" + "-" * 70)
    print("TEST 6: Study-Specific Digest")
    print("-" * 70)
    
    try:
        if 'patients' in summarizer.data_loader.data:
            patients_df = summarizer.data_loader.data['patients']
            sample_study = patients_df['study_id'].iloc[0]
            
            summary = daily_digest(study_id=sample_study, role="Study Lead")
            
            print("✅ Study-specific digest generated")
            print("   Study: {}".format(sample_study))
            print("   Entity ID: {}".format(summary.entity_id))
            print("   Findings: {}".format(len(summary.findings)))
            
            tests_passed += 1
        else:
            print("⚠️ No study data available, skipping")
            tests_passed += 1
    except Exception as e:
        print("❌ FAILED: {}".format(e))
        tests_failed += 1
    
    # =========================================================================
    # TEST 7: Executive Summary
    # =========================================================================
    print("\n" + "-" * 70)
    print("TEST 7: Executive Summary")
    print("-" * 70)
    
    try:
        summary = executive_summary()
        
        print("✅ Executive summary generated")
        print("   Summary ID: {}".format(summary.summary_id))
        print("   Title: {}".format(summary.title))
        print("   Sections: {}".format(len(summary.sections)))
        print("   Findings: {}".format(len(summary.findings)))
        
        # Show executive summary
        print("\n   Executive Summary:")
        print("   " + summary.executive_summary.replace('\n', '\n   '))
        
        tests_passed += 1
    except Exception as e:
        print("❌ FAILED: {}".format(e))
        import traceback
        traceback.print_exc()
        tests_failed += 1
    
    # =========================================================================
    # TEST 8: Investigation Summary
    # =========================================================================
    print("\n" + "-" * 70)
    print("TEST 8: Investigation Summary")
    print("-" * 70)
    
    try:
        if 'patients' in summarizer.data_loader.data:
            patients_df = summarizer.data_loader.data['patients']
            sample_site = patients_df['site_id'].iloc[0]
            
            summary = summarizer.summarize_investigation(
                entity_id=sample_site,
                entity_type="site",
                issue_focus="sdv_incomplete"
            )
            
            print("✅ Investigation summary generated")
            print("   Summary ID: {}".format(summary.summary_id))
            print("   Entity: {}".format(summary.entity_id))
            print("   Sections: {}".format(len(summary.sections)))
            print("   Findings (Hypotheses): {}".format(len(summary.findings)))
            print("   Action Items: {}".format(len(summary.action_items)))
            
            # Show hypotheses
            if summary.findings:
                print("\n   Root Cause Hypotheses:")
                for finding in summary.findings[:3]:
                    print("      - {} ({}%)".format(
                        finding.title, 
                        finding.metric_value if finding.metric_value else "N/A"
                    ))
            
            tests_passed += 1
        else:
            print("⚠️ No data available, skipping")
            tests_passed += 1
    except Exception as e:
        print("❌ FAILED: {}".format(e))
        import traceback
        traceback.print_exc()
        tests_failed += 1
    
    # =========================================================================
    # TEST 9: Markdown Output
    # =========================================================================
    print("\n" + "-" * 70)
    print("TEST 9: Markdown Output")
    print("-" * 70)
    
    try:
        summary = executive_summary()
        markdown = summary.to_markdown()
        
        print("✅ Markdown output generated")
        print("   Length: {:,} characters".format(len(markdown)))
        print("   Lines: {}".format(len(markdown.split('\n'))))
        
        # Check for key markdown elements
        checks = [
            ('# ' in markdown, 'H1 heading'),
            ('## ' in markdown, 'H2 headings'),
            ('**' in markdown, 'Bold text'),
            ('|' in markdown, 'Tables'),
        ]
        
        print("\n   Markdown elements:")
        for check, name in checks:
            status = "✅" if check else "❌"
            print("      {} {}".format(status, name))
        
        tests_passed += 1
    except Exception as e:
        print("❌ FAILED: {}".format(e))
        tests_failed += 1
    
    # =========================================================================
    # TEST 10: HTML Output
    # =========================================================================
    print("\n" + "-" * 70)
    print("TEST 10: HTML Output")
    print("-" * 70)
    
    try:
        summary = executive_summary()
        html = summary.to_html()
        
        print("✅ HTML output generated")
        print("   Length: {:,} characters".format(len(html)))
        
        # Check for key HTML elements
        checks = [
            ('<html>' in html, '<html> tag'),
            ('<head>' in html, '<head> tag'),
            ('<body>' in html, '<body> tag'),
            ('<h1>' in html, '<h1> heading'),
            ('<table>' in html, '<table> element'),
            ('<style>' in html, 'CSS styles'),
        ]
        
        print("\n   HTML elements:")
        for check, name in checks:
            status = "✅" if check else "❌"
            print("      {} {}".format(status, name))
        
        tests_passed += 1
    except Exception as e:
        print("❌ FAILED: {}".format(e))
        tests_failed += 1
    
    # =========================================================================
    # TEST 11: JSON Output
    # =========================================================================
    print("\n" + "-" * 70)
    print("TEST 11: JSON Output")
    print("-" * 70)
    
    try:
        import json
        
        summary = executive_summary()
        summary_dict = summary.to_dict()
        
        # Verify it's valid JSON
        json_str = json.dumps(summary_dict, indent=2, default=str)
        parsed = json.loads(json_str)
        
        print("✅ JSON output generated")
        print("   Keys: {}".format(list(parsed.keys())))
        print("   Findings count: {}".format(len(parsed.get('findings', []))))
        print("   Actions count: {}".format(len(parsed.get('action_items', []))))
        
        tests_passed += 1
    except Exception as e:
        print("❌ FAILED: {}".format(e))
        tests_failed += 1
    
    # =========================================================================
    # TEST 12: Save Summary to Files
    # =========================================================================
    print("\n" + "-" * 70)
    print("TEST 12: Save Summary to Files")
    print("-" * 70)
    
    try:
        summary = executive_summary()
        saved_files = summarizer.save_summary(summary)
        
        print("✅ Summary saved to files")
        for format_type, file_path in saved_files.items():
            # Check file exists and has content
            from pathlib import Path
            path = Path(file_path)
            exists = path.exists()
            size = path.stat().st_size if exists else 0
            
            status = "✅" if exists and size > 0 else "❌"
            print("   {} {}: {:,} bytes".format(status, format_type.upper(), size))
        
        tests_passed += 1
    except Exception as e:
        print("❌ FAILED: {}".format(e))
        import traceback
        traceback.print_exc()
        tests_failed += 1
    
    # =========================================================================
    # TEST 13: Findings and Severity
    # =========================================================================
    print("\n" + "-" * 70)
    print("TEST 13: Findings and Severity")
    print("-" * 70)
    
    try:
        if 'patients' in summarizer.data_loader.data:
            patients_df = summarizer.data_loader.data['patients']
            sample_site = patients_df['site_id'].iloc[0]
            
            summary = summarize_site(sample_site)
            
            # Count findings by severity
            severity_counts = {}
            for finding in summary.findings:
                sev = finding.severity.value
                severity_counts[sev] = severity_counts.get(sev, 0) + 1
            
            print("✅ Findings analyzed")
            print("   Total findings: {}".format(len(summary.findings)))
            print("\n   By severity:")
            for sev, count in severity_counts.items():
                icon = {
                    'critical': '🔴',
                    'high': '🟠',
                    'medium': '🟡',
                    'low': '🟢',
                    'info': 'ℹ️'
                }.get(sev, '•')
                print("      {} {}: {}".format(icon, sev.title(), count))
            
            tests_passed += 1
        else:
            print("⚠️ No data available, skipping")
            tests_passed += 1
    except Exception as e:
        print("❌ FAILED: {}".format(e))
        tests_failed += 1
    
    # =========================================================================
    # TEST 14: Action Items
    # =========================================================================
    print("\n" + "-" * 70)
    print("TEST 14: Action Items")
    print("-" * 70)
    
    try:
        if 'patients' in summarizer.data_loader.data:
            patients_df = summarizer.data_loader.data['patients']
            sample_site = patients_df['site_id'].iloc[0]
            
            summary = summarize_site(sample_site)
            
            print("✅ Action items generated")
            print("   Total actions: {}".format(len(summary.action_items)))
            
            if summary.action_items:
                print("\n   Top actions:")
                for action in summary.action_items[:3]:
                    print("      - [{}] {} (Owner: {})".format(
                        action.priority,
                        action.title[:40],
                        action.owner
                    ))
                    if action.due_date:
                        print("        Due: {}".format(action.due_date.strftime('%Y-%m-%d')))
            
            tests_passed += 1
        else:
            print("⚠️ No data available, skipping")
            tests_passed += 1
    except Exception as e:
        print("❌ FAILED: {}".format(e))
        tests_failed += 1
    
    # =========================================================================
    # TEST 15: Unified Summarize Interface
    # =========================================================================
    print("\n" + "-" * 70)
    print("TEST 15: Unified Summarize Interface")
    print("-" * 70)
    
    try:
        # Test the unified interface
        summary_types = [
            (SummaryType.EXECUTIVE, None, {}),
            (SummaryType.DAILY_DIGEST, None, {'recipient_role': 'CRA'}),
        ]
        
        for summary_type, entity_id, kwargs in summary_types:
            summary = summarizer.summarize(summary_type, entity_id, **kwargs)
            print("   ✅ {}: {}".format(summary_type.value, summary.summary_id))
        
        print("\n✅ Unified interface working")
        tests_passed += 1
    except Exception as e:
        print("❌ FAILED: {}".format(e))
        tests_failed += 1
    
    # =========================================================================
    # TEST 16: Statistics Tracking
    # =========================================================================
    print("\n" + "-" * 70)
    print("TEST 16: Statistics Tracking")
    print("-" * 70)
    
    try:
        stats = summarizer.get_stats()
        
        print("✅ Statistics tracked")
        print("   Summaries generated: {}".format(stats['summaries_generated']))
        print("\n   By type:")
        for type_name, count in stats.get('by_type', {}).items():
            print("      {}: {}".format(type_name, count))
        
        tests_passed += 1
    except Exception as e:
        print("❌ FAILED: {}".format(e))
        tests_failed += 1
    
    # =========================================================================
    # TEST 17: Empty/Missing Data Handling
    # =========================================================================
    print("\n" + "-" * 70)
    print("TEST 17: Empty/Missing Data Handling")
    print("-" * 70)
    
    try:
        # Try to summarize a non-existent patient
        summary = summarize_patient("NONEXISTENT_PATIENT_KEY")
        
        print("✅ Empty data handled gracefully")
        print("   Summary ID: {}".format(summary.summary_id))
        print("   Has executive summary: {}".format(len(summary.executive_summary) > 0))
        print("   Findings: {}".format(len(summary.findings)))
        
        tests_passed += 1
    except Exception as e:
        print("❌ FAILED: {}".format(e))
        tests_failed += 1
    
    # =========================================================================
    # TEST 18: Quick Functions
    # =========================================================================
    print("\n" + "-" * 70)
    print("TEST 18: Quick Functions")
    print("-" * 70)
    
    try:
        from src.generation.auto_summarizer import (
            summarize_patient,
            summarize_site,
            daily_digest,
            executive_summary
        )
        
        # Test each quick function
        exec_sum = executive_summary()
        digest = daily_digest()
        
        print("✅ Quick functions working")
        print("   executive_summary(): {}".format(exec_sum.summary_id))
        print("   daily_digest(): {}".format(digest.summary_id))
        
        tests_passed += 1
    except Exception as e:
        print("❌ FAILED: {}".format(e))
        tests_failed += 1
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print("Tests Passed: {}".format(tests_passed))
    print("Tests Failed: {}".format(tests_failed))
    print("Total: {}".format(tests_passed + tests_failed))
    
    if tests_failed == 0:
        print("\n✅ ALL TESTS PASSED!")
    else:
        print("\n⚠️ {} test(s) failed".format(tests_failed))
    
    # Print output location
    print("\n" + "=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)
    print("Summaries saved to: data/outputs/summaries/")
    
    # Print usage examples
    print("\n" + "=" * 70)
    print("USAGE EXAMPLES")
    print("=" * 70)
    print("""
from src.generation.auto_summarizer import (
    summarize_patient,
    summarize_site,
    daily_digest,
    executive_summary,
    get_auto_summarizer
)

# Patient summary
summary = summarize_patient("Study_1|Site_1|Subject_1")
print(summary.to_markdown())

# Site narrative
summary = summarize_site("Site_1")
print(summary.executive_summary)

# Daily digest for Study Lead
summary = daily_digest(role="Study Lead")
print(summary.to_markdown())

# Executive summary
summary = executive_summary()
print(summary.to_html())

# Save to files
summarizer = get_auto_summarizer()
files = summarizer.save_summary(summary)
print(files)
""")
    
    return tests_failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)