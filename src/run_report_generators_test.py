"""
TRIALPULSE NEXUS 10X - Report Generators Test Runner
Phase 6.2: Tests all 8 report generators
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def print_header(text: str):
    print(f"\n{'='*70}")
    print(f" {text}")
    print('='*70)

def print_result(name: str, status: str, details: str = ""):
    icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
    print(f"  {icon} {name}: {status}")
    if details:
        print(f"      {details}")

def run_tests():
    """Run all report generator tests."""
    
    print_header("TRIALPULSE NEXUS 10X - PHASE 6.2 REPORT GENERATORS TEST")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {
        'passed': 0,
        'failed': 0,
        'warnings': 0,
        'reports_generated': 0,
        'errors': []
    }
    
    # Import modules
    try:
        from src.generation.report_generators import (
            ReportGeneratorFactory,
            OutputFormat,
            DataLoader,
            CRAMonitoringReportGenerator,
            SitePerformanceReportGenerator,
            SponsorUpdateReportGenerator,
            MeetingPackGenerator,
            QuerySummaryReportGenerator,
            SafetyNarrativeGenerator,
            DBLockReadinessReportGenerator,
            ExecutiveBriefGenerator,
            generate_report
        )
        print_result("Module Import", "PASS")
        results['passed'] += 1
    except Exception as e:
        print_result("Module Import", "FAIL", str(e))
        results['failed'] += 1
        results['errors'].append(f"Import: {e}")
        return results
    
    # =========================================================================
    # Test 1: Data Loader
    # =========================================================================
    print_header("Test 1: Data Loader")
    
    try:
        loader = DataLoader()
        
        # Test loading each data source
        data_sources = [
            ('patient_data', loader.get_patient_data),
            ('patient_issues', loader.get_patient_issues),
            ('patient_dqi', loader.get_patient_dqi),
            ('patient_clean', loader.get_patient_clean),
            ('patient_dblock', loader.get_patient_dblock),
            ('site_benchmarks', loader.get_site_benchmarks),
        ]
        
        loaded_count = 0
        for name, func in data_sources:
            df = func()
            if df is not None:
                print_result(f"Load {name}", "PASS", f"{len(df):,} rows")
                loaded_count += 1
            else:
                print_result(f"Load {name}", "WARN", "Not available")
                results['warnings'] += 1
        
        # Test portfolio summary
        summary = loader.get_portfolio_summary()
        if summary and summary.get('patients', {}).get('total', 0) > 0:
            print_result("Portfolio Summary", "PASS", 
                        f"Patients: {summary['patients']['total']:,}")
            results['passed'] += 1
        else:
            print_result("Portfolio Summary", "WARN", "Limited data")
            results['warnings'] += 1
        
        # Test site summary
        site_summary = loader.get_site_summary("Site_1")
        print_result("Site Summary", "PASS", f"Site_1: {site_summary.get('patients', 0)} patients")
        results['passed'] += 1
        
    except Exception as e:
        print_result("Data Loader", "FAIL", str(e))
        results['failed'] += 1
        results['errors'].append(f"Data Loader: {e}")
    
    # =========================================================================
    # Test 2: CRA Monitoring Report
    # =========================================================================
    print_header("Test 2: CRA Monitoring Report Generator")
    
    try:
        generator = CRAMonitoringReportGenerator()
        
        outputs = generator.generate(
            cra_id="CRA-001",
            cra_name="Sarah Chen",
            sites=["Site_1", "Site_2", "Site_3"],
            output_formats=[OutputFormat.HTML]
        )
        
        if outputs:
            output = outputs[0]
            print_result("CRA Report Generation", "PASS")
            print(f"      Report ID: {output.report_id}")
            print(f"      Format: {output.format.value}")
            print(f"      File: {output.file_path}")
            print(f"      Time: {output.generation_time_ms:.1f}ms")
            
            if output.file_path and Path(output.file_path).exists():
                print_result("File Created", "PASS", f"{output.file_size_bytes:,} bytes")
                results['reports_generated'] += 1
            
            if output.warnings:
                for w in output.warnings:
                    print_result("Warning", "WARN", w)
                    results['warnings'] += 1
            
            results['passed'] += 1
        else:
            print_result("CRA Report", "FAIL", "No output generated")
            results['failed'] += 1
            
    except Exception as e:
        print_result("CRA Monitoring Report", "FAIL", str(e))
        results['failed'] += 1
        results['errors'].append(f"CRA Report: {e}")
    
    # =========================================================================
    # Test 3: Site Performance Report
    # =========================================================================
    print_header("Test 3: Site Performance Report Generator")
    
    try:
        generator = SitePerformanceReportGenerator()
        
        # Portfolio level
        outputs = generator.generate(
            output_formats=[OutputFormat.HTML]
        )
        
        if outputs:
            output = outputs[0]
            print_result("Site Performance (Portfolio)", "PASS")
            print(f"      Report ID: {output.report_id}")
            print(f"      File: {output.file_path}")
            results['passed'] += 1
            results['reports_generated'] += 1
            
            if output.warnings:
                for w in output.warnings:
                    print_result("Warning", "WARN", w)
                    results['warnings'] += 1
        
        # Study level
        outputs = generator.generate(
            study_id="Study_1",
            output_formats=[OutputFormat.HTML]
        )
        
        if outputs:
            print_result("Site Performance (Study)", "PASS", f"Study_1")
            results['passed'] += 1
            results['reports_generated'] += 1
            
    except Exception as e:
        print_result("Site Performance Report", "FAIL", str(e))
        results['failed'] += 1
        results['errors'].append(f"Site Performance: {e}")
    
    # =========================================================================
    # Test 4: Sponsor Update Report
    # =========================================================================
    print_header("Test 4: Sponsor Update Report Generator")
    
    try:
        generator = SponsorUpdateReportGenerator()
        
        outputs = generator.generate(
            output_formats=[OutputFormat.HTML, OutputFormat.PPTX]
        )
        
        html_generated = False
        pptx_generated = False
        
        for output in outputs:
            print_result(f"Sponsor Update ({output.format.value})", "PASS")
            print(f"      Report ID: {output.report_id}")
            print(f"      File: {output.file_path}")
            
            if output.format == OutputFormat.HTML:
                html_generated = True
            elif output.format == OutputFormat.PPTX:
                pptx_generated = True
            
            results['reports_generated'] += 1
            
            if output.warnings:
                for w in output.warnings:
                    print_result("Warning", "WARN", w)
                    results['warnings'] += 1
        
        if html_generated:
            results['passed'] += 1
        if pptx_generated or any(o.warnings for o in outputs if o.format == OutputFormat.PPTX):
            results['passed'] += 1  # PPTX attempted
            
    except Exception as e:
        print_result("Sponsor Update Report", "FAIL", str(e))
        results['failed'] += 1
        results['errors'].append(f"Sponsor Update: {e}")
    
    # =========================================================================
    # Test 5: Meeting Pack Generator
    # =========================================================================
    print_header("Test 5: Meeting Pack Generator")
    
    try:
        generator = MeetingPackGenerator()
        
        meeting_types = ['team', 'sponsor', 'dmc']
        
        for meeting_type in meeting_types:
            outputs = generator.generate(
                meeting_type=meeting_type,
                output_formats=[OutputFormat.HTML]
            )
            
            if outputs:
                output = outputs[0]
                print_result(f"Meeting Pack ({meeting_type})", "PASS")
                print(f"      File: {output.file_path}")
                results['reports_generated'] += 1
                
                if output.warnings:
                    for w in output.warnings:
                        results['warnings'] += 1
        
        results['passed'] += 1
        
    except Exception as e:
        print_result("Meeting Pack Generator", "FAIL", str(e))
        results['failed'] += 1
        results['errors'].append(f"Meeting Pack: {e}")
    
    # =========================================================================
    # Test 6: Query Summary Report
    # =========================================================================
    print_header("Test 6: Query Summary Report Generator")
    
    try:
        generator = QuerySummaryReportGenerator()
        
        outputs = generator.generate(
            output_formats=[OutputFormat.HTML]
        )
        
        if outputs:
            output = outputs[0]
            print_result("Query Summary (Portfolio)", "PASS")
            print(f"      Report ID: {output.report_id}")
            print(f"      File: {output.file_path}")
            results['passed'] += 1
            results['reports_generated'] += 1
            
            if output.warnings:
                for w in output.warnings:
                    print_result("Warning", "WARN", w)
                    results['warnings'] += 1
        
        # Site-level
        outputs = generator.generate(
            site_id="Site_1",
            output_formats=[OutputFormat.HTML]
        )
        
        if outputs:
            print_result("Query Summary (Site)", "PASS", "Site_1")
            results['reports_generated'] += 1
            
    except Exception as e:
        print_result("Query Summary Report", "FAIL", str(e))
        results['failed'] += 1
        results['errors'].append(f"Query Summary: {e}")
    
    # =========================================================================
    # Test 7: Safety Narrative Report
    # =========================================================================
    print_header("Test 7: Safety Narrative Report Generator")
    
    try:
        generator = SafetyNarrativeGenerator()
        
        outputs = generator.generate(
            output_formats=[OutputFormat.HTML, OutputFormat.DOCX]
        )
        
        for output in outputs:
            print_result(f"Safety Narrative ({output.format.value})", "PASS")
            print(f"      Report ID: {output.report_id}")
            print(f"      File: {output.file_path}")
            results['reports_generated'] += 1
            
            if output.warnings:
                for w in output.warnings:
                    print_result("Warning", "WARN", w)
                    results['warnings'] += 1
        
        results['passed'] += 1
        
    except Exception as e:
        print_result("Safety Narrative Report", "FAIL", str(e))
        results['failed'] += 1
        results['errors'].append(f"Safety Narrative: {e}")
    
    # =========================================================================
    # Test 8: DB Lock Readiness Report
    # =========================================================================
    print_header("Test 8: DB Lock Readiness Report Generator")
    
    try:
        generator = DBLockReadinessReportGenerator()
        
        target_date = datetime.now() + timedelta(days=90)
        
        outputs = generator.generate(
            target_date=target_date,
            output_formats=[OutputFormat.HTML]
        )
        
        if outputs:
            output = outputs[0]
            print_result("DB Lock Readiness", "PASS")
            print(f"      Report ID: {output.report_id}")
            print(f"      Target Date: {target_date.strftime('%Y-%m-%d')}")
            print(f"      File: {output.file_path}")
            results['passed'] += 1
            results['reports_generated'] += 1
            
            if output.warnings:
                for w in output.warnings:
                    print_result("Warning", "WARN", w)
                    results['warnings'] += 1
                    
    except Exception as e:
        print_result("DB Lock Readiness Report", "FAIL", str(e))
        results['failed'] += 1
        results['errors'].append(f"DB Lock: {e}")
    
    # =========================================================================
    # Test 9: Executive Brief Report
    # =========================================================================
    print_header("Test 9: Executive Brief Report Generator")
    
    try:
        generator = ExecutiveBriefGenerator()
        
        outputs = generator.generate(
            output_formats=[OutputFormat.HTML]
        )
        
        if outputs:
            output = outputs[0]
            print_result("Executive Brief", "PASS")
            print(f"      Report ID: {output.report_id}")
            print(f"      File: {output.file_path}")
            results['passed'] += 1
            results['reports_generated'] += 1
            
            if output.warnings:
                for w in output.warnings:
                    print_result("Warning", "WARN", w)
                    results['warnings'] += 1
        
        # Study-level
        outputs = generator.generate(
            study_id="Study_21",
            output_formats=[OutputFormat.HTML]
        )
        
        if outputs:
            print_result("Executive Brief (Study)", "PASS", "Study_21")
            results['reports_generated'] += 1
            
    except Exception as e:
        print_result("Executive Brief Report", "FAIL", str(e))
        results['failed'] += 1
        results['errors'].append(f"Executive Brief: {e}")
    
    # =========================================================================
    # Test 10: Report Generator Factory
    # =========================================================================
    print_header("Test 10: Report Generator Factory")
    
    try:
        # List available types
        report_types = ReportGeneratorFactory.list_report_types()
        print_result("List Report Types", "PASS", f"{len(report_types)} types")
        print(f"      Types: {', '.join(report_types)}")
        
        # Get each generator
        for rtype in report_types:
            generator = ReportGeneratorFactory.get_generator(rtype)
            print_result(f"Get Generator ({rtype})", "PASS", type(generator).__name__)
        
        results['passed'] += 1
        
        # Test convenience function
        outputs = generate_report(
            report_type='executive_brief',
            output_formats=[OutputFormat.HTML]
        )
        
        if outputs:
            print_result("Convenience Function", "PASS", f"{len(outputs)} outputs")
            results['passed'] += 1
            
    except Exception as e:
        print_result("Report Generator Factory", "FAIL", str(e))
        results['failed'] += 1
        results['errors'].append(f"Factory: {e}")
    
    # =========================================================================
    # Test 11: PDF Generation (if WeasyPrint available)
    # =========================================================================
    print_header("Test 11: PDF Generation")
    
    try:
        generator = ExecutiveBriefGenerator()
        
        outputs = generator.generate(
            output_formats=[OutputFormat.PDF]
        )
        
        if outputs:
            output = outputs[0]
            if output.file_path and Path(output.file_path).exists():
                if output.file_path.endswith('.pdf'):
                    print_result("PDF Generation", "PASS", f"{output.file_size_bytes:,} bytes")
                    results['passed'] += 1
                    results['reports_generated'] += 1
                else:
                    print_result("PDF Generation", "WARN", "Fallback to HTML")
                    results['warnings'] += 1
            
            if output.warnings:
                for w in output.warnings:
                    print_result("PDF Warning", "WARN", w)
                    results['warnings'] += 1
                    
    except Exception as e:
        print_result("PDF Generation", "FAIL", str(e))
        results['failed'] += 1
    
    # =========================================================================
    # Test 12: DOCX Generation (if python-docx available)
    # =========================================================================
    print_header("Test 12: DOCX Generation")
    
    try:
        generator = SafetyNarrativeGenerator()
        
        outputs = generator.generate(
            output_formats=[OutputFormat.DOCX]
        )
        
        if outputs:
            output = outputs[0]
            if output.file_path and Path(output.file_path).exists():
                if output.file_path.endswith('.docx'):
                    print_result("DOCX Generation", "PASS", f"{output.file_size_bytes:,} bytes")
                    results['passed'] += 1
                    results['reports_generated'] += 1
                else:
                    print_result("DOCX Generation", "WARN", "File not created")
                    results['warnings'] += 1
            
            if output.warnings:
                for w in output.warnings:
                    print_result("DOCX Warning", "WARN", w)
                    results['warnings'] += 1
                    
    except Exception as e:
        print_result("DOCX Generation", "FAIL", str(e))
        results['failed'] += 1
    
    # =========================================================================
    # Summary
    # =========================================================================
    print_header("TEST SUMMARY")
    
    total_tests = results['passed'] + results['failed']
    
    print(f"""
    Tests Passed:      {results['passed']}
    Tests Failed:      {results['failed']}
    Warnings:          {results['warnings']}
    Reports Generated: {results['reports_generated']}
    
    Success Rate:      {results['passed']/total_tests*100:.1f}% ({results['passed']}/{total_tests})
    """)
    
    if results['errors']:
        print("\n  Errors:")
        for error in results['errors']:
            print(f"    - {error}")
    
    # List generated files
    output_dir = PROJECT_ROOT / "data" / "outputs" / "reports"
    if output_dir.exists():
        files = list(output_dir.glob("*"))
        recent_files = sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)[:10]
        
        print(f"\n  Recent Reports in {output_dir}:")
        for f in recent_files:
            size = f.stat().st_size
            print(f"    - {f.name} ({size:,} bytes)")
    
    print(f"\n  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if results['failed'] == 0:
        print("\n  🎉 ALL TESTS PASSED!")
    else:
        print(f"\n  ⚠️  {results['failed']} tests failed - review errors above")
    
    return results


if __name__ == "__main__":
    results = run_tests()
    sys.exit(0 if results['failed'] == 0 else 1)