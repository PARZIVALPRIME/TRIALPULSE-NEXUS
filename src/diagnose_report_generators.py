"""
TRIALPULSE NEXUS 10X - Report Generators Diagnostic Script
Run this if you encounter errors to identify the issue.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def print_header(text: str):
    print(f"\n{'='*60}")
    print(f" {text}")
    print('='*60)

def check_item(name: str, condition: bool, details: str = ""):
    icon = "✅" if condition else "❌"
    print(f"  {icon} {name}")
    if details:
        print(f"      {details}")
    return condition

def run_diagnostics():
    """Run comprehensive diagnostics."""
    
    print_header("TRIALPULSE NEXUS 10X - REPORT GENERATORS DIAGNOSTICS")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {sys.version}")
    print(f"Project Root: {PROJECT_ROOT}")
    
    issues = []
    
    # =========================================================================
    # 1. Check Directory Structure
    # =========================================================================
    print_header("1. Directory Structure")
    
    required_dirs = [
        PROJECT_ROOT / "src" / "generation",
        PROJECT_ROOT / "data" / "processed" / "analytics",
        PROJECT_ROOT / "data" / "processed" / "upr",
        PROJECT_ROOT / "data" / "outputs" / "reports",
    ]
    
    for dir_path in required_dirs:
        exists = dir_path.exists()
        if not exists:
            dir_path.mkdir(parents=True, exist_ok=True)
            check_item(str(dir_path.relative_to(PROJECT_ROOT)), True, "Created")
        else:
            check_item(str(dir_path.relative_to(PROJECT_ROOT)), True, "Exists")
    
    # =========================================================================
    # 2. Check Required Files
    # =========================================================================
    print_header("2. Required Files")
    
    required_files = [
        ("src/generation/__init__.py", False),
        ("src/generation/template_engine.py", True),
        ("src/generation/report_generators.py", True),
    ]
    
    for file_path, critical in required_files:
        full_path = PROJECT_ROOT / file_path
        exists = full_path.exists()
        if not exists and critical:
            issues.append(f"Missing critical file: {file_path}")
        check_item(file_path, exists, "Required" if critical else "Optional")
    
    # =========================================================================
    # 3. Check Dependencies
    # =========================================================================
    print_header("3. Python Dependencies")
    
    dependencies = [
        ("pandas", True),
        ("numpy", True),
        ("jinja2", True),
        ("weasyprint", False),  # Optional for PDF
        ("docx", False),  # Optional for DOCX (python-docx)
        ("pptx", False),  # Optional for PPTX (python-pptx)
    ]
    
    for pkg, required in dependencies:
        try:
            __import__(pkg)
            check_item(pkg, True, "Installed")
        except ImportError:
            if required:
                issues.append(f"Missing required package: {pkg}")
                check_item(pkg, False, "MISSING - Required!")
            else:
                check_item(pkg, False, "Not installed (optional)")
    
    # =========================================================================
    # 4. Check Data Files
    # =========================================================================
    print_header("4. Data Files")
    
    data_files = [
        "data/processed/upr/unified_patient_record.parquet",
        "data/processed/analytics/patient_issues.parquet",
        "data/processed/analytics/patient_dqi_enhanced.parquet",
        "data/processed/analytics/patient_clean_status.parquet",
        "data/processed/analytics/patient_dblock_status.parquet",
        "data/processed/analytics/site_benchmarks.parquet",
    ]
    
    for file_path in data_files:
        full_path = PROJECT_ROOT / file_path
        exists = full_path.exists()
        if exists:
            import pandas as pd
            try:
                df = pd.read_parquet(full_path)
                check_item(file_path, True, f"{len(df):,} rows")
            except Exception as e:
                check_item(file_path, False, f"Error reading: {e}")
                issues.append(f"Cannot read {file_path}: {e}")
        else:
            check_item(file_path, False, "Not found")
            issues.append(f"Data file missing: {file_path}")
    
    # =========================================================================
    # 5. Check Template Engine
    # =========================================================================
    print_header("5. Template Engine")
    
    try:
        from src.generation.template_engine import get_template_engine, TemplateEngine
        engine = get_template_engine()
        templates = engine.list_templates()
        check_item("Template Engine Import", True)
        check_item("Templates Available", len(templates) > 0, f"{len(templates)} templates")
        
        # Test rendering
        test_vars = {
            'report_id': 'TEST-001',
            'report_date': datetime.now(),
            'study_id': 'TEST',
            'summary': {'patients': {'total': 100}},
            'key_metrics': {},
            'highlights': ['Test'],
            'concerns': ['None'],
            'next_actions': ['Test']
        }
        
        report = engine.render('executive_brief', test_vars)
        check_item("Template Rendering", len(report.content) > 0, f"{len(report.content)} chars")
        
    except Exception as e:
        check_item("Template Engine", False, str(e))
        issues.append(f"Template Engine error: {e}")
    
    # =========================================================================
    # 6. Check Report Generators
    # =========================================================================
    print_header("6. Report Generators")
    
    try:
        from src.generation.report_generators import (
            ReportGeneratorFactory,
            OutputFormat,
            DataLoader
        )
        
        check_item("Report Generators Import", True)
        
        # Test data loader
        loader = DataLoader()
        summary = loader.get_portfolio_summary()
        check_item("Data Loader", True, f"Portfolio has {summary.get('patients', {}).get('total', 0)} patients")
        
        # Test factory
        report_types = ReportGeneratorFactory.list_report_types()
        check_item("Report Factory", True, f"{len(report_types)} report types")
        
        for rtype in report_types:
            try:
                gen = ReportGeneratorFactory.get_generator(rtype)
                check_item(f"  {rtype}", True, type(gen).__name__)
            except Exception as e:
                check_item(f"  {rtype}", False, str(e))
                issues.append(f"Generator {rtype}: {e}")
        
    except Exception as e:
        check_item("Report Generators", False, str(e))
        issues.append(f"Report Generators error: {e}")
    
    # =========================================================================
    # 7. Test Report Generation
    # =========================================================================
    print_header("7. Test Report Generation")
    
    try:
        from src.generation.report_generators import (
            ExecutiveBriefGenerator,
            OutputFormat
        )
        
        generator = ExecutiveBriefGenerator()
        outputs = generator.generate(output_formats=[OutputFormat.HTML])
        
        if outputs:
            output = outputs[0]
            check_item("Generate Report", True, f"ID: {output.report_id}")
            
            if output.file_path:
                file_exists = Path(output.file_path).exists()
                check_item("Output File", file_exists, output.file_path)
                
                if file_exists:
                    size = Path(output.file_path).stat().st_size
                    check_item("File Size", size > 0, f"{size:,} bytes")
            
            if output.warnings:
                for w in output.warnings:
                    print(f"      ⚠️ Warning: {w}")
        else:
            check_item("Generate Report", False, "No output")
            issues.append("Report generation returned empty output")
            
    except Exception as e:
        check_item("Test Report Generation", False, str(e))
        issues.append(f"Test generation error: {e}")
        
        # Print traceback for debugging
        import traceback
        print("\n  Traceback:")
        traceback.print_exc()
    
    # =========================================================================
    # Summary
    # =========================================================================
    print_header("DIAGNOSTIC SUMMARY")
    
    if issues:
        print(f"\n  ❌ Found {len(issues)} issue(s):\n")
        for i, issue in enumerate(issues, 1):
            print(f"    {i}. {issue}")
        
        print("\n  Recommended Actions:")
        
        if any("Missing required package" in i for i in issues):
            print("    - Run: pip install pandas numpy jinja2")
        
        if any("python-docx" in i or "docx" in i for i in issues):
            print("    - For DOCX: pip install python-docx")
        
        if any("weasyprint" in i for i in issues):
            print("    - For PDF: pip install weasyprint")
        
        if any("pptx" in i for i in issues):
            print("    - For PPTX: pip install python-pptx")
        
        if any("Data file missing" in i for i in issues):
            print("    - Run previous phases to generate data files")
            print("    - Check: python src/run_ingestion.py")
            print("    - Check: python src/run_issue_detector.py")
        
        if any("template_engine" in i.lower() for i in issues):
            print("    - Ensure Phase 6.1 Template Engine is complete")
        
    else:
        print("\n  ✅ All diagnostics passed!")
        print("  Report Generators are ready to use.")
    
    print(f"\n  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return len(issues) == 0


if __name__ == "__main__":
    success = run_diagnostics()
    sys.exit(0 if success else 1)