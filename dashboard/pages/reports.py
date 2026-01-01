"""
TRIALPULSE NEXUS 10X - Reports Page
Phase 7.3: Comprehensive report generation with 8+ report types
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.config.theme import THEME_CONFIG

# Try to import report generators
try:
    from src.generation.report_generators import (
        ReportGeneratorFactory,
        OutputFormat,
        CRAMonitoringReportGenerator,
        SitePerformanceReportGenerator,
        SponsorUpdateReportGenerator,
        MeetingPackGenerator,
        QuerySummaryReportGenerator,
        SafetyNarrativeGenerator,
        DBLockReadinessReportGenerator,
        ExecutiveBriefGenerator
    )
    GENERATORS_AVAILABLE = True
except ImportError as e:
    GENERATORS_AVAILABLE = False
    IMPORT_ERROR = str(e)


# Report type definitions
REPORT_TYPES = {
    "cra_monitoring": {
        "name": "CRA Monitoring Report",
        "icon": "👩‍⚕️",
        "description": "Comprehensive site visit summary for Clinical Research Associates",
        "time_estimate": "~30 seconds",
        "fields": ["cra_name", "sites", "date_range"],
        "formats": ["html", "pdf", "docx"]
    },
    "site_performance": {
        "name": "Site Performance Report",
        "icon": "🏥",
        "description": "Detailed performance metrics and benchmarks for sites",
        "time_estimate": "~30 seconds",
        "fields": ["site_id", "study_id"],
        "formats": ["html", "pdf", "docx", "pptx"]
    },
    "sponsor_update": {
        "name": "Sponsor Status Update",
        "icon": "📈",
        "description": "Executive summary for sponsor stakeholders",
        "time_estimate": "~45 seconds",
        "fields": ["study_id"],
        "formats": ["html", "pdf", "docx", "pptx"]
    },
    "meeting_pack": {
        "name": "Meeting Pack",
        "icon": "📋",
        "description": "Pre-built meeting materials with agenda",
        "time_estimate": "~60 seconds",
        "fields": ["meeting_type", "study_id"],
        "formats": ["html", "pdf", "pptx"]
    },
    "query_summary": {
        "name": "Query Resolution Summary",
        "icon": "❓",
        "description": "Query status, trends, and resolution metrics",
        "time_estimate": "~15 seconds",
        "fields": ["site_id", "study_id"],
        "formats": ["html", "pdf", "csv"]
    },
    "safety_narrative": {
        "name": "Safety Narrative",
        "icon": "🛡️",
        "description": "SAE case narratives for medical review",
        "time_estimate": "~60 seconds",
        "fields": ["patient_key", "study_id"],
        "formats": ["html", "pdf", "docx"]
    },
    "db_lock_readiness": {
        "name": "DB Lock Readiness Report",
        "icon": "🔒",
        "description": "Database lock preparation status and blockers",
        "time_estimate": "~45 seconds",
        "fields": ["study_id", "target_date"],
        "formats": ["html", "pdf", "docx"]
    },
    "executive_brief": {
        "name": "Executive Brief",
        "icon": "👔",
        "description": "C-suite summary with key metrics",
        "time_estimate": "~30 seconds",
        "fields": ["study_id"],
        "formats": ["html", "pdf", "pptx"]
    }
}

MEETING_TYPES = {
    "team": "Team Meeting",
    "sponsor": "Sponsor Call",
    "investigator": "Investigator Meeting", 
    "safety": "Safety Review",
    "oversight": "Oversight Committee"
}


def render_page(user: Any = None):
    """Main render function for Reports page."""
    
    theme = THEME_CONFIG
    
    # Page header
    st.markdown(f"""
        <h1 style='color: {theme.text_primary}; margin-bottom: 0.5rem;'>
            📄 Report Generation Center
        </h1>
        <p style='color: {theme.text_secondary}; margin-bottom: 1.5rem;'>
            Generate professional reports in seconds with AI-powered insights
        </p>
    """, unsafe_allow_html=True)
    
    # Check if generators are available
    if not GENERATORS_AVAILABLE:
        st.warning(f"⚠️ Report generators not fully loaded: {IMPORT_ERROR}")
    
    # Main layout
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        render_report_selector()
    
    with col_right:
        render_report_workspace()
    
    st.markdown("---")
    
    # Recent Reports section
    render_recent_reports()


def render_report_selector():
    """Render report type selector."""
    
    theme = THEME_CONFIG
    
    st.subheader("📑 Select Report Type")
    
    # Initialize session state
    if 'selected_report_type' not in st.session_state:
        st.session_state.selected_report_type = "cra_monitoring"
    
    # Report type radio selection
    report_options = list(REPORT_TYPES.keys())
    report_labels = [f"{REPORT_TYPES[k]['icon']} {REPORT_TYPES[k]['name']}" for k in report_options]
    
    selected_idx = report_options.index(st.session_state.selected_report_type)
    
    selected = st.radio(
        "Choose a report type:",
        report_options,
        index=selected_idx,
        format_func=lambda x: f"{REPORT_TYPES[x]['icon']} {REPORT_TYPES[x]['name']}",
        label_visibility="collapsed"
    )
    
    if selected != st.session_state.selected_report_type:
        st.session_state.selected_report_type = selected
        st.rerun()
    
    # Show description
    report_info = REPORT_TYPES[selected]
    st.caption(f"⏱️ {report_info['time_estimate']}")
    st.info(report_info['description'])


def render_report_workspace():
    """Render the report configuration and generation workspace."""
    
    theme = THEME_CONFIG
    selected_type = st.session_state.get('selected_report_type', 'cra_monitoring')
    report_info = REPORT_TYPES.get(selected_type, {})
    
    st.subheader(f"{report_info.get('icon', '📄')} {report_info.get('name', 'Report')}")
    st.caption(report_info.get('description', ''))
    
    # Configuration form
    with st.form(key="report_config_form"):
        st.markdown("### ⚙️ Configuration")
        
        # Dynamic fields based on report type
        fields = report_info.get('fields', [])
        config = {}
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'cra_name' in fields:
                config['cra_name'] = st.text_input("CRA Name", value="Sarah Chen")
            
            if 'site_id' in fields:
                config['site_id'] = st.selectbox("Site", 
                    ["All Sites", "Site_101", "Site_102", "Site_103", "Site_201", "Site_301"])
            
            if 'study_id' in fields:
                config['study_id'] = st.selectbox("Study",
                    ["All Studies", "Study_21", "Study_22", "Study_23"])
            
            if 'patient_key' in fields:
                config['patient_key'] = st.text_input("Patient Key (Optional)",
                    placeholder="e.g., Study_21_Site_101_001")
        
        with col2:
            if 'meeting_type' in fields:
                config['meeting_type'] = st.selectbox("Meeting Type",
                    list(MEETING_TYPES.keys()),
                    format_func=lambda x: MEETING_TYPES.get(x, x))
            
            if 'target_date' in fields:
                config['target_date'] = st.date_input("Target Date",
                    value=datetime.now() + timedelta(days=30))
            
            if 'date_range' in fields or 'sites' in fields:
                config['date_range'] = st.date_input("Report Period",
                    value=(datetime.now() - timedelta(days=30), datetime.now()))
        
        # Output format selection
        st.markdown("### 📤 Output Format")
        
        available_formats = report_info.get('formats', ['html', 'pdf'])
        
        format_cols = st.columns(len(available_formats))
        selected_formats = []
        
        format_labels = {
            'html': '🌐 HTML',
            'pdf': '📕 PDF',
            'docx': '📘 Word',
            'pptx': '📙 PowerPoint',
            'csv': '📊 CSV'
        }
        
        for i, fmt in enumerate(available_formats):
            with format_cols[i]:
                if st.checkbox(format_labels.get(fmt, fmt.upper()), value=(fmt == 'html'), key=f"fmt_{fmt}"):
                    selected_formats.append(fmt)
        
        # Generate buttons
        st.markdown("---")
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            generate_clicked = st.form_submit_button("🚀 Generate Report", 
                                                      use_container_width=True,
                                                      type="primary")
        
        with col_btn2:
            preview_clicked = st.form_submit_button("👁️ Quick Preview",
                                                     use_container_width=True)
    
    # Handle generation
    if generate_clicked or preview_clicked:
        generate_report(selected_type, config, selected_formats, preview_only=preview_clicked)


def generate_report(report_type: str, config: Dict, formats: List[str], preview_only: bool = False):
    """Generate the selected report."""
    
    with st.spinner('Generating preview...' if preview_only else 'Generating report...'):
        try:
            if GENERATORS_AVAILABLE:
                # Get the generator
                generator = ReportGeneratorFactory.get_generator(report_type)
                
                if generator:
                    # Prepare parameters
                    params = prepare_generator_params(report_type, config)
                    
                    # Set output formats
                    if preview_only:
                        output_formats = [OutputFormat.HTML]
                    else:
                        output_formats = []
                        for f in formats:
                            if hasattr(OutputFormat, f.upper()):
                                output_formats.append(getattr(OutputFormat, f.upper()))
                        if not output_formats:
                            output_formats = [OutputFormat.HTML]
                    
                    params['output_formats'] = output_formats
                    
                    # Generate
                    results = generator.generate(**params)
                    
                    # Display results
                    display_report_results(results, report_type, preview_only)
                else:
                    st.error(f"Report generator not found for type: {report_type}")
            else:
                # Fallback - show sample output
                display_sample_report(report_type, config)
                
        except Exception as e:
            st.error(f"❌ Error generating report: {str(e)}")
            st.exception(e)


def prepare_generator_params(report_type: str, config: Dict) -> Dict:
    """Prepare parameters for the report generator."""
    
    params = {}
    
    if 'cra_name' in config:
        params['cra_name'] = config['cra_name']
    
    if 'site_id' in config and config['site_id'] != 'All Sites':
        params['site_id'] = config['site_id']
    
    if 'study_id' in config and config['study_id'] != 'All Studies':
        params['study_id'] = config['study_id']
    
    if 'meeting_type' in config:
        params['meeting_type'] = config['meeting_type']
    
    if 'patient_key' in config and config['patient_key']:
        params['patient_key'] = config['patient_key']
    
    if 'target_date' in config:
        params['target_date'] = datetime.combine(config['target_date'], datetime.min.time())
    
    params['report_date'] = datetime.now()
    
    return params


def display_report_results(results: List, report_type: str, preview_only: bool):
    """Display generated report results."""
    
    if not results:
        st.warning("No reports generated.")
        return
    
    st.success(f"✅ {'Preview ready!' if preview_only else f'{len(results)} report(s) generated!'}")
    
    for i, result in enumerate(results):
        with st.expander(f"📄 {result.title} ({result.format.value.upper()})", expanded=(i == 0)):
            # Metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Format", result.format.value.upper())
            with col2:
                st.metric("Generated", result.generated_at.strftime("%H:%M:%S"))
            with col3:
                st.metric("Pages", result.page_count or "N/A")
            
            # Content preview for HTML
            if result.html_content and result.format.value == 'html':
                st.markdown("---")
                st.markdown("**Preview:**")
                st.components.v1.html(result.html_content, height=400, scrolling=True)
            
            # Download button
            if result.content:
                st.download_button(
                    label=f"⬇️ Download {result.format.value.upper()}",
                    data=result.content,
                    file_name=f"{result.report_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{result.format.value}",
                    mime=get_mime_type(result.format.value),
                    key=f"download_{i}_{result.format.value}"
                )
            elif result.file_path:
                st.info(f"📁 Saved to: {result.file_path}")


def display_sample_report(report_type: str, config: Dict):
    """Display a sample report when generators aren't available."""
    
    report_info = REPORT_TYPES.get(report_type, {})
    
    st.info("📋 Sample Report Preview (Full generation available when dependencies are loaded)")
    
    # Sample metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Patients", "57,997")
    with col2:
        st.metric("Average DQI", "98.2")
    with col3:
        st.metric("Clean Rate", "53.7%")
    
    # Sample content
    st.markdown(f"""
    ### {report_info.get('icon', '📄')} {report_info.get('name', 'Report')}
    
    **Generated:** {datetime.now().strftime('%B %d, %Y at %H:%M')}
    
    #### Executive Summary
    This is a sample report preview. The full report generation system 
    integrates with the TrialPulse analytics pipeline to provide comprehensive
    insights and actionable recommendations.
    
    #### Key Highlights
    - ✅ Data quality metrics trending upward
    - ⚠️ 3 sites require attention
    - 📊 Query resolution rate: 87%
    """)
    
    # Download sample
    sample_html = f"""
    <html>
    <head><title>{report_info.get('name', 'Report')}</title></head>
    <body style="font-family: Arial, sans-serif; padding: 20px;">
        <h1>{report_info.get('icon', '')} {report_info.get('name', 'Report')}</h1>
        <p>Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}</p>
        <h2>Summary</h2>
        <p>Sample report from TrialPulse Nexus 10X</p>
    </body>
    </html>
    """
    
    st.download_button(
        label="⬇️ Download Sample (HTML)",
        data=sample_html,
        file_name=f"sample_{report_type}_{datetime.now().strftime('%Y%m%d')}.html",
        mime="text/html"
    )


def render_recent_reports():
    """Render recent reports section."""
    
    st.subheader("📚 Recent Reports")
    
    # Sample recent reports data
    recent_reports = [
        {"name": "CRA Monitoring - Sarah Chen", "type": "cra_monitoring", 
         "date": "Today, 09:15", "format": "PDF"},
        {"name": "Site Performance - All Sites", "type": "site_performance", 
         "date": "Today, 08:30", "format": "PPTX"},
        {"name": "Executive Brief - Study_21", "type": "executive_brief", 
         "date": "Yesterday", "format": "PDF"},
        {"name": "DB Lock Readiness", "type": "db_lock_readiness", 
         "date": "Jan 1, 2026", "format": "DOCX"},
    ]
    
    cols = st.columns(4)
    for i, report in enumerate(recent_reports):
        with cols[i]:
            report_info = REPORT_TYPES.get(report['type'], {})
            icon = report_info.get('icon', '📄')
            
            st.markdown(f"""
                **{icon} {report['name'][:20]}...**
                
                📅 {report['date']} | 📎 {report['format']}
            """)


def get_mime_type(format_type: str) -> str:
    """Get MIME type for a file format."""
    mime_types = {
        'html': 'text/html',
        'pdf': 'application/pdf',
        'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        'csv': 'text/csv',
        'json': 'application/json'
    }
    return mime_types.get(format_type, 'application/octet-stream')


if __name__ == "__main__":
    render_page()