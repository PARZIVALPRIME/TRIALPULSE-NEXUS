# Save as: src/generation/template_engine.py

"""
TRIALPULSE NEXUS 10X - Template Engine v1.0
Jinja2-based template system with multi-format support

Features:
- 12 report types
- Variable injection
- Conditional sections
- Multi-format output (HTML, PDF, Word, PPT)
- Component reuse
- Custom filters
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib

from jinja2 import Environment, FileSystemLoader, select_autoescape, BaseLoader, TemplateNotFound

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReportType(Enum):
    """12 supported report types"""
    CRA_MONITORING = "cra_monitoring"
    SITE_PERFORMANCE = "site_performance"
    SPONSOR_UPDATE = "sponsor_update"
    MEETING_PACK = "meeting_pack"
    SAFETY_NARRATIVE = "safety_narrative"
    INSPECTION_PREP = "inspection_prep"
    QUERY_SUMMARY = "query_summary"
    SITE_NEWSLETTER = "site_newsletter"
    EXECUTIVE_BRIEF = "executive_brief"
    DB_LOCK_READINESS = "db_lock_readiness"
    ISSUE_ESCALATION = "issue_escalation"
    DAILY_DIGEST = "daily_digest"


class OutputFormat(Enum):
    """Supported output formats"""
    HTML = "html"
    PDF = "pdf"
    WORD = "docx"
    POWERPOINT = "pptx"
    MARKDOWN = "md"
    JSON = "json"


@dataclass
class ReportMetadata:
    """Metadata for generated reports"""
    report_id: str
    report_type: ReportType
    title: str
    generated_at: datetime
    generated_by: str
    version: str = "1.0"
    classification: str = "Internal"
    expires_at: Optional[datetime] = None
    checksum: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'report_id': self.report_id,
            'report_type': self.report_type.value,
            'title': self.title,
            'generated_at': self.generated_at.isoformat(),
            'generated_by': self.generated_by,
            'version': self.version,
            'classification': self.classification,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'checksum': self.checksum
        }


@dataclass
class ReportTemplate:
    """Template definition"""
    template_id: str
    report_type: ReportType
    name: str
    description: str
    template_file: str
    required_variables: List[str]
    optional_variables: List[str] = field(default_factory=list)
    supported_formats: List[OutputFormat] = field(default_factory=lambda: [OutputFormat.HTML])
    sections: List[str] = field(default_factory=list)
    approver_role: Optional[str] = None
    generation_time_seconds: int = 30


@dataclass
class GeneratedReport:
    """Generated report output"""
    report_id: str
    metadata: ReportMetadata
    content: str
    format: OutputFormat
    file_path: Optional[str] = None
    generation_time_ms: int = 0
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'report_id': self.report_id,
            'metadata': self.metadata.to_dict(),
            'format': self.format.value,
            'file_path': self.file_path,
            'generation_time_ms': self.generation_time_ms,
            'warnings': self.warnings,
            'content_length': len(self.content)
        }


class StringLoader(BaseLoader):
    """Load templates from strings (for inline templates)"""
    
    def __init__(self, templates: Dict[str, str]):
        self.templates = templates
    
    def get_source(self, environment, template):
        if template in self.templates:
            source = self.templates[template]
            return source, template, lambda: True
        raise TemplateNotFound(template)


class TemplateEngine:
    """
    Main template engine for report generation
    
    Features:
    - Jinja2 template rendering
    - Custom filters for clinical trial data
    - Multi-format output support
    - Component/partial reuse
    - Caching for performance
    """
    
    def __init__(self, templates_dir: Optional[str] = None):
        """Initialize template engine"""
        self.templates_dir = templates_dir or str(Path(__file__).parent / "templates")
        self.output_dir = str(Path(__file__).parent.parent.parent / "data" / "outputs" / "reports")
        
        # Create directories if needed
        os.makedirs(self.templates_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize Jinja2 environment
        self.env = self._create_environment()
        
        # Register built-in templates
        self.templates: Dict[str, ReportTemplate] = {}
        self._register_builtin_templates()
        
        # Cache for rendered components
        self.component_cache: Dict[str, str] = {}
        
        # Statistics
        self.stats = {
            'reports_generated': 0,
            'total_generation_time_ms': 0,
            'cache_hits': 0
        }
        
        logger.info(f"TemplateEngine initialized with {len(self.templates)} templates")
    
    def _create_environment(self) -> Environment:
        """Create Jinja2 environment with custom configuration"""
        
        # Try file loader first, fall back to string loader
        try:
            loader = FileSystemLoader(self.templates_dir)
        except:
            loader = StringLoader({})
        
        env = Environment(
            loader=loader,
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True
        )
        
        # Register custom filters
        self._register_filters(env)
        
        # Register custom globals
        self._register_globals(env)
        
        return env
    
    def _register_filters(self, env: Environment):
        """Register custom Jinja2 filters for clinical trial data"""
        
        # Number formatting
        env.filters['format_number'] = lambda x: f"{x:,}" if isinstance(x, (int, float)) else x
        env.filters['format_percent'] = lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) else x
        env.filters['format_decimal'] = lambda x, d=2: f"{x:.{d}f}" if isinstance(x, (int, float)) else x
        
        # Date formatting
        env.filters['format_date'] = lambda x: x.strftime('%Y-%m-%d') if isinstance(x, datetime) else x
        env.filters['format_datetime'] = lambda x: x.strftime('%Y-%m-%d %H:%M') if isinstance(x, datetime) else x
        env.filters['relative_date'] = self._relative_date
        
        # Clinical trial specific
        env.filters['dqi_band'] = self._dqi_to_band
        env.filters['risk_color'] = self._risk_to_color
        env.filters['priority_badge'] = self._priority_badge
        env.filters['status_icon'] = self._status_icon
        env.filters['trend_arrow'] = self._trend_arrow
        
        # Text formatting
        env.filters['truncate_smart'] = self._truncate_smart
        env.filters['pluralize'] = lambda n, s, p: s if n == 1 else p
        env.filters['title_case'] = lambda x: x.title() if isinstance(x, str) else x
        
        # List formatting
        env.filters['bullet_list'] = lambda items: '\n'.join(f"• {item}" for item in items)
        env.filters['numbered_list'] = lambda items: '\n'.join(f"{i+1}. {item}" for i, item in enumerate(items))
        env.filters['join_and'] = lambda items: ', '.join(items[:-1]) + ' and ' + items[-1] if len(items) > 1 else items[0] if items else ''
        
    def _register_globals(self, env: Environment):
        """Register global variables and functions"""
        env.globals['now'] = datetime.now
        env.globals['today'] = datetime.now().date
        env.globals['range'] = range
        env.globals['len'] = len
        env.globals['min'] = min
        env.globals['max'] = max
        env.globals['sum'] = sum
        env.globals['sorted'] = sorted
        
    # Custom filter implementations
    def _relative_date(self, dt: datetime) -> str:
        """Convert datetime to relative string"""
        if not isinstance(dt, datetime):
            return str(dt)
        
        now = datetime.now()
        diff = now - dt
        
        if diff.days == 0:
            return "Today"
        elif diff.days == 1:
            return "Yesterday"
        elif diff.days < 7:
            return f"{diff.days} days ago"
        elif diff.days < 30:
            weeks = diff.days // 7
            return f"{weeks} week{'s' if weeks > 1 else ''} ago"
        else:
            months = diff.days // 30
            return f"{months} month{'s' if months > 1 else ''} ago"
    
    def _dqi_to_band(self, score: float) -> str:
        """Convert DQI score to band name"""
        if score >= 95:
            return "Pristine"
        elif score >= 85:
            return "Excellent"
        elif score >= 75:
            return "Good"
        elif score >= 65:
            return "Fair"
        elif score >= 50:
            return "Poor"
        elif score >= 25:
            return "Critical"
        else:
            return "Emergency"
    
    def _risk_to_color(self, risk: str) -> str:
        """Convert risk level to color"""
        colors = {
            'low': '#28a745',      # Green
            'medium': '#ffc107',   # Yellow
            'high': '#fd7e14',     # Orange
            'critical': '#dc3545'  # Red
        }
        return colors.get(risk.lower(), '#6c757d')
    
    def _priority_badge(self, priority: str) -> str:
        """Generate priority badge HTML"""
        colors = {
            'critical': 'danger',
            'high': 'warning',
            'medium': 'info',
            'low': 'secondary',
            'optional': 'light'
        }
        color = colors.get(priority.lower(), 'secondary')
        return f'<span class="badge bg-{color}">{priority.upper()}</span>'
    
    def _status_icon(self, status: str) -> str:
        """Convert status to icon"""
        icons = {
            'complete': '✅',
            'completed': '✅',
            'in_progress': '🔄',
            'pending': '⏳',
            'blocked': '🚫',
            'failed': '❌',
            'warning': '⚠️',
            'ready': '🟢',
            'not_ready': '🔴'
        }
        return icons.get(status.lower(), '•')
    
    def _trend_arrow(self, trend: float) -> str:
        """Convert trend value to arrow"""
        if trend > 0.05:
            return '↑'
        elif trend < -0.05:
            return '↓'
        else:
            return '→'
    
    def _truncate_smart(self, text: str, length: int = 100) -> str:
        """Truncate text at word boundary"""
        if len(text) <= length:
            return text
        truncated = text[:length].rsplit(' ', 1)[0]
        return truncated + '...'
    
    def _register_builtin_templates(self):
        """Register the 12 built-in report templates"""
        
        templates = [
            ReportTemplate(
                template_id="cra_monitoring",
                report_type=ReportType.CRA_MONITORING,
                name="CRA Monitoring Report",
                description="Comprehensive monitoring report for CRA site visits",
                template_file="cra_monitoring.html",
                required_variables=['site_id', 'visit_date', 'cra_name', 'site_data'],
                optional_variables=['findings', 'actions', 'follow_ups'],
                supported_formats=[OutputFormat.HTML, OutputFormat.PDF, OutputFormat.WORD],
                sections=['header', 'site_overview', 'data_quality', 'issues', 'actions', 'signature'],
                approver_role="Study Lead",
                generation_time_seconds=30
            ),
            ReportTemplate(
                template_id="site_performance",
                report_type=ReportType.SITE_PERFORMANCE,
                name="Site Performance Summary",
                description="Weekly/monthly site performance metrics and trends",
                template_file="site_performance.html",
                required_variables=['site_id', 'period_start', 'period_end', 'metrics'],
                optional_variables=['benchmarks', 'trends', 'recommendations'],
                supported_formats=[OutputFormat.HTML, OutputFormat.PDF],
                sections=['header', 'kpis', 'trends', 'benchmarks', 'recommendations'],
                approver_role="CTM",
                generation_time_seconds=30
            ),
            ReportTemplate(
                template_id="sponsor_update",
                report_type=ReportType.SPONSOR_UPDATE,
                name="Sponsor Status Update",
                description="Executive summary for sponsor stakeholders",
                template_file="sponsor_update.html",
                required_variables=['study_id', 'report_date', 'study_metrics'],
                optional_variables=['milestones', 'risks', 'highlights'],
                supported_formats=[OutputFormat.HTML, OutputFormat.PDF, OutputFormat.POWERPOINT],
                sections=['executive_summary', 'enrollment', 'data_quality', 'safety', 'timeline', 'risks'],
                approver_role="Study Lead",
                generation_time_seconds=45
            ),
            ReportTemplate(
                template_id="meeting_pack",
                report_type=ReportType.MEETING_PACK,
                name="Meeting Pack Generator",
                description="Complete slide deck for study team meetings",
                template_file="meeting_pack.html",
                required_variables=['meeting_type', 'meeting_date', 'attendees', 'study_data'],
                optional_variables=['agenda', 'previous_actions', 'discussion_topics'],
                supported_formats=[OutputFormat.HTML, OutputFormat.POWERPOINT],
                sections=['title', 'agenda', 'status', 'metrics', 'issues', 'actions', 'next_steps'],
                approver_role="Study Lead",
                generation_time_seconds=60
            ),
            ReportTemplate(
                template_id="safety_narrative",
                report_type=ReportType.SAFETY_NARRATIVE,
                name="Safety Narrative",
                description="SAE narrative for regulatory submission",
                template_file="safety_narrative.html",
                required_variables=['sae_id', 'patient_id', 'event_details'],
                optional_variables=['medical_history', 'concomitant_meds', 'outcome'],
                supported_formats=[OutputFormat.HTML, OutputFormat.WORD],
                sections=['patient_info', 'event_description', 'medical_history', 'treatment', 'outcome', 'causality'],
                approver_role="Safety Physician",
                generation_time_seconds=60
            ),
            ReportTemplate(
                template_id="inspection_prep",
                report_type=ReportType.INSPECTION_PREP,
                name="Inspection Readiness Report",
                description="Comprehensive inspection preparation package",
                template_file="inspection_prep.html",
                required_variables=['site_id', 'inspection_date', 'inspector_type'],
                optional_variables=['focus_areas', 'document_checklist', 'risk_areas'],
                supported_formats=[OutputFormat.HTML, OutputFormat.PDF, OutputFormat.WORD],
                sections=['overview', 'document_status', 'data_quality', 'deviations', 'risk_areas', 'checklist'],
                approver_role="PI",
                generation_time_seconds=60
            ),
            ReportTemplate(
                template_id="query_summary",
                report_type=ReportType.QUERY_SUMMARY,
                name="Query Resolution Summary",
                description="Query status and aging report",
                template_file="query_summary.html",
                required_variables=['entity_id', 'query_data'],
                optional_variables=['aging_breakdown', 'top_issues', 'trends'],
                supported_formats=[OutputFormat.HTML, OutputFormat.PDF],
                sections=['summary', 'aging', 'by_category', 'by_site', 'trends'],
                generation_time_seconds=15
            ),
            ReportTemplate(
                template_id="site_newsletter",
                report_type=ReportType.SITE_NEWSLETTER,
                name="Site Newsletter",
                description="Monthly newsletter for site teams",
                template_file="site_newsletter.html",
                required_variables=['month', 'year', 'study_updates'],
                optional_variables=['tips', 'reminders', 'recognition'],
                supported_formats=[OutputFormat.HTML],
                sections=['header', 'updates', 'reminders', 'tips', 'recognition', 'contact'],
                approver_role="CTM",
                generation_time_seconds=45
            ),
            ReportTemplate(
                template_id="executive_brief",
                report_type=ReportType.EXECUTIVE_BRIEF,
                name="Executive Brief",
                description="One-page executive summary",
                template_file="executive_brief.html",
                required_variables=['study_id', 'report_date', 'key_metrics'],
                optional_variables=['highlights', 'concerns', 'decisions_needed'],
                supported_formats=[OutputFormat.HTML, OutputFormat.PDF],
                sections=['headline', 'metrics', 'status', 'actions'],
                approver_role="Study Lead",
                generation_time_seconds=30
            ),
            ReportTemplate(
                template_id="db_lock_readiness",
                report_type=ReportType.DB_LOCK_READINESS,
                name="Database Lock Readiness Report",
                description="Comprehensive DB lock readiness assessment",
                template_file="db_lock_readiness.html",
                required_variables=['study_id', 'target_date', 'readiness_data'],
                optional_variables=['blockers', 'timeline', 'actions'],
                supported_formats=[OutputFormat.HTML, OutputFormat.PDF],
                sections=['summary', 'readiness_score', 'blockers', 'by_site', 'timeline', 'actions'],
                approver_role="Data Manager",
                generation_time_seconds=45
            ),
            ReportTemplate(
                template_id="issue_escalation",
                report_type=ReportType.ISSUE_ESCALATION,
                name="Issue Escalation Report",
                description="Formal issue escalation documentation",
                template_file="issue_escalation.html",
                required_variables=['issue_id', 'issue_details', 'escalation_level'],
                optional_variables=['root_cause', 'proposed_actions', 'timeline'],
                supported_formats=[OutputFormat.HTML, OutputFormat.PDF, OutputFormat.WORD],
                sections=['issue_summary', 'impact', 'root_cause', 'actions', 'timeline', 'approval'],
                approver_role="Study Lead",
                generation_time_seconds=30
            ),
            ReportTemplate(
                template_id="daily_digest",
                report_type=ReportType.DAILY_DIGEST,
                name="Daily Digest",
                description="Daily summary of key activities and alerts",
                template_file="daily_digest.html",
                required_variables=['digest_date', 'recipient_role', 'summary_data'],
                optional_variables=['alerts', 'tasks', 'updates'],
                supported_formats=[OutputFormat.HTML],
                sections=['header', 'alerts', 'tasks', 'metrics', 'updates'],
                generation_time_seconds=15
            )
        ]
        
        for template in templates:
            self.templates[template.template_id] = template
    
    def get_template(self, template_id: str) -> Optional[ReportTemplate]:
        """Get template by ID"""
        return self.templates.get(template_id)
    
    def list_templates(self) -> List[Dict]:
        """List all available templates"""
        return [
            {
                'template_id': t.template_id,
                'name': t.name,
                'type': t.report_type.value,
                'description': t.description,
                'required_variables': t.required_variables,
                'supported_formats': [f.value for f in t.supported_formats],
                'approver': t.approver_role,
                'generation_time': t.generation_time_seconds
            }
            for t in self.templates.values()
        ]
    
    def validate_variables(self, template_id: str, variables: Dict[str, Any]) -> tuple:
        """
        Validate that all required variables are provided
        
        Returns:
            (is_valid, missing_vars, extra_vars)
        """
        template = self.get_template(template_id)
        if not template:
            return False, [f"Template '{template_id}' not found"], []
        
        required = set(template.required_variables)
        provided = set(variables.keys())
        
        missing = required - provided
        extra = provided - required - set(template.optional_variables)
        
        return len(missing) == 0, list(missing), list(extra)
    
    def render(
        self,
        template_id: str,
        variables: Dict[str, Any],
        output_format: OutputFormat = OutputFormat.HTML,
        include_metadata: bool = True,
        generated_by: str = "System"
    ) -> GeneratedReport:
        """
        Render a report from template
        
        Args:
            template_id: Template identifier
            variables: Variables to inject into template
            output_format: Desired output format
            include_metadata: Include report metadata header
            generated_by: User/system that generated the report
            
        Returns:
            GeneratedReport with rendered content
        """
        import time
        start_time = time.time()
        
        template = self.get_template(template_id)
        if not template:
            raise ValueError(f"Template '{template_id}' not found")
        
        # Validate variables
        is_valid, missing, extra = self.validate_variables(template_id, variables)
        warnings = []
        
        if not is_valid:
            raise ValueError(f"Missing required variables: {missing}")
        
        if extra:
            warnings.append(f"Extra variables provided (ignored): {extra}")
        
        # Generate report ID
        report_id = f"RPT-{template_id.upper()}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Create metadata
        metadata = ReportMetadata(
            report_id=report_id,
            report_type=template.report_type,
            title=template.name,
            generated_at=datetime.now(),
            generated_by=generated_by,
            expires_at=datetime.now() + timedelta(days=30) if template.report_type == ReportType.DAILY_DIGEST else None
        )
        
        # Add metadata to variables
        render_vars = {**variables}
        if include_metadata:
            render_vars['_metadata'] = metadata.to_dict()
            render_vars['_template'] = {
                'id': template.template_id,
                'name': template.name,
                'sections': template.sections
            }
        
        # Render template
        try:
            content = self._render_template(template, render_vars, output_format)
        except Exception as e:
            logger.error(f"Template rendering failed: {e}")
            raise
        
        # Calculate checksum
        metadata.checksum = hashlib.md5(content.encode()).hexdigest()
        
        # Calculate generation time
        generation_time_ms = int((time.time() - start_time) * 1000)
        
        # Update stats
        self.stats['reports_generated'] += 1
        self.stats['total_generation_time_ms'] += generation_time_ms
        
        report = GeneratedReport(
            report_id=report_id,
            metadata=metadata,
            content=content,
            format=output_format,
            generation_time_ms=generation_time_ms,
            warnings=warnings
        )
        
        logger.info(f"Generated report {report_id} in {generation_time_ms}ms")
        
        return report
    
    def _render_template(
        self,
        template: ReportTemplate,
        variables: Dict[str, Any],
        output_format: OutputFormat
    ) -> str:
        """Render template with variables"""
        
        # Get the inline template for the report type
        template_content = self._get_inline_template(template.template_id)
        
        # Create a new environment with string loader for this template
        string_env = Environment(
            loader=StringLoader({template.template_file: template_content}),
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Copy filters and globals from main environment
        string_env.filters = self.env.filters.copy()
        string_env.globals = self.env.globals.copy()
        
        # Render
        jinja_template = string_env.get_template(template.template_file)
        rendered = jinja_template.render(**variables)
        
        return rendered
    
    def _get_inline_template(self, template_id: str) -> str:
        """Get inline template content for a report type"""
        
        templates = {
            'cra_monitoring': self._template_cra_monitoring(),
            'site_performance': self._template_site_performance(),
            'sponsor_update': self._template_sponsor_update(),
            'meeting_pack': self._template_meeting_pack(),
            'safety_narrative': self._template_safety_narrative(),
            'inspection_prep': self._template_inspection_prep(),
            'query_summary': self._template_query_summary(),
            'site_newsletter': self._template_site_newsletter(),
            'executive_brief': self._template_executive_brief(),
            'db_lock_readiness': self._template_db_lock_readiness(),
            'issue_escalation': self._template_issue_escalation(),
            'daily_digest': self._template_daily_digest()
        }
        
        return templates.get(template_id, self._template_default())
    
    # ==================== INLINE TEMPLATES ====================
    
    def _template_cra_monitoring(self) -> str:
        return '''<!DOCTYPE html>
<html>
<head>
    <title>CRA Monitoring Report - {{ site_id }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; color: #333; }
        .header { background: #1a365d; color: white; padding: 20px; margin-bottom: 20px; }
        .header h1 { margin: 0; }
        .header .meta { font-size: 14px; margin-top: 10px; opacity: 0.9; }
        .section { margin-bottom: 30px; }
        .section h2 { color: #1a365d; border-bottom: 2px solid #1a365d; padding-bottom: 10px; }
        .metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; }
        .metric-card { background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; }
        .metric-value { font-size: 28px; font-weight: bold; color: #1a365d; }
        .metric-label { font-size: 12px; color: #666; margin-top: 5px; }
        table { width: 100%; border-collapse: collapse; margin-top: 15px; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #f8f9fa; font-weight: 600; }
        .status-good { color: #28a745; }
        .status-warning { color: #ffc107; }
        .status-critical { color: #dc3545; }
        .badge { padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }
        .badge-success { background: #d4edda; color: #155724; }
        .badge-warning { background: #fff3cd; color: #856404; }
        .badge-danger { background: #f8d7da; color: #721c24; }
        .findings { background: #fff3cd; padding: 15px; border-radius: 8px; margin-top: 15px; }
        .actions { background: #e7f3ff; padding: 15px; border-radius: 8px; margin-top: 15px; }
        .signature-block { margin-top: 40px; padding-top: 20px; border-top: 2px solid #ddd; }
        .signature-line { margin-top: 40px; border-top: 1px solid #333; width: 300px; }
        .footer { margin-top: 40px; font-size: 12px; color: #666; text-align: center; }
    </style>
</head>
<body>
    <div class="header">
        <h1>CRA Monitoring Report</h1>
        <div class="meta">
            <strong>Site:</strong> {{ site_id }} | 
            <strong>Visit Date:</strong> {{ visit_date | format_date }} | 
            <strong>CRA:</strong> {{ cra_name }}
        </div>
    </div>
    
    <div class="section">
        <h2>Site Overview</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{{ site_data.total_patients | default(0) | format_number }}</div>
                <div class="metric-label">Total Patients</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ site_data.dqi_score | default(0) | format_decimal(1) }}</div>
                <div class="metric-label">DQI Score</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ site_data.clean_rate | default(0) | format_percent }}</div>
                <div class="metric-label">Clean Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ site_data.open_queries | default(0) }}</div>
                <div class="metric-label">Open Queries</div>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>Data Quality Metrics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Current</th>
                <th>Target</th>
                <th>Status</th>
            </tr>
            {% for metric in site_data.metrics | default([]) %}
            <tr>
                <td>{{ metric.name }}</td>
                <td>{{ metric.value | format_decimal(1) }}</td>
                <td>{{ metric.target | format_decimal(1) }}</td>
                <td>
                    {% if metric.value >= metric.target %}
                    <span class="badge badge-success">On Target</span>
                    {% elif metric.value >= metric.target * 0.9 %}
                    <span class="badge badge-warning">Near Target</span>
                    {% else %}
                    <span class="badge badge-danger">Below Target</span>
                    {% endif %}
                </td>
            </tr>
            {% else %}
            <tr><td colspan="4">No metrics available</td></tr>
            {% endfor %}
        </table>
    </div>
    
    {% if findings %}
    <div class="section">
        <h2>Findings</h2>
        <div class="findings">
            <ul>
            {% for finding in findings %}
                <li><strong>{{ finding.category }}:</strong> {{ finding.description }}</li>
            {% endfor %}
            </ul>
        </div>
    </div>
    {% endif %}
    
    {% if actions %}
    <div class="section">
        <h2>Action Items</h2>
        <div class="actions">
            <table>
                <tr>
                    <th>Action</th>
                    <th>Owner</th>
                    <th>Due Date</th>
                    <th>Priority</th>
                </tr>
                {% for action in actions %}
                <tr>
                    <td>{{ action.description }}</td>
                    <td>{{ action.owner }}</td>
                    <td>{{ action.due_date | format_date }}</td>
                    <td>{{ action.priority | priority_badge }}</td>
                </tr>
                {% endfor %}
            </table>
        </div>
    </div>
    {% endif %}
    
    <div class="signature-block">
        <h2>Signatures</h2>
        <p><strong>CRA:</strong> {{ cra_name }}</p>
        <div class="signature-line"></div>
        <p style="margin-top: 5px;">Date: _____________</p>
        
        <p style="margin-top: 30px;"><strong>Site Coordinator:</strong></p>
        <div class="signature-line"></div>
        <p style="margin-top: 5px;">Date: _____________</p>
    </div>
    
    <div class="footer">
        <p>Report ID: {{ _metadata.report_id }} | Generated: {{ _metadata.generated_at }} | Classification: {{ _metadata.classification }}</p>
    </div>
</body>
</html>'''

    def _template_site_performance(self) -> str:
        return '''<!DOCTYPE html>
<html>
<head>
    <title>Site Performance Report - {{ site_id }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; color: #333; }
        .header { background: linear-gradient(135deg, #1a365d 0%, #2d5a87 100%); color: white; padding: 25px; margin-bottom: 25px; border-radius: 8px; }
        .header h1 { margin: 0; font-size: 24px; }
        .period { font-size: 14px; margin-top: 8px; opacity: 0.9; }
        .kpi-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 15px; margin-bottom: 30px; }
        .kpi-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; }
        .kpi-value { font-size: 32px; font-weight: bold; }
        .kpi-label { font-size: 12px; color: #666; margin-top: 5px; }
        .kpi-trend { font-size: 14px; margin-top: 8px; }
        .trend-up { color: #28a745; }
        .trend-down { color: #dc3545; }
        .trend-flat { color: #6c757d; }
        .section { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
        .section h2 { margin-top: 0; color: #1a365d; font-size: 18px; }
        .benchmark-bar { background: #e9ecef; height: 24px; border-radius: 4px; position: relative; margin: 10px 0; }
        .benchmark-fill { height: 100%; border-radius: 4px; }
        .benchmark-marker { position: absolute; top: -5px; width: 3px; height: 34px; background: #333; }
        .percentile-label { font-size: 11px; color: #666; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #eee; }
        th { background: #f8f9fa; font-weight: 600; font-size: 13px; }
        .recommendation { background: #e7f3ff; padding: 15px; border-radius: 8px; margin-top: 10px; border-left: 4px solid #007bff; }
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Site Performance Summary</h1>
        <div class="period">
            <strong>{{ site_id }}</strong> | 
            Period: {{ period_start | format_date }} to {{ period_end | format_date }}
        </div>
    </div>
    
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-value" style="color: {% if metrics.dqi >= 85 %}#28a745{% elif metrics.dqi >= 70 %}#ffc107{% else %}#dc3545{% endif %}">
                {{ metrics.dqi | format_decimal(1) }}
            </div>
            <div class="kpi-label">DQI Score</div>
            <div class="kpi-trend {% if metrics.dqi_trend > 0 %}trend-up{% elif metrics.dqi_trend < 0 %}trend-down{% else %}trend-flat{% endif %}">
                {{ metrics.dqi_trend | trend_arrow }} {{ metrics.dqi_trend | abs | format_decimal(1) }}
            </div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">{{ metrics.clean_rate | format_percent }}</div>
            <div class="kpi-label">Clean Rate</div>
            <div class="kpi-trend {% if metrics.clean_trend > 0 %}trend-up{% elif metrics.clean_trend < 0 %}trend-down{% else %}trend-flat{% endif %}">
                {{ metrics.clean_trend | trend_arrow }}
            </div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">{{ metrics.query_resolution_days | format_decimal(1) }}</div>
            <div class="kpi-label">Avg Query Days</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">{{ metrics.sdv_rate | format_percent }}</div>
            <div class="kpi-label">SDV Complete</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">{{ metrics.patients }}</div>
            <div class="kpi-label">Active Patients</div>
        </div>
    </div>
    
    {% if benchmarks %}
    <div class="section">
        <h2>📈 Performance vs Benchmarks</h2>
        {% for bench in benchmarks %}
        <div style="margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <strong>{{ bench.metric }}</strong>
                <span class="percentile-label">Percentile: {{ bench.percentile }}th</span>
            </div>
            <div class="benchmark-bar">
                <div class="benchmark-fill" style="width: {{ bench.percentile }}%; background: {% if bench.percentile >= 75 %}#28a745{% elif bench.percentile >= 50 %}#ffc107{% else %}#dc3545{% endif %};"></div>
                <div class="benchmark-marker" style="left: 50%;"></div>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 11px; color: #666;">
                <span>Your Value: {{ bench.value | format_decimal(1) }}</span>
                <span>Median: {{ bench.median | format_decimal(1) }}</span>
            </div>
        </div>
        {% endfor %}
    </div>
    {% endif %}
    
    {% if recommendations %}
    <div class="section">
        <h2>💡 Recommendations</h2>
        {% for rec in recommendations %}
        <div class="recommendation">
            <strong>{{ rec.title }}</strong>
            <p style="margin: 5px 0 0 0;">{{ rec.description }}</p>
        </div>
        {% endfor %}
    </div>
    {% endif %}
    
    <div style="text-align: center; color: #666; font-size: 12px; margin-top: 30px;">
        Report ID: {{ _metadata.report_id }} | Generated: {{ _metadata.generated_at }}
    </div>
</body>
</html>'''

    def _template_executive_brief(self) -> str:
        return '''<!DOCTYPE html>
<html>
<head>
    <title>Executive Brief - {{ study_id }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 40px; background: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .header { background: linear-gradient(135deg, #1a365d 0%, #2d5a87 100%); color: white; padding: 30px; }
        .header h1 { margin: 0; font-size: 28px; }
        .header .subtitle { font-size: 16px; margin-top: 8px; opacity: 0.9; }
        .content { padding: 30px; }
        .headline { background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 25px; border-left: 4px solid #1a365d; }
        .headline h2 { margin: 0 0 10px 0; color: #1a365d; font-size: 18px; }
        .kpi-row { display: flex; justify-content: space-between; margin-bottom: 25px; }
        .kpi { text-align: center; flex: 1; padding: 15px; }
        .kpi-value { font-size: 36px; font-weight: bold; color: #1a365d; }
        .kpi-label { font-size: 13px; color: #666; margin-top: 5px; }
        .status-section { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 25px; }
        .status-card { padding: 20px; border-radius: 8px; }
        .status-good { background: #d4edda; border-left: 4px solid #28a745; }
        .status-warning { background: #fff3cd; border-left: 4px solid #ffc107; }
        .status-critical { background: #f8d7da; border-left: 4px solid #dc3545; }
        .status-card h3 { margin: 0 0 10px 0; font-size: 14px; }
        .action-list { list-style: none; padding: 0; margin: 0; }
        .action-list li { padding: 10px 0; border-bottom: 1px solid #eee; display: flex; align-items: center; }
        .action-list li:last-child { border-bottom: none; }
        .action-priority { width: 80px; font-size: 11px; font-weight: bold; text-transform: uppercase; }
        .footer { background: #f8f9fa; padding: 20px; text-align: center; font-size: 12px; color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📋 Executive Brief</h1>
            <div class="subtitle">{{ study_id }} | {{ report_date | format_date }}</div>
        </div>
        
        <div class="content">
            {% if highlights %}
            <div class="headline">
                <h2>🎯 Key Headline</h2>
                <p style="margin: 0; font-size: 16px;">{{ highlights[0] if highlights else 'Study progressing as planned.' }}</p>
            </div>
            {% endif %}
            
            <div class="kpi-row">
                <div class="kpi">
                    <div class="kpi-value">{{ key_metrics.patients | format_number }}</div>
                    <div class="kpi-label">Total Patients</div>
                </div>
                <div class="kpi">
                    <div class="kpi-value">{{ key_metrics.dqi | format_decimal(1) }}</div>
                    <div class="kpi-label">DQI Score</div>
                </div>
                <div class="kpi">
                    <div class="kpi-value">{{ key_metrics.clean_rate | format_percent }}</div>
                    <div class="kpi-label">Clean Rate</div>
                </div>
                <div class="kpi">
                    <div class="kpi-value">{{ key_metrics.dblock_ready | format_percent }}</div>
                    <div class="kpi-label">DB Lock Ready</div>
                </div>
            </div>
            
            <div class="status-section">
                <div class="status-card status-good">
                    <h3>✅ On Track</h3>
                    <ul style="margin: 0; padding-left: 20px; font-size: 14px;">
                        {% for item in key_metrics.on_track | default(['Data quality improving', 'Enrollment on target']) %}
                        <li>{{ item }}</li>
                        {% endfor %}
                    </ul>
                </div>
                <div class="status-card {% if concerns %}status-warning{% else %}status-good{% endif %}">
                    <h3>{% if concerns %}⚠️ Attention Needed{% else %}✅ No Concerns{% endif %}</h3>
                    <ul style="margin: 0; padding-left: 20px; font-size: 14px;">
                        {% for item in concerns | default(['No major concerns']) %}
                        <li>{{ item }}</li>
                        {% endfor %}
                    </ul>
                </div>
            </div>
            
            {% if decisions_needed %}
            <div>
                <h3 style="color: #1a365d; margin-bottom: 15px;">🔔 Decisions Needed</h3>
                <ul class="action-list">
                    {% for decision in decisions_needed %}
                    <li>
                        <span class="action-priority" style="color: {% if decision.priority == 'critical' %}#dc3545{% elif decision.priority == 'high' %}#fd7e14{% else %}#28a745{% endif %}">
                            {{ decision.priority }}
                        </span>
                        <span>{{ decision.description }}</span>
                    </li>
                    {% endfor %}
                </ul>
            </div>
            {% endif %}
        </div>
        
        <div class="footer">
            Report ID: {{ _metadata.report_id }} | Prepared by: {{ _metadata.generated_by }} | {{ _metadata.generated_at }}
        </div>
    </div>
</body>
</html>'''

    def _template_db_lock_readiness(self) -> str:
        return '''<!DOCTYPE html>
<html>
<head>
    <title>DB Lock Readiness - {{ study_id }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; color: #333; }
        .header { background: #1a365d; color: white; padding: 25px; border-radius: 8px; margin-bottom: 25px; }
        .header h1 { margin: 0; }
        .target-date { font-size: 18px; margin-top: 10px; }
        .score-section { display: flex; gap: 30px; margin-bottom: 30px; }
        .score-circle { width: 180px; height: 180px; border-radius: 50%; display: flex; flex-direction: column; justify-content: center; align-items: center; color: white; }
        .score-value { font-size: 48px; font-weight: bold; }
        .score-label { font-size: 14px; margin-top: 5px; }
        .score-good { background: linear-gradient(135deg, #28a745, #20c997); }
        .score-warning { background: linear-gradient(135deg, #ffc107, #fd7e14); }
        .score-critical { background: linear-gradient(135deg, #dc3545, #c82333); }
        .breakdown { flex: 1; }
        .breakdown h3 { margin-top: 0; color: #1a365d; }
        .progress-bar { background: #e9ecef; height: 30px; border-radius: 4px; margin: 10px 0; overflow: hidden; }
        .progress-fill { height: 100%; display: flex; align-items: center; padding-left: 10px; color: white; font-weight: bold; }
        .blocker-section { background: #fff3cd; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .blocker-list { list-style: none; padding: 0; margin: 15px 0 0 0; }
        .blocker-item { display: flex; justify-content: space-between; padding: 12px; background: white; margin-bottom: 8px; border-radius: 4px; border-left: 4px solid #dc3545; }
        .site-table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        .site-table th, .site-table td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        .site-table th { background: #f8f9fa; font-weight: 600; }
        .badge { padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: bold; }
        .badge-ready { background: #d4edda; color: #155724; }
        .badge-pending { background: #fff3cd; color: #856404; }
        .badge-blocked { background: #f8d7da; color: #721c24; }
        .timeline { margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 8px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔒 Database Lock Readiness Report</h1>
        <div class="target-date">
            Study: {{ study_id }} | Target Date: {{ target_date | format_date }}
        </div>
    </div>
    
    <div class="score-section">
        <div class="score-circle {% if readiness_data.ready_rate >= 80 %}score-good{% elif readiness_data.ready_rate >= 50 %}score-warning{% else %}score-critical{% endif %}">
            <div class="score-value">{{ readiness_data.ready_rate | format_decimal(0) }}%</div>
            <div class="score-label">Ready for Lock</div>
        </div>
        
        <div class="breakdown">
            <h3>Readiness Breakdown</h3>
            {% for category in readiness_data.categories | default([]) %}
            <div>
                <div style="display: flex; justify-content: space-between; font-size: 14px;">
                    <span>{{ category.name }}</span>
                    <span>{{ category.rate | format_percent }}</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {{ category.rate }}%; background: {% if category.rate >= 90 %}#28a745{% elif category.rate >= 70 %}#ffc107{% else %}#dc3545{% endif %};">
                    </div>
                </div>
            </div>
            {% endfor %}
        </div>
    </div>
    
    {% if blockers %}
    <div class="blocker-section">
        <h3 style="margin-top: 0;">🚫 Top Blockers</h3>
        <ul class="blocker-list">
            {% for blocker in blockers[:5] %}
            <li class="blocker-item">
                <span><strong>{{ blocker.type | title_case }}</strong>: {{ blocker.description }}</span>
                <span>{{ blocker.count | format_number }} patients</span>
            </li>
            {% endfor %}
        </ul>
    </div>
    {% endif %}
    
    <h3>Site Readiness Summary</h3>
    <table class="site-table">
        <thead>
            <tr>
                <th>Site</th>
                <th>Patients</th>
                <th>Ready</th>
                <th>Pending</th>
                <th>Blocked</th>
                <th>Status</th>
            </tr>
        </thead>
        <tbody>
            {% for site in readiness_data.sites | default([]) %}
            <tr>
                <td>{{ site.site_id }}</td>
                <td>{{ site.patients }}</td>
                <td>{{ site.ready }}</td>
                <td>{{ site.pending }}</td>
                <td>{{ site.blocked }}</td>
                <td>
                    {% if site.blocked == 0 and site.pending == 0 %}
                    <span class="badge badge-ready">Ready</span>
                    {% elif site.blocked == 0 %}
                    <span class="badge badge-pending">Pending</span>
                    {% else %}
                    <span class="badge badge-blocked">Blocked</span>
                    {% endif %}
                </td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    
    {% if timeline %}
    <div class="timeline">
        <h3 style="margin-top: 0;">📅 Projected Timeline</h3>
        <p><strong>Current Trajectory:</strong> {{ timeline.projected_date | format_date }}</p>
        <p><strong>Days to Target:</strong> {{ timeline.days_remaining }} days</p>
        <p><strong>Confidence:</strong> {{ timeline.confidence | format_percent }}</p>
    </div>
    {% endif %}
    
    <div style="text-align: center; color: #666; font-size: 12px; margin-top: 30px;">
        Report ID: {{ _metadata.report_id }} | Generated: {{ _metadata.generated_at }}
    </div>
</body>
</html>'''

    def _template_daily_digest(self) -> str:
        return '''<!DOCTYPE html>
<html>
<head>
    <title>Daily Digest - {{ digest_date | format_date }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 0; background: #f5f5f5; }
        .container { max-width: 600px; margin: 0 auto; background: white; }
        .header { background: #1a365d; color: white; padding: 20px; text-align: center; }
        .header h1 { margin: 0; font-size: 22px; }
        .header .date { font-size: 14px; margin-top: 5px; opacity: 0.9; }
        .section { padding: 20px; border-bottom: 1px solid #eee; }
        .section h2 { color: #1a365d; font-size: 16px; margin: 0 0 15px 0; }
        .alert-card { padding: 12px; border-radius: 6px; margin-bottom: 10px; }
        .alert-critical { background: #f8d7da; border-left: 4px solid #dc3545; }
        .alert-high { background: #fff3cd; border-left: 4px solid #fd7e14; }
        .alert-normal { background: #e7f3ff; border-left: 4px solid #007bff; }
        .task-list { list-style: none; padding: 0; margin: 0; }
        .task-item { display: flex; align-items: center; padding: 10px 0; border-bottom: 1px solid #eee; }
        .task-item:last-child { border-bottom: none; }
        .task-checkbox { width: 20px; height: 20px; border: 2px solid #ddd; border-radius: 4px; margin-right: 12px; }
        .task-priority { font-size: 11px; font-weight: bold; padding: 2px 8px; border-radius: 10px; margin-left: auto; }
        .priority-critical { background: #dc3545; color: white; }
        .priority-high { background: #fd7e14; color: white; }
        .priority-medium { background: #ffc107; color: #333; }
        .metrics-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
        .metric-box { text-align: center; padding: 15px; background: #f8f9fa; border-radius: 6px; }
        .metric-value { font-size: 24px; font-weight: bold; color: #1a365d; }
        .metric-label { font-size: 11px; color: #666; margin-top: 5px; }
        .footer { padding: 20px; text-align: center; font-size: 12px; color: #666; background: #f8f9fa; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📬 Daily Digest</h1>
            <div class="date">{{ digest_date | format_date }} | {{ recipient_role | title_case }}</div>
        </div>
        
        {% if alerts %}
        <div class="section">
            <h2>🚨 Alerts ({{ alerts | length }})</h2>
            {% for alert in alerts %}
            <div class="alert-card alert-{{ alert.severity | lower }}">
                <strong>{{ alert.title }}</strong>
                <p style="margin: 5px 0 0 0; font-size: 14px;">{{ alert.message }}</p>
            </div>
            {% endfor %}
        </div>
        {% endif %}
        
        {% if tasks %}
        <div class="section">
            <h2>✅ Your Tasks ({{ tasks | length }})</h2>
            <ul class="task-list">
                {% for task in tasks %}
                <li class="task-item">
                    <div class="task-checkbox"></div>
                    <span>{{ task.description }}</span>
                    <span class="task-priority priority-{{ task.priority | lower }}">{{ task.priority }}</span>
                </li>
                {% endfor %}
            </ul>
        </div>
        {% endif %}
        
        <div class="section">
            <h2>📊 Today's Metrics</h2>
            <div class="metrics-grid">
                <div class="metric-box">
                    <div class="metric-value">{{ summary_data.new_issues | default(0) }}</div>
                    <div class="metric-label">New Issues</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">{{ summary_data.resolved_today | default(0) }}</div>
                    <div class="metric-label">Resolved Today</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">{{ summary_data.pending_actions | default(0) }}</div>
                    <div class="metric-label">Pending Actions</div>
                </div>
            </div>
        </div>
        
        {% if updates %}
        <div class="section">
            <h2>📝 Updates</h2>
            {% for update in updates %}
            <p style="margin: 10px 0;"><strong>{{ update.time }}:</strong> {{ update.message }}</p>
            {% endfor %}
        </div>
        {% endif %}
        
        <div class="footer">
            <p>This is an automated digest from TrialPulse Nexus</p>
            <p>Report ID: {{ _metadata.report_id }}</p>
        </div>
    </div>
</body>
</html>'''

    def _template_query_summary(self) -> str:
        return '''<!DOCTYPE html>
<html>
<head>
    <title>Query Summary - {{ entity_id }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { background: #1a365d; color: white; padding: 20px; border-radius: 8px; }
        .header h1 { margin: 0; }
        .summary-cards { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }
        .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; }
        .card-value { font-size: 32px; font-weight: bold; }
        .card-label { font-size: 12px; color: #666; margin-top: 5px; }
        .aging-chart { margin: 20px 0; }
        .aging-bar { display: flex; height: 40px; border-radius: 4px; overflow: hidden; margin: 10px 0; }
        .aging-segment { display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 14px; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #f8f9fa; }
    </style>
</head>
<body>
    <div class="header">
        <h1>📋 Query Resolution Summary</h1>
        <p style="margin: 5px 0 0 0;">{{ entity_id }} | Generated: {{ now() | format_datetime }}</p>
    </div>
    
    <div class="summary-cards">
        <div class="card">
            <div class="card-value" style="color: #1a365d;">{{ query_data.total | format_number }}</div>
            <div class="card-label">Total Queries</div>
        </div>
        <div class="card">
            <div class="card-value" style="color: #dc3545;">{{ query_data.open | format_number }}</div>
            <div class="card-label">Open</div>
        </div>
        <div class="card">
            <div class="card-value" style="color: #28a745;">{{ query_data.resolved | format_number }}</div>
            <div class="card-label">Resolved</div>
        </div>
        <div class="card">
            <div class="card-value" style="color: #ffc107;">{{ query_data.avg_days | format_decimal(1) }}</div>
            <div class="card-label">Avg Days Open</div>
        </div>
    </div>
    
    {% if aging_breakdown %}
    <div class="aging-chart">
        <h3>Query Aging</h3>
        <div class="aging-bar">
            {% for age in aging_breakdown %}
            <div class="aging-segment" style="width: {{ age.percent }}%; background: {{ age.color }};">
                {{ age.count }}
            </div>
            {% endfor %}
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 12px; color: #666;">
            {% for age in aging_breakdown %}
            <span>{{ age.label }}: {{ age.count }} ({{ age.percent | format_decimal(0) }}%)</span>
            {% endfor %}
        </div>
    </div>
    {% endif %}
    
    {% if top_issues %}
    <h3>Top Query Categories</h3>
    <table>
        <tr>
            <th>Category</th>
            <th>Count</th>
            <th>% of Total</th>
            <th>Avg Days</th>
        </tr>
        {% for issue in top_issues %}
        <tr>
            <td>{{ issue.category }}</td>
            <td>{{ issue.count }}</td>
            <td>{{ issue.percent | format_percent }}</td>
            <td>{{ issue.avg_days | format_decimal(1) }}</td>
        </tr>
        {% endfor %}
    </table>
    {% endif %}
    
    <div style="text-align: center; color: #666; font-size: 12px; margin-top: 30px;">
        Report ID: {{ _metadata.report_id }}
    </div>
</body>
</html>'''

    def _template_sponsor_update(self) -> str:
        return '''<!DOCTYPE html>
<html>
<head>
    <title>Sponsor Update - {{ study_id }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { background: linear-gradient(135deg, #1a365d, #2d5a87); color: white; padding: 30px; border-radius: 8px; }
        .metrics-row { display: flex; gap: 20px; margin: 25px 0; }
        .metric { flex: 1; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); text-align: center; }
        .metric-value { font-size: 36px; font-weight: bold; color: #1a365d; }
        .metric-label { color: #666; margin-top: 5px; }
        .section { background: white; padding: 25px; border-radius: 8px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
        .section h2 { color: #1a365d; margin-top: 0; border-bottom: 2px solid #e9ecef; padding-bottom: 10px; }
        .highlight { background: #e7f3ff; padding: 15px; border-radius: 6px; border-left: 4px solid #007bff; }
        .risk { background: #fff3cd; padding: 15px; border-radius: 6px; border-left: 4px solid #ffc107; }
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Sponsor Status Update</h1>
        <p>{{ study_id }} | {{ report_date | format_date }}</p>
    </div>
    
    <div class="metrics-row">
        <div class="metric">
            <div class="metric-value">{{ study_metrics.patients | format_number }}</div>
            <div class="metric-label">Total Patients</div>
        </div>
        <div class="metric">
            <div class="metric-value">{{ study_metrics.sites }}</div>
            <div class="metric-label">Active Sites</div>
        </div>
        <div class="metric">
            <div class="metric-value">{{ study_metrics.dqi | format_decimal(1) }}</div>
            <div class="metric-label">DQI Score</div>
        </div>
        <div class="metric">
            <div class="metric-value">{{ study_metrics.dblock_ready | format_percent }}</div>
            <div class="metric-label">DB Lock Ready</div>
        </div>
    </div>
    
    {% if highlights %}
    <div class="section">
        <h2>✨ Key Highlights</h2>
        {% for h in highlights %}
        <div class="highlight" style="margin-bottom: 10px;">{{ h }}</div>
        {% endfor %}
    </div>
    {% endif %}
    
    {% if risks %}
    <div class="section">
        <h2>⚠️ Risks & Mitigations</h2>
        {% for r in risks %}
        <div class="risk" style="margin-bottom: 10px;">
            <strong>{{ r.risk }}</strong><br>
            <span style="color: #28a745;">Mitigation:</span> {{ r.mitigation }}
        </div>
        {% endfor %}
    </div>
    {% endif %}
    
    <div style="text-align: center; color: #666; font-size: 12px; margin-top: 30px;">
        Report ID: {{ _metadata.report_id }} | Confidential
    </div>
</body>
</html>'''

    def _template_meeting_pack(self) -> str:
        return '''<!DOCTYPE html>
<html>
<head>
    <title>{{ meeting_type }} - {{ meeting_date | format_date }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 0; }
        .slide { page-break-after: always; padding: 40px; min-height: 700px; }
        .slide:last-child { page-break-after: avoid; }
        .title-slide { background: linear-gradient(135deg, #1a365d, #2d5a87); color: white; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; }
        .title-slide h1 { font-size: 42px; margin-bottom: 20px; }
        .slide h2 { color: #1a365d; border-bottom: 3px solid #1a365d; padding-bottom: 10px; }
        .agenda-list { list-style: none; padding: 0; }
        .agenda-list li { padding: 15px; margin: 10px 0; background: #f8f9fa; border-radius: 6px; font-size: 18px; }
        .metrics-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }
        .metric-card { background: #f8f9fa; padding: 25px; border-radius: 8px; text-align: center; }
        .metric-value { font-size: 48px; font-weight: bold; color: #1a365d; }
    </style>
</head>
<body>
    <!-- Title Slide -->
    <div class="slide title-slide">
        <h1>{{ meeting_type }}</h1>
        <p style="font-size: 24px;">{{ meeting_date | format_date }}</p>
        <p style="margin-top: 30px;">Attendees: {{ attendees | join(', ') }}</p>
    </div>
    
    <!-- Agenda Slide -->
    <div class="slide">
        <h2>📋 Agenda</h2>
        <ul class="agenda-list">
            {% for item in agenda | default(['Study Status', 'Data Quality Review', 'Open Issues', 'Action Items', 'Next Steps']) %}
            <li>{{ loop.index }}. {{ item }}</li>
            {% endfor %}
        </ul>
    </div>
    
    <!-- Status Slide -->
    <div class="slide">
        <h2>📊 Study Status</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{{ study_data.patients | format_number }}</div>
                <div>Total Patients</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ study_data.dqi | format_decimal(1) }}</div>
                <div>DQI Score</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ study_data.clean_rate | format_percent }}</div>
                <div>Clean Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ study_data.sites }}</div>
                <div>Active Sites</div>
            </div>
        </div>
    </div>
    
    <!-- Actions Slide -->
    <div class="slide">
        <h2>✅ Action Items</h2>
        <table style="width: 100%; border-collapse: collapse;">
            <tr style="background: #1a365d; color: white;">
                <th style="padding: 12px; text-align: left;">Action</th>
                <th style="padding: 12px;">Owner</th>
                <th style="padding: 12px;">Due Date</th>
            </tr>
            {% for action in previous_actions | default([]) %}
            <tr style="border-bottom: 1px solid #ddd;">
                <td style="padding: 12px;">{{ action.description }}</td>
                <td style="padding: 12px; text-align: center;">{{ action.owner }}</td>
                <td style="padding: 12px; text-align: center;">{{ action.due_date | format_date }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>
</body>
</html>'''

    def _template_safety_narrative(self) -> str:
        return '''<!DOCTYPE html>
<html>
<head>
    <title>Safety Narrative - {{ sae_id }}</title>
    <style>
        body { font-family: 'Times New Roman', serif; margin: 50px; line-height: 1.6; }
        .header { border-bottom: 2px solid #333; padding-bottom: 15px; margin-bottom: 25px; }
        .header h1 { margin: 0; font-size: 20px; }
        .field { margin: 15px 0; }
        .field-label { font-weight: bold; display: inline-block; width: 180px; }
        .section { margin: 25px 0; }
        .section h2 { font-size: 16px; border-bottom: 1px solid #999; padding-bottom: 5px; }
        .narrative { background: #f9f9f9; padding: 20px; border-left: 3px solid #1a365d; margin: 15px 0; }
        .footer { margin-top: 40px; border-top: 1px solid #ddd; padding-top: 15px; font-size: 12px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>SERIOUS ADVERSE EVENT NARRATIVE</h1>
        <p><strong>SAE ID:</strong> {{ sae_id }} | <strong>Patient:</strong> {{ patient_id }}</p>
    </div>
    
    <div class="section">
        <h2>Patient Information</h2>
        <div class="field"><span class="field-label">Subject ID:</span> {{ patient_id }}</div>
        <div class="field"><span class="field-label">Age/Sex:</span> {{ event_details.age | default('N/A') }} / {{ event_details.sex | default('N/A') }}</div>
        <div class="field"><span class="field-label">Study Site:</span> {{ event_details.site_id | default('N/A') }}</div>
        <div class="field"><span class="field-label">Enrollment Date:</span> {{ event_details.enrollment_date | default('N/A') }}</div>
    </div>
    
    <div class="section">
        <h2>Event Description</h2>
        <div class="field"><span class="field-label">Event Term:</span> {{ event_details.event_term }}</div>
        <div class="field"><span class="field-label">MedDRA PT:</span> {{ event_details.meddra_pt | default('Pending coding') }}</div>
        <div class="field"><span class="field-label">Onset Date:</span> {{ event_details.onset_date | format_date if event_details.onset_date else 'N/A' }}</div>
        <div class="field"><span class="field-label">Resolution Date:</span> {{ event_details.resolution_date | format_date if event_details.resolution_date else 'Ongoing' }}</div>
        <div class="field"><span class="field-label">Severity:</span> {{ event_details.severity | default('N/A') }}</div>
        <div class="field"><span class="field-label">Seriousness Criteria:</span> {{ event_details.seriousness_criteria | default('N/A') }}</div>
        
        <div class="narrative">
            <strong>Narrative:</strong><br>
            {{ event_details.narrative | default('Narrative pending.') }}
        </div>
    </div>
    
    {% if medical_history %}
    <div class="section">
        <h2>Relevant Medical History</h2>
        <ul>
            {% for condition in medical_history %}
            <li>{{ condition.condition }} {% if condition.start_date %}({{ condition.start_date }}){% endif %}</li>
            {% endfor %}
        </ul>
    </div>
    {% endif %}
    
    {% if concomitant_meds %}
    <div class="section">
        <h2>Concomitant Medications</h2>
        <table style="width: 100%; border-collapse: collapse;">
            <tr style="background: #f0f0f0;">
                <th style="padding: 8px; text-align: left; border: 1px solid #ddd;">Medication</th>
                <th style="padding: 8px; text-align: left; border: 1px solid #ddd;">Indication</th>
                <th style="padding: 8px; text-align: left; border: 1px solid #ddd;">Start Date</th>
                <th style="padding: 8px; text-align: left; border: 1px solid #ddd;">Ongoing</th>
            </tr>
            {% for med in concomitant_meds %}
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;">{{ med.name }}</td>
                <td style="padding: 8px; border: 1px solid #ddd;">{{ med.indication | default('N/A') }}</td>
                <td style="padding: 8px; border: 1px solid #ddd;">{{ med.start_date | default('N/A') }}</td>
                <td style="padding: 8px; border: 1px solid #ddd;">{{ 'Yes' if med.ongoing else 'No' }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>
    {% endif %}
    
    <div class="section">
        <h2>Treatment and Outcome</h2>
        <div class="field"><span class="field-label">Treatment Given:</span> {{ event_details.treatment | default('None reported') }}</div>
        <div class="field"><span class="field-label">Outcome:</span> {{ outcome | default('Pending') }}</div>
        <div class="field"><span class="field-label">Action Taken with Study Drug:</span> {{ event_details.action_taken | default('None') }}</div>
    </div>
    
    <div class="section">
        <h2>Causality Assessment</h2>
        <div class="field"><span class="field-label">Investigator Assessment:</span> {{ event_details.investigator_causality | default('Pending') }}</div>
        <div class="field"><span class="field-label">Sponsor Assessment:</span> {{ event_details.sponsor_causality | default('Pending') }}</div>
        <div class="narrative">
            <strong>Causality Rationale:</strong><br>
            {{ event_details.causality_rationale | default('Assessment pending.') }}
        </div>
    </div>
    
    <div class="footer">
        <p><strong>Report ID:</strong> {{ _metadata.report_id }}</p>
        <p><strong>Generated:</strong> {{ _metadata.generated_at }} | <strong>Classification:</strong> {{ _metadata.classification }}</p>
        <p><em>This document contains confidential patient information and is intended for regulatory purposes only.</em></p>
    </div>
</body>
</html>'''

    def _template_inspection_prep(self) -> str:
        return '''<!DOCTYPE html>
<html>
<head>
    <title>Inspection Readiness - {{ site_id }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { background: #c82333; color: white; padding: 25px; border-radius: 8px; }
        .header h1 { margin: 0; }
        .alert-banner { background: #fff3cd; border: 2px solid #ffc107; padding: 15px; border-radius: 8px; margin: 20px 0; text-align: center; }
        .section { margin: 25px 0; }
        .section h2 { color: #1a365d; border-bottom: 2px solid #1a365d; padding-bottom: 10px; }
        .checklist { list-style: none; padding: 0; }
        .checklist li { padding: 12px; margin: 8px 0; background: #f8f9fa; border-radius: 6px; display: flex; align-items: center; }
        .check-icon { width: 24px; height: 24px; border-radius: 50%; margin-right: 12px; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; }
        .check-pass { background: #28a745; }
        .check-fail { background: #dc3545; }
        .check-pending { background: #ffc107; color: #333; }
        .risk-card { padding: 15px; border-radius: 8px; margin: 10px 0; }
        .risk-high { background: #f8d7da; border-left: 4px solid #dc3545; }
        .risk-medium { background: #fff3cd; border-left: 4px solid #ffc107; }
        .risk-low { background: #d4edda; border-left: 4px solid #28a745; }
        table { width: 100%; border-collapse: collapse; margin-top: 15px; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #f8f9fa; }
        .status-badge { padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: bold; }
        .status-complete { background: #d4edda; color: #155724; }
        .status-incomplete { background: #f8d7da; color: #721c24; }
        .status-pending { background: #fff3cd; color: #856404; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 Inspection Readiness Report</h1>
        <p style="margin: 10px 0 0 0;">
            <strong>Site:</strong> {{ site_id }} | 
            <strong>Inspection Date:</strong> {{ inspection_date | format_date }} |
            <strong>Inspector:</strong> {{ inspector_type }}
        </p>
    </div>
    
    <div class="alert-banner">
        <strong>⏰ Days Until Inspection: {{ (inspection_date - now()).days if inspection_date else 'TBD' }}</strong>
    </div>
    
    <div class="section">
        <h2>📋 Document Checklist</h2>
        <ul class="checklist">
            {% for doc in document_checklist | default([
                {'name': 'Informed Consent Forms', 'status': 'complete'},
                {'name': 'Source Documents', 'status': 'pending'},
                {'name': 'Delegation Log', 'status': 'complete'},
                {'name': 'Training Records', 'status': 'complete'},
                {'name': 'Protocol & Amendments', 'status': 'complete'},
                {'name': 'IRB/EC Approvals', 'status': 'complete'},
                {'name': 'Drug Accountability', 'status': 'pending'},
                {'name': 'Lab Certifications', 'status': 'complete'}
            ]) %}
            <li>
                <span class="check-icon {% if doc.status == 'complete' %}check-pass{% elif doc.status == 'pending' %}check-pending{% else %}check-fail{% endif %}">
                    {% if doc.status == 'complete' %}✓{% elif doc.status == 'pending' %}!{% else %}✗{% endif %}
                </span>
                {{ doc.name }}
                <span class="status-badge {% if doc.status == 'complete' %}status-complete{% elif doc.status == 'pending' %}status-pending{% else %}status-incomplete{% endif %}" style="margin-left: auto;">
                    {{ doc.status | title_case }}
                </span>
            </li>
            {% endfor %}
        </ul>
    </div>
    
    <div class="section">
        <h2>📊 Data Quality Summary</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Current</th>
                <th>Target</th>
                <th>Status</th>
            </tr>
            <tr>
                <td>Query Resolution Rate</td>
                <td>{{ focus_areas.query_rate | default(92) | format_percent }}</td>
                <td>95%</td>
                <td><span class="status-badge {% if focus_areas.query_rate | default(92) >= 95 %}status-complete{% else %}status-pending{% endif %}">{% if focus_areas.query_rate | default(92) >= 95 %}On Target{% else %}Action Needed{% endif %}</span></td>
            </tr>
            <tr>
                <td>SDV Completion</td>
                <td>{{ focus_areas.sdv_rate | default(88) | format_percent }}</td>
                <td>100%</td>
                <td><span class="status-badge {% if focus_areas.sdv_rate | default(88) >= 100 %}status-complete{% else %}status-pending{% endif %}">{% if focus_areas.sdv_rate | default(88) >= 100 %}Complete{% else %}In Progress{% endif %}</span></td>
            </tr>
            <tr>
                <td>Open Protocol Deviations</td>
                <td>{{ focus_areas.deviations | default(3) }}</td>
                <td>0</td>
                <td><span class="status-badge {% if focus_areas.deviations | default(3) == 0 %}status-complete{% else %}status-incomplete{% endif %}">{% if focus_areas.deviations | default(3) == 0 %}None{% else %}Review Required{% endif %}</span></td>
            </tr>
        </table>
    </div>
    
    {% if risk_areas %}
    <div class="section">
        <h2>⚠️ Risk Areas</h2>
        {% for risk in risk_areas %}
        <div class="risk-card risk-{{ risk.level | lower }}">
            <strong>{{ risk.area }}</strong>
            <p style="margin: 5px 0 0 0;">{{ risk.description }}</p>
            {% if risk.mitigation %}
            <p style="margin: 5px 0 0 0; color: #28a745;"><strong>Mitigation:</strong> {{ risk.mitigation }}</p>
            {% endif %}
        </div>
        {% endfor %}
    </div>
    {% endif %}
    
    <div class="section">
        <h2>✅ Pre-Inspection Actions</h2>
        <table>
            <tr>
                <th>Action</th>
                <th>Owner</th>
                <th>Due Date</th>
                <th>Status</th>
            </tr>
            {% for action in focus_areas.actions | default([
                {'action': 'Complete SDV for remaining CRFs', 'owner': 'CRA', 'due': 'D-7', 'status': 'pending'},
                {'action': 'Resolve open queries', 'owner': 'Site', 'due': 'D-5', 'status': 'pending'},
                {'action': 'Update delegation log', 'owner': 'Site', 'due': 'D-3', 'status': 'complete'},
                {'action': 'Site file review', 'owner': 'CRA', 'due': 'D-2', 'status': 'pending'}
            ]) %}
            <tr>
                <td>{{ action.action }}</td>
                <td>{{ action.owner }}</td>
                <td>{{ action.due }}</td>
                <td><span class="status-badge {% if action.status == 'complete' %}status-complete{% else %}status-pending{% endif %}">{{ action.status | title_case }}</span></td>
            </tr>
            {% endfor %}
        </table>
    </div>
    
    <div style="background: #e7f3ff; padding: 20px; border-radius: 8px; margin-top: 30px;">
        <h3 style="margin-top: 0; color: #1a365d;">📞 Key Contacts</h3>
        <p><strong>CRA:</strong> {{ focus_areas.cra_name | default('Contact CTM') }} | <strong>CTM:</strong> {{ focus_areas.ctm_name | default('Contact Study Lead') }}</p>
        <p><strong>Study Lead:</strong> {{ focus_areas.study_lead | default('TBD') }}</p>
    </div>
    
    <div style="text-align: center; color: #666; font-size: 12px; margin-top: 30px;">
        Report ID: {{ _metadata.report_id }} | Generated: {{ _metadata.generated_at }} | Requires PI Approval
    </div>
</body>
</html>'''

    def _template_site_newsletter(self) -> str:
        return '''<!DOCTYPE html>
<html>
<head>
    <title>Site Newsletter - {{ month }} {{ year }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 0; background: #f5f5f5; }
        .container { max-width: 650px; margin: 0 auto; background: white; }
        .header { background: linear-gradient(135deg, #1a365d, #2d5a87); color: white; padding: 30px; text-align: center; }
        .header h1 { margin: 0; font-size: 28px; }
        .header .date { font-size: 16px; margin-top: 10px; opacity: 0.9; }
        .content { padding: 30px; }
        .section { margin-bottom: 30px; }
        .section h2 { color: #1a365d; font-size: 20px; margin-bottom: 15px; padding-bottom: 10px; border-bottom: 2px solid #e9ecef; }
        .update-card { background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 15px; }
        .update-card h3 { margin: 0 0 10px 0; color: #1a365d; font-size: 16px; }
        .reminder-box { background: #fff3cd; padding: 20px; border-radius: 8px; border-left: 4px solid #ffc107; }
        .tip-box { background: #d4edda; padding: 20px; border-radius: 8px; border-left: 4px solid #28a745; }
        .recognition { background: linear-gradient(135deg, #ffd700, #ffed4a); padding: 20px; border-radius: 8px; text-align: center; }
        .recognition h3 { margin: 0 0 10px 0; }
        .contact-section { background: #1a365d; color: white; padding: 25px; text-align: center; }
        .footer { padding: 20px; text-align: center; font-size: 12px; color: #666; background: #f8f9fa; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📰 Site Newsletter</h1>
            <div class="date">{{ month }} {{ year }}</div>
        </div>
        
        <div class="content">
            <div class="section">
                <h2>📢 Study Updates</h2>
                {% for update in study_updates | default([]) %}
                <div class="update-card">
                    <h3>{{ update.title }}</h3>
                    <p style="margin: 0;">{{ update.content }}</p>
                </div>
                {% else %}
                <p>No major updates this month. Study progressing as planned.</p>
                {% endfor %}
            </div>
            
            {% if reminders %}
            <div class="section">
                <h2>⏰ Reminders</h2>
                <div class="reminder-box">
                    <ul style="margin: 0; padding-left: 20px;">
                        {% for reminder in reminders %}
                        <li style="margin-bottom: 8px;">{{ reminder }}</li>
                        {% endfor %}
                    </ul>
                </div>
            </div>
            {% endif %}
            
            {% if tips %}
            <div class="section">
                <h2>💡 Tips & Best Practices</h2>
                {% for tip in tips %}
                <div class="tip-box" style="margin-bottom: 15px;">
                    <strong>{{ tip.title }}</strong>
                    <p style="margin: 5px 0 0 0;">{{ tip.content }}</p>
                </div>
                {% endfor %}
            </div>
            {% endif %}
            
            {% if recognition %}
            <div class="section">
                <h2>🏆 Site Recognition</h2>
                {% for rec in recognition %}
                <div class="recognition" style="margin-bottom: 15px;">
                    <h3>⭐ {{ rec.site }}</h3>
                    <p style="margin: 0;">{{ rec.achievement }}</p>
                </div>
                {% endfor %}
            </div>
            {% endif %}
        </div>
        
        <div class="contact-section">
            <h3 style="margin: 0 0 15px 0;">📞 Need Help?</h3>
            <p style="margin: 0;">Contact your CRA or reach out to the study team</p>
            <p style="margin: 10px 0 0 0;">Email: study-support@example.com</p>
        </div>
        
        <div class="footer">
            <p>This newsletter is for site personnel only. Please do not forward externally.</p>
            <p>Report ID: {{ _metadata.report_id }}</p>
        </div>
    </div>
</body>
</html>'''

    def _template_issue_escalation(self) -> str:
        return '''<!DOCTYPE html>
<html>
<head>
    <title>Issue Escalation - {{ issue_id }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { background: #dc3545; color: white; padding: 25px; border-radius: 8px; }
        .header h1 { margin: 0; display: flex; align-items: center; }
        .escalation-level { background: white; color: #dc3545; padding: 5px 15px; border-radius: 20px; font-size: 14px; margin-left: 15px; }
        .summary-box { background: #f8f9fa; padding: 25px; border-radius: 8px; margin: 25px 0; }
        .field-row { display: flex; margin: 10px 0; }
        .field-label { width: 150px; font-weight: bold; color: #666; }
        .field-value { flex: 1; }
        .section { margin: 25px 0; }
        .section h2 { color: #1a365d; border-bottom: 2px solid #1a365d; padding-bottom: 10px; }
        .impact-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin: 15px 0; }
        .impact-card { background: #fff3cd; padding: 15px; border-radius: 8px; text-align: center; }
        .impact-value { font-size: 24px; font-weight: bold; color: #856404; }
        .impact-label { font-size: 12px; color: #666; }
        .action-table { width: 100%; border-collapse: collapse; margin: 15px 0; }
        .action-table th, .action-table td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        .action-table th { background: #f8f9fa; }
        .timeline { background: #e7f3ff; padding: 20px; border-radius: 8px; margin: 20px 0; }
        .approval-section { background: #f8f9fa; padding: 25px; border-radius: 8px; margin-top: 30px; }
        .signature-line { border-top: 1px solid #333; width: 250px; margin-top: 40px; padding-top: 5px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>
            🚨 Issue Escalation
            <span class="escalation-level">Level {{ escalation_level }}</span>
        </h1>
        <p style="margin: 10px 0 0 0;">Issue ID: {{ issue_id }} | Date: {{ now() | format_date }}</p>
    </div>
    
    <div class="summary-box">
        <h3 style="margin-top: 0; color: #1a365d;">Issue Summary</h3>
        <div class="field-row">
            <span class="field-label">Issue Type:</span>
            <span class="field-value">{{ issue_details.type | title_case }}</span>
        </div>
        <div class="field-row">
            <span class="field-label">Entity:</span>
            <span class="field-value">{{ issue_details.entity_id }}</span>
        </div>
        <div class="field-row">
            <span class="field-label">First Identified:</span>
            <span class="field-value">{{ issue_details.identified_date | format_date if issue_details.identified_date else 'N/A' }}</span>
        </div>
        <div class="field-row">
            <span class="field-label">Days Outstanding:</span>
            <span class="field-value">{{ issue_details.days_outstanding | default(0) }} days</span>
        </div>
        <div class="field-row">
            <span class="field-label">Description:</span>
            <span class="field-value">{{ issue_details.description }}</span>
        </div>
    </div>
    
    <div class="section">
        <h2>📊 Impact Assessment</h2>
        <div class="impact-grid">
            <div class="impact-card">
                <div class="impact-value">{{ issue_details.patients_affected | default(0) }}</div>
                <div class="impact-label">Patients Affected</div>
            </div>
            <div class="impact-card">
                <div class="impact-value">{{ issue_details.dqi_impact | default(0) | format_decimal(1) }}</div>
                <div class="impact-label">DQI Impact</div>
            </div>
            <div class="impact-card">
                <div class="impact-value">{{ issue_details.timeline_impact | default('N/A') }}</div>
                <div class="impact-label">Timeline Impact</div>
            </div>
        </div>
    </div>
    
    {% if root_cause %}
    <div class="section">
        <h2>🔍 Root Cause Analysis</h2>
        <p><strong>Primary Root Cause:</strong> {{ root_cause.primary }}</p>
        {% if root_cause.contributing_factors %}
        <p><strong>Contributing Factors:</strong></p>
        <ul>
            {% for factor in root_cause.contributing_factors %}
            <li>{{ factor }}</li>
            {% endfor %}
        </ul>
        {% endif %}
    </div>
    {% endif %}
    
    {% if proposed_actions %}
    <div class="section">
        <h2>✅ Proposed Actions</h2>
        <table class="action-table">
            <tr>
                <th>Action</th>
                <th>Owner</th>
                <th>Due Date</th>
                <th>Priority</th>
            </tr>
            {% for action in proposed_actions %}
            <tr>
                <td>{{ action.description }}</td>
                <td>{{ action.owner }}</td>
                <td>{{ action.due_date | format_date if action.due_date else 'TBD' }}</td>
                <td>{{ action.priority | priority_badge }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>
    {% endif %}
    
    {% if timeline %}
    <div class="timeline">
        <h3 style="margin-top: 0;">📅 Resolution Timeline</h3>
        <p><strong>Target Resolution:</strong> {{ timeline.target_date | format_date if timeline.target_date else 'TBD' }}</p>
        <p><strong>Current Status:</strong> {{ timeline.status | default('In Progress') }}</p>
        <p><strong>Next Milestone:</strong> {{ timeline.next_milestone | default('Pending action plan approval') }}</p>
    </div>
    {% endif %}
    
    <div class="approval-section">
        <h3 style="margin-top: 0;">📝 Approval Required</h3>
        <p>This escalation requires approval from the following:</p>
        <ul>
            <li>Study Lead</li>
            {% if escalation_level >= 3 %}
            <li>Medical Monitor</li>
            {% endif %}
            {% if escalation_level >= 4 %}
            <li>Sponsor Representative</li>
            {% endif %}
        </ul>
        
        <div style="display: flex; gap: 50px; margin-top: 30px;">
            <div>
                <div class="signature-line">Study Lead Signature</div>
                <p>Date: _____________</p>
            </div>
            {% if escalation_level >= 3 %}
            <div>
                <div class="signature-line">Medical Monitor Signature</div>
                <p>Date: _____________</p>
            </div>
            {% endif %}
        </div>
    </div>
    
    <div style="text-align: center; color: #666; font-size: 12px; margin-top: 30px;">
        Report ID: {{ _metadata.report_id }} | Generated: {{ _metadata.generated_at }} | Classification: {{ _metadata.classification }}
    </div>
</body>
</html>'''

    def _template_default(self) -> str:
        """Default template for unknown report types"""
        return '''<!DOCTYPE html>
<html>
<head>
    <title>Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { background: #1a365d; color: white; padding: 20px; border-radius: 8px; }
        .content { margin-top: 20px; }
        pre { background: #f8f9fa; padding: 15px; border-radius: 4px; overflow-x: auto; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Report</h1>
        <p>Generated: {{ now() | format_datetime }}</p>
    </div>
    <div class="content">
        <h2>Data</h2>
        <pre>{{ _metadata | tojson(indent=2) if _metadata else 'No metadata' }}</pre>
    </div>
</body>
</html>'''

    def save_report(
        self,
        report: GeneratedReport,
        filename: Optional[str] = None
    ) -> str:
        """Save generated report to file"""
        
        if filename is None:
            ext = report.format.value
            filename = f"{report.report_id}.{ext}"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report.content)
        
        report.file_path = filepath
        logger.info(f"Saved report to {filepath}")
        
        return filepath
    
    def get_stats(self) -> Dict:
        """Get template engine statistics"""
        return {
            **self.stats,
            'templates_registered': len(self.templates),
            'avg_generation_time_ms': (
                self.stats['total_generation_time_ms'] / self.stats['reports_generated']
                if self.stats['reports_generated'] > 0 else 0
            )
        }


# Factory function
_engine_instance: Optional[TemplateEngine] = None

def get_template_engine() -> TemplateEngine:
    """Get or create template engine instance"""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = TemplateEngine()
    return _engine_instance