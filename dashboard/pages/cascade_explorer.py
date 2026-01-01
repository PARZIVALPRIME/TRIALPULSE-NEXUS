"""
TRIALPULSE NEXUS 10X - Cascade Explorer Dashboard v1.0
Interactive dependency graph visualization with cascade impact analysis.

Features:
- Interactive dependency graph (Plotly network)
- Cascade impact analysis
- "Fix X → Unlocks Y" recommendations
- Critical path to DB Lock
- What-if scenario simulator
- Node drill-down with patient details
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import json
import networkx as nx

# ============================================================
# CONFIGURATION
# ============================================================

class IssueType(Enum):
    """Issue types for cascade analysis."""
    MISSING_VISITS = "missing_visits"
    MISSING_PAGES = "missing_pages"
    OPEN_QUERIES = "open_queries"
    SDV_INCOMPLETE = "sdv_incomplete"
    SIGNATURE_GAPS = "signature_gaps"
    BROKEN_SIGNATURES = "broken_signatures"
    SAE_DM_PENDING = "sae_dm_pending"
    SAE_SAFETY_PENDING = "sae_safety_pending"
    MEDDRA_UNCODED = "meddra_uncoded"
    WHODRUG_UNCODED = "whodrug_uncoded"
    LAB_ISSUES = "lab_issues"
    EDRR_ISSUES = "edrr_issues"
    INACTIVATED_FORMS = "inactivated_forms"
    HIGH_QUERY_VOLUME = "high_query_volume"
    DB_LOCK = "db_lock"


# Issue dependencies: source -> targets (fixing source unlocks targets)
ISSUE_DEPENDENCIES = {
    "missing_visits": ["missing_pages", "open_queries", "sdv_incomplete", "signature_gaps"],
    "missing_pages": ["open_queries", "sdv_incomplete", "signature_gaps"],
    "open_queries": ["signature_gaps", "db_lock"],
    "sdv_incomplete": ["signature_gaps", "db_lock"],
    "signature_gaps": ["db_lock"],
    "broken_signatures": ["signature_gaps", "db_lock"],
    "sae_dm_pending": ["sae_safety_pending", "db_lock"],
    "sae_safety_pending": ["db_lock"],
    "meddra_uncoded": ["db_lock"],
    "whodrug_uncoded": ["db_lock"],
    "lab_issues": ["db_lock"],
    "edrr_issues": ["db_lock"],
    "inactivated_forms": ["db_lock"],
    "high_query_volume": ["open_queries"],
}

# Issue display configuration
ISSUE_CONFIG = {
    "missing_visits": {"name": "Missing Visits", "color": "#e74c3c", "icon": "📅", "weight": 100},
    "missing_pages": {"name": "Missing Pages", "color": "#c0392b", "icon": "📄", "weight": 95},
    "open_queries": {"name": "Open Queries", "color": "#f39c12", "icon": "❓", "weight": 80},
    "sdv_incomplete": {"name": "SDV Incomplete", "color": "#e67e22", "icon": "🔍", "weight": 75},
    "signature_gaps": {"name": "Signature Gaps", "color": "#9b59b6", "icon": "✍️", "weight": 70},
    "broken_signatures": {"name": "Broken Signatures", "color": "#8e44ad", "icon": "💔", "weight": 65},
    "sae_dm_pending": {"name": "SAE-DM Pending", "color": "#e74c3c", "icon": "⚠️", "weight": 100},
    "sae_safety_pending": {"name": "SAE-Safety Pending", "color": "#c0392b", "icon": "🚨", "weight": 95},
    "meddra_uncoded": {"name": "MedDRA Uncoded", "color": "#3498db", "icon": "🏷️", "weight": 50},
    "whodrug_uncoded": {"name": "WHODrug Uncoded", "color": "#2980b9", "icon": "💊", "weight": 50},
    "lab_issues": {"name": "Lab Issues", "color": "#1abc9c", "icon": "🧪", "weight": 60},
    "edrr_issues": {"name": "EDRR Issues", "color": "#16a085", "icon": "🔗", "weight": 55},
    "inactivated_forms": {"name": "Inactivated Forms", "color": "#95a5a6", "icon": "📋", "weight": 40},
    "high_query_volume": {"name": "High Query Volume", "color": "#f1c40f", "icon": "📊", "weight": 45},
    "db_lock": {"name": "DB Lock Ready", "color": "#27ae60", "icon": "🔒", "weight": 0},
}

# Responsible parties
RESPONSIBLE_PARTIES = {
    "missing_visits": "Site Coordinator",
    "missing_pages": "CRA",
    "open_queries": "Data Manager",
    "sdv_incomplete": "CRA",
    "signature_gaps": "Site / PI",
    "broken_signatures": "Site / PI",
    "sae_dm_pending": "Safety Data Manager",
    "sae_safety_pending": "Safety Physician",
    "meddra_uncoded": "Medical Coder",
    "whodrug_uncoded": "Medical Coder",
    "lab_issues": "Data Manager",
    "edrr_issues": "Data Manager",
    "inactivated_forms": "Data Manager",
    "high_query_volume": "CRA",
    "db_lock": "Study Lead",
}


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class CascadeNode:
    """Represents a node in the cascade graph."""
    issue_type: str
    name: str
    patient_count: int
    issue_count: int
    unlock_score: float  # PageRank-like score
    color: str
    icon: str
    responsible: str
    position: Tuple[float, float] = (0, 0)
    
    @property
    def size(self) -> int:
        """Node size based on patient count."""
        return max(20, min(100, 20 + self.patient_count // 100))
    
    @property
    def impact_level(self) -> str:
        """Impact level based on unlock score."""
        if self.unlock_score >= 80:
            return "Critical"
        elif self.unlock_score >= 60:
            return "High"
        elif self.unlock_score >= 40:
            return "Medium"
        else:
            return "Low"


@dataclass
class CascadeEdge:
    """Represents an edge in the cascade graph."""
    source: str
    target: str
    weight: float
    patients_affected: int
    
    @property
    def label(self) -> str:
        return f"{self.patients_affected} patients"


@dataclass
class CascadeImpact:
    """Result of cascade impact analysis."""
    source_issue: str
    direct_unlocks: List[str]
    cascade_chain: List[str]
    patients_unblocked: int
    dqi_improvement: float
    days_saved: float
    effort_hours: float
    roi_score: float
    priority: str


@dataclass
class CriticalPath:
    """Critical path to DB Lock."""
    path: List[str]
    total_patients: int
    total_effort: float
    estimated_days: int
    bottleneck: str
    

@dataclass
class WhatIfScenario:
    """What-if scenario result."""
    scenario_name: str
    action: str
    before_state: Dict[str, int]
    after_state: Dict[str, int]
    patients_unlocked: int
    dqi_change: float
    days_acceleration: float
    probability_success: float
    recommendations: List[str]


# ============================================================
# DATA LOADER
# ============================================================

class CascadeDataLoader:
    """Loads and manages cascade analysis data."""
    
    def __init__(self):
        self.base_path = Path("data/processed")
        self._cache = {}
        self._cache_time = {}
        self._cache_ttl = 300  # 5 minutes
    
    def _load_cached(self, key: str, loader_func) -> pd.DataFrame:
        """Load data with caching."""
        now = datetime.now().timestamp()
        if key in self._cache and (now - self._cache_time.get(key, 0)) < self._cache_ttl:
            return self._cache[key]
        
        data = loader_func()
        self._cache[key] = data
        self._cache_time[key] = now
        return data
    
    def load_patient_issues(self) -> pd.DataFrame:
        """Load patient issues data."""
        def loader():
            path = self.base_path / "analytics" / "patient_issues.parquet"
            if path.exists():
                return pd.read_parquet(path)
            return pd.DataFrame()
        return self._load_cached("patient_issues", loader)
    
    def load_cascade_analysis(self) -> pd.DataFrame:
        """Load pre-computed cascade analysis."""
        def loader():
            path = self.base_path / "analytics" / "patient_cascade_analysis.parquet"
            if path.exists():
                return pd.read_parquet(path)
            return pd.DataFrame()
        return self._load_cached("cascade_analysis", loader)
    
    def load_upr(self) -> pd.DataFrame:
        """Load unified patient record."""
        def loader():
            path = self.base_path / "upr" / "unified_patient_record.parquet"
            if path.exists():
                return pd.read_parquet(path)
            return pd.DataFrame()
        return self._load_cached("upr", loader)
    
    def load_dblock_status(self) -> pd.DataFrame:
        """Load DB lock status."""
        def loader():
            path = self.base_path / "analytics" / "patient_dblock_status.parquet"
            if path.exists():
                return pd.read_parquet(path)
            return pd.DataFrame()
        return self._load_cached("dblock_status", loader)
    
    def get_issue_counts(self, study_id: str = None, site_id: str = None) -> Dict[str, int]:
        """Get issue counts by type."""
        df = self.load_patient_issues()
        if df.empty:
            return {issue: 0 for issue in ISSUE_CONFIG.keys()}
        
        # Apply filters
        if study_id and study_id != "All Studies":
            if 'study_id' in df.columns:
                df = df[df['study_id'] == study_id]
        if site_id and site_id != "All Sites":
            if 'site_id' in df.columns:
                df = df[df['site_id'] == site_id]
        
        counts = {}
        for issue_type in ISSUE_CONFIG.keys():
            if issue_type == "db_lock":
                # DB lock is the target, count ready patients
                dblock_df = self.load_dblock_status()
                if not dblock_df.empty:
                    ready_col = None
                    for col in ['db_lock_tier1_ready', 'dblock_tier1_ready']:
                        if col in dblock_df.columns:
                            ready_col = col
                            break
                    if ready_col:
                        counts[issue_type] = int((~dblock_df[ready_col].fillna(False)).sum())
                    else:
                        counts[issue_type] = len(dblock_df)
                else:
                    counts[issue_type] = 0
            else:
                # Check for issue_* column
                col_name = f"issue_{issue_type}"
                if col_name in df.columns:
                    counts[issue_type] = int(df[col_name].fillna(False).sum())
                else:
                    counts[issue_type] = 0
        
        return counts
    
    def get_cascade_nodes(self, study_id: str = None, site_id: str = None) -> List[CascadeNode]:
        """Build cascade nodes with metrics."""
        issue_counts = self.get_issue_counts(study_id, site_id)
        
        # Calculate unlock scores using PageRank-like algorithm
        unlock_scores = self._calculate_unlock_scores(issue_counts)
        
        # Build node positions (hierarchical layout)
        positions = self._calculate_node_positions()
        
        nodes = []
        for issue_type, config in ISSUE_CONFIG.items():
            count = issue_counts.get(issue_type, 0)
            nodes.append(CascadeNode(
                issue_type=issue_type,
                name=config["name"],
                patient_count=count,
                issue_count=count,
                unlock_score=unlock_scores.get(issue_type, 0),
                color=config["color"],
                icon=config["icon"],
                responsible=RESPONSIBLE_PARTIES.get(issue_type, "Unknown"),
                position=positions.get(issue_type, (0, 0))
            ))
        
        return nodes
    
    def get_cascade_edges(self, study_id: str = None, site_id: str = None) -> List[CascadeEdge]:
        """Build cascade edges with weights."""
        issue_counts = self.get_issue_counts(study_id, site_id)
        
        edges = []
        for source, targets in ISSUE_DEPENDENCIES.items():
            source_count = issue_counts.get(source, 0)
            for target in targets:
                # Weight based on source count and dependency strength
                target_count = issue_counts.get(target, 0)
                weight = min(source_count, target_count) if source_count > 0 else 0
                
                edges.append(CascadeEdge(
                    source=source,
                    target=target,
                    weight=weight,
                    patients_affected=source_count
                ))
        
        return edges
    
    def _calculate_unlock_scores(self, issue_counts: Dict[str, int]) -> Dict[str, float]:
        """Calculate unlock scores using modified PageRank."""
        # Build NetworkX graph
        G = nx.DiGraph()
        
        for issue_type in ISSUE_CONFIG.keys():
            G.add_node(issue_type)
        
        for source, targets in ISSUE_DEPENDENCIES.items():
            for target in targets:
                weight = ISSUE_CONFIG.get(source, {}).get("weight", 50)
                G.add_edge(source, target, weight=weight)
        
        # Calculate PageRank
        try:
            pagerank = nx.pagerank(G, weight='weight')
        except:
            pagerank = {node: 1.0 / len(G.nodes()) for node in G.nodes()}
        
        # Normalize to 0-100 scale
        max_pr = max(pagerank.values()) if pagerank else 1
        scores = {k: (v / max_pr) * 100 for k, v in pagerank.items()}
        
        # Adjust by issue count
        for issue_type, count in issue_counts.items():
            if count > 0 and issue_type in scores:
                scores[issue_type] = min(100, scores[issue_type] * (1 + np.log10(count + 1) / 5))
        
        return scores
    
    def _calculate_node_positions(self) -> Dict[str, Tuple[float, float]]:
        """Calculate hierarchical node positions."""
        # Layer-based layout
        layers = {
            0: ["missing_visits", "high_query_volume"],  # Root causes
            1: ["missing_pages", "sae_dm_pending"],
            2: ["open_queries", "sdv_incomplete", "sae_safety_pending"],
            3: ["signature_gaps", "broken_signatures", "meddra_uncoded", "whodrug_uncoded"],
            4: ["lab_issues", "edrr_issues", "inactivated_forms"],
            5: ["db_lock"],  # Target
        }
        
        positions = {}
        for layer, nodes in layers.items():
            y = 1 - (layer / 5)  # Top to bottom
            for i, node in enumerate(nodes):
                x = (i + 1) / (len(nodes) + 1)  # Spread horizontally
                positions[node] = (x, y)
        
        return positions
    
    def get_cascade_impact(self, issue_type: str, study_id: str = None) -> CascadeImpact:
        """Calculate cascade impact of fixing an issue type."""
        issue_counts = self.get_issue_counts(study_id)
        
        # Direct unlocks
        direct_unlocks = ISSUE_DEPENDENCIES.get(issue_type, [])
        
        # Build full cascade chain (BFS)
        cascade_chain = []
        visited = set()
        queue = list(direct_unlocks)
        while queue:
            current = queue.pop(0)
            if current not in visited:
                visited.add(current)
                cascade_chain.append(current)
                queue.extend(ISSUE_DEPENDENCIES.get(current, []))
        
        # Calculate metrics
        source_count = issue_counts.get(issue_type, 0)
        
        # Estimate patients unblocked (cascading effect)
        patients_unblocked = source_count
        for unlocked in cascade_chain:
            # Partial unblock effect
            patients_unblocked += int(issue_counts.get(unlocked, 0) * 0.3)
        
        # DQI improvement (weighted by issue importance)
        weight = ISSUE_CONFIG.get(issue_type, {}).get("weight", 50)
        dqi_improvement = (source_count / 1000) * (weight / 100) * 5  # Max ~5 points per 1000 patients
        
        # Effort estimation
        effort_hours = source_count * 0.1  # 6 minutes per issue
        
        # Days saved (based on cascade depth)
        days_saved = len(cascade_chain) * 2 + source_count / 500
        
        # ROI score
        roi_score = (patients_unblocked * 10 + dqi_improvement * 100) / max(effort_hours, 1)
        
        # Priority
        if roi_score >= 50 or "sae" in issue_type:
            priority = "Critical"
        elif roi_score >= 30:
            priority = "High"
        elif roi_score >= 15:
            priority = "Medium"
        else:
            priority = "Low"
        
        return CascadeImpact(
            source_issue=issue_type,
            direct_unlocks=direct_unlocks,
            cascade_chain=cascade_chain,
            patients_unblocked=patients_unblocked,
            dqi_improvement=round(dqi_improvement, 2),
            days_saved=round(days_saved, 1),
            effort_hours=round(effort_hours, 1),
            roi_score=round(roi_score, 1),
            priority=priority
        )
    
    def get_critical_path(self, study_id: str = None) -> CriticalPath:
        """Find critical path to DB Lock."""
        issue_counts = self.get_issue_counts(study_id)
        
        # Build graph
        G = nx.DiGraph()
        for issue_type in ISSUE_CONFIG.keys():
            count = issue_counts.get(issue_type, 0)
            G.add_node(issue_type, count=count)
        
        for source, targets in ISSUE_DEPENDENCIES.items():
            for target in targets:
                weight = issue_counts.get(source, 0) + 1
                G.add_edge(source, target, weight=weight)
        
        # Find path with most patients affected
        paths_to_dblock = []
        for node in G.nodes():
            if node != "db_lock":
                try:
                    paths = list(nx.all_simple_paths(G, node, "db_lock"))
                    for path in paths:
                        total = sum(issue_counts.get(n, 0) for n in path[:-1])
                        paths_to_dblock.append((path, total))
                except nx.NetworkXNoPath:
                    pass
        
        if not paths_to_dblock:
            return CriticalPath(
                path=["db_lock"],
                total_patients=0,
                total_effort=0,
                estimated_days=0,
                bottleneck="None"
            )
        
        # Sort by patient count (descending)
        paths_to_dblock.sort(key=lambda x: x[1], reverse=True)
        critical_path, total_patients = paths_to_dblock[0]
        
        # Find bottleneck (node with most issues)
        bottleneck = max(critical_path[:-1], key=lambda n: issue_counts.get(n, 0))
        
        # Estimate effort and days
        total_effort = total_patients * 0.1
        estimated_days = int(total_effort / 8) + 1  # 8 hours per day
        
        return CriticalPath(
            path=critical_path,
            total_patients=total_patients,
            total_effort=round(total_effort, 1),
            estimated_days=estimated_days,
            bottleneck=bottleneck
        )
    
    def run_what_if_scenario(
        self,
        scenario_type: str,
        issue_type: str = None,
        reduction_percent: int = 50,
        study_id: str = None
    ) -> WhatIfScenario:
        """Run what-if scenario simulation."""
        issue_counts = self.get_issue_counts(study_id)
        before_state = dict(issue_counts)
        after_state = dict(issue_counts)
        
        if scenario_type == "fix_issue":
            # Reduce specific issue by percentage
            if issue_type and issue_type in after_state:
                original = after_state[issue_type]
                reduction = int(original * reduction_percent / 100)
                after_state[issue_type] = original - reduction
                
                # Cascade effect
                for unlocked in ISSUE_DEPENDENCIES.get(issue_type, []):
                    if unlocked in after_state:
                        cascade_reduction = int(reduction * 0.3)
                        after_state[unlocked] = max(0, after_state[unlocked] - cascade_reduction)
                
                patients_unlocked = reduction
                dqi_change = (reduction / 1000) * 2
                days_acceleration = reduction / 200
                probability = 0.85
                
                recommendations = [
                    f"Focus on {ISSUE_CONFIG[issue_type]['name']} first",
                    f"Assign {RESPONSIBLE_PARTIES[issue_type]} to lead resolution",
                    f"Set target: {reduction_percent}% reduction in 2 weeks"
                ]
        
        elif scenario_type == "add_resource":
            # Adding resource increases resolution capacity
            patients_unlocked = sum(issue_counts.values()) // 10
            dqi_change = 3.0
            days_acceleration = 7
            probability = 0.75
            
            recommendations = [
                "Add 1 additional CRA for site monitoring",
                "Prioritize high-impact sites",
                "Review weekly progress"
            ]
        
        elif scenario_type == "close_site":
            # Simulate site closure impact
            patients_unlocked = 0  # Actually negative impact
            dqi_change = -1.0
            days_acceleration = -14
            probability = 0.90
            
            recommendations = [
                "Avoid site closure if possible",
                "Transfer patients to nearby sites",
                "Document all ongoing issues"
            ]
        
        else:
            patients_unlocked = 0
            dqi_change = 0
            days_acceleration = 0
            probability = 0.5
            recommendations = ["Select a valid scenario"]
        
        return WhatIfScenario(
            scenario_name=scenario_type.replace("_", " ").title(),
            action=f"{scenario_type} - {issue_type or 'general'}",
            before_state=before_state,
            after_state=after_state,
            patients_unlocked=patients_unlocked,
            dqi_change=round(dqi_change, 2),
            days_acceleration=round(days_acceleration, 1),
            probability_success=probability,
            recommendations=recommendations
        )
    
    def get_top_opportunities(self, n: int = 10, study_id: str = None) -> List[CascadeImpact]:
        """Get top cascade opportunities ranked by ROI."""
        opportunities = []
        
        for issue_type in ISSUE_CONFIG.keys():
            if issue_type != "db_lock":
                impact = self.get_cascade_impact(issue_type, study_id)
                if impact.patients_unblocked > 0:
                    opportunities.append(impact)
        
        # Sort by ROI score
        opportunities.sort(key=lambda x: x.roi_score, reverse=True)
        
        return opportunities[:n]
    
    def get_studies_list(self) -> List[str]:
        """Get list of studies."""
        df = self.load_upr()
        if df.empty:
            return ["All Studies"]
        if 'study_id' in df.columns:
            studies = ["All Studies"] + sorted(df['study_id'].dropna().unique().tolist())
            return studies
        return ["All Studies"]
    
    def get_sites_list(self, study_id: str = None) -> List[str]:
        """Get list of sites."""
        df = self.load_upr()
        if df.empty:
            return ["All Sites"]
        
        if study_id and study_id != "All Studies" and 'study_id' in df.columns:
            df = df[df['study_id'] == study_id]
        
        if 'site_id' in df.columns:
            sites = ["All Sites"] + sorted(df['site_id'].dropna().unique().tolist()[:50])
            return sites
        return ["All Sites"]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_impact_color(level: str) -> str:
    """Get color for impact level."""
    colors = {
        "Critical": "#e74c3c",
        "High": "#f39c12",
        "Medium": "#3498db",
        "Low": "#27ae60"
    }
    return colors.get(level, "#95a5a6")


def get_priority_color(priority: str) -> str:
    """Get color for priority."""
    colors = {
        "Critical": "#e74c3c",
        "High": "#f39c12",
        "Medium": "#3498db",
        "Low": "#27ae60"
    }
    return colors.get(priority, "#95a5a6")


def format_number(value: float) -> str:
    """Format number for display."""
    if value >= 1000000:
        return f"{value/1000000:.1f}M"
    elif value >= 1000:
        return f"{value/1000:.1f}K"
    else:
        return f"{value:,.0f}"


# ============================================================
# VISUALIZATION COMPONENTS
# ============================================================

def create_cascade_graph(
    nodes: List[CascadeNode],
    edges: List[CascadeEdge],
    selected_node: str = None,
    highlight_path: List[str] = None
) -> go.Figure:
    """Create interactive cascade dependency graph."""
    
    # Create node traces
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    node_sizes = []
    node_symbols = []
    
    for node in nodes:
        x, y = node.position
        node_x.append(x)
        node_y.append(y)
        
        # Hover text
        text = f"""
        <b>{node.icon} {node.name}</b><br>
        Patients: {format_number(node.patient_count)}<br>
        Unlock Score: {node.unlock_score:.1f}<br>
        Impact: {node.impact_level}<br>
        Responsible: {node.responsible}
        """
        node_text.append(text)
        
        # Color
        if selected_node and node.issue_type == selected_node:
            node_colors.append("#FFD700")  # Gold for selected
        elif highlight_path and node.issue_type in highlight_path:
            node_colors.append("#00CED1")  # Cyan for path
        else:
            node_colors.append(node.color)
        
        # Size based on patient count
        node_sizes.append(node.size)
        
        # Symbol
        if node.issue_type == "db_lock":
            node_symbols.append("diamond")
        else:
            node_symbols.append("circle")
    
    # Create edge traces
    edge_x = []
    edge_y = []
    edge_colors = []
    
    for edge in edges:
        # Find source and target positions
        source_node = next((n for n in nodes if n.issue_type == edge.source), None)
        target_node = next((n for n in nodes if n.issue_type == edge.target), None)
        
        if source_node and target_node:
            x0, y0 = source_node.position
            x1, y1 = target_node.position
            
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # Highlight edges in path
            if highlight_path and edge.source in highlight_path and edge.target in highlight_path:
                edge_colors.append("#00CED1")
            else:
                edge_colors.append("rgba(150, 150, 150, 0.5)")
    
    # Create figure
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x,
        y=edge_y,
        mode='lines',
        line=dict(width=1, color='rgba(150, 150, 150, 0.5)'),
        hoverinfo='none',
        showlegend=False
    ))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=2, color='white'),
            symbol=node_symbols
        ),
        text=[n.icon for n in nodes],
        textposition="middle center",
        textfont=dict(size=12),
        hovertext=node_text,
        hoverinfo='text',
        showlegend=False,
        customdata=[n.issue_type for n in nodes]
    ))
    
    # Add node labels below
    fig.add_trace(go.Scatter(
        x=node_x,
        y=[y - 0.06 for y in node_y],
        mode='text',
        text=[n.name for n in nodes],
        textposition="bottom center",
        textfont=dict(size=9, color='#2c3e50'),
        hoverinfo='none',
        showlegend=False
    ))
    
    # Layout
    fig.update_layout(
        title=dict(
            text="<b>Cascade Dependency Graph</b><br><sup>Fix upstream issues to unlock downstream → DB Lock</sup>",
            x=0.5,
            xanchor='center'
        ),
        showlegend=False,
        hovermode='closest',
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-0.1, 1.1]
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-0.15, 1.15]
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=500,
        margin=dict(l=20, r=20, t=60, b=20),
        annotations=[
            dict(
                x=0.5,
                y=-0.1,
                text="🔒 DB Lock Ready (Target)",
                showarrow=False,
                font=dict(size=12, color='#27ae60')
            ),
            dict(
                x=0.5,
                y=1.1,
                text="📅 Root Causes (Start Here)",
                showarrow=False,
                font=dict(size=12, color='#e74c3c')
            )
        ]
    )
    
    return fig


def create_impact_chart(opportunities: List[CascadeImpact]) -> go.Figure:
    """Create cascade impact comparison chart."""
    
    issues = [ISSUE_CONFIG.get(o.source_issue, {}).get("name", o.source_issue) for o in opportunities]
    roi_scores = [o.roi_score for o in opportunities]
    patients = [o.patients_unblocked for o in opportunities]
    colors = [ISSUE_CONFIG.get(o.source_issue, {}).get("color", "#95a5a6") for o in opportunities]
    
    fig = go.Figure()
    
    # ROI bars
    fig.add_trace(go.Bar(
        y=issues,
        x=roi_scores,
        orientation='h',
        marker_color=colors,
        text=[f"ROI: {r:.1f}" for r in roi_scores],
        textposition='outside',
        hovertemplate="<b>%{y}</b><br>ROI Score: %{x:.1f}<br>Patients: %{customdata}<extra></extra>",
        customdata=patients
    ))
    
    fig.update_layout(
        title="<b>Cascade Impact by Issue Type</b><br><sup>Higher ROI = Better fix-to-impact ratio</sup>",
        xaxis_title="ROI Score",
        yaxis_title="",
        height=400,
        margin=dict(l=150, r=50, t=60, b=40),
        xaxis=dict(range=[0, max(roi_scores) * 1.2] if roi_scores else [0, 100]),
        yaxis=dict(autorange="reversed")
    )
    
    return fig


def create_critical_path_chart(path: CriticalPath, issue_counts: Dict[str, int]) -> go.Figure:
    """Create critical path visualization."""
    
    if not path.path or len(path.path) <= 1:
        fig = go.Figure()
        fig.add_annotation(
            text="No critical path identified",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False
        )
        return fig
    
    # Build path data
    labels = [ISSUE_CONFIG.get(p, {}).get("name", p) for p in path.path]
    counts = [issue_counts.get(p, 0) for p in path.path]
    colors = [ISSUE_CONFIG.get(p, {}).get("color", "#95a5a6") for p in path.path]
    icons = [ISSUE_CONFIG.get(p, {}).get("icon", "●") for p in path.path]
    
    fig = go.Figure()
    
    # Create funnel-like chart
    fig.add_trace(go.Funnel(
        y=labels,
        x=counts,
        textposition="inside",
        textinfo="value+percent initial",
        marker=dict(color=colors),
        connector=dict(line=dict(color="royalblue", dash="dot", width=2))
    ))
    
    fig.update_layout(
        title=f"<b>Critical Path to DB Lock</b><br><sup>Bottleneck: {ISSUE_CONFIG.get(path.bottleneck, {}).get('name', path.bottleneck)}</sup>",
        height=350,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig


def create_whatif_comparison(scenario: WhatIfScenario) -> go.Figure:
    """Create before/after comparison chart."""
    
    # Get top 8 issues by before count
    sorted_issues = sorted(
        scenario.before_state.items(),
        key=lambda x: x[1],
        reverse=True
    )[:8]
    
    issues = [ISSUE_CONFIG.get(i[0], {}).get("name", i[0]) for i in sorted_issues]
    before = [scenario.before_state.get(i[0], 0) for i in sorted_issues]
    after = [scenario.after_state.get(i[0], 0) for i in sorted_issues]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Before',
        y=issues,
        x=before,
        orientation='h',
        marker_color='#e74c3c',
        opacity=0.7
    ))
    
    fig.add_trace(go.Bar(
        name='After',
        y=issues,
        x=after,
        orientation='h',
        marker_color='#27ae60',
        opacity=0.7
    ))
    
    fig.update_layout(
        title=f"<b>What-If: {scenario.scenario_name}</b>",
        barmode='group',
        xaxis_title="Patient Count",
        yaxis_title="",
        height=350,
        margin=dict(l=150, r=20, t=60, b=40),
        yaxis=dict(autorange="reversed"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


# ============================================================
# RENDER FUNCTIONS
# ============================================================

def render_page(user=None):
    """Main render function for Cascade Explorer."""
    
    # Initialize data loader
    loader = CascadeDataLoader()
    
    # Page header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h1 style="color: white; margin: 0;">🌊 Cascade Explorer</h1>
        <p style="color: rgba(255,255,255,0.8); margin: 5px 0 0 0;">
            Interactive dependency graph • Impact analysis • What-if simulation
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Filters
    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
    
    with col1:
        studies = loader.get_studies_list()
        selected_study = st.selectbox("📚 Study", studies, key="cascade_study")
    
    with col2:
        sites = loader.get_sites_list(selected_study if selected_study != "All Studies" else None)
        selected_site = st.selectbox("🏥 Site", sites, key="cascade_site")
    
    with col3:
        view_mode = st.selectbox(
            "🎯 View Mode",
            ["Dependency Graph", "Impact Analysis", "Critical Path", "What-If Simulator"],
            key="cascade_view"
        )
    
    with col4:
        if st.button("🔄 Refresh", key="cascade_refresh"):
            loader._cache.clear()
            st.rerun()
    
    st.divider()
    
    # Get data
    study_filter = selected_study if selected_study != "All Studies" else None
    site_filter = selected_site if selected_site != "All Sites" else None
    
    nodes = loader.get_cascade_nodes(study_filter, site_filter)
    edges = loader.get_cascade_edges(study_filter, site_filter)
    issue_counts = loader.get_issue_counts(study_filter, site_filter)
    
    # Render based on view mode
    if view_mode == "Dependency Graph":
        render_dependency_graph_view(loader, nodes, edges, issue_counts, study_filter)
    elif view_mode == "Impact Analysis":
        render_impact_analysis_view(loader, study_filter)
    elif view_mode == "Critical Path":
        render_critical_path_view(loader, issue_counts, study_filter)
    elif view_mode == "What-If Simulator":
        render_whatif_view(loader, study_filter)


def render_dependency_graph_view(
    loader: CascadeDataLoader,
    nodes: List[CascadeNode],
    edges: List[CascadeEdge],
    issue_counts: Dict[str, int],
    study_id: str = None
):
    """Render dependency graph view."""
    
    # Summary metrics
    total_issues = sum(issue_counts.get(k, 0) for k in issue_counts if k != "db_lock")
    critical_path = loader.get_critical_path(study_id)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Issues", format_number(total_issues))
    with col2:
        st.metric("Issue Types", len([n for n in nodes if n.patient_count > 0]))
    with col3:
        st.metric("Dependencies", len(edges))
    with col4:
        st.metric("Critical Path", f"{len(critical_path.path)} steps")
    with col5:
        st.metric("Bottleneck", ISSUE_CONFIG.get(critical_path.bottleneck, {}).get("icon", "❓"))
    
    st.divider()
    
    # Main layout
    col_graph, col_detail = st.columns([3, 2])
    
    with col_graph:
        # Node selection for highlighting
        selected_node = st.selectbox(
            "🔍 Select node to highlight cascade:",
            ["None"] + [f"{ISSUE_CONFIG[n.issue_type]['icon']} {n.name}" for n in nodes if n.patient_count > 0],
            key="selected_node"
        )
        
        # Extract issue type from selection
        highlight_node = None
        highlight_path = None
        if selected_node != "None":
            for node in nodes:
                if node.name in selected_node:
                    highlight_node = node.issue_type
                    # Get cascade chain
                    impact = loader.get_cascade_impact(highlight_node, study_id)
                    highlight_path = [highlight_node] + impact.cascade_chain
                    break
        
        # Create and display graph
        fig = create_cascade_graph(nodes, edges, highlight_node, highlight_path)
        st.plotly_chart(fig, use_container_width=True)
    
    with col_detail:
        st.markdown("### 📊 Issue Summary")
        
        # Sort by patient count
        sorted_nodes = sorted(
            [n for n in nodes if n.patient_count > 0],
            key=lambda x: x.patient_count,
            reverse=True
        )
        
        for node in sorted_nodes[:8]:
            with st.container():
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {node.color}22, white);
                            border-left: 4px solid {node.color};
                            padding: 10px; margin: 5px 0; border-radius: 5px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: bold;">{node.icon} {node.name}</span>
                        <span style="background: {node.color}; color: white; padding: 2px 8px; 
                                     border-radius: 10px; font-size: 0.9em;">
                            {format_number(node.patient_count)}
                        </span>
                    </div>
                    <div style="font-size: 0.85em; color: #666; margin-top: 5px;">
                        Unlock Score: {node.unlock_score:.1f} • {node.responsible}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Selected node detail
        if highlight_node:
            st.markdown("### 🎯 Selected Node Impact")
            impact = loader.get_cascade_impact(highlight_node, study_id)
            
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 15px; border-radius: 10px;">
                <h4 style="margin: 0;">Fix {ISSUE_CONFIG[highlight_node]['name']}</h4>
                <hr style="margin: 10px 0;">
                <p><b>Direct Unlocks:</b> {', '.join([ISSUE_CONFIG.get(u, {}).get('name', u) for u in impact.direct_unlocks]) or 'None'}</p>
                <p><b>Cascade Chain:</b> {len(impact.cascade_chain)} steps</p>
                <p><b>Patients Unblocked:</b> {format_number(impact.patients_unblocked)}</p>
                <p><b>DQI Improvement:</b> +{impact.dqi_improvement:.1f} points</p>
                <p><b>Days Saved:</b> {impact.days_saved:.0f}</p>
                <p><b>Effort:</b> {impact.effort_hours:.0f} hours</p>
                <p><b>ROI Score:</b> <span style="color: {get_priority_color(impact.priority)}; font-weight: bold;">{impact.roi_score:.1f}</span></p>
            </div>
            """, unsafe_allow_html=True)


def render_impact_analysis_view(loader: CascadeDataLoader, study_id: str = None):
    """Render impact analysis view."""
    
    st.markdown("### 📈 Cascade Impact Analysis")
    st.markdown("*Fix these issues first for maximum downstream impact*")
    
    # Get top opportunities
    opportunities = loader.get_top_opportunities(10, study_id)
    
    if not opportunities:
        st.warning("No cascade opportunities found. Data may be missing.")
        return
    
    # Summary cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_unblocked = sum(o.patients_unblocked for o in opportunities)
        st.metric("Total Patients Unblockable", format_number(total_unblocked))
    
    with col2:
        total_dqi = sum(o.dqi_improvement for o in opportunities)
        st.metric("Potential DQI Improvement", f"+{total_dqi:.1f}")
    
    with col3:
        critical_count = len([o for o in opportunities if o.priority in ["Critical", "High"]])
        st.metric("High Priority Actions", critical_count)
    
    st.divider()
    
    # Impact chart
    col_chart, col_table = st.columns([2, 1])
    
    with col_chart:
        fig = create_impact_chart(opportunities)
        st.plotly_chart(fig, use_container_width=True)
    
    with col_table:
        st.markdown("### 🎯 Top Actions")
        for i, opp in enumerate(opportunities[:5], 1):
            priority_color = get_priority_color(opp.priority)
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {priority_color}11, white);
                        border-left: 4px solid {priority_color};
                        padding: 10px; margin: 8px 0; border-radius: 5px;">
                <div style="font-weight: bold;">
                    #{i} {ISSUE_CONFIG.get(opp.source_issue, {}).get('icon', '')} {ISSUE_CONFIG.get(opp.source_issue, {}).get('name', opp.source_issue)}
                </div>
                <div style="font-size: 0.85em; color: #666;">
                    🎯 {opp.patients_unblocked} patients • ⏱️ {opp.effort_hours:.0f}h • ROI: {opp.roi_score:.1f}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Detailed table
    with st.expander("📋 Full Impact Analysis Table"):
        table_data = []
        for opp in opportunities:
            table_data.append({
                "Issue": ISSUE_CONFIG.get(opp.source_issue, {}).get("name", opp.source_issue),
                "Priority": opp.priority,
                "Patients Unblocked": opp.patients_unblocked,
                "DQI Improvement": f"+{opp.dqi_improvement:.1f}",
                "Days Saved": opp.days_saved,
                "Effort (hrs)": opp.effort_hours,
                "ROI Score": opp.roi_score,
                "Cascade Chain": " → ".join([ISSUE_CONFIG.get(c, {}).get("name", c) for c in opp.cascade_chain[:3]]) + ("..." if len(opp.cascade_chain) > 3 else "")
            })
        
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)


def render_critical_path_view(
    loader: CascadeDataLoader,
    issue_counts: Dict[str, int],
    study_id: str = None
):
    """Render critical path view."""
    
    st.markdown("### 🛤️ Critical Path to DB Lock")
    st.markdown("*This is the sequence of issues blocking the most patients from DB Lock*")
    
    path = loader.get_critical_path(study_id)
    
    if not path.path or len(path.path) <= 1:
        st.info("No significant critical path identified. Most patients may already be ready for DB Lock.")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Path Length", f"{len(path.path)} steps")
    with col2:
        st.metric("Total Patients", format_number(path.total_patients))
    with col3:
        st.metric("Estimated Effort", f"{path.total_effort:.0f} hours")
    with col4:
        st.metric("Est. Days to Clear", f"{path.estimated_days}")
    
    st.divider()
    
    # Path visualization
    col_chart, col_detail = st.columns([2, 1])
    
    with col_chart:
        fig = create_critical_path_chart(path, issue_counts)
        st.plotly_chart(fig, use_container_width=True)
    
    with col_detail:
        st.markdown("### 📍 Path Steps")
        for i, step in enumerate(path.path):
            config = ISSUE_CONFIG.get(step, {})
            is_bottleneck = step == path.bottleneck
            
            step_style = "border: 2px solid #e74c3c;" if is_bottleneck else ""
            bottleneck_badge = '<span style="background: #e74c3c; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.8em;">BOTTLENECK</span>' if is_bottleneck else ""
            
            st.markdown(f"""
            <div style="background: {config.get('color', '#95a5a6')}11;
                        padding: 12px; margin: 8px 0; border-radius: 8px; {step_style}">
                <div style="font-weight: bold;">
                    Step {i+1}: {config.get('icon', '●')} {config.get('name', step)}
                    {bottleneck_badge}
                </div>
                <div style="font-size: 0.85em; color: #666; margin-top: 5px;">
                    Patients: {format_number(issue_counts.get(step, 0))} • {RESPONSIBLE_PARTIES.get(step, 'Unknown')}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Arrow between steps
            if i < len(path.path) - 1:
                st.markdown("<div style='text-align: center; color: #3498db; font-size: 1.5em;'>↓</div>", unsafe_allow_html=True)
    
    # Recommendations
    st.divider()
    st.markdown("### 💡 Recommendations")
    
    bottleneck_config = ISSUE_CONFIG.get(path.bottleneck, {})
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #e74c3c22, white);
                border-left: 4px solid #e74c3c;
                padding: 15px; border-radius: 5px;">
        <h4 style="margin: 0 0 10px 0;">🎯 Focus on Bottleneck: {bottleneck_config.get('name', path.bottleneck)}</h4>
        <ul style="margin: 0; padding-left: 20px;">
            <li>Assign additional {RESPONSIBLE_PARTIES.get(path.bottleneck, 'resources')} to clear backlog</li>
            <li>Set daily targets: {max(1, issue_counts.get(path.bottleneck, 0) // path.estimated_days)} issues/day</li>
            <li>Monitor progress with weekly cascade reviews</li>
            <li>Escalate blockers immediately to Study Lead</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


def render_whatif_view(loader: CascadeDataLoader, study_id: str = None):
    """Render what-if simulator view."""
    
    st.markdown("### 🔮 What-If Scenario Simulator")
    st.markdown("*Explore the impact of different interventions*")
    
    # Scenario selection
    col1, col2, col3 = st.columns(3)
    
    with col1:
        scenario_type = st.selectbox(
            "Scenario Type",
            ["fix_issue", "add_resource", "close_site"],
            format_func=lambda x: {
                "fix_issue": "🔧 Fix Specific Issue",
                "add_resource": "👥 Add Resource",
                "close_site": "🏥 Close Site"
            }.get(x, x)
        )
    
    with col2:
        if scenario_type == "fix_issue":
            issue_options = [
                k for k, v in loader.get_issue_counts(study_id).items()
                if v > 0 and k != "db_lock"
            ]
            selected_issue = st.selectbox(
                "Issue to Fix",
                issue_options,
                format_func=lambda x: f"{ISSUE_CONFIG.get(x, {}).get('icon', '')} {ISSUE_CONFIG.get(x, {}).get('name', x)}"
            ) if issue_options else None
        else:
            selected_issue = None
    
    with col3:
        if scenario_type == "fix_issue":
            reduction_pct = st.slider("Reduction %", 10, 100, 50, 10)
        else:
            reduction_pct = 50
    
    # Run simulation
    if st.button("🚀 Run Simulation", type="primary"):
        with st.spinner("Running simulation..."):
            scenario = loader.run_what_if_scenario(
                scenario_type,
                selected_issue,
                reduction_pct,
                study_id
            )
        
        # Display results
        st.divider()
        
        # Impact summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            color = "#27ae60" if scenario.patients_unlocked > 0 else "#e74c3c"
            st.markdown(f"""
            <div style="text-align: center; padding: 15px; background: {color}11; border-radius: 10px;">
                <div style="font-size: 2em; font-weight: bold; color: {color};">
                    {'+' if scenario.patients_unlocked > 0 else ''}{format_number(scenario.patients_unlocked)}
                </div>
                <div style="color: #666;">Patients Unlocked</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            color = "#27ae60" if scenario.dqi_change > 0 else "#e74c3c"
            st.markdown(f"""
            <div style="text-align: center; padding: 15px; background: {color}11; border-radius: 10px;">
                <div style="font-size: 2em; font-weight: bold; color: {color};">
                    {'+' if scenario.dqi_change > 0 else ''}{scenario.dqi_change:.1f}
                </div>
                <div style="color: #666;">DQI Change</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            color = "#27ae60" if scenario.days_acceleration > 0 else "#e74c3c"
            st.markdown(f"""
            <div style="text-align: center; padding: 15px; background: {color}11; border-radius: 10px;">
                <div style="font-size: 2em; font-weight: bold; color: {color};">
                    {'+' if scenario.days_acceleration > 0 else ''}{scenario.days_acceleration:.0f}
                </div>
                <div style="color: #666;">Days Acceleration</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            prob_color = "#27ae60" if scenario.probability_success >= 0.7 else "#f39c12" if scenario.probability_success >= 0.5 else "#e74c3c"
            st.markdown(f"""
            <div style="text-align: center; padding: 15px; background: {prob_color}11; border-radius: 10px;">
                <div style="font-size: 2em; font-weight: bold; color: {prob_color};">
                    {scenario.probability_success:.0%}
                </div>
                <div style="color: #666;">Success Probability</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Before/After comparison
        col_chart, col_rec = st.columns([2, 1])
        
        with col_chart:
            fig = create_whatif_comparison(scenario)
            st.plotly_chart(fig, use_container_width=True)
        
        with col_rec:
            st.markdown("### 💡 Recommendations")
            for i, rec in enumerate(scenario.recommendations, 1):
                st.markdown(f"""
                <div style="background: #f8f9fa; padding: 10px; margin: 5px 0; border-radius: 5px;">
                    <span style="color: #3498db; font-weight: bold;">{i}.</span> {rec}
                </div>
                """, unsafe_allow_html=True)
            
            # Action button
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("📋 Create Action Plan", key="create_action_plan"):
                st.success("Action plan created! Check the Resolver for next steps.")


# ============================================================
# TEST FUNCTION
# ============================================================

def test_cascade_explorer():
    """Test cascade explorer functionality."""
    print("=" * 60)
    print("TRIALPULSE NEXUS 10X - CASCADE EXPLORER TEST")
    print("=" * 60)
    
    loader = CascadeDataLoader()
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Data Loader Initialization
    print("\nTEST 1: Data Loader Initialization")
    try:
        assert loader is not None
        print("   ✅ Data loader initialized")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        tests_failed += 1
    
    # Test 2: Load Issue Counts
    print("\nTEST 2: Load Issue Counts")
    try:
        counts = loader.get_issue_counts()
        total = sum(counts.values())
        print(f"   Total issues: {total}")
        print(f"   Issue types with data: {len([k for k, v in counts.items() if v > 0])}")
        print("   ✅ Issue counts loaded")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        tests_failed += 1
    
    # Test 3: Build Cascade Nodes
    print("\nTEST 3: Build Cascade Nodes")
    try:
        nodes = loader.get_cascade_nodes()
        print(f"   Nodes created: {len(nodes)}")
        print(f"   Nodes with patients: {len([n for n in nodes if n.patient_count > 0])}")
        print("   ✅ Cascade nodes built")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        tests_failed += 1
    
    # Test 4: Build Cascade Edges
    print("\nTEST 4: Build Cascade Edges")
    try:
        edges = loader.get_cascade_edges()
        print(f"   Edges created: {len(edges)}")
        print("   ✅ Cascade edges built")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        tests_failed += 1
    
    # Test 5: Cascade Impact Calculation
    print("\nTEST 5: Cascade Impact Calculation")
    try:
        impact = loader.get_cascade_impact("sdv_incomplete")
        print(f"   Issue: SDV Incomplete")
        print(f"   Direct unlocks: {impact.direct_unlocks}")
        print(f"   Cascade chain: {len(impact.cascade_chain)} steps")
        print(f"   Patients unblocked: {impact.patients_unblocked}")
        print(f"   ROI Score: {impact.roi_score}")
        print("   ✅ Cascade impact calculated")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        tests_failed += 1
    
    # Test 6: Critical Path
    print("\nTEST 6: Critical Path to DB Lock")
    try:
        path = loader.get_critical_path()
        print(f"   Path: {' → '.join(path.path)}")
        print(f"   Total patients: {path.total_patients}")
        print(f"   Bottleneck: {path.bottleneck}")
        print(f"   Estimated days: {path.estimated_days}")
        print("   ✅ Critical path calculated")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        tests_failed += 1
    
    # Test 7: What-If Scenario
    print("\nTEST 7: What-If Scenario")
    try:
        scenario = loader.run_what_if_scenario("fix_issue", "sdv_incomplete", 50)
        print(f"   Scenario: {scenario.scenario_name}")
        print(f"   Patients unlocked: {scenario.patients_unlocked}")
        print(f"   DQI change: {scenario.dqi_change:+.1f}")
        print(f"   Days acceleration: {scenario.days_acceleration:+.0f}")
        print(f"   Success probability: {scenario.probability_success:.0%}")
        print("   ✅ What-if scenario simulated")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        tests_failed += 1
    
    # Test 8: Top Opportunities
    print("\nTEST 8: Top Opportunities")
    try:
        opportunities = loader.get_top_opportunities(5)
        print(f"   Top {len(opportunities)} opportunities:")
        for opp in opportunities[:3]:
            print(f"      - {opp.source_issue}: ROI {opp.roi_score:.1f}, {opp.patients_unblocked} patients")
        print("   ✅ Top opportunities retrieved")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        tests_failed += 1
    
    # Test 9: Graph Creation
    print("\nTEST 9: Graph Visualization")
    try:
        nodes = loader.get_cascade_nodes()
        edges = loader.get_cascade_edges()
        fig = create_cascade_graph(nodes, edges)
        assert fig is not None
        print(f"   Graph traces: {len(fig.data)}")
        print("   ✅ Graph created successfully")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        tests_failed += 1
    
    # Test 10: Impact Chart
    print("\nTEST 10: Impact Chart")
    try:
        opportunities = loader.get_top_opportunities(10)
        fig = create_impact_chart(opportunities)
        assert fig is not None
        print("   ✅ Impact chart created")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        tests_failed += 1
    
    # Test 11: Render Function
    print("\nTEST 11: Render Page Function")
    try:
        assert callable(render_page)
        print("   ✅ render_page is callable")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        tests_failed += 1
    
    # Test 12: Helper Functions
    print("\nTEST 12: Helper Functions")
    try:
        assert get_impact_color("Critical") == "#e74c3c"
        assert get_impact_color("Low") == "#27ae60"
        assert format_number(1500) == "1.5K"
        assert format_number(50) == "50"
        print("   ✅ Helper functions working")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        tests_failed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"RESULTS: {tests_passed}/{tests_passed + tests_failed} tests passed")
    if tests_failed == 0:
        print("✅ ALL TESTS PASSED!")
    else:
        print(f"❌ {tests_failed} tests failed")
    print("=" * 60)
    
    return tests_failed == 0


if __name__ == "__main__":
    test_cascade_explorer()