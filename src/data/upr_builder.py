"""
TRIALPULSE NEXUS 10X - Unified Patient Record (UPR) Builder (v1.2)
===================================================================
Fixed: Smart join strategy - uses study_id + subject_id when patient_key match is low.

Author: TrialPulse Team
Version: 1.2.0
"""

import os
import re
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
import traceback
import warnings

import pandas as pd
import numpy as np
from loguru import logger

warnings.filterwarnings('ignore', category=UserWarning)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import DATA_PROCESSED, LOGS_DIR


# ============================================
# CONSTANTS
# ============================================

CLEANED_DATA_DIR = DATA_PROCESSED / "cleaned"
UPR_OUTPUT_DIR = DATA_PROCESSED / "upr"
UPR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DETAIL_TABLES = {
    'visit_projection_agg.parquet': 'visit',
    'missing_lab_ranges_agg.parquet': 'lab',
    'sae_dashboard_dm_agg.parquet': 'sae_dm',
    'sae_dashboard_safety_agg.parquet': 'sae_safety',
    'inactivated_forms_agg.parquet': 'inactivated',
    'missing_pages_agg.parquet': 'pages',
    'compiled_edrr_agg.parquet': 'edrr',
    'coding_meddra_agg.parquet': 'meddra',
    'coding_whodrug_agg.parquet': 'whodrug',
}

EXCLUDE_COLUMNS = ['study_id', 'site_id', 'subject_id', 'patient_key']


# ============================================
# LOGGING & UTILITIES
# ============================================

def setup_logger() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"upr_builder_{timestamp}.log"
    logger.remove()
    logger.add(log_file, format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}", level="DEBUG", rotation="10 MB")
    logger.add(lambda msg: print(msg, end=""), format="<level>{level:<8}</level> | {message}\n", level="INFO", colorize=True)
    return log_file


def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif pd.isna(obj):
        return None
    return obj


# ============================================
# DATA CLASSES
# ============================================

@dataclass
class JoinStats:
    table_name: str
    rows_in_detail: int
    rows_matched: int
    rows_unmatched: int
    columns_added: int
    match_rate: float = 0.0
    join_method: str = "patient_key"


@dataclass
class UPRManifest:
    run_id: str
    start_time: str
    end_time: str = ""
    status: str = "running"
    schema_version: str = "1.2.0"
    main_spine_rows: int = 0
    main_spine_columns: int = 0
    tables_joined: int = 0
    join_stats: Dict[str, Dict] = field(default_factory=dict)
    upr_rows: int = 0
    upr_columns: int = 0
    columns_with_nulls: Dict[str, float] = field(default_factory=dict)
    errors: List[Dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return asdict(self)


# ============================================
# UPR BUILDER
# ============================================

class UPRBuilder:
    def __init__(self, input_dir: Path = None, output_dir: Path = None):
        self.input_dir = input_dir or CLEANED_DATA_DIR
        self.output_dir = output_dir or UPR_OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = setup_logger()
        self.start_time = datetime.now()
        
        self.manifest = UPRManifest(
            run_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            start_time=datetime.now().isoformat()
        )
        
        self.upr: Optional[pd.DataFrame] = None
    
    def load_main_spine(self) -> pd.DataFrame:
        logger.info("Loading main spine (CPID EDC Metrics)...")
        
        spine_path = self.input_dir / "cpid_edc_metrics.parquet"
        if not spine_path.exists():
            raise FileNotFoundError(f"Main spine not found: {spine_path}")
        
        df = pd.read_parquet(spine_path)
        
        self.manifest.main_spine_rows = len(df)
        self.manifest.main_spine_columns = len(df.columns)
        
        logger.info(f"  ✓ Loaded: {len(df):,} patients, {len(df.columns)} columns")
        
        if 'study_id' in df.columns:
            study_counts = df['study_id'].value_counts()
            logger.info(f"  Studies: {len(study_counts)}")
            for study, count in study_counts.head(5).items():
                logger.info(f"    {study}: {count:,} patients")
            if len(study_counts) > 5:
                logger.info(f"    ... and {len(study_counts) - 5} more studies")
        
        if 'subject_status_clean' in df.columns:
            status_counts = df['subject_status_clean'].value_counts()
            logger.info(f"  Subject Status:")
            for status, count in status_counts.items():
                logger.info(f"    {status}: {count:,}")
        
        return df
    
    def load_detail_table(self, filename: str) -> Optional[pd.DataFrame]:
        file_path = self.input_dir / filename
        if not file_path.exists():
            logger.warning(f"  ⚠️ File not found: {filename}")
            return None
        return pd.read_parquet(file_path)
    
    def join_detail_table(self, main_df: pd.DataFrame, detail_df: pd.DataFrame, 
                          table_name: str, prefix: str) -> Tuple[pd.DataFrame, JoinStats]:
        """
        Join a detail table to the main spine.
        Uses smart join strategy: patient_key first, then study_id + subject_id fallback.
        """
        
        stats = JoinStats(
            table_name=table_name,
            rows_in_detail=len(detail_df),
            rows_matched=0,
            rows_unmatched=0,
            columns_added=0,
            join_method="patient_key"
        )
        
        detail_columns = [c for c in detail_df.columns if c not in EXCLUDE_COLUMNS]
        
        if not detail_columns:
            logger.warning(f"  ⚠️ No columns to join from {table_name}")
            return main_df, stats
        
        # Check patient_key match rate first
        matched_keys = set(main_df['patient_key']) & set(detail_df['patient_key'])
        patient_key_match_rate = len(matched_keys) / len(detail_df) if len(detail_df) > 0 else 0
        
        # Rename columns with prefix
        rename_map = {col: f"{prefix}_{col}" for col in detail_columns}
        
        # Choose join strategy based on match rate
        if patient_key_match_rate >= 0.5:
            # Good match rate - use patient_key
            detail_renamed = detail_df[['patient_key'] + detail_columns].copy()
            detail_renamed = detail_renamed.rename(columns=rename_map)
            
            result = main_df.merge(detail_renamed, on='patient_key', how='left')
            
            stats.rows_matched = len(matched_keys)
            stats.rows_unmatched = len(detail_df) - len(matched_keys)
            stats.columns_added = len(detail_columns)
            stats.match_rate = patient_key_match_rate * 100
            stats.join_method = "patient_key"
            
            logger.info(f"  ✓ {table_name}: +{stats.columns_added} cols, "
                       f"{stats.rows_matched:,}/{stats.rows_in_detail:,} matched ({stats.match_rate:.1f}%) [patient_key]")
        
        else:
            # Low match rate - try study_id + subject_id join
            if 'study_id' in detail_df.columns and 'subject_id' in detail_df.columns:
                # Check subject match rate
                main_subjects = set(zip(main_df['study_id'].fillna(''), main_df['subject_id'].fillna('')))
                detail_subjects = set(zip(detail_df['study_id'].fillna(''), detail_df['subject_id'].fillna('')))
                subject_matches = main_subjects & detail_subjects
                subject_match_rate = len(subject_matches) / len(detail_df) if len(detail_df) > 0 else 0
                
                if subject_match_rate > 0:
                    # Use study_id + subject_id join
                    detail_renamed = detail_df[['study_id', 'subject_id'] + detail_columns].copy()
                    detail_renamed = detail_renamed.rename(columns=rename_map)
                    
                    # Drop duplicates on join keys (keep first)
                    detail_renamed = detail_renamed.drop_duplicates(subset=['study_id', 'subject_id'], keep='first')
                    
                    result = main_df.merge(
                        detail_renamed, 
                        on=['study_id', 'subject_id'], 
                        how='left'
                    )
                    
                    stats.rows_matched = len(subject_matches)
                    stats.rows_unmatched = len(detail_df) - len(subject_matches)
                    stats.columns_added = len(detail_columns)
                    stats.match_rate = subject_match_rate * 100
                    stats.join_method = "study+subject"
                    
                    logger.info(f"  ✓ {table_name}: +{stats.columns_added} cols, "
                               f"{stats.rows_matched:,}/{stats.rows_in_detail:,} matched ({stats.match_rate:.1f}%) [study+subject]")
                else:
                    # No matches
                    detail_renamed = detail_df[['patient_key'] + detail_columns].copy()
                    detail_renamed = detail_renamed.rename(columns=rename_map)
                    result = main_df.merge(detail_renamed, on='patient_key', how='left')
                    
                    stats.rows_matched = 0
                    stats.rows_unmatched = len(detail_df)
                    stats.columns_added = len(detail_columns)
                    stats.match_rate = 0
                    stats.join_method = "none"
                    
                    logger.warning(f"  ⚠️ {table_name}: +{stats.columns_added} cols, 0/{stats.rows_in_detail:,} matched (0.0%)")
            else:
                # No study_id/subject_id columns
                detail_renamed = detail_df[['patient_key'] + detail_columns].copy()
                detail_renamed = detail_renamed.rename(columns=rename_map)
                result = main_df.merge(detail_renamed, on='patient_key', how='left')
                
                stats.rows_matched = len(matched_keys)
                stats.rows_unmatched = len(detail_df) - len(matched_keys)
                stats.columns_added = len(detail_columns)
                stats.match_rate = patient_key_match_rate * 100
                stats.join_method = "patient_key"
                
                logger.info(f"  ✓ {table_name}: +{stats.columns_added} cols, "
                           f"{stats.rows_matched:,}/{stats.rows_in_detail:,} matched ({stats.match_rate:.1f}%)")
        
        return result, stats
    
    def fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Filling missing values...")
        df = df.copy()
        filled_counts = {}
        
        for col in df.columns:
            null_count = df[col].isna().sum()
            if null_count == 0:
                continue
            
            col_lower = col.lower()
            
            if any(x in col_lower for x in ['count', 'total', 'rate', 'days', 'avg', 'max', 'min',
                                              'coded', 'uncoded', 'pending', 'completed',
                                              'queries', 'issues', 'forms', 'pages', 'visits',
                                              'signatures', 'crfs', 'sdv', 'pds']):
                df[col] = df[col].fillna(0)
                filled_counts[col] = null_count
            elif 'status' in col_lower:
                df[col] = df[col].fillna('Unknown')
                filled_counts[col] = null_count
            elif any(x in col_lower for x in ['_id', 'key', '_ts', 'timestamp']):
                pass
            elif df[col].dtype == 'object':
                df[col] = df[col].fillna('')
                filled_counts[col] = null_count
        
        if filled_counts:
            logger.info(f"  ✓ Filled {len(filled_counts)} columns with defaults")
        
        return df
    
    def calculate_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Calculating derived metrics...")
        df = df.copy()
        metrics_added = []
        
        # Total Issues
        issue_cols = [c for c in df.columns if any(x in c.lower() for x in 
                     ['issue_count', 'pending', 'open_queries', 'missing'])]
        if issue_cols:
            df['total_issues_all_sources'] = df[issue_cols].sum(axis=1)
            metrics_added.append('total_issues_all_sources')
        
        # Coding totals
        if 'meddra_coding_meddra_total' in df.columns and 'whodrug_coding_whodrug_total' in df.columns:
            df['total_coding_terms'] = df['meddra_coding_meddra_total'] + df['whodrug_coding_whodrug_total']
            df['total_coded_terms'] = df.get('meddra_coding_meddra_coded', 0) + df.get('whodrug_coding_whodrug_coded', 0)
            df['total_uncoded_terms'] = df.get('meddra_coding_meddra_uncoded', 0) + df.get('whodrug_coding_whodrug_uncoded', 0)
            df['coding_completion_rate'] = np.where(
                df['total_coding_terms'] > 0,
                (df['total_coded_terms'] / df['total_coding_terms'] * 100).clip(0, 100),
                100.0
            )
            metrics_added.extend(['total_coding_terms', 'total_coded_terms', 'total_uncoded_terms', 'coding_completion_rate'])
        
        # SAE totals
        if 'sae_dm_sae_dm_total' in df.columns and 'sae_safety_sae_safety_total' in df.columns:
            df['total_sae_issues'] = df['sae_dm_sae_dm_total'] + df['sae_safety_sae_safety_total']
            df['total_sae_pending'] = df.get('sae_dm_sae_dm_pending', 0) + df.get('sae_safety_sae_safety_pending', 0)
            metrics_added.extend(['total_sae_issues', 'total_sae_pending'])
        
        # Has issues flags
        for col_check, flag_name in [
            ('visit_missing_visit_count', 'has_missing_visits'),
            ('pages_missing_page_count', 'has_missing_pages'),
            ('lab_lab_issue_count', 'has_lab_issues'),
            ('edrr_edrr_issue_count', 'has_edrr_issues'),
        ]:
            if col_check in df.columns:
                df[flag_name] = df[col_check] > 0
                metrics_added.append(flag_name)
        
        # Completeness score
        completeness_factors = []
        for col in ['visit_missing_visit_count', 'pages_missing_page_count', 'open_queries_calculated', 'total_uncoded_terms']:
            if col in df.columns:
                completeness_factors.append(df[col] == 0)
        
        if completeness_factors:
            df['completeness_score'] = sum(completeness_factors) / len(completeness_factors) * 100
            metrics_added.append('completeness_score')
        
        # Risk level
        if 'total_issues_all_sources' in df.columns:
            conditions = [
                df['total_issues_all_sources'] == 0,
                df['total_issues_all_sources'] <= 5,
                df['total_issues_all_sources'] <= 20,
                df['total_issues_all_sources'] > 20
            ]
            choices = ['Low', 'Medium', 'High', 'Critical']
            df['risk_level'] = np.select(conditions, choices, default='Unknown')
            metrics_added.append('risk_level')
        
        logger.info(f"  ✓ Added {len(metrics_added)} derived metrics")
        return df
    
    def calculate_null_percentages(self, df: pd.DataFrame) -> Dict[str, float]:
        null_pcts = {}
        for col in df.columns:
            null_pct = (df[col].isna().sum() / len(df) * 100) if len(df) > 0 else 0
            if null_pct > 0:
                null_pcts[col] = round(null_pct, 2)
        return null_pcts
    
    def validate_upr(self, df: pd.DataFrame) -> List[str]:
        logger.info("Validating UPR...")
        warnings_list = []
        
        dup_count = df['patient_key'].duplicated().sum()
        if dup_count > 0:
            warnings_list.append(f"Found {dup_count} duplicate patient_keys")
            logger.warning(f"  ⚠️ {dup_count} duplicate patient_keys")
        else:
            logger.info(f"  ✓ No duplicate patient_keys")
        
        required_cols = ['patient_key', 'study_id', 'site_id', 'subject_id']
        missing_required = [c for c in required_cols if c not in df.columns]
        if missing_required:
            warnings_list.append(f"Missing required columns: {missing_required}")
            logger.warning(f"  ⚠️ Missing: {missing_required}")
        else:
            logger.info(f"  ✓ All required columns present")
        
        logger.info(f"  ✓ UPR has {len(df):,} patients, {len(df.columns)} columns")
        
        return warnings_list
    
    def _clean_types_for_parquet(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].apply(lambda x: str(x) if pd.notna(x) and x is not None else None)
            elif df[col].dtype == 'bool':
                df[col] = df[col].astype(bool)
        return df
    
    def _save_upr(self):
        logger.info("  Cleaning column types...")
        self.upr = self._clean_types_for_parquet(self.upr)
        
        # Main UPR
        upr_path = self.output_dir / "unified_patient_record.parquet"
        self.upr.to_parquet(upr_path, index=False)
        logger.info(f"  ✓ Saved: {upr_path.name} ({len(self.upr):,} rows, {len(self.upr.columns)} columns)")
        
        # Sample CSV
        summary_cols = ['patient_key', 'study_id', 'site_id', 'subject_id', 'subject_status_clean',
                        'region', 'country', 'total_issues_all_sources', 'risk_level', 'completeness_score']
        summary_cols = [c for c in summary_cols if c in self.upr.columns]
        self.upr[summary_cols].head(1000).to_csv(self.output_dir / "upr_sample.csv", index=False)
        logger.info(f"  ✓ Saved: upr_sample.csv")
        
        # Column catalog
        catalog = [{
            'column': col,
            'dtype': str(self.upr[col].dtype),
            'non_null': int(self.upr[col].notna().sum()),
            'null_pct': round(self.upr[col].isna().sum() / len(self.upr) * 100, 2)
        } for col in self.upr.columns]
        pd.DataFrame(catalog).to_csv(self.output_dir / "upr_column_catalog.csv", index=False)
        logger.info(f"  ✓ Saved: upr_column_catalog.csv ({len(catalog)} columns)")
    
    def build(self) -> pd.DataFrame:
        logger.info("=" * 70)
        logger.info("TRIALPULSE NEXUS 10X - UPR BUILDER (v1.2)")
        logger.info("=" * 70)
        logger.info(f"Input: {self.input_dir}")
        logger.info(f"Output: {self.output_dir}")
        logger.info("")
        
        try:
            # Step 1
            logger.info("=" * 70)
            logger.info("STEP 1: LOADING MAIN SPINE")
            logger.info("=" * 70)
            self.upr = self.load_main_spine()
            
            # Step 2
            logger.info("\n" + "=" * 70)
            logger.info("STEP 2: JOINING DETAIL TABLES")
            logger.info("=" * 70)
            for filename, prefix in DETAIL_TABLES.items():
                detail_df = self.load_detail_table(filename)
                if detail_df is not None:
                    self.upr, stats = self.join_detail_table(self.upr, detail_df, filename.replace('.parquet', ''), prefix)
                    self.manifest.join_stats[filename] = asdict(stats)
                    self.manifest.tables_joined += 1
            
            # Step 3
            logger.info("\n" + "=" * 70)
            logger.info("STEP 3: HANDLING MISSING VALUES")
            logger.info("=" * 70)
            self.upr = self.fill_missing_values(self.upr)
            
            # Step 4
            logger.info("\n" + "=" * 70)
            logger.info("STEP 4: CALCULATING DERIVED METRICS")
            logger.info("=" * 70)
            self.upr = self.calculate_derived_metrics(self.upr)
            
            # Step 5
            self.manifest.columns_with_nulls = self.calculate_null_percentages(self.upr)
            
            logger.info("\n" + "=" * 70)
            logger.info("STEP 5: VALIDATION")
            logger.info("=" * 70)
            self.manifest.warnings = self.validate_upr(self.upr)
            
            # Metadata
            self.upr['_upr_built_ts'] = datetime.now().isoformat()
            self.upr['_upr_version'] = '1.2.0'
            
            self.manifest.upr_rows = len(self.upr)
            self.manifest.upr_columns = len(self.upr.columns)
            self.manifest.status = "completed"
            self.manifest.end_time = datetime.now().isoformat()
            
            # Step 6
            logger.info("\n" + "=" * 70)
            logger.info("STEP 6: SAVING UPR")
            logger.info("=" * 70)
            self._save_upr()
            
        except Exception as e:
            self.manifest.status = "failed"
            self.manifest.errors.append({'error': str(e), 'traceback': traceback.format_exc()})
            logger.error(f"UPR build failed: {traceback.format_exc()}")
            raise
        
        # Manifest
        with open(self.output_dir / "upr_manifest.json", 'w') as f:
            json.dump(convert_to_serializable(self.manifest.to_dict()), f, indent=2)
        
        self._print_summary()
        return self.upr
    
    def _print_summary(self):
        logger.info("\n" + "=" * 70)
        logger.info("UPR BUILD COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Status: {self.manifest.status}")
        logger.info(f"\nMain Spine: {self.manifest.main_spine_rows:,} rows, {self.manifest.main_spine_columns} cols")
        logger.info(f"\nJoins:")
        for name, stats in self.manifest.join_stats.items():
            logger.info(f"  {name}: {stats['rows_matched']:,}/{stats['rows_in_detail']:,} ({stats['match_rate']:.1f}%) [{stats.get('join_method', 'unknown')}]")
        logger.info(f"\nFinal UPR: {self.manifest.upr_rows:,} rows, {self.manifest.upr_columns} cols")
        
        if self.manifest.warnings:
            logger.warning(f"\nWarnings: {self.manifest.warnings}")
        if not self.manifest.errors:
            logger.info("\n✅ NO ERRORS!")


def main():
    builder = UPRBuilder()
    upr = builder.build()
    if builder.manifest.status == "completed":
        print(f"\n✅ UPR BUILD SUCCESS! {len(upr):,} patients, {len(upr.columns)} columns")
        return 0
    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())