# src/fix_missing_methods.py
"""Add missing methods back to EscalationEngine"""

from pathlib import Path

def fix_missing_methods():
    file_path = Path("src/collaboration/escalation_engine.py")
    content = file_path.read_text(encoding='utf-8')
    
    # The methods to add - insert them before the SINGLETON ACCESSOR section
    missing_methods = '''
    # =========================================================================
    # HISTORY & STATISTICS
    # =========================================================================
    
    def get_escalation_history(self, escalation_id: str) -> List[Dict]:
        """Get escalation history"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM escalation_history 
                WHERE escalation_id = ?
                ORDER BY changed_at ASC
            """, (escalation_id,))
            
            rows = cursor.fetchall()
            columns = [col[0] for col in cursor.description]
            
            return [dict(zip(columns, row)) for row in rows]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get escalation statistics"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Total escalations
            cursor.execute("SELECT COUNT(*) FROM escalations")
            stats['total_escalations'] = cursor.fetchone()[0]
            
            # By status
            cursor.execute("""
                SELECT status, COUNT(*) FROM escalations GROUP BY status
            """)
            stats['by_status'] = {row[0]: row[1] for row in cursor.fetchall()}
            
            # By level
            cursor.execute("""
                SELECT level, COUNT(*) FROM escalations GROUP BY level
            """)
            stats['by_level'] = {f"L{row[0]}": row[1] for row in cursor.fetchall()}
            
            # Active escalations
            cursor.execute("""
                SELECT COUNT(*) FROM escalations 
                WHERE status IN ('pending', 'acknowledged', 'in_progress')
            """)
            stats['active_escalations'] = cursor.fetchone()[0]
            
            # By trigger
            cursor.execute("""
                SELECT trigger, COUNT(*) FROM escalations GROUP BY trigger
            """)
            stats['by_trigger'] = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Average resolution time (hours)
            cursor.execute("""
                SELECT AVG(
                    (julianday(resolved_at) - julianday(escalated_at)) * 24
                ) FROM escalations 
                WHERE resolved_at IS NOT NULL
            """)
            result = cursor.fetchone()[0]
            stats['avg_resolution_hours'] = round(result, 1) if result else 0
            
            # Escalations in last 24 hours
            yesterday = (datetime.now() - timedelta(hours=24)).isoformat()
            cursor.execute("""
                SELECT COUNT(*) FROM escalations WHERE escalated_at > ?
            """, (yesterday,))
            stats['last_24_hours'] = cursor.fetchone()[0]
            
            # Escalations in last 7 days
            last_week = (datetime.now() - timedelta(days=7)).isoformat()
            cursor.execute("""
                SELECT COUNT(*) FROM escalations WHERE escalated_at > ?
            """, (last_week,))
            stats['last_7_days'] = cursor.fetchone()[0]
            
            # Rules count
            stats['rules_count'] = len(self.rules)
            stats['active_rules'] = len(self.get_active_rules())
            
            return stats
    
    def get_audit_trail(self, escalation_id: str) -> List[Dict]:
        """Get audit trail for an escalation"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM escalation_audit 
                WHERE escalation_id = ?
                ORDER BY created_at ASC
            """, (escalation_id,))
            
            rows = cursor.fetchall()
            columns = [col[0] for col in cursor.description]
            
            return [dict(zip(columns, row)) for row in rows]


'''
    
    # Find the insertion point - before SINGLETON ACCESSOR
    insertion_marker = "# =============================================================================\n# SINGLETON ACCESSOR"
    
    if insertion_marker in content:
        # Check if methods already exist
        if "def get_escalation_history(self" in content:
            print("⚠️ get_escalation_history already exists")
        if "def get_statistics(self" in content:
            print("⚠️ get_statistics already exists")
        if "def get_audit_trail(self" in content:
            print("⚠️ get_audit_trail already exists")
        
        # Only add if missing
        if "def get_escalation_history(self" not in content:
            content = content.replace(insertion_marker, missing_methods + insertion_marker)
            file_path.write_text(content, encoding='utf-8')
            print("✅ Added missing methods (get_escalation_history, get_statistics, get_audit_trail)")
            return True
        else:
            print("⚠️ Methods seem to already exist but may be malformed. Manual check needed.")
            return False
    else:
        print("❌ Could not find insertion point. Manual fix needed.")
        return False


if __name__ == "__main__":
    fix_missing_methods()