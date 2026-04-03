"""
FencerAI Report Generator
======================
Version: 2.0 | Last Updated: 2026-04-02

Generates HTML reports for fencing sessions.
Creates post-session reports with stats, alerts, and drill recommendations.

Example:
    generator = ReportGenerator()
    html = generator.generate_session_report(session_data)
    generator.save_report(html, "outputs/reports/bout_20260402.html")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class SessionReportData:
    """Data for generating a session report."""
    session_name: str
    date: str
    duration_seconds: float
    son_score: int
    opp_score: int
    son_fencer_id: str
    opp_fencer_id: str
    alerts: List[Dict[str, Any]]
    action_stats: Dict[str, int]
    alert_stats: Dict[str, int]
    frequent_alerts: List[tuple]  # (message, count)
    location: Optional[str] = None
    notes: Optional[str] = None


class ReportGenerator:
    """
    Generates HTML reports for fencing sessions.

    Creates:
    - Session overview (date, duration, score)
    - Alert summary by category
    - Action statistics
    - Frequent issues ("3 Things to Fix")
    - Drill recommendations

    Example:
        >>> gen = ReportGenerator()
        >>> data = ReportData(...)
        >>> html = gen.generate_session_report(data)
        >>> gen.save_report(html, "report.html")
    """

    def __init__(self):
        """Initialize report generator."""
        self._drill_recommendations = self._build_drill_map()

    def generate_session_report(self, data: SessionReportData) -> str:
        """
        Generate HTML report for a session.

        Args:
            data: SessionReportData with session information

        Returns:
            HTML string
        """
        win_loss = self._get_win_loss(data.son_score, data.opp_score)
        duration_min = data.duration_seconds / 60.0

        # Build HTML
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FencerAI Report: {data.session_name}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background: #1a1a2e;
            color: #eee;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        h1 {{ margin: 0 0 10px 0; }}
        .subtitle {{ opacity: 0.9; }}
        .score-panel {{
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 40px;
            padding: 30px;
            background: #16213e;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .score {{
            font-size: 72px;
            font-weight: bold;
        }}
        .son-score {{ color: #ff6b6b; }}
        .opp-score {{ color: #4ecdc4; }}
        .vs {{ font-size: 24px; opacity: 0.7; }}
        .win-loss {{
            font-size: 18px;
            padding: 5px 15px;
            border-radius: 20px;
            background: {self._get_win_loss_color(win_loss)};
        }}
        .section {{
            background: #16213e;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        h2 {{ margin-top: 0; color: #667eea; }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }}
        .stat-card {{
            background: #1a1a2e;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-value {{ font-size: 28px; font-weight: bold; color: #667eea; }}
        .stat-label {{ font-size: 12px; opacity: 0.7; margin-top: 5px; }}
        .things-to-fix {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding: 20px;
            border-radius: 10px;
        }}
        .things-to-fix h2 {{ color: #fff; }}
        .fix-item {{
            padding: 10px;
            background: rgba(255,255,255,0.2);
            border-radius: 5px;
            margin-bottom: 10px;
        }}
        .alert-category {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 15px;
            font-size: 12px;
            margin-right: 10px;
        }}
        .category-distance {{ background: #4ecdc4; }}
        .category-attack {{ background: #ff6b6b; }}
        .category-defense {{ background: #667eea; }}
        .category-recovery {{ background: #f093fb; }}
        .category-general {{ background: #888; }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #333;
        }}
        th {{ color: #667eea; }}
        .drill-recommendation {{
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
        }}
        .footer {{
            text-align: center;
            opacity: 0.5;
            margin-top: 30px;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>FencerAI Session Report</h1>
        <div class="subtitle">
            {data.session_name} | {data.date[:10]} | {duration_min:.1f} min
            {f" | {data.location}" if data.location else ""}
        </div>
    </div>

    <div class="score-panel">
        <div class="score son-score">{data.son_score}</div>
        <div>
            <div class="vs">vs</div>
            <div class="win-loss">{win_loss}</div>
        </div>
        <div class="score opp-score">{data.opp_score}</div>
    </div>

    <div class="section">
        <h2>Session Statistics</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{duration_min:.1f}</div>
                <div class="stat-label">Minutes</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{sum(data.action_stats.values())}</div>
                <div class="stat-label">Total Actions</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(data.alerts)}</div>
                <div class="stat-label">Coaching Alerts</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{data.action_stats.get('attack', 0)}</div>
                <div class="stat-label">Attacks</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{data.action_stats.get('fleche', 0)}</div>
                <div class="stat-label">Fleches</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{data.action_stats.get('parry', 0)}</div>
                <div class="stat-label">Parries</div>
            </div>
        </div>
    </div>
"""

        # Things to Fix section
        if data.frequent_alerts:
            html += """
    <div class="things-to-fix">
        <h2>3 Things to Focus On</h2>
"""
            for msg, count in data.frequent_alerts[:3]:
                html += f"""
        <div class="fix-item">
            <strong>({count}x)</strong> {msg}
        </div>
"""
            html += """
    </div>
"""

        # Action breakdown
        if data.action_stats:
            html += """
    <div class="section">
        <h2>Action Breakdown</h2>
        <table>
            <tr><th>Action</th><th>Count</th></tr>
"""
            for action, count in sorted(data.action_stats.items(), key=lambda x: -x[1]):
                html += f"""
            <tr><td>{action.capitalize()}</td><td>{count}</td></tr>
"""
            html += """
        </table>
    </div>
"""

        # Alert breakdown
        if data.alert_stats:
            html += """
    <div class="section">
        <h2>Alert Categories</h2>
        <div class="stats-grid">
"""
            for cat, count in data.alert_stats.items():
                html += f"""
            <div class="stat-card">
                <div class="stat-value">{count}</div>
                <div class="stat-label">{cat.capitalize()}</div>
            </div>
"""
            html += """
        </div>
    </div>
"""

        # Drill recommendations
        drills = self._get_drill_recommendations(data)
        if drills:
            html += """
    <div class="section">
        <h2>Recommended Drills</h2>
"""
            for drill in drills:
                html += f"""
        <div class="drill-recommendation">
            <strong>{drill['name']}</strong><br>
            <small>{drill['description']}</small>
        </div>
"""
            html += """
    </div>
"""

        # Notes
        if data.notes:
            html += f"""
    <div class="section">
        <h2>Notes</h2>
        <p>{data.notes}</p>
    </div>
"""

        html += f"""
    <div class="footer">
        Generated by FencerAI | {datetime.now().strftime('%Y-%m-%d %H:%M')}
    </div>
</body>
</html>
"""
        return html

    def _get_win_loss(self, son_score: int, opp_score: int) -> str:
        """Get win/loss/tie string."""
        if son_score > opp_score:
            return "WIN"
        elif son_score < opp_score:
            return "LOSS"
        else:
            return "TIE"

    def _get_win_loss_color(self, win_loss: str) -> str:
        """Get background color for win/loss badge."""
        colors = {
            "WIN": "#38ef7d",
            "LOSS": "#f5576c",
            "TIE": "#667eea",
        }
        return colors.get(win_loss, "#667eea")

    def _build_drill_map(self) -> Dict[str, List[Dict]]:
        """Build map of issues to drill recommendations."""
        return {
            "recovery": [
                {"name": "Recovery Lunge", "description": "Practice quick recovery steps after lunges"},
                {"name": "Defensive Retreat", "description": "Fast backward movement to safe distance"},
            ],
            "arm_extension": [
                {"name": "Extension Drills", "description": "Wall partner drills for full arm extension"},
                {"name": "Target Practice", "description": "Hit targets at full extension"},
            ],
            "distance": [
                {"name": "Distance Control", "description": "Advance-retreat combinations at varying speeds"},
                {"name": "Measure Drills", "description": "Practice closing to optimal attack distance"},
            ],
            "predictable": [
                {"name": "Compound Attacks", "description": "Feint-attack combinations to vary timing"},
                {"name": "Multiple Prep Actions", "description": "Change attack preparation patterns"},
            ],
            "parry": [
                {"name": " Parry Response Drills", "description": "Parry-riposte sequences"},
                {"name": "Counter-Parries", "description": "Practice second-intention parries"},
            ],
            "fleche": [
                {"name": "Fleche Timing", "description": "Practice fleche in isolation"},
                {"name": "Counter-Fleche", "description": "Defensive responses to fleche attacks"},
            ],
        }

    def _get_drill_recommendations(self, data: SessionReportData) -> List[Dict]:
        """Get drill recommendations based on frequent issues."""
        recommendations = []

        # Check frequent alerts for drill matches
        for msg, count in data.frequent_alerts:
            msg_lower = msg.lower()
            if "recovery" in msg_lower or "riposte" in msg_lower:
                recommendations.extend(self._drill_recommendations.get("recovery", []))
            if "arm" in msg_lower or "extend" in msg_lower:
                recommendations.extend(self._drill_recommendations.get("arm_extension", []))
            if "distance" in msg_lower:
                recommendations.extend(self._drill_recommendations.get("distance", []))
            if "predictable" in msg_lower or "vary" in msg_lower:
                recommendations.extend(self._drill_recommendations.get("predictable", []))
            if "parry" in msg_lower:
                recommendations.extend(self._drill_recommendations.get("parry", []))
            if "fleche" in msg_lower:
                recommendations.extend(self._drill_recommendations.get("fleche", []))

        # Also check action stats
        if data.action_stats.get("fleche", 0) > 5:
            recommendations.extend(self._drill_recommendations.get("fleche", []))

        # Deduplicate and limit
        seen = set()
        unique = []
        for r in recommendations:
            if r["name"] not in seen:
                seen.add(r["name"])
                unique.append(r)

        return unique[:4]  # Max 4 recommendations

    def save_report(self, html: str, filepath: str) -> Path:
        """
        Save HTML report to file.

        Args:
            html: HTML string
            filepath: Output path

        Returns:
            Path to saved file
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html)
        return path
