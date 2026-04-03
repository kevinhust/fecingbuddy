"""
FencerAI Reporting Module
======================
Version: 2.0 | Last Updated: 2026-04-02

Session recording, historical database, and HTML report generation.
"""

from __future__ import annotations

from src.reporting.history_db import HistoryDatabase, SessionRecord, AlertRecordDB, ActionRecordDB
from src.reporting.report_generator import ReportGenerator, SessionReportData

__all__ = [
    "HistoryDatabase",
    "SessionRecord",
    "AlertRecordDB",
    "ActionRecordDB",
    "ReportGenerator",
    "SessionReportData",
]
