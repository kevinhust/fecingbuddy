"""
Tests for src/reporting/history_db.py and src/reporting/report_generator.py
"""

import pytest
import tempfile
import os
import sys
from pathlib import Path

sys.path.insert(0, '/Users/kevinwang/Documents/20Projects/fencingbuddy')

from src.reporting.history_db import (
    HistoryDatabase,
    SessionRecord,
    AlertRecordDB,
    ActionRecordDB,
)
from src.reporting.report_generator import ReportGenerator, SessionReportData


class TestSessionRecord:
    """Test SessionRecord dataclass."""

    def test_session_record_creation(self):
        """Test SessionRecord can be created."""
        record = SessionRecord(
            id=1,
            session_name="Test Session",
            date="2026-04-02T10:00:00",
            duration_seconds=300.0,
            son_score=5,
            opp_score=3,
            son_fencer_id="son",
            opp_fencer_id="opponent",
            location="Club",
            notes="Good practice",
            alert_count=10,
            total_actions=50,
        )
        assert record.id == 1
        assert record.session_name == "Test Session"
        assert record.son_score == 5
        assert record.opp_score == 3


class TestAlertRecordDB:
    """Test AlertRecordDB dataclass."""

    def test_alert_record_creation(self):
        """Test AlertRecordDB can be created."""
        record = AlertRecordDB(
            id=1,
            session_id=1,
            timestamp=1.5,
            message="Extend your arm",
            priority=2,
            category="attack",
            fencer_id=0,
        )
        assert record.id == 1
        assert record.message == "Extend your arm"
        assert record.priority == 2


class TestActionRecordDB:
    """Test ActionRecordDB dataclass."""

    def test_action_record_creation(self):
        """Test ActionRecordDB can be created."""
        record = ActionRecordDB(
            id=1,
            session_id=1,
            timestamp=2.0,
            action_type="attack",
            confidence=0.85,
            fencer_id=0,
        )
        assert record.id == 1
        assert record.action_type == "attack"
        assert record.confidence == 0.85


class TestHistoryDatabase:
    """Test HistoryDatabase class."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_history.db")
            db = HistoryDatabase(db_path=db_path)
            yield db
            db.close()

    def test_initialization(self, temp_db):
        """Test HistoryDatabase initializes correctly."""
        assert temp_db.db_path.exists()

    def test_add_session(self, temp_db):
        """Test adding a session to the database."""
        session_id = temp_db.add_session(
            session_name="Test Bout",
            duration_seconds=600.0,
            son_score=5,
            opp_score=2,
            son_fencer_id="son_001",
            opp_fencer_id="opp_001",
        )
        assert session_id == 1

    def test_add_session_with_alerts(self, temp_db):
        """Test adding a session with alerts."""
        alerts = [
            {"timestamp": 1.0, "message": "Alert 1", "priority": 2, "category": "attack"},
            {"timestamp": 2.0, "message": "Alert 2", "priority": 3, "category": "distance"},
        ]
        session_id = temp_db.add_session(
            session_name="Test Bout",
            duration_seconds=600.0,
            son_score=5,
            opp_score=2,
            alerts=alerts,
        )
        assert session_id == 1

        alerts_retrieved = temp_db.get_session_alerts(session_id)
        assert len(alerts_retrieved) == 2
        assert alerts_retrieved[0].message == "Alert 1"
        assert alerts_retrieved[1].category == "distance"

    def test_add_session_with_actions(self, temp_db):
        """Test adding a session with actions."""
        actions = [
            {"timestamp": 1.0, "action_type": "attack", "confidence": 0.9, "fencer_id": 0},
            {"timestamp": 2.0, "action_type": "parry", "confidence": 0.8, "fencer_id": 1},
        ]
        session_id = temp_db.add_session(
            session_name="Test Bout",
            duration_seconds=600.0,
            son_score=5,
            opp_score=2,
            actions=actions,
        )
        assert session_id == 1

        actions_retrieved = temp_db.get_session_actions(session_id)
        assert len(actions_retrieved) == 2
        assert actions_retrieved[0].action_type == "attack"
        assert actions_retrieved[1].fencer_id == 1

    def test_get_sessions(self, temp_db):
        """Test retrieving sessions."""
        # Add multiple sessions
        temp_db.add_session("Session 1", 300.0, 5, 3)
        temp_db.add_session("Session 2", 400.0, 2, 5)
        temp_db.add_session("Session 3", 500.0, 5, 5)

        sessions = temp_db.get_sessions(limit=10)
        assert len(sessions) == 3
        assert sessions[0].session_name == "Session 3"  # Most recent first

    def test_get_sessions_with_limit(self, temp_db):
        """Test session limit."""
        for i in range(5):
            temp_db.add_session(f"Session {i}", 300.0, 5, 3)

        sessions = temp_db.get_sessions(limit=2)
        assert len(sessions) == 2

    def test_get_sessions_with_fencer_filter(self, temp_db):
        """Test filtering sessions by fencer."""
        temp_db.add_session("Session 1", 300.0, 5, 3, son_fencer_id="son_001")
        temp_db.add_session("Session 2", 300.0, 3, 5, son_fencer_id="son_002")

        sessions = temp_db.get_sessions(fencer_id="son_001")
        assert len(sessions) == 1
        assert sessions[0].session_name == "Session 1"

    def test_get_alert_stats(self, temp_db):
        """Test alert statistics."""
        alerts = [
            {"timestamp": 1.0, "message": "A1", "priority": 2, "category": "attack"},
            {"timestamp": 2.0, "message": "A2", "priority": 2, "category": "attack"},
            {"timestamp": 3.0, "message": "A3", "priority": 3, "category": "distance"},
        ]
        session_id = temp_db.add_session(
            "Test", 300.0, 5, 3, alerts=alerts
        )

        stats = temp_db.get_alert_stats(session_id)
        assert stats["attack"] == 2
        assert stats["distance"] == 1

    def test_get_action_stats(self, temp_db):
        """Test action statistics."""
        actions = [
            {"timestamp": 1.0, "action_type": "attack", "confidence": 0.9, "fencer_id": 0},
            {"timestamp": 2.0, "action_type": "attack", "confidence": 0.8, "fencer_id": 0},
            {"timestamp": 3.0, "action_type": "parry", "confidence": 0.7, "fencer_id": 0},
        ]
        session_id = temp_db.add_session(
            "Test", 300.0, 5, 3, actions=actions
        )

        stats = temp_db.get_action_stats(session_id)
        assert stats["attack"] == 2
        assert stats["parry"] == 1

    def test_close_and_reopen(self, temp_db):
        """Test closing and reopening database."""
        temp_db.add_session("Test", 300.0, 5, 3)
        temp_db.close()

        # Reopen and verify
        db_path = str(temp_db.db_path)
        db2 = HistoryDatabase(db_path=db_path)
        sessions = db2.get_sessions()
        assert len(sessions) == 1
        db2.close()


class TestReportGenerator:
    """Test ReportGenerator class."""

    def test_initialization(self):
        """Test ReportGenerator initializes correctly."""
        gen = ReportGenerator()
        assert gen is not None

    def test_generate_session_report(self):
        """Test HTML report generation."""
        gen = ReportGenerator()
        data = SessionReportData(
            session_name="Test Bout",
            date="2026-04-02T10:00:00",
            duration_seconds=600.0,
            son_score=5,
            opp_score=3,
            son_fencer_id="son_001",
            opp_fencer_id="opp_001",
            alerts=[
                {"message": "Extend arm", "priority": 2, "category": "attack"},
            ],
            action_stats={"attack": 10, "parry": 5},
            alert_stats={"attack": 3, "distance": 2},
            frequent_alerts=[("Extend arm", 3)],
        )

        html = gen.generate_session_report(data)

        # Verify HTML contains key elements
        assert "FencerAI Session Report" in html
        assert "Test Bout" in html
        assert ">5<" in html  # Son score
        assert ">3<" in html  # Opp score
        assert "WIN" in html
        assert "Extend arm" in html

    def test_save_report(self):
        """Test saving HTML report to file."""
        gen = ReportGenerator()
        data = SessionReportData(
            session_name="Test",
            date="2026-04-02",
            duration_seconds=300.0,
            son_score=5,
            opp_score=3,
            son_fencer_id="son",
            opp_fencer_id="opp",
            alerts=[],
            action_stats={},
            alert_stats={},
            frequent_alerts=[],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "report.html")
            path = gen.save_report("test content", filepath)
            assert path.exists()
            assert path.read_text() == "test content"

    def test_win_loss_tie(self):
        """Test win/loss/tie detection."""
        gen = ReportGenerator()

        # Win
        data = SessionReportData(
            session_name="Test", date="2026-04-02", duration_seconds=300.0,
            son_score=5, opp_score=3, son_fencer_id="s", opp_fencer_id="o",
            alerts=[], action_stats={}, alert_stats={}, frequent_alerts=[]
        )
        html = gen.generate_session_report(data)
        assert "WIN" in html

        # Loss
        data.son_score = 2
        data.opp_score = 5
        html = gen.generate_session_report(data)
        assert "LOSS" in html

        # Tie
        data.son_score = 5
        data.opp_score = 5
        html = gen.generate_session_report(data)
        assert "TIE" in html

    def test_drill_recommendations(self):
        """Test drill recommendations are included."""
        gen = ReportGenerator()
        data = SessionReportData(
            session_name="Test",
            date="2026-04-02",
            duration_seconds=300.0,
            son_score=5,
            opp_score=3,
            son_fencer_id="son",
            opp_fencer_id="opp",
            alerts=[],
            action_stats={},
            alert_stats={},
            frequent_alerts=[("Recovery needs work", 5)],
        )

        html = gen.generate_session_report(data)
        # Recovery alerts should trigger recovery drill recommendations
        assert "Recovery" in html or "Drill" in html
