"""
FencerAI History Database
=======================
Version: 2.0 | Last Updated: 2026-04-02

SQLite-based historical data storage for fencing sessions.
Stores session metadata, scores, alerts, and action statistics.

Example:
    db = HistoryDatabase()
    db.add_session(session_data)
    sessions = db.get_sessions(fencer_id=0, limit=10)
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class SessionRecord:
    """A recorded session in the database."""
    id: int
    session_name: str
    date: str  # ISO format
    duration_seconds: float
    son_score: int
    opp_score: int
    son_fencer_id: str
    opp_fencer_id: str
    location: Optional[str]
    notes: Optional[str]
    alert_count: int
    total_actions: int


@dataclass
class AlertRecordDB:
    """An alert from a session."""
    id: int
    session_id: int
    timestamp: float
    message: str
    priority: int
    category: str
    fencer_id: Optional[int]


@dataclass
class ActionRecordDB:
    """An action from a session."""
    id: int
    session_id: int
    timestamp: float
    action_type: str
    confidence: float
    fencer_id: int


class HistoryDatabase:
    """
    SQLite database for storing and querying session history.

    Stores:
    - Session metadata (scores, date, duration)
    - Alert history per session
    - Action history per session

    Example:
        >>> db = HistoryDatabase()
        >>> db.add_session(session_data)
        >>> recent = db.get_sessions(limit=5)
    """

    def __init__(self, db_path: str = "outputs/fencerai_history.db"):
        """
        Initialize history database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_name TEXT NOT NULL,
                date TEXT NOT NULL,
                duration_seconds REAL NOT NULL,
                son_score INTEGER NOT NULL,
                opp_score INTEGER NOT NULL,
                son_fencer_id TEXT,
                opp_fencer_id TEXT,
                location TEXT,
                notes TEXT,
                alert_count INTEGER DEFAULT 0,
                total_actions INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Alerts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                message TEXT NOT NULL,
                priority INTEGER NOT NULL,
                category TEXT,
                fencer_id INTEGER,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)

        # Actions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                action_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                fencer_id INTEGER NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)

        # Indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_alerts_session
            ON alerts(session_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_actions_session
            ON actions(session_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_date
            ON sessions(date)
        """)

        conn.commit()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def add_session(
        self,
        session_name: str,
        duration_seconds: float,
        son_score: int,
        opp_score: int,
        alerts: Optional[List[Dict]] = None,
        actions: Optional[List[Dict]] = None,
        son_fencer_id: str = "son",
        opp_fencer_id: str = "opponent",
        location: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> int:
        """
        Add a new session to the database.

        Args:
            session_name: Name of the session
            duration_seconds: Duration in seconds
            son_score: Son's final score
            opp_score: Opponent's final score
            alerts: List of alert dicts
            actions: List of action dicts
            son_fencer_id: Identifier for son
            opp_fencer_id: Identifier for opponent
            location: Optional location
            notes: Optional notes

        Returns:
            Session ID
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Insert session
        cursor.execute("""
            INSERT INTO sessions
            (session_name, date, duration_seconds, son_score, opp_score,
             son_fencer_id, opp_fencer_id, location, notes, alert_count, total_actions)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_name,
            datetime.now().isoformat(),
            duration_seconds,
            son_score,
            opp_score,
            son_fencer_id,
            opp_fencer_id,
            location,
            notes,
            len(alerts) if alerts else 0,
            len(actions) if actions else 0,
        ))
        session_id = cursor.lastrowid

        # Insert alerts
        if alerts:
            for alert in alerts:
                cursor.execute("""
                    INSERT INTO alerts
                    (session_id, timestamp, message, priority, category, fencer_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    alert.get("timestamp", 0),
                    alert.get("message", ""),
                    alert.get("priority", 3),
                    alert.get("category", "general"),
                    alert.get("fencer_id"),
                ))

        # Insert actions
        if actions:
            for action in actions:
                cursor.execute("""
                    INSERT INTO actions
                    (session_id, timestamp, action_type, confidence, fencer_id)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    session_id,
                    action.get("timestamp", 0),
                    action.get("action_type", "idle"),
                    action.get("confidence", 0.5),
                    action.get("fencer_id", 0),
                ))

        conn.commit()
        return session_id

    def get_sessions(
        self,
        fencer_id: Optional[str] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> List[SessionRecord]:
        """
        Get recent sessions.

        Args:
            fencer_id: Filter by fencer (son_fencer_id or opp_fencer_id)
            limit: Maximum number of sessions
            offset: Offset for pagination

        Returns:
            List of SessionRecord
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        query = "SELECT * FROM sessions"
        params: List[Any] = []

        if fencer_id:
            query += " WHERE son_fencer_id = ? OR opp_fencer_id = ?"
            params.extend([fencer_id, fencer_id])

        query += " ORDER BY date DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor.execute(query, params)
        rows = cursor.fetchall()

        return [
            SessionRecord(
                id=row["id"],
                session_name=row["session_name"],
                date=row["date"],
                duration_seconds=row["duration_seconds"],
                son_score=row["son_score"],
                opp_score=row["opp_score"],
                son_fencer_id=row["son_fencer_id"],
                opp_fencer_id=row["opp_fencer_id"],
                location=row["location"],
                notes=row["notes"],
                alert_count=row["alert_count"],
                total_actions=row["total_actions"],
            )
            for row in rows
        ]

    def get_session_alerts(self, session_id: int) -> List[AlertRecordDB]:
        """Get alerts for a session."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM alerts WHERE session_id = ? ORDER BY timestamp",
            (session_id,)
        )
        rows = cursor.fetchall()
        return [
            AlertRecordDB(
                id=row["id"],
                session_id=row["session_id"],
                timestamp=row["timestamp"],
                message=row["message"],
                priority=row["priority"],
                category=row["category"],
                fencer_id=row["fencer_id"],
            )
            for row in rows
        ]

    def get_session_actions(self, session_id: int) -> List[ActionRecordDB]:
        """Get actions for a session."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM actions WHERE session_id = ? ORDER BY timestamp",
            (session_id,)
        )
        rows = cursor.fetchall()
        return [
            ActionRecordDB(
                id=row["id"],
                session_id=row["session_id"],
                timestamp=row["timestamp"],
                action_type=row["action_type"],
                confidence=row["confidence"],
                fencer_id=row["fencer_id"],
            )
            for row in rows
        ]

    def get_alert_stats(self, session_id: int) -> Dict[str, int]:
        """Get alert statistics for a session."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT category, COUNT(*) as count
            FROM alerts
            WHERE session_id = ?
            GROUP BY category
        """, (session_id,))
        rows = cursor.fetchall()
        return {row["category"]: row["count"] for row in rows}

    def get_action_stats(self, session_id: int) -> Dict[str, int]:
        """Get action statistics for a session."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT action_type, COUNT(*) as count
            FROM actions
            WHERE session_id = ?
            GROUP BY action_type
        """, (session_id,))
        rows = cursor.fetchall()
        return {row["action_type"]: row["count"] for row in rows}

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.close()
