import uuid
import threading
from typing import Dict, Optional, List
from datetime import datetime, timezone
from app.models import (
    QuestionEvaluation,
    PerformanceReport,
    VivaHistoryResponse
)


class SessionData:
    def __init__(self, session_id: str, student_name: str, subject: str, difficulty: str, first_question: str):
        self.session_id: str = session_id
        self.student_name: str = student_name
        self.subject: str = subject
        self.difficulty: str = difficulty
        self.status: str = "in_progress"
        self.current_question_number: int = 1
        self.current_question: str = first_question
        self.evaluations: List[QuestionEvaluation] = []
        self.performance_report: Optional[PerformanceReport] = None
        self.created_at: datetime = datetime.now(timezone.utc)

    @property
    def is_completed(self) -> bool:
        return self.status == "completed" or len(self.evaluations) >= 3


class SessionStore:
    def __init__(self):
        self._sessions: Dict[str, SessionData] = {}
        self._lock = threading.Lock()

    def create_session(self, student_name: str, subject: str, difficulty: str, first_question: str) -> SessionData:
        session_id = str(uuid.uuid4())
        session = SessionData(
            session_id=session_id,
            student_name=student_name,
            subject=subject,
            difficulty=difficulty,
            first_question=first_question
        )
        with self._lock:
            self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[SessionData]:
        with self._lock:
            return self._sessions.get(session_id)

    def record_answer_and_evaluation(
        self,
        session_id: str,
        evaluation: QuestionEvaluation,
        next_question: Optional[str] = None
    ) -> Optional[SessionData]:
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return None

            session.evaluations.append(evaluation)

            if len(session.evaluations) >= 3:
                session.status = "completed"
                session.current_question = None
            else:
                session.current_question_number = len(session.evaluations) + 1
                session.current_question = next_question

            return session

    def attach_performance_report(self, session_id: str, report: PerformanceReport) -> bool:
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False
            session.performance_report = report
            session.status = "completed"
            return True

    def clear(self):
        """Helper for testing to reset sessions."""
        with self._lock:
            self._sessions.clear()


# Global store instance
session_store = SessionStore()
