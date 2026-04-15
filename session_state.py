"""Session state management for student learning sessions.

Tracks:
- Current image and question
- Response history
- Performance metrics over time
- Adaptive selector state
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class ResponseRecord:
    """A single student response record."""

    image_id: str
    category: str
    question_type: str
    question_text: str
    student_answer: str
    correctness_score: float
    verdict: str
    feedback: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def is_correct(self) -> bool:
        return self.verdict == "CORRECT"


@dataclass
class SessionState:
    """Manages the state of a student learning session."""

    session_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    student_name: str = "Student"
    responses: list[ResponseRecord] = field(default_factory=list)
    current_image_id: Optional[str] = None
    current_question: Optional[dict] = None
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    images_seen: set = field(default_factory=set)

    @property
    def total_questions(self) -> int:
        return len(self.responses)

    @property
    def correct_count(self) -> int:
        return sum(1 for r in self.responses if r.is_correct)

    @property
    def accuracy(self) -> float:
        if not self.responses:
            return 0.0
        return self.correct_count / self.total_questions

    @property
    def almost_right_count(self) -> int:
        return sum(1 for r in self.responses if r.verdict == "ALMOST_RIGHT")

    @property
    def incorrect_count(self) -> int:
        return sum(1 for r in self.responses if r.verdict == "INCORRECT")

    def add_response(self, record: ResponseRecord) -> None:
        self.responses.append(record)
        self.images_seen.add(record.image_id)

    def get_category_performance(self) -> dict[str, dict]:
        """Get performance breakdown by category."""
        cats: dict[str, dict] = {}
        for r in self.responses:
            if r.category not in cats:
                cats[r.category] = {"total": 0, "correct": 0, "scores": []}
            cats[r.category]["total"] += 1
            if r.is_correct:
                cats[r.category]["correct"] += 1
            cats[r.category]["scores"].append(r.correctness_score)

        for cat in cats:
            total = cats[cat]["total"]
            cats[cat]["accuracy"] = cats[cat]["correct"] / total if total > 0 else 0
            cats[cat]["avg_score"] = (
                sum(cats[cat]["scores"]) / len(cats[cat]["scores"])
                if cats[cat]["scores"]
                else 0
            )

        return cats

    def get_recent_trend(self, n: int = 5) -> list[float]:
        """Get the last n correctness scores to show improvement trend."""
        return [r.correctness_score for r in self.responses[-n:]]
