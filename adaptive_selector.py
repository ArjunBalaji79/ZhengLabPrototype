"""Adaptive image selection engine using Bayesian skill modeling.

Uses a Beta-Binomial model per diagnostic category to track student
performance and adaptively select images targeting skill gaps.
Implements Thompson sampling for exploration-exploitation balance.
"""

import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats

from config import (
    PRIOR_ALPHA,
    PRIOR_BETA,
    EXPLORATION_WEIGHT,
    MIN_IMAGES_BEFORE_ADAPTING,
)


@dataclass
class CategorySkill:
    """Tracks a student's skill in a specific diagnostic category."""

    category: str
    alpha: float = PRIOR_ALPHA  # Pseudo-count of correct answers
    beta: float = PRIOR_BETA   # Pseudo-count of incorrect answers
    total_seen: int = 0
    history: list = field(default_factory=list)

    @property
    def estimated_accuracy(self) -> float:
        """Expected accuracy (mean of Beta distribution)."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def estimated_error_rate(self) -> float:
        """Expected error rate = 1 - accuracy."""
        return self.beta / (self.alpha + self.beta)

    @property
    def uncertainty(self) -> float:
        """Variance of the Beta distribution — higher = less certain."""
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    def update(self, correct: bool) -> None:
        """Update skill model after a student response."""
        if correct:
            self.alpha += 1.0
        else:
            self.beta += 1.0
        self.total_seen += 1
        self.history.append(correct)

    def sample_error_rate(self) -> float:
        """Thompson sampling: draw from posterior to estimate error rate."""
        # Sample accuracy from Beta(alpha, beta), return 1 - accuracy
        sampled_accuracy = np.random.beta(self.alpha, self.beta)
        return 1.0 - sampled_accuracy

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "alpha": self.alpha,
            "beta": self.beta,
            "total_seen": self.total_seen,
            "estimated_accuracy": round(self.estimated_accuracy, 3),
            "estimated_error_rate": round(self.estimated_error_rate, 3),
        }


class AdaptiveSelector:
    """Selects images adaptively based on inferred student skill gaps.

    Strategy:
    - For the first MIN_IMAGES_BEFORE_ADAPTING images: uniform random selection
    - After that: Thompson sampling — draw error rates from posterior,
      select category with highest sampled error rate (= most likely to struggle)
    - Exploration factor ensures we don't get stuck in one category
    """

    def __init__(self, categories: list[str]):
        self.skills: dict[str, CategorySkill] = {
            cat: CategorySkill(category=cat) for cat in categories
        }
        self.total_images_shown = 0

    @property
    def is_adapting(self) -> bool:
        """Whether we have enough data to start adaptive selection."""
        return self.total_images_shown >= MIN_IMAGES_BEFORE_ADAPTING

    def select_category(self) -> str:
        """Select the next diagnostic category to test the student on."""
        self.total_images_shown += 1

        # Phase 1: Uniform exploration
        if not self.is_adapting:
            return random.choice(list(self.skills.keys()))

        # Phase 2: Thompson sampling with exploration
        if random.random() < EXPLORATION_WEIGHT:
            # Explore: pick a random category
            return random.choice(list(self.skills.keys()))

        # Exploit: sample from posteriors, pick highest error rate
        sampled_errors = {
            cat: skill.sample_error_rate()
            for cat, skill in self.skills.items()
        }
        return max(sampled_errors, key=sampled_errors.get)

    def update(self, category: str, correct: bool) -> None:
        """Update the skill model after a student response."""
        if category in self.skills:
            self.skills[category].update(correct)

    def get_skill_summary(self) -> list[dict]:
        """Get a summary of the student's skill across all categories."""
        return [skill.to_dict() for skill in self.skills.values()]

    def get_weakest_categories(self, n: int = 3) -> list[str]:
        """Return the n categories with highest estimated error rate."""
        sorted_skills = sorted(
            self.skills.values(),
            key=lambda s: s.estimated_error_rate,
            reverse=True,
        )
        return [s.category for s in sorted_skills[:n]]

    def get_selection_probabilities(self) -> dict[str, float]:
        """Get the current selection probability for each category.

        Useful for visualization — shows how the adaptive engine
        is shifting its focus.
        """
        if not self.is_adapting:
            n = len(self.skills)
            return {cat: 1.0 / n for cat in self.skills}

        # Approximate selection probabilities via error rates
        error_rates = {
            cat: skill.estimated_error_rate
            for cat, skill in self.skills.items()
        }

        # Mix with exploration
        uniform = 1.0 / len(self.skills)
        total_error = sum(error_rates.values())

        if total_error == 0:
            return {cat: uniform for cat in self.skills}

        probs = {}
        for cat in self.skills:
            exploit_prob = error_rates[cat] / total_error
            probs[cat] = (
                EXPLORATION_WEIGHT * uniform
                + (1 - EXPLORATION_WEIGHT) * exploit_prob
            )

        # Normalize
        total = sum(probs.values())
        return {cat: p / total for cat, p in probs.items()}
