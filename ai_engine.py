"""AI Engine for answer evaluation and question generation.

Uses Cerebras API (OpenAI-compatible) with Qwen 3 235B to:
1. Evaluate student answers against ground truth
2. Generate questions based on image metadata
3. Provide educational feedback
"""

import json
import re
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from config import (
    CEREBRAS_API_KEY,
    CEREBRAS_MODEL,
    CEREBRAS_BASE_URL,
    SYSTEM_PROMPT,
    EVALUATION_PROMPT,
    QUESTION_GEN_PROMPT,
    SOCRATIC_HINT_PROMPT,
    SOCRATIC_MAX_TURNS,
    CORRECT_THRESHOLD,
    ALMOST_THRESHOLD,
)


def _parse_json_response(text: str) -> dict:
    """Robustly parse JSON from LLM response, handling markdown fences and think tags."""
    text = text.strip()
    # Strip <think>...</think> blocks (Qwen reasoning mode)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {"error": "Failed to parse response", "raw": text}


class AIEngine:
    """Handles all AI interactions for the teaching system."""

    def __init__(self, api_key: str = CEREBRAS_API_KEY):
        self.last_error: str | None = None
        self.init_error: str | None = None
        if OpenAI is None:
            self.client = None
            self.init_error = "openai package not installed"
        elif not api_key:
            self.client = None
            self.init_error = "CEREBRAS_API_KEY is empty — secret not loaded"
        else:
            try:
                self.client = OpenAI(
                    api_key=api_key,
                    base_url=CEREBRAS_BASE_URL,
                )
            except Exception as e:
                self.client = None
                self.init_error = f"OpenAI client init failed: {type(e).__name__}: {e}"
        self.model = CEREBRAS_MODEL

    @property
    def is_available(self) -> bool:
        return self.client is not None and CEREBRAS_API_KEY != ""

    def _record_error(self, where: str, exc: Exception) -> None:
        self.last_error = f"{where}: {type(exc).__name__}: {exc}"
        print(f"AI {where} error: {exc}")

    def evaluate_answer(
        self,
        image_path: str | Path,
        ground_truth: str,
        question: str,
        student_answer: str,
    ) -> dict:
        """Evaluate a student's answer against ground truth using Qwen 3 235B.

        Text-only evaluation — the model compares the student's answer
        to the ground truth findings. The image is shown to the student
        in the UI but not sent to the model.
        """
        if not self.is_available:
            return self._mock_evaluation(ground_truth, student_answer)

        prompt = EVALUATION_PROMPT.format(
            ground_truth=ground_truth,
            question=question,
            student_answer=student_answer,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1000,
            )

            result = _parse_json_response(response.choices[0].message.content)

            # Ensure verdict is set based on score if not parsed
            if "correctness_score" in result and "verdict" not in result:
                score = result["correctness_score"]
                if score >= CORRECT_THRESHOLD:
                    result["verdict"] = "CORRECT"
                elif score >= ALMOST_THRESHOLD:
                    result["verdict"] = "ALMOST_RIGHT"
                else:
                    result["verdict"] = "INCORRECT"

            self.last_error = None
            return result
        except Exception as e:
            self._record_error("evaluation", e)
            return self._mock_evaluation(ground_truth, student_answer)

    def generate_question(
        self,
        image_path: str | Path,
        category: str,
        ground_truth: str,
        question_type: str = "multiple_choice",
    ) -> dict:
        """Generate a question about a medical image.

        Uses ground truth metadata to generate contextual questions.
        """
        if not self.is_available:
            return self._mock_question(category, ground_truth, question_type)

        type_instructions = {
            "multiple_choice": "Generate a multiple choice question with exactly 4 options (A-D). Only one should be correct.",
            "open_ended": "Generate an open-ended question that asks the student to describe what they observe. The correct_answer should be the ideal response.",
            "true_false": "Generate a statement about the image that is either true or false. Include a claim that the student must evaluate. The correct_answer should be 'True' or 'False'.",
        }

        prompt = QUESTION_GEN_PROMPT.format(
            question_type=question_type,
            category=category,
            ground_truth=ground_truth,
            type_specific_instructions=type_instructions.get(
                question_type, type_instructions["multiple_choice"]
            ),
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=800,
            )

            self.last_error = None
            return _parse_json_response(response.choices[0].message.content)
        except Exception as e:
            self._record_error("question_generation", e)
            return self._mock_question(category, ground_truth, question_type)

    def generate_socratic_hint(
        self,
        ground_truth: str,
        question: str,
        conversation_history: list[dict],
        student_response: str,
        turn_number: int,
    ) -> dict:
        """Generate a Socratic hint that guides the student without revealing the answer."""
        max_turns = SOCRATIC_MAX_TURNS

        # Build conversation history string
        history_lines = []
        for exchange in conversation_history:
            history_lines.append(f"Student: {exchange['student']}")
            history_lines.append(f"Tutor: {exchange['tutor']}")
        history_str = "\n".join(history_lines) if history_lines else "(This is the student's first answer.)"

        # Vary guidance based on turn number
        if turn_number < max_turns:
            turn_guidance = (
                "Give a focused hint or ask a probing question to nudge the student "
                "toward what they're missing. Be encouraging but don't give it away."
            )
        else:
            turn_guidance = (
                "This is the final exchange. Give a strong, nearly direct hint that "
                "points clearly at what the student is missing, while still framing "
                "it as a question. After this, the full answer will be revealed."
            )

        if not self.is_available:
            return self._mock_socratic_hint(ground_truth, student_response, turn_number, max_turns)

        prompt = SOCRATIC_HINT_PROMPT.format(
            ground_truth=ground_truth,
            question=question,
            conversation_history=history_str,
            student_response=student_response,
            turn_number=turn_number,
            max_turns=max_turns,
            turn_guidance=turn_guidance,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.5,
                max_tokens=500,
            )
            self.last_error = None
            return _parse_json_response(response.choices[0].message.content)
        except Exception as e:
            self._record_error("socratic_hint", e)
            return self._mock_socratic_hint(ground_truth, student_response, turn_number, max_turns)

    def _mock_socratic_hint(
        self, ground_truth: str, student_response: str, turn: int, max_turns: int
    ) -> dict:
        """Mock Socratic hint for demo mode."""
        hints = [
            "That's a start — but look more carefully at the edges of the bone. Do you notice any disruption in the cortex?",
            "You're getting closer. Think about the specific location — is the abnormality more proximal or distal? What does that tell you about the mechanism?",
            "Almost there. Consider: if you trace the outline of the structure, where exactly does the continuity break? What type of finding would that suggest?",
        ]
        idx = min(turn - 1, len(hints) - 1)
        return {
            "hint_text": f"[Demo Mode] {hints[idx]}",
            "student_on_track": turn > 1,
            "estimated_closeness": min(0.3 * turn, 0.9),
        }

    def _mock_evaluation(self, ground_truth: str, student_answer: str) -> dict:
        """Mock evaluation for demo mode without API key."""
        gt_words = set(ground_truth.lower().split())
        sa_words = set(student_answer.lower().split())
        overlap = len(gt_words & sa_words) / max(len(gt_words), 1)

        if overlap > 0.5:
            verdict = "CORRECT"
            score = 0.85
        elif overlap > 0.2:
            verdict = "ALMOST_RIGHT"
            score = 0.55
        else:
            verdict = "INCORRECT"
            score = 0.15

        return {
            "correctness_score": score,
            "verdict": verdict,
            "feedback": f"[Demo Mode] Your answer showed {'good' if score > 0.5 else 'partial'} understanding. The key findings were: {ground_truth}",
            "missed_findings": ["See ground truth for complete findings"] if score < 0.8 else [],
            "incorrect_claims": [],
            "key_regions": ["central region", "peripheral areas"],
            "reflection_prompt": "What specific features in the image led you to your conclusion?",
            "teaching_point": f"Key finding: {ground_truth}",
        }

    def _mock_question(
        self, category: str, ground_truth: str, question_type: str
    ) -> dict:
        """Mock question generation for demo mode."""
        if question_type == "multiple_choice":
            return {
                "question_text": f"What is the most likely finding in this {category} image?",
                "options": [
                    f"A) {ground_truth}",
                    "B) No significant abnormality detected",
                    "C) Motion artifact obscuring findings",
                    "D) Indeterminate — further imaging required",
                ],
                "correct_answer": f"A) {ground_truth}",
                "explanation": f"The image shows {ground_truth}.",
                "difficulty": "medium",
            }
        elif question_type == "true_false":
            return {
                "question_text": f"True or False: This image shows {ground_truth}.",
                "options": ["True", "False"],
                "correct_answer": "True",
                "explanation": f"The image indeed shows {ground_truth}.",
                "difficulty": "medium",
            }
        else:
            return {
                "question_text": "Describe any abnormalities you observe in this image. Be specific about location and characteristics.",
                "options": [],
                "correct_answer": ground_truth,
                "explanation": f"The key findings include: {ground_truth}.",
                "difficulty": "medium",
            }
