# MedTeach AI — Prompt Architecture

This document describes every prompt used by the system, where it lives, and how the pieces fit together at runtime.

All prompt templates are defined in [config.py](config.py) and consumed by [ai_engine.py](ai_engine.py). The model is **Qwen 3 235B Instruct** (`qwen-3-235b-a22b-instruct-2507`) served via the Cerebras API through the OpenAI-compatible SDK.

---

## 1. Overview

The system uses **four distinct prompts**, each for a separate stage of the learning loop:

| # | Prompt | Purpose | Temperature |
|---|--------|---------|-------------|
| 1 | `SYSTEM_PROMPT` | Persistent role/persona for every call | — |
| 2 | `QUESTION_GEN_PROMPT` | Generates MCQ / open-ended / T-F questions from image metadata | 0.7 |
| 3 | `EVALUATION_PROMPT` | Grades the student's final answer against ground truth | 0.3 |
| 4 | `SOCRATIC_HINT_PROMPT` | Produces guiding hints without revealing the answer | 0.5 |

Every call pairs the system prompt with exactly one task prompt. Responses are requested as **strict JSON** and parsed by `_parse_json_response()` in [ai_engine.py:32](ai_engine.py#L32), which strips `<think>` reasoning blocks and markdown fences before `json.loads`.

---

## 2. The System Prompt

Defined at [config.py:59](config.py#L59). Sent as the `system` role message on every Cerebras call.

```
You are MedTeach AI, an expert radiology educator. You help medical students
learn to interpret radiological images through structured feedback.

Your role:
1. Evaluate student answers against ground truth findings
2. Provide constructive, educational feedback
3. Highlight what the student got right, what they missed, and what they misidentified
4. Reference specific regions of the image in your feedback
5. Encourage reflection and learning

Be supportive but accurate. Medical education requires precision — do not
overlook errors, but frame feedback constructively.
```

**Why this shape:** it anchors the model to a single persona (radiology educator), enumerates its responsibilities so the downstream JSON fields feel natural to fill, and explicitly trades off tone (supportive) against rigor (precise) — which matters in a medical context where a flattering grader would be actively harmful.

---

## 3. Question Generation

Defined at [config.py:120](config.py#L120). Called from `AIEngine.generate_question()` at [ai_engine.py:119](ai_engine.py#L119).

The template takes `category`, `ground_truth`, and a `question_type` (`multiple_choice`, `open_ended`, or `true_false`). Python code in `generate_question()` injects a `type_specific_instructions` block so a single template supports all three formats.

Returns JSON with `question_text`, `options`, `correct_answer`, `explanation`, `difficulty`. The higher temperature (0.7) encourages variety so repeat images don't produce identical stems.

**Why text-only:** the image is *not* sent to the model for question generation. The ground truth in `data/metadata.json` is authoritative — passing the pixels would just invite the model to hallucinate findings that contradict the label.

---

## 4. Answer Evaluation

Defined at [config.py:70](config.py#L70). Called from `AIEngine.evaluate_answer()` at [ai_engine.py:69](ai_engine.py#L69).

Inputs: `ground_truth`, `question`, `student_answer`. Returns JSON with:

- `correctness_score` — float 0.0–1.0
- `verdict` — `CORRECT` / `ALMOST_RIGHT` / `INCORRECT`
- `feedback` — prose explanation
- `missed_findings` — list
- `incorrect_claims` — list
- `key_regions` — image regions to focus on
- `reflection_prompt` — metacognitive question
- `teaching_point` — one-line takeaway

Temperature is low (0.3) for grading stability. If the model returns a score but omits `verdict`, [ai_engine.py:105-112](ai_engine.py#L105-L112) derives it from `CORRECT_THRESHOLD` (0.8) and `ALMOST_THRESHOLD` (0.4) defined in [config.py:49](config.py#L49).

**Why text-only evaluation:** same reason as question generation. The student sees the image in the Streamlit UI, but the grader compares their prose against the authoritative ground truth string. This avoids the model "seeing" something in the pixels that the label doesn't claim and then inventing a new correct answer mid-grading.

---

## 5. Socratic Hints

Defined at [config.py:95](config.py#L95). Called from `AIEngine.generate_socratic_hint()` at [ai_engine.py:164](ai_engine.py#L164).

This is the multi-turn loop. `SOCRATIC_MAX_TURNS = 3` (set at [config.py:93](config.py#L93)). On each turn:

1. Conversation history is flattened into a `Student: ... Tutor: ...` transcript.
2. `turn_guidance` is swapped depending on whether this is a mid-turn hint or the final exchange:
   - Mid-turn → focused hint, don't give it away
   - Final turn → "strong, nearly direct hint ... while still framing it as a question"
3. The template injects `turn_number`, `max_turns`, and the guidance string.

Returns `hint_text`, `student_on_track` (bool), `estimated_closeness` (0–1). Temperature 0.5 balances variety with coherence across turns.

**Why the turn-count escalation:** a fixed Socratic policy either gives up too early (frustrating students who are one nudge away) or stalls forever (frustrating students who are lost). Graduating the hint strength on the final turn guarantees forward progress while still preserving the discovery-learning arc on earlier turns.

---

## 6. JSON Parsing

All four prompts demand strict JSON responses. `_parse_json_response()` at [ai_engine.py:32](ai_engine.py#L32) handles three failure modes observed with Qwen:

1. **`<think>...</think>` reasoning blocks** — stripped with a regex before parsing.
2. **Markdown code fences** (` ```json ... ``` `) — stripped.
3. **Partial JSON with surrounding prose** — falls back to extracting the first `{...}` block.

If everything fails, it returns `{"error": ..., "raw": ...}` so callers can degrade gracefully.

---

## 7. Graceful Degradation (Demo Mode)

If `CEREBRAS_API_KEY` is unset, `AIEngine.is_available` returns `False` and every method routes to a mock implementation (`_mock_evaluation`, `_mock_question`, `_mock_socratic_hint`). The mocks use simple word-overlap scoring so the UI can be demoed end-to-end without hitting the API — useful for offline screenshots and faculty previews.

---

## 8. Data Flow Through a Single Session

```
1. adaptive_selector picks an image (Bayesian Beta-Binomial per category)
        │
        ▼
2. generate_question(image, category, ground_truth, type)
        → SYSTEM_PROMPT + QUESTION_GEN_PROMPT → Qwen → JSON question
        │
        ▼
3. Student answers in the Streamlit chat
        │
        ▼
4a. If answer looks partial → generate_socratic_hint(...) up to 3 turns
        → SYSTEM_PROMPT + SOCRATIC_HINT_PROMPT → Qwen → JSON hint
        │
        ▼
4b. Final answer → evaluate_answer(ground_truth, question, student_answer)
        → SYSTEM_PROMPT + EVALUATION_PROMPT → Qwen → JSON verdict
        │
        ▼
5. session_state updates Beta(α, β) for the category → loop to step 1
```

---

## 9. Design Constraints Worth Remembering

- **Ground truth is the source of truth, not the pixels.** The model grades prose-vs-prose. The image exists to train the student's eye, not the LLM.
- **Low temperature for grading, high temperature for question generation.** Consistency where it matters, variety where it doesn't.
- **Structured JSON everywhere.** No free-form responses — the UI depends on predictable field names.
- **Mock fallbacks for every method.** The prototype must always run, even without an API key, because demos happen on flaky wifi.
