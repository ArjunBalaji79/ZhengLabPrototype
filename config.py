"""Configuration and constants for MedTeach AI."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# API Keys
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "")
CEREBRAS_MODEL = "qwen-3-235b-a22b-instruct-2507"
CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
SAMPLE_IMAGES_DIR = DATA_DIR / "sample_images"
ASSETS_DIR = BASE_DIR / "assets"

# Diagnostic Categories
CATEGORIES = {
    "fracture": {
        "name": "Fracture Detection",
        "description": "Identifying bone fractures in X-ray images",
        "subcategories": ["displaced", "non-displaced", "hairline", "comminuted"]
    },
    "chest_pathology": {
        "name": "Chest Pathology",
        "description": "Identifying abnormalities in chest X-rays",
        "subcategories": ["cardiomegaly", "pneumothorax", "pleural_effusion", "consolidation", "normal"]
    },
    "dental": {
        "name": "Dental Pathology",
        "description": "Identifying dental abnormalities in X-rays",
        "subcategories": ["caries", "periodontal", "periapical", "impaction"]
    }
}

# Question Types
QUESTION_TYPES = ["multiple_choice", "open_ended", "true_false"]

# Adaptive Selection Parameters
PRIOR_ALPHA = 1.0  # Beta prior: initial correct count
PRIOR_BETA = 1.0   # Beta prior: initial incorrect count
EXPLORATION_WEIGHT = 0.3  # Thompson sampling exploration factor
MIN_IMAGES_BEFORE_ADAPTING = 3  # Minimum images before adaptive selection kicks in

# Feedback Thresholds
CORRECT_THRESHOLD = 0.8
ALMOST_THRESHOLD = 0.4

# UI Configuration
APP_TITLE = "MedTeach AI"
APP_SUBTITLE = "Adaptive Medical Image Interpretation Trainer"
APP_ICON = "🩻"
MAX_SESSION_IMAGES = 50

# Prompt Templates
SYSTEM_PROMPT = """You are MedTeach AI, an expert radiology educator. You help medical students learn to interpret radiological images through structured feedback.

Your role:
1. Evaluate student answers against ground truth findings
2. Provide constructive, educational feedback
3. Highlight what the student got right, what they missed, and what they misidentified
4. Reference specific regions of the image in your feedback
5. Encourage reflection and learning

Be supportive but accurate. Medical education requires precision — do not overlook errors, but frame feedback constructively."""

EVALUATION_PROMPT = """Evaluate the student's answer for this medical image.

**Ground Truth Findings:**
{ground_truth}

**Question Asked:**
{question}

**Student's Answer:**
{student_answer}

Respond in this exact JSON format:
{{
    "correctness_score": <float 0.0-1.0>,
    "verdict": "<CORRECT|ALMOST_RIGHT|INCORRECT>",
    "feedback": "<constructive feedback explaining what was right and wrong>",
    "missed_findings": ["<list of findings the student missed>"],
    "incorrect_claims": ["<list of incorrect statements the student made>"],
    "key_regions": ["<list of image regions the student should focus on>"],
    "reflection_prompt": "<a question to help the student reflect on their answer>",
    "teaching_point": "<one key educational takeaway>"
}}"""

SOCRATIC_MAX_TURNS = 3  # Maximum back-and-forth exchanges before revealing answer

SOCRATIC_HINT_PROMPT = """You are guiding a medical student toward the correct interpretation of a radiological image using the Socratic method. Do NOT reveal the answer directly. Instead, ask probing questions and give targeted hints to help them reason toward the correct findings.

**Ground Truth Findings:**
{ground_truth}

**Original Question:**
{question}

**Conversation so far:**
{conversation_history}

**Student's latest response:**
{student_response}

This is exchange {turn_number} of {max_turns}.

{turn_guidance}

Respond in this exact JSON format:
{{
    "hint_text": "<your Socratic response — a guiding question or hint, NOT the answer>",
    "student_on_track": <true if student is getting closer, false if going off track>,
    "estimated_closeness": <float 0.0-1.0 how close the student is to the correct answer>
}}"""

QUESTION_GEN_PROMPT = """Generate a {question_type} question for a medical student about this radiological image.

**Image Details:**
- Category: {category}
- Ground Truth: {ground_truth}

{type_specific_instructions}

Respond in this exact JSON format:
{{
    "question_text": "<the question>",
    "options": [<list of options if multiple choice, else empty list>],
    "correct_answer": "<the correct answer>",
    "explanation": "<brief explanation of the correct answer>",
    "difficulty": "<easy|medium|hard>"
}}"""
