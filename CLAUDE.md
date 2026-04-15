# MedTeach AI - Interactive Medical Image Education System

## Project Overview

An adaptive AI-powered teaching system for medical image interpretation. Built as a prototype for Professor Tian Zheng's Convergence Design Studio at Columbia University, in collaboration with the aiX Faculty Fellowship Program.

The system helps medical students learn to interpret radiological images (chest X-rays, pediatric fracture X-rays, dental X-rays) through an interactive, chat-based learning experience with adaptive difficulty.

## Core Requirements (from Professor Zheng's spec)

1. **Randomly show an image to a student**
2. **Ask the student to assess the image** — multiple choice, open-ended questions, or T/F with justification
3. **AI analyzes the student's answer** for accuracy via chat
4. **Feedback with visual evidence** — right/almost right/wrong, plus annotations or heatmaps on the image tied to the correct answer. Reflection prompts for what they noticed, missed, or misread.
5. **Adaptive image selection** — infer student skill gaps after a few images; shift selection probability toward images where the student is more likely to make mistakes.

## Architecture

### Tech Stack
- **Frontend**: Streamlit (for rapid prototyping; Professor Zheng's lab is familiar with it)
- **Backend**: Python
- **AI**: OpenAI GPT-4o (multimodal — handles both image analysis and text evaluation) via API
- **Image Processing**: Pillow, matplotlib for heatmap/annotation overlays
- **Data**: GRAZPEDWRI-DX pediatric fracture dataset (freely downloadable, has bounding box ground truth)
- **Adaptive Engine**: Bayesian skill model tracking per-category performance

### Project Structure

```
medteach-ai/
├── CLAUDE.md                    # This file
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variables template
├── app.py                       # Main Streamlit application
├── config.py                    # Configuration and constants
├── src/
│   ├── __init__.py
│   ├── ai_engine.py             # Multimodal LLM integration for answer evaluation
│   ├── adaptive_selector.py     # Bayesian skill model + adaptive image selection
│   ├── image_manager.py         # Image loading, annotation overlay, heatmap generation
│   ├── question_generator.py    # Question generation (MCQ, open-ended, T/F)
│   ├── feedback_engine.py       # Feedback generation with visual evidence
│   └── session_state.py         # Student session and performance tracking
├── data/
│   ├── sample_images/           # Sample X-ray images for demo (subset)
│   └── metadata.json            # Image metadata, labels, ground truth
├── assets/
│   └── style.css                # Custom Streamlit styling
└── tests/
    └── test_adaptive.py         # Tests for adaptive selection engine
```

## Key Design Decisions

### Image Selection Strategy
- Use a **Beta-Binomial Bayesian model** per diagnostic category
- Each category (e.g., "fracture detection", "cardiomegaly", "pneumothorax") has a Beta(α, β) prior
- After each student response: update α (correct) or β (incorrect)
- Selection probability ∝ P(mistake) = β / (α + β) — higher error rate categories get sampled more
- Add exploration factor (Thompson sampling) to avoid getting stuck

### Question Types
1. **Multiple Choice**: "What abnormality is most likely present in this image?" with 4 options
2. **Open-Ended**: "Describe any abnormalities you observe in this chest X-ray."
3. **True/False + Justification**: "This X-ray shows a displaced fracture of the radius. True or False? Explain your reasoning."

### Answer Evaluation
- Send the image + student answer + ground truth to GPT-4o
- Structured evaluation: correctness score (0-1), specific feedback, missed findings, visual regions to highlight
- Three-tier feedback: ✅ Correct, ⚠️ Almost Right, ❌ Incorrect

### Visual Feedback
- Overlay bounding boxes from ground truth annotations
- Generate attention heatmaps highlighting relevant regions
- Side-by-side: student's focus vs. correct findings

## Development Guidelines

### Code Style
- Python 3.10+
- Type hints everywhere
- Docstrings for all public functions
- Keep Streamlit components modular and reusable

### Environment Variables
- `OPENAI_API_KEY` — for GPT-4o multimodal calls
- `DATA_DIR` — path to image dataset

### Running the App
```bash
streamlit run app.py
```

### Demo Mode
The app should work in demo mode with sample images even without the full dataset downloaded. Include 5-10 sample X-ray images in `data/sample_images/` for immediate demonstration.

## Important Notes

- This is a PROTOTYPE / MOCKUP — optimize for demonstrating the concept, not production scale
- The slides attached by Zohaib (Vision Transformers for Fracture Detection) are CONFIDENTIAL — do not share
- MIMIC-CXR requires CITI training for access — use GRAZPEDWRI-DX or sample images for the prototype
- Focus on the INTERACTION FLOW first, then polish the AI accuracy
- Make it visually impressive — this needs to wow faculty fellows (Zohaib, Sharon) and Professor Zheng
