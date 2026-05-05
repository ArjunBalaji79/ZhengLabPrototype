"""MedTeach AI - Adaptive Medical Image Interpretation Trainer.

An interactive AI-powered teaching system for medical image interpretation.
Built as a prototype for Professor Tian Zheng's Convergence Design Studio
at Columbia University.

Run: streamlit run app.py
"""

import json
import random
from pathlib import Path

import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image as PILImage, ImageDraw

from config import APP_TITLE, APP_SUBTITLE, APP_ICON, CATEGORIES, QUESTION_TYPES, ASSETS_DIR, SOCRATIC_MAX_TURNS
from adaptive_selector import AdaptiveSelector
from ai_engine import AIEngine
from image_manager import ImageManager
from session_state import SessionState, ResponseRecord


# ─── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
css_path = ASSETS_DIR / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ─── Initialize Session State ──────────────────────────────────────────────────

def init_session():
    """Initialize all session state variables."""
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.image_manager = ImageManager()
        st.session_state.ai_engine = AIEngine()
        st.session_state.session = SessionState()

        available_cats = st.session_state.image_manager.get_available_categories()
        if not available_cats:
            available_cats = list(CATEGORIES.keys())
        st.session_state.selector = AdaptiveSelector(available_cats)

        st.session_state.current_image = None
        st.session_state.current_question = None
        st.session_state.awaiting_answer = False
        st.session_state.show_feedback = False
        st.session_state.last_evaluation = None
        st.session_state.question_type_pref = "multiple_choice"
        st.session_state.phase = "welcome"  # welcome, question, guided, feedback, summary
        st.session_state.student_rects = []
        st.session_state.annotation_result = None
        st.session_state.socratic_history = []  # list of {"student": ..., "tutor": ...}
        st.session_state.socratic_turn = 0
        st.session_state.initial_answer = None


init_session()

im: ImageManager = st.session_state.image_manager
ai: AIEngine = st.session_state.ai_engine
session: SessionState = st.session_state.session
selector: AdaptiveSelector = st.session_state.selector


# ─── Helper Functions ──────────────────────────────────────────────────────────

def load_next_image():
    """Select and load the next image using adaptive selection."""
    category = selector.select_category()
    image = im.get_random_image(category)

    # Fallback: try any category
    if image is None:
        image = im.get_random_image()

    if image is None:
        st.error("No images available. Please add images to data/sample_images/")
        return False

    st.session_state.current_image = image

    # Generate question
    question = ai.generate_question(
        image_path=image.filepath,
        category=image.category,
        ground_truth=image.ground_truth,
        question_type=st.session_state.question_type_pref,
    )
    st.session_state.current_question = question
    st.session_state.awaiting_answer = True
    st.session_state.show_feedback = False
    st.session_state.last_evaluation = None
    st.session_state.phase = "question"
    st.session_state.socratic_history = []
    st.session_state.socratic_turn = 0
    st.session_state.initial_answer = None
    return True


def evaluate_answer(answer: str):
    """Evaluate the student's answer and show feedback."""
    image = st.session_state.current_image
    question = st.session_state.current_question

    evaluation = ai.evaluate_answer(
        image_path=image.filepath,
        ground_truth=image.ground_truth,
        question=question.get("question_text", ""),
        student_answer=answer,
    )

    # Blend annotation score into overall score (30% annotation, 70% answer)
    annotation_result = st.session_state.get("annotation_result")
    if annotation_result and image.annotations:
        ann_score = annotation_result["annotation_score"]
        text_score = evaluation.get("correctness_score", 0)
        blended = 0.7 * text_score + 0.3 * ann_score
        evaluation["correctness_score"] = round(blended, 2)
        evaluation["annotation_score"] = ann_score

        # Re-derive verdict from blended score
        from config import CORRECT_THRESHOLD, ALMOST_THRESHOLD
        if blended >= CORRECT_THRESHOLD:
            evaluation["verdict"] = "CORRECT"
        elif blended >= ALMOST_THRESHOLD:
            evaluation["verdict"] = "ALMOST_RIGHT"
        else:
            evaluation["verdict"] = "INCORRECT"

    # Update adaptive selector
    is_correct = evaluation.get("verdict") == "CORRECT"
    selector.update(image.category, is_correct)

    # Record response
    record = ResponseRecord(
        image_id=image.image_id,
        category=image.category,
        question_type=st.session_state.question_type_pref,
        question_text=question.get("question_text", ""),
        student_answer=answer,
        correctness_score=evaluation.get("correctness_score", 0),
        verdict=evaluation.get("verdict", "INCORRECT"),
        feedback=evaluation.get("feedback", ""),
    )
    session.add_response(record)

    st.session_state.last_evaluation = evaluation
    st.session_state.awaiting_answer = False
    st.session_state.show_feedback = True
    st.session_state.phase = "feedback"


# ─── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"# {APP_ICON} {APP_TITLE}")
    st.caption(APP_SUBTITLE)
    st.divider()

    # Session info
    st.markdown("### 📊 Session Progress")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total", session.total_questions)
    with col2:
        st.metric("Correct", session.correct_count)
    with col3:
        st.metric("Accuracy", f"{session.accuracy:.0%}" if session.total_questions > 0 else "—")

    if session.total_questions > 0:
        # Performance trend
        scores = session.get_recent_trend(10)
        if len(scores) > 1:
            fig, ax = plt.subplots(figsize=(4, 1.5))
            fig.patch.set_facecolor('#111827')
            ax.set_facecolor('#111827')
            ax.plot(scores, color='#60a5fa', linewidth=2, marker='o', markersize=4)
            ax.fill_between(range(len(scores)), scores, alpha=0.15, color='#60a5fa')
            ax.set_ylim(-0.05, 1.05)
            ax.set_ylabel('Score', color='#9ca3af', fontsize=8)
            ax.tick_params(colors='#9ca3af', labelsize=7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('#374151')
            ax.spines['left'].set_color('#374151')
            st.pyplot(fig, width="stretch")
            plt.close(fig)

    st.divider()

    # Adaptive Selection Visualization
    if selector.is_adapting:
        st.markdown("### 🎯 Adaptive Focus")
        st.caption("Image selection probability by category")
        probs = selector.get_selection_probabilities()
        for cat, prob in sorted(probs.items(), key=lambda x: -x[1]):
            cat_name = CATEGORIES.get(cat, {}).get("name", cat.replace("_", " ").title())
            st.progress(prob, text=f"{cat_name}: {prob:.0%}")

    st.divider()

    # Settings
    st.markdown("### ⚙️ Settings")
    qt = st.selectbox(
        "Question Type",
        options=QUESTION_TYPES,
        format_func=lambda x: x.replace("_", " ").title(),
        index=QUESTION_TYPES.index(st.session_state.question_type_pref),
    )
    st.session_state.question_type_pref = qt

    api_status = "🟢 Connected" if ai.is_available else "🟡 Demo Mode"
    st.caption(f"AI Engine: {api_status}")

    with st.expander("🔍 AI Diagnostics", expanded=not ai.is_available or bool(ai.last_error)):
        import os
        st.markdown("**Cerebras (eval + question gen)**")
        cerebras_key_present = bool(os.getenv("CEREBRAS_API_KEY"))
        cerebras_key_len = len(os.getenv("CEREBRAS_API_KEY", ""))
        st.write(f"- Env var set: {cerebras_key_present} (length {cerebras_key_len})")
        st.write(f"- Model: `{ai.model}`")
        st.write(f"- Client initialized: {ai.client is not None}")
        if ai.init_error:
            st.error(f"Init error: {ai.init_error}")

        st.markdown("**Gemini (Socratic hints)**")
        gemini_key_present = bool(os.getenv("GEMINI_API_KEY"))
        gemini_key_len = len(os.getenv("GEMINI_API_KEY", ""))
        st.write(f"- Env var set: {gemini_key_present} (length {gemini_key_len})")
        st.write(f"- Model: `{ai.gemini_model}`")
        st.write(f"- Client initialized: {ai.gemini_client is not None}")
        if ai.gemini_init_error:
            st.error(f"Init error: {ai.gemini_init_error}")

        if ai.last_error:
            st.error(f"Last API error: {ai.last_error}")
        if ai.is_available and ai.gemini_client is not None and not ai.last_error:
            st.success("No errors recorded yet.")

    st.divider()

    # Reset
    if st.button("🔄 Reset Session", width="stretch"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# ─── Main Content ──────────────────────────────────────────────────────────────

# Welcome Phase
if st.session_state.phase == "welcome":
    st.markdown(f"""
    # {APP_ICON} {APP_TITLE}
    ### {APP_SUBTITLE}

    Welcome to **MedTeach AI**, an adaptive learning system that helps you practice
    interpreting medical images. The system will:

    1. 🩻 Show you a radiological image
    2. ❓ Ask you to assess the findings
    3. 🤖 Evaluate your answer using AI
    4. 📝 Provide detailed feedback with visual annotations
    5. 🎯 **Adapt to your skill gaps** — focusing on areas where you need more practice

    ---
    """)

    col_start, col_space = st.columns([1, 2])
    with col_start:
        if st.button("▶️ Start Learning Session", width="stretch", type="primary"):
            if load_next_image():
                st.rerun()

    # Show available datasets
    with st.expander("📁 Available Image Sets"):
        cats = im.get_available_categories()
        if cats:
            for cat in cats:
                images = im.get_images_by_category(cat)
                cat_name = CATEGORIES.get(cat, {}).get("name", cat.replace("_", " ").title())
                st.write(f"**{cat_name}**: {len(images)} images")
        else:
            st.warning("No images loaded. Add images to `data/sample_images/` and update `data/metadata.json`.")


# Question Phase
elif st.session_state.phase == "question":
    image = st.session_state.current_image
    question = st.session_state.current_question

    if image and question:
        # Initialize annotation state for this image
        if "pending_clicks" not in st.session_state:
            st.session_state.pending_clicks = []  # list of (x, y) clicks
        if "drawn_rects" not in st.session_state:
            st.session_state.drawn_rects = []  # completed rectangles

        # Header with progress
        progress_text = f"Image {session.total_questions + 1}"
        if selector.is_adapting:
            progress_text += " -- Adaptive Selection Active"
        st.markdown(f"#### {progress_text}")

        col_img, col_q = st.columns([1, 1])

        with col_img:
            img = image.load()
            img_w, img_h = img.size

            # Draw existing rectangles and pending click on the image
            display_img = img.copy()
            draw = ImageDraw.Draw(display_img)

            for rect in st.session_state.drawn_rects:
                x, y, w, h = rect["x"], rect["y"], rect["w"], rect["h"]
                draw.rectangle([x, y, x + w, y + h], outline="yellow", width=2)

            if len(st.session_state.pending_clicks) == 1:
                px, py = st.session_state.pending_clicks[0]
                # Draw crosshair at first click
                draw.line([(px - 10, py), (px + 10, py)], fill="yellow", width=2)
                draw.line([(px, py - 10), (px, py + 10)], fill="yellow", width=2)

            st.markdown("##### Mark regions of concern")
            n_rects = len(st.session_state.drawn_rects)
            if len(st.session_state.pending_clicks) == 0:
                st.caption(f"Click the **top-left** corner of an abnormal region. ({n_rects} marked)")
            else:
                st.caption(f"Now click the **bottom-right** corner to complete the rectangle. ({n_rects} marked)")

            # Clickable image
            coords = streamlit_image_coordinates(
                display_img,
                key=f"img_click_{image.image_id}_{n_rects}_{len(st.session_state.pending_clicks)}",
            )

            if coords is not None:
                click_x, click_y = coords["x"], coords["y"]
                if len(st.session_state.pending_clicks) == 0:
                    # First click — top-left corner
                    st.session_state.pending_clicks = [(click_x, click_y)]
                    st.rerun()
                else:
                    # Second click — bottom-right corner, complete the rectangle
                    x1, y1 = st.session_state.pending_clicks[0]
                    x2, y2 = click_x, click_y
                    # Normalize so top-left < bottom-right
                    lx, ly = min(x1, x2), min(y1, y2)
                    rx, ry = max(x1, x2), max(y1, y2)
                    w, h = rx - lx, ry - ly
                    if w > 5 and h > 5:  # ignore tiny accidental clicks
                        st.session_state.drawn_rects.append(
                            {"x": lx, "y": ly, "w": w, "h": h}
                        )
                    st.session_state.pending_clicks = []
                    st.rerun()

            # Undo / clear buttons
            undo_col, clear_col = st.columns(2)
            with undo_col:
                if st.session_state.drawn_rects and st.button("Undo last"):
                    st.session_state.drawn_rects.pop()
                    st.session_state.pending_clicks = []
                    st.rerun()
            with clear_col:
                if st.session_state.drawn_rects and st.button("Clear all"):
                    st.session_state.drawn_rects = []
                    st.session_state.pending_clicks = []
                    st.rerun()

        with col_q:
            # Question
            q_text = question.get("question_text", "Describe what you observe in this image.")
            st.markdown("### Question")
            st.markdown(f"**{q_text}**")

            q_type = st.session_state.question_type_pref
            answer = None
            submit = False

            if q_type == "multiple_choice" and question.get("options"):
                answer = st.radio(
                    "Select your answer:",
                    options=question["options"],
                    index=None,
                    label_visibility="collapsed",
                )
                submit = answer and st.button("Submit Answer", type="primary", width="stretch")

            elif q_type == "true_false":
                tf_col1, tf_col2 = st.columns(2)
                with tf_col1:
                    if st.button("True", width="stretch"):
                        answer = "True"
                        submit = True
                with tf_col2:
                    if st.button("False", width="stretch"):
                        answer = "False"
                        submit = True

                justification = st.text_area(
                    "Provide your justification (optional):",
                    placeholder="Explain your reasoning...",
                    height=100,
                )
                if justification and st.button("Submit with Justification", type="primary", width="stretch"):
                    answer = f"{'True' if 'true' in justification.lower() else 'False'}. Justification: {justification}"
                    submit = True

            else:  # open_ended
                answer = st.text_area(
                    "Your assessment:",
                    placeholder="Describe any abnormalities you observe. Be specific about location, characteristics, and severity...",
                    height=150,
                )
                submit = answer and st.button("Submit Answer", type="primary", width="stretch")

            if submit and answer:
                student_rects = list(st.session_state.drawn_rects)
                st.session_state.student_rects = student_rects
                annotation_result = im.score_student_annotations(image, student_rects)
                st.session_state.annotation_result = annotation_result

                # Clear annotation state
                st.session_state.pending_clicks = []
                st.session_state.drawn_rects = []

                # Start Socratic guided phase instead of immediate evaluation
                st.session_state.initial_answer = answer
                st.session_state.socratic_turn = 1
                st.session_state.socratic_history = []

                # Generate first Socratic hint
                hint = ai.generate_socratic_hint(
                    ground_truth=image.ground_truth,
                    question=question.get("question_text", ""),
                    conversation_history=[],
                    student_response=answer,
                    turn_number=1,
                )
                st.session_state.socratic_history.append({
                    "student": answer,
                    "tutor": hint.get("hint_text", "Can you look more carefully at the image?"),
                })
                st.session_state.phase = "guided"
                st.rerun()

        # Skip button
        st.divider()
        if st.button("Skip this image"):
            st.session_state.student_rects = []
            st.session_state.annotation_result = None
            st.session_state.pending_clicks = []
            st.session_state.drawn_rects = []
            if load_next_image():
                st.rerun()


# Guided (Socratic) Phase
elif st.session_state.phase == "guided":
    image = st.session_state.current_image
    question = st.session_state.current_question

    if image and question:
        st.markdown("#### Guided Discussion")

        col_img_g, col_chat = st.columns([1, 1])

        with col_img_g:
            img = image.load()
            st.image(img, use_container_width=True)

        with col_chat:
            # Render conversation history as a chat
            for exchange in st.session_state.socratic_history:
                with st.chat_message("user"):
                    st.markdown(exchange["student"])
                with st.chat_message("assistant", avatar="🩻"):
                    st.markdown(exchange["tutor"])

            turn = st.session_state.socratic_turn
            remaining = SOCRATIC_MAX_TURNS - turn

            if remaining > 0:
                st.caption(f"{remaining} exchange{'s' if remaining != 1 else ''} remaining before the answer is revealed.")
                followup = st.text_area(
                    "Your response:",
                    placeholder="Revise your thinking based on the hint above...",
                    height=100,
                    key=f"socratic_input_{turn}",
                )
                if followup and st.button("Reply", type="primary", key=f"socratic_submit_{turn}"):
                    st.session_state.socratic_turn += 1
                    hint = ai.generate_socratic_hint(
                        ground_truth=image.ground_truth,
                        question=question.get("question_text", ""),
                        conversation_history=st.session_state.socratic_history,
                        student_response=followup,
                        turn_number=st.session_state.socratic_turn,
                    )
                    st.session_state.socratic_history.append({
                        "student": followup,
                        "tutor": hint.get("hint_text", "Think about what else you might be missing."),
                    })

                    # If this was the last turn, go to feedback
                    if st.session_state.socratic_turn >= SOCRATIC_MAX_TURNS:
                        # Build combined answer from all student responses for evaluation
                        all_student_text = " | ".join(
                            ex["student"] for ex in st.session_state.socratic_history
                        )
                        evaluate_answer(all_student_text)

                    st.rerun()
            else:
                # All turns used — show reveal button
                st.info("Let's see how you did. Click below to see the full feedback and correct answer.")
                if st.button("Show Answer & Feedback", type="primary", width="stretch"):
                    # Already evaluated on last turn; go to feedback
                    st.rerun()

        st.divider()
        # Allow skipping the guided discussion
        if st.button("Skip discussion — show answer now"):
            all_student_text = " | ".join(
                ex["student"] for ex in st.session_state.socratic_history
            )
            evaluate_answer(all_student_text)
            st.rerun()


# Feedback Phase
elif st.session_state.phase == "feedback":
    image = st.session_state.current_image
    evaluation = st.session_state.last_evaluation

    if image and evaluation:
        verdict = evaluation.get("verdict", "INCORRECT")

        # Verdict banner
        if verdict == "CORRECT":
            st.success("✅ **Correct!** Great job identifying the findings.")
        elif verdict == "ALMOST_RIGHT":
            st.warning("⚠️ **Almost Right!** You identified some findings but missed key details.")
        else:
            st.error("❌ **Incorrect.** Let's review the findings together.")

        # Annotation scoring
        annotation_result = st.session_state.get("annotation_result")
        student_rects = st.session_state.get("student_rects", [])

        if annotation_result:
            ann_score = annotation_result["annotation_score"]
            ann_detail = annotation_result["detail"]
            if ann_score >= 0.8:
                st.success(f"Region marking: **{ann_score:.0%}** -- {ann_detail}")
            elif ann_score >= 0.4:
                st.warning(f"Region marking: **{ann_score:.0%}** -- {ann_detail}")
            else:
                st.error(f"Region marking: **{ann_score:.0%}** -- {ann_detail}")

        # Visual feedback
        col_orig, col_annotated = st.columns(2)

        with col_orig:
            st.markdown("##### Your Marks")
            if student_rects and image.annotations:
                marked_img = im.annotate_with_student_marks(image, student_rects)
                st.image(marked_img, width="stretch")
                st.caption("Green = ground truth, Yellow = your marks")
            else:
                img = image.load()
                st.image(img, width="stretch")

        with col_annotated:
            st.markdown("##### Key Findings")
            if image.annotations:
                annotated = im.annotate_image(image)
                st.image(annotated, width="stretch")
            else:
                heatmap = im.generate_heatmap_overlay(image)
                st.image(heatmap, width="stretch")

        st.divider()

        # Detailed feedback
        col_fb, col_ref = st.columns([2, 1])

        with col_fb:
            st.markdown("### 📝 Feedback")
            st.markdown(evaluation.get("feedback", ""))

            # Missed findings
            missed = evaluation.get("missed_findings", [])
            if missed:
                st.markdown("**Missed Findings:**")
                for finding in missed:
                    st.markdown(f"- 🔍 {finding}")

            # Incorrect claims
            incorrect = evaluation.get("incorrect_claims", [])
            if incorrect:
                st.markdown("**Incorrect Statements:**")
                for claim in incorrect:
                    st.markdown(f"- ⚠️ {claim}")

            # Teaching point
            teaching = evaluation.get("teaching_point", "")
            if teaching:
                st.info(f"💡 **Key Takeaway:** {teaching}")

        with col_ref:
            st.markdown("### 🪞 Reflection")
            reflection_prompt = evaluation.get(
                "reflection_prompt",
                "What did you notice first? What might you have overlooked?",
            )
            st.markdown(f"*{reflection_prompt}*")

            reflection = st.text_area(
                "Your reflection:",
                placeholder="Think about what you noticed, missed, or misread...",
                height=120,
                key="reflection_box",
            )

            # Ground truth reveal
            with st.expander("📋 Full Ground Truth"):
                st.markdown(f"**{image.ground_truth}**")

        st.divider()

        # Navigation
        col_next, col_summary = st.columns(2)

        with col_next:
            if st.button("➡️ Next Image", type="primary", width="stretch"):
                if load_next_image():
                    st.rerun()

        with col_summary:
            if session.total_questions >= 3:
                if st.button("📊 View Session Summary", width="stretch"):
                    st.session_state.phase = "summary"
                    st.rerun()


# Summary Phase
elif st.session_state.phase == "summary":
    st.markdown(f"# 📊 Session Summary")
    st.markdown(f"**{session.total_questions}** images reviewed")

    # Overall metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Questions", session.total_questions)
    with col2:
        st.metric("Correct", session.correct_count)
    with col3:
        st.metric("Almost Right", session.almost_right_count)
    with col4:
        st.metric("Overall Accuracy", f"{session.accuracy:.0%}")

    st.divider()

    # Category breakdown
    st.markdown("### Performance by Category")
    cat_perf = session.get_category_performance()

    if cat_perf:
        cols = st.columns(len(cat_perf))
        for i, (cat, perf) in enumerate(cat_perf.items()):
            cat_name = CATEGORIES.get(cat, {}).get("name", cat.replace("_", " ").title())
            with cols[i]:
                st.markdown(f"**{cat_name}**")
                st.metric("Accuracy", f"{perf['accuracy']:.0%}")
                st.metric("Questions", perf["total"])
                st.progress(perf["accuracy"])

    st.divider()

    # Skill gaps
    st.markdown("### 🎯 Areas for Improvement")
    weak = selector.get_weakest_categories(3)
    for cat in weak:
        skill = selector.skills[cat]
        cat_name = CATEGORIES.get(cat, {}).get("name", cat.replace("_", " ").title())
        st.markdown(
            f"- **{cat_name}**: estimated accuracy {skill.estimated_accuracy:.0%} "
            f"({skill.total_seen} images seen)"
        )

    # Adaptive selection visualization
    if selector.is_adapting:
        st.markdown("### 📈 Adaptive Selection Probabilities")
        probs = selector.get_selection_probabilities()
        fig, ax = plt.subplots(figsize=(8, 3))
        fig.patch.set_facecolor('#111827')
        ax.set_facecolor('#111827')

        cats = list(probs.keys())
        vals = [probs[c] for c in cats]
        names = [CATEGORIES.get(c, {}).get("name", c.replace("_", " ").title()) for c in cats]

        bars = ax.barh(names, vals, color='#3b82f6', edgecolor='#60a5fa', linewidth=0.5)
        ax.set_xlabel("Selection Probability", color='#9ca3af')
        ax.tick_params(colors='#9ca3af')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#374151')
        ax.spines['left'].set_color('#374151')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:.0%}', va='center', color='#d1d5db', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    st.divider()

    # Response history
    with st.expander("📜 Full Response History"):
        for i, r in enumerate(session.responses, 1):
            icon = {"CORRECT": "✅", "ALMOST_RIGHT": "⚠️", "INCORRECT": "❌"}.get(r.verdict, "❓")
            cat_name = CATEGORIES.get(r.category, {}).get("name", r.category)
            st.markdown(f"**{i}. {icon} {cat_name}** — Score: {r.correctness_score:.0%}")
            st.caption(f"Q: {r.question_text[:100]}...")
            st.caption(f"A: {r.student_answer[:100]}...")
            st.divider()

    # Continue or reset
    col_cont, col_reset = st.columns(2)
    with col_cont:
        if st.button("➡️ Continue Learning", type="primary", width="stretch"):
            if load_next_image():
                st.rerun()
    with col_reset:
        if st.button("🔄 Start New Session", width="stretch"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
