# MedTeach AI

Adaptive medical image interpretation trainer for radiology education. Students interpret X-ray images, receive AI-powered feedback with visual annotations, and the system adapts to focus on their weak areas.

Built for Professor Tian Zheng's Convergence Design Studio at Columbia University, as part of the aiX Faculty Fellowship Program.

## How it works

1. Student is shown a radiological image (chest X-ray or pediatric wrist X-ray)
2. System generates a question (multiple choice, open-ended, or true/false)
3. Student submits their interpretation
4. AI evaluates the response against ground truth annotations
5. Visual feedback overlays bounding boxes and heatmaps on the image
6. Bayesian skill model updates, shifting future image selection toward weak areas

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Add your Cerebras API key to .env
```

### Populating the image dataset

The app requires processed X-ray images in `data/sample_images/`. Raw datasets go in `GRAPE/` (GRAZPEDWRI-DX) and `VinBigData/` (VinBigData Chest X-ray), then run:

```bash
python download_images.py
```

This converts DICOMs and 16-bit PNGs to display-ready JPEGs, maps real bounding box annotations from the dataset CSVs/YOLO labels, and generates `data/metadata.json`.

### Running the app

```bash
streamlit run app.py
```

## Datasets

| Dataset | Type | Annotations | Source |
|---------|------|------------|--------|
| VinBigData Chest X-ray | Chest X-rays (DICOM) | Radiologist bounding boxes for 14 pathology classes | [Kaggle](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection) |
| GRAZPEDWRI-DX | Pediatric wrist X-rays (PNG) | YOLO bounding boxes for fractures, bone anomalies, soft tissue findings | [Figshare](https://figshare.com/articles/dataset/GRAZPEDWRI-DX/14825193) |

Bounding boxes from both datasets are real annotations by radiologists -- not synthetic or approximated.

## Architecture

| Component | File | Role |
|-----------|------|------|
| App | `app.py` | Streamlit UI, session flow |
| AI Engine | `ai_engine.py` | LLM integration (Cerebras API, OpenAI-compatible) for question generation and answer evaluation |
| Adaptive Selector | `adaptive_selector.py` | Beta-Binomial Bayesian model with Thompson sampling |
| Image Manager | `image_manager.py` | Image loading, bounding box overlay, heatmap generation |
| Session State | `session_state.py` | Student performance tracking |
| Config | `config.py` | Prompt templates, model settings, constants |
| Image Processor | `download_images.py` | Dataset conversion and metadata generation |

## Tech stack

- **Frontend**: Streamlit
- **AI**: Cerebras Cloud API (Qwen 3 235B)
- **Image processing**: Pillow, matplotlib, pydicom
- **Adaptive engine**: scipy, numpy (Beta distribution + Thompson sampling)
- **Python**: 3.10+

## Contributors

- Arjun Balaji (ab6136@columbia.edu)
- Xixi Chen (xc2829@columbia.edu)

## Acknowledgments

Professor Tian Zheng, Department of Statistics, Columbia University. Zohaib and Sharon (aiX Faculty Fellows) for the medical imaging education use case.
