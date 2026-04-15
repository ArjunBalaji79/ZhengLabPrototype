# Next Steps

## 1. Expand the image dataset

The prototype runs on ~9 images. Download more from VinBigData (15k chest X-rays with bounding boxes) and GRAZPEDWRI-DX (20k wrist X-rays with YOLO labels). Both datasets are freely available. The `download_images.py` script already handles batch conversion -- just add more raw files to the `VinBigData/` and `GRAPE/` directories and re-run it.

## 2. Validate the AI model choice

The app currently targets Cerebras Cloud API with Qwen 3 235B. This model has not been confirmed to support vision/multimodal input. If it doesn't, switch to one of:
- **Google Gemini 2.0 Flash** via `google-genai` SDK (strong multimodal, free tier available)
- **OpenAI GPT-4o** via `openai` SDK (best multimodal quality, requires paid API key)

The `ai_engine.py` file uses the OpenAI-compatible client format, so switching providers only requires changing the base URL and model name in `config.py`.

## 3. Add a dental X-ray module

Dental was dropped from the prototype due to lack of annotated datasets. The best candidate is the DENTEX challenge dataset (panoramic dental X-rays with bounding boxes for caries, periapical lesions, and impacted teeth). Available via Grand Challenge. Adding it means: downloading the data, adding a processing function in `download_images.py`, and adding `"dental"` entries to the metadata.

## 4. Fix Streamlit deprecation warnings

Replace all `use_container_width=True/False` calls in `app.py` with the new `width` parameter (`width="stretch"` or `width="content"`). These are cosmetic but will break after the Streamlit deprecation deadline.
