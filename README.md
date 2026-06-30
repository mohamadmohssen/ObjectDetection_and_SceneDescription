# YOLO Scene Captioning

A live camera app that detects objects (YOLO), writes a caption for the scene (GIT model),
then rewrites that caption in a chosen personality style (funny, serious, coach, etc.)
using a local LLM through Ollama.

## What you need before you start

1. Python 3.10+ installed
2. A webcam
3. Ollama installed and running
4. Internet connection for the very first run only (to download the YOLO model files)

## Step 1 — Install Ollama

Download and install it from https://ollama.com

After installing, pull the model this project uses:

```
ollama pull llama3
```

Make sure Ollama is running in the background before you start the app.
On most systems it starts automatically after install. If not, run:

```
ollama serve
```

## Step 2 — Install Python packages

From the project folder, run:

```
pip install -r requirements.txt
```

## Step 3 — Get the GIT caption model

The caption model (microsoft/git-base) is set to load from your local Hugging Face
cache only - it will not auto-download. Before first run, download it once with
internet access:

```
python -c "from transformers import AutoProcessor, AutoModelForCausalLM; AutoProcessor.from_pretrained('microsoft/git-base'); AutoModelForCausalLM.from_pretrained('microsoft/git-base')"
```

## Step 4 — Run it

```
python main.py
```

The first time you run it, the YOLO weight files (yolov8m.pt and yolov8m-seg.pt)
will download automatically. After that they're saved locally and won't download again.

## Controls

- Click the [DET]/[SEG] button (top left) — switch between object detection and segmentation
- Click [Mode] (top right) — choose the caption personality (funny, serious, coach, etc.)
- Click [Stats] — show a live panel counting detected object types
- Press Q — quit
- Press Esc — close an open dropdown
