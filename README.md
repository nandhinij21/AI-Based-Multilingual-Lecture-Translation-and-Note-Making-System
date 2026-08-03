# AI-Based-Multilingual-Lecture-Translation-and-Note-Making-System

An end-to-end multimodal pipeline for translating educational YouTube video lectures and generating structured, student-friendly lecture notes across multiple Indic languages with 100% Technical Vocabulary & Mathematical Formula Preservation.

---

## 📌 Project Overview

Digital learning platforms have democratized education, yet millions of students face significant language barriers when accessing high-quality STEM content, which is predominantly delivered in English, Hindi, or code-switched Hinglish. Existing Machine Translation (MT) systems frequently mistranslate domain-specific technical vocabulary and fail to handle mathematical expressions or code-switching.

This repository implements a 6-Stage Automated Multimodal Pipeline that ingests YouTube lecture videos, transcribes multi-lingual Indian speech, identifies technical and mathematical entities using a custom MuRIL–CRF model, translates the content without corrupting domain terms using an innovative Entity-Shielded Translation (EST) framework, synthesizes localized neural dubbing, and generates structured educational notes.

---

## 🚀 Key Features

- Code-Switched Speech Recognition: Handles rapid Indian English, Hindi, and code-mixed speech (Hinglish) using int8-quantized Faster-Whisper.
- Domain Entity Recognition: Extracts technical terms (TECH) and mathematical expressions (FORMULA) using a fine-tuned MuRIL + CRF token classification architecture.
- Entity-Shielded Translation (EST): Ensures a 100% Term Preservation Rate (TPR) by masking/shielding technical entities from machine translation corruption via IndicTrans2.
- Synchronized Localized Dubbing: Generates multi-speaker, pitch-matched, gender-aware audio dubbed tracks using Microsoft Edge Neural TTS in Tamil, Telugu, Marathi, and Bengali.
- AI Note Generation: Synthesizes structured, student-friendly educational study notes (summaries, key concepts, Q&As, MCQs, and learning outcomes) using Llama-3.1-8b-instant via the Groq API while keeping code/technical terms in English.

---

## 🏗 System Architecture (6-Stage Pipeline)

[ YouTube URL ]
│
▼
┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐
│ Stage 1 │ ───► │ Stage 2 │ ───► │ Stage 3 │ ───► │ Stage 4 │ ───► │ Stage 5 │ ───► │ Stage 6 │
│ ASR Engine│ │ MuRIL-CRF │ │ Entity │ │IndicTrans2│ │ Neural TTS│ │ Groq AI │
│Transcribe │ │ Domain NER│ │ Masking │ │ Translation│ │ Local Dub │ │ Notes Gen │
└───────────┘ └───────────┘ └───────────┘ └───────────┘ └───────────┘ └───────────┘

### Stage Summary

1. Stage 1 & 2 (Audio Extraction & NER Processing): Ingests audio via yt-dlp & FFmpeg. Transcribes code-switched speech using Faster-Whisper (large-v3-turbo) and identifies domain-specific technical entities using MuRIL + CRF.
2. Stage 3 & 4 (Entity Masking & Machine Translation): Shields extracted technical entities/formulas and translates transcripts into regional Indian languages using IndicTrans2.
3. Stage 5 (Neural Dubbing & Audio-Video Sync): Synthesizes regional voice tracks via edge-tts and syncs audio duration to match video frames.
4. Stage 6 (AI Educational Note Generation): Uses Groq + Llama-3.1-8b-instant to generate structured, multi-lingual study notes (\*\_stage6_notes.json).

---

## 📂 Repository Structure

.
├── Notebooks/ # Experimental Jupyter Notebooks
│ ├── notebook-translation.ipynb # Stage 3 & 4 Translation prototyping
│ └── notebook-stage5.ipynb # Stage 5 Audio Dubbing prototyping
├── Scripts/ # Standalone execution scripts and runtime data
│ ├── stage1&2_extract&convert.py # Combined Audio Extraction, ASR, & NER
│ ├── stage6_notes.py # Stage 6 AI Note Generator script
│ ├── .env # API Keys & Local Configuration
│ ├── www.youtube.com_cookies # Cookies file for YouTube audio extraction
│ ├── audio/ # Extracted audio tracks (.wav/media)
│ └── output/ # Pipeline output files (transcripts, NER, translated JSONs, notes)
│ ├── <video_id>\_large-v3-turbo.json
│ ├── <video_id>\_large-v3-turbo_ner.json
│ ├── <video_id>\_large-v3-turbo_translated.json
│ └── <video_id>\_stage6_notes.json
├── .gitignore # Git exclusion rules
├── requirements.txt # Project Dependencies
└── README.md # Documentation

---

## ⚙️ Installation & Setup

### 1. System Dependencies

Ensure FFmpeg and Deno are installed on your system:

# Ubuntu/Linux

sudo apt-get update && sudo apt-get install -y ffmpeg

# Install Deno (Required for yt-dlp JS engine)

curl -fsSL https://deno.land/install.sh | sh -s -- -y

### 2. Python Environment & Setup

Create and activate a virtual environment, then install dependencies:

python -m venv .venv
source .venv/bin/activate # On Windows: .venv\Scripts\activate

pip install -r requirements.txt

### 3. Environment Variables

Add your GROQ_API_KEY in Scripts/.env:

GROQ_API_KEY=your_groq_api_key_here

---

## 🏃 Quickstart / Execution Guide

### 1. Extract Audio & Run ASR + NER (Stages 1 & 2)

cd Scripts
python stage1&2_extract&convert.py

### 2. Entity-Shielded Translation & Dubbing (Stages 3, 4 & 5)

Run the corresponding Jupyter notebooks inside Notebooks/ or run the converted python scripts:

- Execute Notebooks/notebook-translation.ipynb for Indic language translation.
- Execute Notebooks/notebook-stage5.ipynb for audio dubbing and synchronization.

### 3. Generate AI Educational Notes (Stage 6)

cd Scripts
python stage6_notes.py

---

## 📊 Experimental Results

- Domain Entity Recognition (MuRIL–CRF): 0.98 F1-Score on TECH entities and 0.87 F1-Score on FORMULA tags.
- Term Preservation Rate (TPR): Maintained 100.0% TPR across all supported Indic translation targets (Tamil, Telugu, Marathi, Bengali).
- Translation Quality: +5.15 SacreBLEU points average improvement over standard unshielded translation baselines.

---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
