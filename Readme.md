# 🎓 AI-Based Multilingual Lecture Translation & Educational Note Generation

> An end-to-end AI pipeline for translating educational YouTube lectures into multiple Indic languages while preserving **100% technical terminology and mathematical formulas**, followed by automatic AI-generated study notes.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Whisper](https://img.shields.io/badge/Faster--Whisper-large--v3--turbo-orange)
![IndicTrans2](https://img.shields.io/badge/Translation-IndicTrans2-red)
![MuRIL](https://img.shields.io/badge/NER-MuRIL--CRF-purple)

---

# 📖 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Pipeline Architecture](#pipeline-architecture)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Tech Stack](#tech-stack)
- [Future Improvements](#future-improvements)
- [License](#license)

---

# 📌 Overview

Millions of students struggle to access quality educational content because most online STEM lectures are available only in English or code-switched Hinglish. Traditional machine translation systems often mistranslate technical vocabulary and mathematical expressions, making learning difficult.

This project presents a **6-stage multimodal AI pipeline** that:

- Downloads YouTube lectures
- Performs multilingual speech recognition
- Detects technical terminology using a custom **MuRIL-CRF** model
- Preserves domain entities during translation through **Entity-Shielded Translation (EST)**
- Generates synchronized dubbed audio
- Produces AI-generated educational notes

---

# ✨ Features

## 🎙 Speech Recognition

- Faster-Whisper (large-v3-turbo)
- Handles English, Hindi, and Hinglish
- Word-level timestamps
- Speaker-aware transcription

## 🧠 Technical Entity Recognition

- Fine-tuned MuRIL + CRF model
- Detects:
  - TECH entities
  - FORMULA entities
- Designed specifically for educational lectures

## 🌐 Entity-Shielded Translation (EST)

- Prevents technical vocabulary corruption
- Preserves formulas
- Uses IndicTrans2
- Achieves **100% Technical Term Preservation Rate**

## 🔊 Neural Dubbing

- Microsoft Edge Neural TTS
- Pitch-aware synchronization
- Gender-aware voice selection
- Supports:
  - Tamil
  - Telugu
  - Marathi
  - Bengali

## 📝 AI Educational Notes

Powered by:

- Llama-3.1-8B-Instant
- Groq API

Automatically generates:

- Lecture Summary
- Key Concepts
- Important Definitions
- MCQs
- Question & Answers
- Learning Outcomes

---

# 🏗 Pipeline Architecture

```text
                 YouTube Lecture
                        │
                        ▼
┌───────────────────────────────────────────┐
│ Stage 1                                   │
│ Audio Extraction + Speech Recognition      │
│ (yt-dlp + FFmpeg + Faster-Whisper)         │
└───────────────────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────┐
│ Stage 2                                   │
│ Domain NER (MuRIL + CRF)                  │
└───────────────────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────┐
│ Stage 3                                   │
│ Entity Shielding                          │
└───────────────────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────┐
│ Stage 4                                   │
│ IndicTrans2 Translation                   │
└───────────────────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────┐
│ Stage 5                                   │
│ Neural Dubbing (Edge TTS)                 │
└───────────────────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────┐
│ Stage 6                                   │
│ AI Educational Note Generation            │
│ (Groq + Llama-3.1-8B)                     │
└───────────────────────────────────────────┘
```

---

# 📂 Repository Structure

```
.
├── Notebooks
│   ├── notebook-translation.ipynb
│   └── notebook-stage5.ipynb
│
├── Scripts
│   ├── stage1&2_extract&convert.py
│   ├── stage6_notes.py
│   ├── .env
│   ├── www.youtube.com_cookies
│   │
│   ├── audio
│   │
│   └── output
│       ├── transcript.json
│       ├── ner.json
│       ├── translated.json
│       └── stage6_notes.json
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# ⚙ Installation

## 1. Clone Repository

```bash
git clone https://github.com/your-username/project-name.git

cd project-name
```

---

## 2. Install FFmpeg

### Ubuntu

```bash
sudo apt update
sudo apt install ffmpeg
```

### Windows

Download from:

https://ffmpeg.org/download.html

---

## 3. Install Deno

```bash
curl -fsSL https://deno.land/install.sh | sh
```

---

## 4. Create Virtual Environment

```bash
python -m venv .venv
```

Activate:

Linux/macOS

```bash
source .venv/bin/activate
```

Windows

```bash
.venv\Scripts\activate
```

---

## 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 6. Configure Environment

Create a `.env` file inside the `Scripts/` directory.

```env
GROQ_API_KEY=your_api_key_here
```

---

# 🚀 Usage

## Stage 1 & 2

Audio Extraction + Speech Recognition + NER

```bash
cd Scripts

python stage1&2_extract&convert.py
```

---

## Stage 3 & 4

Run the translation notebook

```
Notebooks/notebook-translation.ipynb
```

---

## Stage 5

Run the dubbing notebook

```
Notebooks/notebook-stage5.ipynb
```

---

## Stage 6

Generate educational notes

```bash
cd Scripts

python stage6_notes.py
```

---

# 📊 Results

| Metric                      | Result                                          |
| --------------------------- | ----------------------------------------------- |
| TECH Entity F1              | **0.98**                                        |
| FORMULA Entity F1           | **0.87**                                        |
| Technical Term Preservation | **100%**                                        |
| SacreBLEU Improvement       | **+5.15**                                       |
| Supported Languages         | English, Hindi, Tamil, Telugu, Marathi, Bengali |

---

# 🛠 Tech Stack

### Speech Recognition

- Faster-Whisper
- FFmpeg
- yt-dlp

### NLP

- MuRIL
- CRF
- Transformers
- PyTorch

### Translation

- IndicTrans2

### Voice Synthesis

- Microsoft Edge TTS

### LLM

- Groq API
- Llama-3.1-8B-Instant

### Programming

- Python
- Jupyter Notebook

---

# 🔮 Future Improvements

- Support additional Indic languages
- Speaker cloning instead of generic TTS
- Real-time lecture translation
- Live subtitle generation
- Web application deployment
- PDF lecture note generation
- Automatic slide extraction

---

# 📜 License

This project is licensed under the **MIT License**.

See the `LICENSE` file for more information.

---

## 🔐 Environment & Secrets (important)

This repository must not contain API keys, tokens, or other secrets. Keep credentials out of source
and load them from environment variables or a secure secrets manager.

- Create a `.env` file inside the `Scripts/` directory for local development.
- Do NOT commit `.env` to the repository. `.gitignore` already excludes environment files.

Example `.env` keys used by this project:

```env
# Scripts/.env
GROQ_API_KEY=your_groq_api_key_here
HF_API_KEY=your_huggingface_token_here
```

I've added `Scripts/.env.example` to the repo with placeholders — copy it to `Scripts/.env` and fill
in real values locally.

Security notes:

- Revoke and rotate any API keys that were previously committed (Groq, Hugging Face, etc.).
- Consider using a secrets manager (GitHub Actions secrets, Azure Key Vault, AWS Secrets Manager,
  or similar) for CI and production.
- Never paste long-lived tokens into notebooks or committed files; prefer environment variables.

If you want, I can add a small helper script to load `.env` safely and fail with a clear message
when required keys are missing.
