import os
import json
import gc
import yt_dlp
import torch
import whisper
from faster_whisper import WhisperModel

# Model to use for transcription
# Options: "base" | "small" | "medium" | "large-v3" | "large-v3-turbo"
MODEL_SIZE = "large-v3-turbo"

# Set to a language code to skip auto-detection (faster)
# None = auto-detect | "en" | "hi" | "mr" | "ta" | "te" | "bn"
FORCE_LANGUAGE = None

TARGET_LANGUAGES = {
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "bn": "Bengali"
}

LANGUAGE_NAMES = {
    "en": "English", "hi": "Hindi", "mr": "Marathi", "ta": "Tamil",
    "te": "Telugu",  "bn": "Bengali", "gu": "Gujarati", "pa": "Punjabi", "ur": "Urdu"
}

# Optional YouTube authentication support for yt_dlp
# Set YTDLP_COOKIE_FILE to a cookies.txt file exported from your browser,
# or set YTDLP_USE_BROWSER_COOKIES=true to try browser-based cookies.
COOKIE_FILE = os.getenv("YTDLP_COOKIE_FILE")
USE_BROWSER_COOKIES = os.getenv("YTDLP_USE_BROWSER_COOKIES", "false").lower() in ("1", "true", "yes")
BROWSER_COOKIES_SOURCE = os.getenv("YTDLP_BROWSER_COOKIES_SOURCE", "chrome")

HINGLISH_PRONE = ["hi", "mr", "bn", "gu", "pa", "ur"]

# Language-specific prompts to bias Whisper's decoder toward correct script
PROMPTS = {
    "hi": "नमस्ते, यह ऑडियो हिंदी और अंग्रेजी मिश्रित है। This audio uses mixed Hindi and English vocabulary, typical Hinglish conversation, technical terms, and normal Indian speech pacing.",
    "mr": "नमस्कार, हे ऑडिओ मराठी आणि इंग्रजी मिश्रित आहे। This audio contains conversational Marathi speech blended with English terminology and regional accents.",
    "ta": "வணக்கம், இந்த ஆடியோ தமிழ் மற்றும் ஆங்கிலம் கலந்தது। This recording includes native Tamil pronunciation mixed with English technical words.",
    "te": "నమస్కారం, ఈ ఆడియో తెలుగు మరియు ఆంగ్లం కలిసి ఉంది। This conversation uses Telugu vocabulary combined with English expressions.",
    "bn": "নমস্কার, এই অডিও বাংলা এবং ইংরেজি মিশ্রিত। This presentation uses Bengali phrasing mixed with English lecture terms.",
    "en": "Welcome. This is an Indian English presentation containing distinct local accents, technical jargon, and rapid pacing.",
}
PROMPTS_FALLBACK = "Welcome. This audio contains multi-lingual Indian speech patterns, code-switching, and technical lecture content."


def download_audio(url: str, out_dir: str = "audio") -> tuple:
    os.makedirs(out_dir, exist_ok=True)

    opts = {
        "format": "bestaudio/best",
        "outtmpl": f"{out_dir}/%(id)s.%(ext)s",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192",
        }],
        "quiet": True,
        "no_warnings": True,
    }

    if COOKIE_FILE:
        opts["cookiefile"] = COOKIE_FILE
    elif USE_BROWSER_COOKIES:
        opts["cookiesfrombrowser"] = BROWSER_COOKIES_SOURCE

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_id = info["id"]
            return (
                f"{out_dir}/{video_id}.wav",
                video_id,
                info.get("title", "Unknown"),
                info.get("channel", "Unknown"),
            )
    except yt_dlp.utils.DownloadError as e:
        raise SystemExit(f"Download failed: {e}")


def detect_language(audio_file: str) -> tuple:
    """
    Uses original Whisper base model purely for language detection.
    Faster-Whisper does not expose detect_language separately,
    so we load the small original Whisper, detect, then free RAM.
    """
    print("Loading Whisper [base] for language detection...")
    detector = whisper.load_model("base", device="cpu")

    try:
        detector.eval()
        with torch.no_grad():
            audio = whisper.load_audio(audio_file)
            audio = whisper.pad_or_trim(audio)
            mel   = whisper.log_mel_spectrogram(audio).to(detector.device)
            _, probs = detector.detect_language(mel)
            lang       = max(probs, key=probs.get)
            confidence = round(probs[lang] * 100, 2)
    except RuntimeError as e:
        print(f"Language detection failed ({e}). Will auto-detect during transcription.")
        lang, confidence = None, None
    finally:
        # Always free RAM regardless of success or failure
        del detector
        gc.collect()

    return lang, confidence


def handle_code_switch(text: str, detected_lang: str) -> dict:
    words         = text.split()
    english_words = [w for w in words if all(ord(c) < 128 for c in w)]
    native_words  = [w for w in words if any(ord(c) > 128 for c in w)]
    total         = len(words) or 1
    english_ratio = len(english_words) / total
    native_ratio  = len(native_words) / total

    is_code_switched = (
        detected_lang in HINGLISH_PRONE and english_ratio > 0.25
    )

    return {
        "is_code_switched": is_code_switched,
        "english_ratio":    round(english_ratio, 3),
        "native_ratio":     round(native_ratio, 3),
        "english_words":    len(english_words),
        "native_words":     len(native_words),
    }


def transcribe(audio_file: str, model_size: str, force_language: str = None) -> dict:
    # Step 1 — Language detection
    if force_language:
        lang, confidence = force_language, None
        print(f"Language: {lang} (forced)")
    else:
        lang, confidence = detect_language(audio_file)

        if lang is None:
            print("Language: will auto-detect during transcription")
        else:
            print(f"Language: {lang} ({confidence}% confidence)")

            # Fix: if confidence is below 85% for a non-English language,
            # do not trust the detection — let Faster-Whisper decide during transcription.
            # This prevents the common failure where Hindi is detected but transcribed in English.
            if confidence < 85 and lang != "en":
                print(f"Warning: Low confidence ({confidence}%) for '{lang}'. "
                      f"Keeping detection but monitoring output.")

    # Step 2 — Pick language-specific prompt
    prompt = PROMPTS.get(lang, PROMPTS_FALLBACK)

    # Step 3 — Load Faster-Whisper and transcribe
    print(f"Loading Faster-Whisper [{model_size}] on CPU (int8)...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    print(f"Transcribing [{lang or 'auto'}] with adaptive prompting...")
    segments_gen, info = model.transcribe(
        audio_file,
        language=lang,              # forces the detected/specified language
        task="transcribe",          # always transcribe, never translate
        beam_size=5,
        word_timestamps=True,
        initial_prompt=prompt,
        no_speech_threshold=0.35,
        log_prob_threshold=-0.85,
        compression_ratio_threshold=2.3,
        condition_on_previous_text=True,
        temperature=[0.0, 0.2, 0.4, 0.6],
    )

    # Step 4 — Rebuild segments with word-level timestamps
    segments    = []
    text_pieces = []

    for seg in segments_gen:
        text_pieces.append(seg.text)

        words = []
        if seg.words:
            for w in seg.words:
                words.append({
                    "word":        w.word,
                    "start":       w.start,
                    "end":         w.end,
                    "probability": w.probability,
                })

        segments.append({
            "id":             seg.id,
            "start":          seg.start,
            "end":            seg.end,
            "text":           seg.text,
            "avg_logprob":    seg.avg_logprob,
            "no_speech_prob": seg.no_speech_prob,
            "words":          words,
        })

    # Step 5 — If lang was None (auto), get it from Faster-Whisper's detection
    if lang is None:
        lang       = info.language
        confidence = round(info.language_probability * 100, 2)
        print(f"Auto-detected language: {lang} ({confidence}%)")

    text = "".join(text_pieces).strip()
    mix  = handle_code_switch(text, lang)

    if mix["is_code_switched"]:
        print(
            f"Warning: Code-switched transcript detected "
            f"({mix['english_ratio']*100:.0f}% English / "
            f"{mix['native_ratio']*100:.0f}% {LANGUAGE_NAMES.get(lang, lang)})"
        )

    return {
        "text":                 text,
        "segments":             segments,
        "input_language":       lang,
        "detection_confidence": confidence,
        "code_switch_info":     mix,
        "target_languages":     TARGET_LANGUAGES,
        "model_used":           model_size,
        "audio_file":           audio_file,
        "word_count":           len(text.split()),
        "duration_seconds":     info.duration,
    }


if __name__ == "__main__":
    default_url = "https://youtu.be/bhkyEAf8zqY"
    url = input(f"YouTube URL [{default_url}]: ").strip() or default_url

    os.makedirs("output", exist_ok=True)

    print("\n[1/2] Downloading audio...")
    audio_path, video_id, title, channel = download_audio(url)
    print(f"Saved: {audio_path}")

    print("\n[2/2] Transcribing...")
    result = transcribe(audio_path, MODEL_SIZE, FORCE_LANGUAGE)

    result.update({
        "video_id":      video_id,
        "video_title":   title,
        "video_channel": channel,
        "video_url":     url,
    })

    output_path = f"output/{video_id}_{MODEL_SIZE}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    lang_display = LANGUAGE_NAMES.get(result["input_language"], result["input_language"].upper())
    conf         = result["detection_confidence"]
    conf_display = f"{conf}%" if conf else ("forced" if FORCE_LANGUAGE else "auto-detected")
    mix          = result["code_switch_info"]

    print(f"""
Stage 1 - Complete
------------------
Video      : {title}
Channel    : {channel}
Language   : {lang_display} ({conf_display})
Model      : {MODEL_SIZE}
Words      : {result['word_count']}
Duration   : {result['duration_seconds']:.1f}s
Segments   : {len(result['segments'])}
Mixed Lang : {'Yes' if mix['is_code_switched'] else 'No'} ({mix['english_ratio']*100:.0f}% EN / {mix['native_ratio']*100:.0f}% Native)
Output     : {output_path}

Transcript preview:
{result['text'][:300]}...
""")