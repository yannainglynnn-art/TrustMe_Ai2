#!/usr/bin/env python3
"""Whisper transcription via the Groq API (free tier, no card needed).

Reads a media file, extracts/compresses the audio with ffmpeg, splits it
into chunks under Groq's 25 MB upload limit, transcribes each chunk with
segment + word timestamps, and writes:
  transcript.json - full segments + words (the "time map")
  transcript.srt  - human-checkable subtitles

Needs GROQ_API_KEY in the environment (repo secret on GitHub Actions).
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import requests

API_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
MODEL = "whisper-large-v3-turbo"
CHUNK_SECONDS = 600              # 10-minute chunks
MAX_BYTES = 24 * 1024 * 1024     # stay just under the 25 MB API limit


def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def prepare_chunks(media: Path, work: Path) -> list:
    """Convert to 16 kHz mono mp3 (32 kbps); split into 10-min chunks if large."""
    work.mkdir(parents=True, exist_ok=True)
    full = work / "audio.mp3"
    run(["ffmpeg", "-y", "-i", str(media), "-vn", "-ac", "1", "-ar", "16000",
         "-codec:a", "libmp3lame", "-b:a", "32k", str(full)])
    if full.stat().st_size <= MAX_BYTES:
        return [full]
    run(["ffmpeg", "-y", "-i", str(full), "-f", "segment",
         "-segment_time", str(CHUNK_SECONDS),
         "-codec:a", "libmp3lame", "-b:a", "32k", str(work / "chunk_%03d.mp3")])
    return sorted(work.glob("chunk_*.mp3"))


def transcribe_chunk(path: Path, api_key: str, language: str) -> dict:
    with open(path, "rb") as f:
        files = {"file": (path.name, f, "audio/mpeg")}
        data = {"model": MODEL, "response_format": "verbose_json",
                "timestamp_granularities[]": ["word", "segment"]}
        if language:
            data["language"] = language
        r = requests.post(API_URL, headers={"Authorization": f"Bearer {api_key}"},
                          files=files, data=data, timeout=600)
    if r.status_code != 200:
        print(f"Groq API error {r.status_code}: {r.text[:500]}", file=sys.stderr)
        sys.exit(1)
    return r.json()


def fmt_ts(sec: float) -> str:
    ms = int(round(sec * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("media")
    ap.add_argument("--language", default="", help="e.g. en, ko, zh - blank = auto-detect")
    ap.add_argument("--outdir", default="out")
    args = ap.parse_args()

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        print("GROQ_API_KEY is not set. On GitHub: repo Settings > Secrets and "
              "variables > Actions > New repository secret.", file=sys.stderr)
        sys.exit(1)

    media = Path(args.media)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    work = Path("work_audio")
    work.mkdir(exist_ok=True)

    chunks = prepare_chunks(media, work)
    multi = len(chunks) > 1
    print(f"{len(chunks)} audio chunk(s)")

    segments, words, language = [], [], ""
    for i, ch in enumerate(chunks):
        offset = i * CHUNK_SECONDS if multi else 0.0
        res = transcribe_chunk(ch, api_key, args.language)
        language = language or res.get("language", "")
        for seg in res.get("segments", []):
            segments.append({"start": seg["start"] + offset,
                             "end": seg["end"] + offset,
                             "text": seg["text"].strip()})
        for w in res.get("words", []):
            words.append({"start": w["start"] + offset,
                          "end": w["end"] + offset,
                          "word": w["word"]})
        print(f"chunk {i + 1}/{len(chunks)} done: "
              f"{len(res.get('segments', []))} segments")

    segments.sort(key=lambda s: s["start"])
    words.sort(key=lambda w: w["start"])

    payload = {"language": language, "model": MODEL,
               "segments": segments, "words": words}
    (outdir / "transcript.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    srt = []
    for i, seg in enumerate(segments, 1):
        srt.append(f"{i}\n{fmt_ts(seg['start'])} --> {fmt_ts(seg['end'])}\n{seg['text']}\n")
    (outdir / "transcript.srt").write_text("\n".join(srt), encoding="utf-8")

    total = segments[-1]["end"] if segments else 0
    print(f"DONE language={language} segments={len(segments)} "
          f"words={len(words)} media_end={total:.1f}s")
    for seg in segments[:3]:
        print(f"  [{seg['start']:7.2f} -> {seg['end']:7.2f}] {seg['text'][:70]}")


if __name__ == "__main__":
    main()
