#!/usr/bin/env python3
"""
OCR Hindi/Gujarati article images with Gemini, translate to English,
and generate an English headline.

Install:
    pip install -U google-genai

Set your Gemini API key:
    PowerShell:  $env:GEMINI_API_KEY="YOUR_API_KEY"
    macOS/Linux: export GEMINI_API_KEY="YOUR_API_KEY"

Configure the global variables below, then run:
    python demo.py
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from google import genai
except ImportError:  # pragma: no cover - user environment check
    genai = None

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
# Configuration
# Default to "input" folder if it exists, otherwise current directory sample
INPUTS: list[str] = ["input"] if Path("input").exists() else ["1.jpeg"]
OUTPUT_PATH = "gemini_ocr_results.json"
MARKDOWN_OUTPUT_PATH: str | None = None
# Set your Gemini API key in environment or .env file (GEMINI_API_KEY)
API_KEY: str | None = os.getenv("GEMINI_API_KEY")
MODEL = DEFAULT_MODEL
COMBINE_IMAGES = False
INLINE_LIMIT_MB = 18.0
RETRIES = 3

OCR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "detected_language": {
            "type": "string",
            "description": "Detected article language: Hindi, Gujarati, Hindi and Gujarati, or Unknown.",
        },
        "extracted_text": {
            "type": "string",
            "description": "Full OCR text in the original script, preserving paragraph order.",
        },
        "notes": {
            "type": "string",
            "description": "Short notes about unreadable text, mixed scripts, or layout issues.",
        },
    },
    "required": ["detected_language", "extracted_text"],
}

TRANSLATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "english_translation": {
            "type": "string",
            "description": "Faithful English translation of the OCR text.",
        }
    },
    "required": ["english_translation"],
}

HEADLINE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "headline": {
            "type": "string",
            "description": "A concise English news headline based on the article.",
        }
    },
    "required": ["headline"],
}


@dataclass
class ArticleResult:
    source_images: list[str]
    detected_language: str
    ocr_text: str
    english_translation: str
    english_headline: str
    ocr_notes: str = ""


def die(message: str) -> None:
    print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(1)


def collect_images(inputs: Iterable[str]) -> list[Path]:
    images: list[Path] = []

    for raw in inputs:
        path = Path(raw).expanduser()
        if not path.exists():
            die(f"input path does not exist: {path}")

        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS:
                    images.append(child)
        elif path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)
        else:
            print(f"Skipping unsupported input: {path}", file=sys.stderr)

    if not images:
        die("no image files found. Supported: jpg, jpeg, png, webp, bmp, tif, tiff")

    return images


def guess_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    return mime_type or "image/jpeg"


def image_input_part(client: Any, image_path: Path, inline_limit_mb: float) -> \
        dict[str, Any]:
    mime_type = guess_mime_type(image_path)
    inline_limit_bytes = int(inline_limit_mb * 1024 * 1024)

    if image_path.stat().st_size <= inline_limit_bytes:
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return {
            "type": "image",
            "data": encoded,
            "mime_type": mime_type,
        }

    uploaded = client.files.upload(file=str(image_path))
    return {
        "type": "image",
        "uri": uploaded.uri,
        "mime_type": getattr(uploaded, "mime_type", mime_type),
    }


def parse_json_response(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    return json.loads(cleaned)


def retry_call(operation: Callable[[], Any], retries: int = 3,
               base_delay: float = 2.0) -> Any:
    last_error: Exception | None = None

    for attempt in range(retries + 1):
        try:
            return operation()
        except Exception as exc:  # Google SDK exception classes may vary by version.
            last_error = exc
            if attempt == retries:
                break
            time.sleep(base_delay * (2 ** attempt))

    raise last_error or RuntimeError("Gemini request failed")


def json_interaction(
        client: Any,
        *,
        model: str,
        input_payload: str | list[dict[str, Any]],
        schema: dict[str, Any],
        system_instruction: str,
        temperature: float = 0.0,
        retries: int = 3,
) -> dict[str, Any]:
    def run() -> Any:
        return client.interactions.create(
            model=model,
            system_instruction=system_instruction,
            input=input_payload,
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": schema,
            },
            generation_config={
                "temperature": temperature,
            },
        )

    interaction = retry_call(run, retries=retries)
    return parse_json_response(interaction.output_text)


def ocr_image(client: Any, image_path: Path, model: str,
              inline_limit_mb: float, retries: int) -> dict[str, Any]:
    prompt = """
Perform OCR on this Indian newspaper/article image.

Rules:
- Extract all visible article text exactly as written.
- Focus on Hindi Devanagari and Gujarati script.
- Preserve paragraph order and line/section order as much as possible.
- Do not translate, summarize, modernize, or correct spellings.
- If a word is unreadable, write [unclear] in that position.
- Ignore phone status bars, app UI, page numbers, ads, watermarks, or captions unless they are part of the article.
- Return JSON only.
""".strip()

    image_part = image_input_part(client, image_path, inline_limit_mb)
    return json_interaction(
        client,
        model=model,
        input_payload=[
            {"type": "text", "text": prompt},
            image_part,
        ],
        schema=OCR_SCHEMA,
        system_instruction="You are a precise OCR engine for Hindi and Gujarati news/article images.",
        temperature=0.0,
        retries=retries,
    )


def translate_to_english(client: Any, ocr_text: str, model: str,
                         retries: int) -> str:
    prompt = f"""
Translate the following Hindi/Gujarati OCR text into clear English.

Requirements:
- Preserve the article's meaning, names, places, dates, numbers, and quotes.
- Keep paragraph breaks where possible.
- Do not add facts that are not in the OCR text.
- Keep [unclear] markers where the OCR is unreadable.
- Return JSON only.

OCR text:
{ocr_text}
""".strip()

    data = json_interaction(
        client,
        model=model,
        input_payload=prompt,
        schema=TRANSLATION_SCHEMA,
        system_instruction="You are a careful English translator for Hindi and Gujarati journalism.",
        temperature=0.1,
        retries=retries,
    )
    return str(data.get("english_translation", "")).strip()


def generate_headline(client: Any, article_text: str, model: str,
                      retries: int) -> str:
    prompt = f"""
Generate one concise English news headline from the article content below.

Rules:
- 8 to 14 words if possible.
- Use a neutral news style.
- Do not use clickbait.
- Do not invent details.
- Prefer the main event, issue, person, place, or policy from the article.
- Return JSON only.

Article content:
{article_text}
""".strip()

    data = json_interaction(
        client,
        model=model,
        input_payload=prompt,
        schema=HEADLINE_SCHEMA,
        system_instruction="You write accurate English newspaper headlines from article text.",
        temperature=0.2,
        retries=retries,
    )
    return str(data.get("headline", "")).strip()


def process_one_image(
        client: Any,
        image_path: Path,
        *,
        model: str,
        inline_limit_mb: float,
        retries: int,
) -> ArticleResult:
    ocr = ocr_image(client, image_path, model, inline_limit_mb, retries)
    ocr_text = str(ocr.get("extracted_text", "")).strip()
    detected_language = str(
        ocr.get("detected_language", "Unknown")).strip() or "Unknown"
    notes = str(ocr.get("notes", "")).strip()

    english_translation = translate_to_english(client, ocr_text, model,
                                               retries) if ocr_text else ""
    headline_source = english_translation or ocr_text
    english_headline = generate_headline(client, headline_source, model,
                                         retries) if headline_source else ""

    return ArticleResult(
        source_images=[str(image_path)],
        detected_language=detected_language,
        ocr_text=ocr_text,
        english_translation=english_translation,
        english_headline=english_headline,
        ocr_notes=notes,
    )


def process_combined_article(
        client: Any,
        image_paths: list[Path],
        *,
        model: str,
        inline_limit_mb: float,
        retries: int,
) -> ArticleResult:
    page_ocr: list[dict[str, str]] = []
    detected_languages: set[str] = set()
    notes: list[str] = []

    for image_path in image_paths:
        print(f"OCR: {image_path}", file=sys.stderr)
        ocr = ocr_image(client, image_path, model, inline_limit_mb, retries)
        text = str(ocr.get("extracted_text", "")).strip()
        lang = str(
            ocr.get("detected_language", "Unknown")).strip() or "Unknown"
        note = str(ocr.get("notes", "")).strip()

        detected_languages.add(lang)
        if note:
            notes.append(f"{image_path.name}: {note}")
        page_ocr.append({"image": str(image_path), "text": text})

    combined_ocr = "\n\n".join(
        f"--- Source image: {Path(page['image']).name} ---\n{page['text']}" for
        page in page_ocr
    ).strip()

    english_translation = translate_to_english(client, combined_ocr, model,
                                               retries) if combined_ocr else ""
    headline_source = english_translation or combined_ocr
    english_headline = generate_headline(client, headline_source, model,
                                         retries) if headline_source else ""

    return ArticleResult(
        source_images=[str(path) for path in image_paths],
        detected_language=", ".join(sorted(detected_languages)) or "Unknown",
        ocr_text=combined_ocr,
        english_translation=english_translation,
        english_headline=english_headline,
        ocr_notes="\n".join(notes),
    )


def write_json(result: ArticleResult | list[ArticleResult],
               output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(result, list):
        payload: Any = [asdict(item) for item in result]
    else:
        payload = asdict(result)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                           encoding="utf-8")


def write_markdown(result: ArticleResult | list[ArticleResult],
                   output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    items = result if isinstance(result, list) else [result]
    parts: list[str] = []

    for index, item in enumerate(items, start=1):
        title = item.english_headline or f"Article {index}"
        parts.append(f"# {title}\n")
        parts.append("## Source Images\n")
        for image in item.source_images:
            parts.append(f"- {image}\n")
        parts.append(f"\n## Detected Language\n{item.detected_language}\n")
        if item.ocr_notes:
            parts.append(f"\n## OCR Notes\n{item.ocr_notes}\n")
        parts.append(f"\n## OCR Text\n\n{item.ocr_text}\n")
        parts.append(
            f"\n## English Translation\n\n{item.english_translation}\n")
        parts.append("\n---\n")

    output_path.write_text("\n".join(parts), encoding="utf-8")


def build_client(api_key: str | None) -> Any:
    if genai is None:
        die("missing dependency. Install it with: pip install -U google-genai")

    if api_key:
        return genai.Client(api_key=api_key)

    if not os.getenv("GEMINI_API_KEY"):
        die("GEMINI_API_KEY is not set. Set it in your environment or API_KEY.")

    return genai.Client()


def main() -> None:
    if not INPUTS:
        die("set INPUTS at the top of this file to one or more image files/folders.")

    client = build_client(API_KEY)
    images = collect_images(INPUTS)

    if COMBINE_IMAGES:
        result: ArticleResult | list[ArticleResult] = process_combined_article(
            client,
            images,
            model=MODEL,
            inline_limit_mb=INLINE_LIMIT_MB,
            retries=RETRIES,
        )
    else:
        results: list[ArticleResult] = []
        for image_path in images:
            print(f"Processing: {image_path}", file=sys.stderr)
            results.append(
                process_one_image(
                    client,
                    image_path,
                    model=MODEL,
                    inline_limit_mb=INLINE_LIMIT_MB,
                    retries=RETRIES,
                )
            )
        result = results

    write_json(result, Path(OUTPUT_PATH))
    print(f"Wrote JSON: {OUTPUT_PATH}", file=sys.stderr)

    if MARKDOWN_OUTPUT_PATH:
        write_markdown(result, Path(MARKDOWN_OUTPUT_PATH))
        print(f"Wrote Markdown: {MARKDOWN_OUTPUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
 