# NewsLens OCR

A high-accuracy Optical Character Recognition (OCR) and article analysis pipeline designed for multilingual newspaper clippings (Gujarati and Hindi), powered by Google Gemini.

---

## Features

- **Multilingual Script Support**: Specifically prompted and tuned for reading complex Gujarati and Hindi newspaper fonts and multi-column article layouts.
- **Batch Processing with Key Rotation (`sample.py`)**:
  - Automatically iterates through images in the `input/` folder.
  - Rotates across multiple Gemini API keys when rate limits (HTTP 429) are encountered.
  - Skips already processed files to allow safe restarts.
  - Saves raw extracted script text into `output/<image_name>.txt`.
- **Structured Article Extraction & Translation (`demo.py`)**:
  - Detects article language (Hindi, Gujarati, mixed).
  - Preserves layout, paragraph order, and script fidelity.
  - Automatically generates an English translation and a concise news headline.
  - Exports results to structured JSON (`gemini_ocr_results.json`) or Markdown.
- **Safe Environment Configuration**: Secure API key management via `.env` without exposing secrets in source code.

---

## Directory Structure

```text
OCR/
├── input/                  # Put your input article images here (JPG, PNG, WebP)
├── output/                 # Destination for extracted text output files
├── demo.py                 # Structured OCR, translation, and headline extraction
├── sample.py               # Batch OCR processor with multi-key rotation
├── requirements.txt        # Python package dependencies
├── .env.example            # Environment variables template
├── .gitignore              # Git ignore rules for caches, outputs, and secrets
├── LICENSE                 # MIT License
└── README.md               # Project documentation
```

---

## Getting Started

### 1. Prerequisites

- Python 3.9 or higher
- A Google Gemini API key from [Google AI Studio](https://aistudio.google.com/)

### 2. Clone and Setup Environment

```bash
# Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Create and activate a virtual environment
# Windows:
python -m venv venv
venv\Scripts\activate

# macOS / Linux:
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure API Keys

Copy `.env.example` to `.env`:

```bash
# Windows PowerShell:
Copy-Item .env.example .env

# Bash / macOS / Linux:
cp .env.example .env
```

Open `.env` and fill in your Gemini API key:

```env
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: multiple keys separated by commas for sample.py rotation
GEMINI_API_KEYS=key1,key2,key3
```

---

## Usage

### Option A: Batch Extraction with Key Rotation (`sample.py`)

Place your article images into the `input/` folder, then run:

```bash
python sample.py
```

The script will process each image and write the extracted raw text directly to `output/<filename>.txt`.

### Option B: Structured Extraction & Translation (`demo.py`)

To extract text, detect the language, translate to English, and generate a headline:

```bash
python demo.py
```

Results are saved to `gemini_ocr_results.json`.

---

## Pushing to GitHub

If you are setting up this repository on GitHub for the first time:

1. **Create a new repository** on GitHub (do not initialize with README or .gitignore since we already have them).
2. **Stage and commit all files**:
   ```bash
   git add .
   git commit -m "Initial commit: Gemini newspaper OCR pipeline"
   ```
3. **Set main branch and link your remote**:
   ```bash
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git push -u origin main
   ```

---

## License

This project is licensed under the [MIT License](LICENSE).
