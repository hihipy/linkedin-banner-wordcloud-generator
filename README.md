# linkedin-banner-wordcloud-generator

A Python desktop application that reads your resume (PDF, DOCX, or TXT),
sends it to an AI provider to extract and weight your most important
professional terms, and renders a word cloud sized for LinkedIn banners
(1584 × 396 pixels).

Supports five AI providers — Claude, ChatGPT, Gemini, Mistral, and Groq —
and works on macOS, Windows, and Linux.

---

## Example

A fashion and graphic design resume processed with a dark background and
Vibrant palette.

**Input — resume**

![Sample resume](examples/resume_input.png)

**Output — LinkedIn banner**

![Generated LinkedIn banner](examples/banner_output.png)

---

## Features

- **AI-Powered Extraction** — Sends your resume to an AI model that understands
  your industry and role, extracts relevant skills and domain terms, and assigns
  importance weights based on how central each term is to your professional
  identity. Works for any field: technology, finance, healthcare, law,
  marketing, engineering, education, and more.
- **Universal Prompt** — The extraction prompt is industry-agnostic. A data
  scientist gets `pandas`, `Machine Learning`, and `ETL`. A lawyer gets
  `Contract Law`, `Litigation`, and `Due Diligence`. A nurse gets `Patient
  Care`, `HIPAA`, and `EHR`. The AI reads the resume and decides what matters.
- **Smart Deduplication** — Automatically removes single-word terms already
  covered by a higher-weight compound (e.g. standalone `BI` is dropped when
  `Power BI` is already present at a higher weight).
- **Interactive Term Editor** — Review, sort, remove, and manually add terms
  before generating. Promote runner-up terms into the main list with one click.
  Auto-backfills from runner-ups when you remove a term.
- **Live Separator Toggle** — Switch between `Data_Science` and `Data Science`
  display style at any time. The change applies to the treeview, the export
  file, and the word cloud output instantly — no re-extraction needed.
- **Appearance Controls** — Choose light or dark background and one of eight
  colour palettes (Vibrant, Mono, Ocean, Hot, Rainbow, Viridis, Plasma,
  Inferno). The Appearance panel only appears after a successful extraction.
- **Temp-First Save Flow** — Generate runs silently to a temp file. Nothing
  lands in your Downloads folder until you click **Save As**, choose a
  location, and confirm. Regenerating replaces the temp file automatically.
- **File Size Management** — Ensures the output PNG stays under 3 MB using
  colour quantisation if needed.
- **Export Terms** — Save the full ranked list (main terms + runner-ups +
  excluded) to a `.txt` file for reference or further editing.
- **Dark / Light Theme** — UI theme is detected from the OS at startup
  (macOS, Windows, Linux all supported).
- **Cross-Platform** — Tested on macOS, Windows, and Linux. Auto-installs
  missing Python packages on first run.
- **Logging** — Detailed logs written to `~/.wordcloud_generator/app.log`.

---

## Project structure

```
linkedin-banner-wordcloud-generator/
├── linkedin_banner_wordcloud_generator.py   # Main application
├── examples/
│   ├── resume_input.png                     # Sample resume
│   └── banner_output.png                    # Generated banner
└── providers/
    ├── __init__.py      # Provider registry and get_provider() factory
    ├── base.py          # BaseProvider abstract class
    ├── anthropic.py     # Claude (Anthropic)
    ├── openai.py        # ChatGPT (OpenAI)
    ├── google.py        # Gemini (Google)
    ├── mistral.py       # Mistral AI
    └── groq.py          # Groq (Llama)
```

---

## Requirements

- Python 3.9+
- An API key for at least one supported AI provider (see table below)

All other Python dependencies install automatically on first run.

> **Linux note:** tkinter is not available via pip. Install it through your
> package manager first:
> ```bash
> sudo apt install python3-tk      # Debian / Ubuntu
> sudo dnf install python3-tkinter  # Fedora / RHEL
> ```

---

## AI providers

| Provider | Free tier | Key format | Install |
|---|---|---|---|
| **Claude** (Anthropic) | — | `sk-ant-api03-…` | `pip install anthropic` |
| **ChatGPT** (OpenAI) | — | `sk-proj-…` | `pip install openai` |
| **Gemini** (Google) | ✓ | `AIza…` | `pip install google-genai` |
| **Groq** | ✓ generous | `gsk_…` | `pip install groq` |
| **Mistral** | — | long random string | `pip install mistralai` |

API keys are stored locally at `~/.wordcloud_generator/config.json`.
Nothing is sent anywhere except the AI provider you choose.

---

## Getting started

1. Clone or download the repository so you have the
   `linkedin_banner_wordcloud_generator.py` file and the `providers/` folder
   together in the same directory.

2. Install at least one AI provider package:
   ```bash
   pip install anthropic      # Claude — recommended
   # or
   pip install groq           # Groq — free tier, no credit card
   ```

3. Run the application:
   ```bash
   python linkedin_banner_wordcloud_generator.py
   ```
   On first run the app installs `wordcloud`, `matplotlib`, `Pillow`,
   `pdfplumber`, `python-docx`, and `darkdetect` automatically.

4. Click **⚙ Settings**, select your provider, paste your API key, click
   **Test Connection**, then **Save**.

---

## Usage

1. Click **Browse …** and select your resume (PDF, DOCX, or TXT).
2. Click **Analyse Resume**. The AI reads your resume and returns up to
   60 weighted terms plus 30 runner-ups.
3. Review the **Extracted Terms** table. You can:
   - Click column headers to sort by term name or weight.
   - Select a row and click **− Remove Selected** to drop a term (the
     highest-weight runner-up auto-fills the gap).
   - Type a term in **Add term** and click **+ Add** to include something
     the AI missed.
   - Pick a runner-up from the dropdown and click **↑ Promote** to move it
     into the main list.
4. In the **Appearance** panel (revealed after extraction):
   - Choose **Light** or **Dark** background.
   - Choose a colour palette from the dropdown.
   - Choose **Underscore** or **Space** separator — updates the table and
     the cloud output live.
5. Click **Generate Word Cloud**. The banner renders to a temp file and
   appears in the preview panel.
6. Click **Save As …** to choose a filename and save location (defaults to
   Downloads). Clicking **Generate Word Cloud** again re-renders with a
   fresh random layout; the previous temp file is deleted.
7. Optionally click **Export Terms (.txt)** to save the full ranked term
   list for reference.

---

## Technical details

| Detail | Value |
|---|---|
| Output dimensions | 1584 × 396 px (LinkedIn banner) |
| Max terms (main) | 60 |
| Runner-ups | 30 |
| Weight range | 1,000 – 10,000 |
| Max resume chars sent to AI | 15,000 |
| PNG size cap | 3 MB (colour-quantised if exceeded) |
| Word cloud library | [`wordcloud`](https://github.com/amueller/word_cloud) |
| Rendering | `matplotlib` → PNG at 300 dpi, previewed with `Pillow` |
| Config / key storage | `~/.wordcloud_generator/config.json` |
| Log file | `~/.wordcloud_generator/app.log` |

---

## Privacy and security

- Your resume text is sent to the AI provider you choose. Do not process
  files containing passwords, private keys, or proprietary trade secrets.
  Check your provider's data usage policy before proceeding.
- API keys are stored in plaintext in `config.json` on your local machine.
  Do not share this file or commit it to a repository.
- No data is collected by this tool. Nothing is logged, transmitted, or
  stored anywhere other than your local machine and the AI provider you choose.

---

## License

Licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

You are free to use, share, and adapt this work — including for use at your
job — under these terms:

- **Attribution** — Credit the original author.
- **NonCommercial** — Not for selling or building commercial products.
- **ShareAlike** — Derivatives must use the same license.