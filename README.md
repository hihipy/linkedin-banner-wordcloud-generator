# LinkedIn Banner Word Cloud Generator

> Turn your resume into a professional LinkedIn banner — powered by AI.

A Python desktop app that reads your resume (PDF, DOCX, or TXT), sends it to
an AI provider to extract and weight your most important professional terms, and
renders a word cloud sized for LinkedIn banners **(1584 × 396 px)**.

Works for **any industry** — technology, finance, healthcare, law, marketing,
engineering, education, and more. The AI reads your resume and decides what
matters for your field.

Supports **Claude, ChatGPT, Gemini, Mistral, and Groq** and runs on
macOS, Windows, and Linux.

---

## How it works

| Step | What happens |
|---|---|
| Browse | Pick your resume (PDF, DOCX, or TXT) |
| Analyse | AI extracts up to 60 weighted terms + 30 runner-ups |
| Edit | Remove, add, or promote terms in the live table |
| Style | Choose background, palette, and separator |
| Generate | Preview renders instantly in the app |
| Save As | Export the PNG wherever you want |

---

## Features

### AI-powered extraction

The prompt is industry-agnostic — it reads your resume, identifies your field,
and extracts what actually matters for your role:

- A **data scientist** gets `pandas`, `Machine Learning`, `ETL`
- A **lawyer** gets `Contract Law`, `Litigation`, `Due Diligence`
- A **nurse** gets `Patient Care`, `HIPAA`, `EHR`
- A **marketer** gets `Brand Strategy`, `SEO`, `HubSpot`

Terms are weighted by how central they are to your professional identity
(1,000 – 10,000), not just by frequency.

### Term editor

- Sort the table by term name or weight
- Remove unwanted terms — runner-ups auto-backfill the gap
- Add terms the AI missed, comma-separated
- Promote any runner-up into the main list with one click

### Appearance

- **Background:** Light or Dark
- **Palette:** Vibrant, Mono, Ocean, Hot, Rainbow, Viridis, Plasma, Inferno
- **Separator:** `Data_Science` (underscore) or `Data Science` (space) —
  changes apply live to the table and the cloud output, no re-extraction needed

### Save flow

Generate writes to a **temp file** — nothing lands in your Downloads until you
click **Save As** and confirm. Regenerating replaces the temp file silently.
The previous temp is cleaned up automatically on regeneration or app close.

### Other

- Smart deduplication — standalone `BI` is dropped when `Power BI` is already present at a higher weight
- Content validation — detects scanned image PDFs, password-protected files, and binary garbage before sending to the AI
- File size cap — colour-quantises the PNG if it exceeds 3 MB
- Dark / light theme follows the OS automatically at startup
- Auto-installs missing Python packages on first run
- Detailed logs at `~/.wordcloud_generator/app.log`

---

## Project structure

```
linkedin-banner-wordcloud-generator/
│
├── linkedin_banner_wordcloud_generator.py   # Main application
│
└── providers/
    ├── __init__.py      # Registry + get_provider() factory
    ├── base.py          # BaseProvider abstract class
    ├── anthropic.py     # Claude  (Anthropic)
    ├── openai.py        # ChatGPT (OpenAI)
    ├── google.py        # Gemini  (Google)
    ├── mistral.py       # Mistral AI
    └── groq.py          # Groq    (Llama)
```

---

## Requirements

- **Python 3.9+**
- An API key for at least one AI provider (see table below)

All other Python dependencies install automatically on first run.

> **Linux users:** tkinter is not available via pip — install it through your
> package manager before running the app:
> ```bash
> sudo apt install python3-tk       # Debian / Ubuntu
> sudo dnf install python3-tkinter  # Fedora / RHEL
> ```

---

## AI providers

| Provider | Free tier | Key starts with | pip package |
|---|---|---|---|
| **Claude** — Anthropic | — | `sk-ant-api03-…` | `anthropic` |
| **ChatGPT** — OpenAI | — | `sk-proj-…` | `openai` |
| **Gemini** — Google | ✓ | `AIza…` | `google-genai` |
| **Groq** | ✓ generous | `gsk_…` | `groq` |
| **Mistral** | — | long random string | `mistralai` |

API keys are stored locally at `~/.wordcloud_generator/config.json`.
Nothing is sent anywhere except the provider you choose.

---

## Getting started

**1. Get the files**

Clone or download the repo so that `linkedin_banner_wordcloud_generator.py`
and the `providers/` folder are in the same directory.

**2. Install an AI provider**

```bash
pip install anthropic   # Claude — best results
# or
pip install groq        # Groq   — free tier, no credit card needed
```

**3. Run the app**

```bash
python linkedin_banner_wordcloud_generator.py
```

On first run, `wordcloud`, `matplotlib`, `Pillow`, `pdfplumber`, `python-docx`,
and `darkdetect` install automatically.

**4. Connect a provider**

Click **⚙ Settings**, select your provider, paste your API key,
click **Test Connection**, then **Save**.

---

## Technical details

| | |
|---|---|
| Output dimensions | 1584 × 396 px |
| Max terms | 60 main + 30 runner-ups |
| Weight range | 1,000 – 10,000 |
| Resume chars sent to AI | up to 15,000 |
| PNG size cap | 3 MB |
| DPI | 300 |
| Word cloud library | [wordcloud](https://github.com/amueller/word_cloud) |
| Rendering | matplotlib → PNG, previewed with Pillow |
| Key storage | `~/.wordcloud_generator/config.json` |
| Log file | `~/.wordcloud_generator/app.log` |

---

## Privacy & security

**Your resume text is sent to the AI provider you choose.** Do not process
files containing passwords, private keys, or proprietary trade secrets.
Review your provider's data-usage policy before using the app with
sensitive documents.

API keys are stored in plaintext on your local machine. Do not share
`config.json` or commit it to a repository.

This tool collects no data. Nothing is logged, transmitted, or stored anywhere
other than your local machine and the AI provider you select.

---

## License

[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

Free to use, share, and adapt — including at your job — under these terms:

- **Attribution** — Credit the original author
- **NonCommercial** — Not for selling or building commercial products
- **ShareAlike** — Derivatives must use the same license
