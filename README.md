# LinkedIn Banner Word Cloud Generator

A Python application that reads your resume (PDF, DOCX, or TXT), extracts professional terms using NLP, and generates a word cloud sized for LinkedIn banners (1584×396 pixels). Supports 18 languages with automatic language detection.

## Features

- **Resume Parsing:** Reads PDF, DOCX, and TXT files — extracts text automatically.
- **NLP-Powered Extraction:** Uses spaCy for named-entity recognition, noun phrase extraction, and part-of-speech filtering. Uses YAKE for statistical keyword ranking.
- **Smart Filtering:** Automatically excludes your name, location, email, dates, section headers, action verbs, and other resume artifacts via header parsing and NER.
- **Interactive Term Editor:** Review, remove, and promote terms in a sortable table. Add custom terms manually or promote runner-ups.
- **Background Color Selection:** Choose between light or dark backgrounds.
- **Color Palettes:** Multiple predefined palettes — Vibrant, Mono, Ocean, Hot, Rainbow, Viridis, Plasma, and Inferno.
- **Compound Term Separator:** Display terms as `Data_Science` or `Data Science`.
- **Multilingual Support:** 18 languages with per-language exclusion lists and spaCy models.
- **File Size Management:** Ensures output is under 3MB using color quantization.
- **Export:** Saves the word cloud as PNG and exports ranked terms (main + runner-ups + excluded) to a text file.

## Requirements

Python 3.9+ — all other dependencies install automatically on first run.

## Usage

1. Place `linkedin_banner_wordcloud_generator.py` and `exclusions.json` in the same folder.
2. Run the script:

```bash
python linkedin_banner_wordcloud_generator.py
```

3. On first run, missing packages (spaCy, YAKE, wordcloud, Pillow, pdfplumber, etc.) and the spaCy English model are installed automatically.
4. Click **Browse** to select your resume file.
5. Click **Analyse Resume** to extract and score terms.
6. Review the term table — click column headers to sort, remove unwanted terms, promote runner-ups.
7. Choose your background, palette, and separator style under **Appearance**.
8. Click **Generate Word Cloud** to preview, then **Save As** to export the PNG.

## Customization

Edit `exclusions.json` to control what gets filtered out. Categories include:

- **section_headers** — Resume headings (Education, Experience, etc.)
- **action_verbs_and_filler** — Managed, Developed, Responsible for, etc.
- **degree_terms** — Bachelor, Master, PhD, etc.
- **date_terms** — January, Present, etc.
- **metadata** — Page, Total, References, etc.
- **never_alone** — Words that are valid in compounds but not solo (e.g., "level", "key", "field")
- **banned_substrings** — Patterns to reject in compound terms (e.g., "stack python", "json output")

All categories are organized by language. Changes take effect on the next run.

## Technical Details

- **Dimensions:** 1584×396 pixels (LinkedIn banner size)
- **NLP Pipeline:** spaCy (NER, noun chunks, POS tagging) + YAKE (statistical keyword extraction)
- **Term Limit:** Top 60 terms displayed, next 30 available as runner-ups
- **Weight Range:** 1,000–10,000 (log-scaled from frequency counts)
- **Word Cloud:** Generated using the `wordcloud` library
- **Visualization:** Previewed with `matplotlib`, exported with Pillow
- **Logging:** Detailed logs saved to `~/.wordcloud_generator/app.log`

## License

This project is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

You are free to:
- Use, share, and adapt this work
- Use it at your job

Under these terms:
- **Attribution** — Credit the original author
- **NonCommercial** — No selling or commercial products
- **ShareAlike** — Derivatives must use the same license
