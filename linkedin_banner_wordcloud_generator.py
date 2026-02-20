# !! MUST come before ANY third-party import — confection triggers the
# !! warning the instant it is imported (even as a transitive dependency).
import warnings                                          # noqa: E402
warnings.filterwarnings(
    "ignore",
    message=r".*Pydantic V1.*not compatible.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*unable to infer type.*",
    category=UserWarning,
)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import json
import logging
import math
import os
import random
import re
import string
import sys
import threading
import tkinter as tk
import unicodedata
from collections import Counter
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR = Path.home() / ".wordcloud_generator"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
# Keep console quiet — only warnings and errors.
logging.getLogger().handlers[1].setLevel(logging.WARNING)
log = logging.getLogger("wordcloud")
log.info("="*60)
log.info("LinkedIn Banner Word Cloud Generator — starting up")
log.info("Python %s on %s", sys.version, sys.platform)


# ---------------------------------------------------------------------------
# Auto-install — installs missing packages on first run, then continues.
# ---------------------------------------------------------------------------
def _auto_install():
    """Check for missing packages and install them automatically."""
    import subprocess
    # Map: import name → pip package name
    deps = {
        "spacy": "spacy",
        "yake": "yake",
        "langdetect": "langdetect",
        "wordcloud": "wordcloud",
        "matplotlib": "matplotlib",
        "PIL": "Pillow",
        "pdfplumber": "pdfplumber",
        "docx": "python-docx",
    }
    missing = []
    for mod, pip_name in deps.items():
        try:
            __import__(mod)
        except ImportError:
            missing.append(pip_name)

    if missing:
        print(f"\n  Installing {len(missing)} missing package(s): {', '.join(missing)}")
        print("  (one-time setup, may take a minute)\n")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet"] + missing
        )
        print()

    # Ensure at least one spaCy model is downloaded.
    try:
        import spacy
        spacy.load("en_core_web_sm")
    except Exception:
        print("  Downloading spaCy English model (one-time) ...")
        subprocess.check_call(
            [sys.executable, "-m", "spacy", "download", "en_core_web_sm", "--quiet"]
        )
        print()

_auto_install()

import spacy                             # noqa: E402
import matplotlib                        # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from PIL import Image, ImageTk           # noqa: E402
from wordcloud import WordCloud          # noqa: E402
log.info("spaCy version %s", spacy.__version__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BANNER_WIDTH = 1584
BANNER_HEIGHT = 396
MAX_FILE_SIZE_MB = 3.0
MAX_TERMS = 60
WEIGHT_MIN = 1000
WEIGHT_MAX = 10000
NOUN_PHRASE_MULTIPLIER = 3
YAKE_MULTIPLIER_CAP = 8
TOKEN_MULTIPLIER = 1
MANUAL_ADD_WEIGHT = 7000
WINDOW_MIN_WIDTH = 1060
WINDOW_MIN_HEIGHT = 860

FILLER_PATTERNS = [
    re.compile(r"^\d+$"),
    re.compile(r"^[\d\-\(\)\+\s\.]+$"),
    re.compile(r"^[a-zA-Z0-9._%+-]+@"),
    re.compile(r"^https?://"),
    re.compile(r"^www\."),
]

PRESERVE_UPPER = frozenset({
    "SQL", "AWS", "API", "CSS", "HTML", "XML", "JSON", "REST", "GPU",
    "CPU", "NLP", "AI", "ML", "ETL", "KPI", "ROI", "CRM", "ERP",
    "SaaS", "IaaS", "PaaS", "CI", "CD", "GCP", "VBA", "DAX", "RPA",
    "OCR", "NER", "LLM", "DNS", "TCP", "UDP", "HTTP", "SSH", "VPN",
    "IAM", "SDK", "IDE", "ORM", "NoSQL", "GIS", "BI", "DBA", "QA",
    "UX", "UI", "PM", "PMP", "CPA", "CFA", "MBA", "PhD", "MSc",
    "BSc", "GAAP", "IFRS", "SOX", "HIPAA", "FERPA", "GDPR", "CCPA",
    "SPSS", "SAS", "MATLAB", "R",
})


# ---------------------------------------------------------------------------
# Load exclusion lists from JSON
# ---------------------------------------------------------------------------

def _load_exclusions() -> dict:
    """Load exclusion lists from ``exclusions.json``."""
    candidates = [
        Path(__file__).resolve().parent / "exclusions.json",
        Path.cwd() / "exclusions.json",
    ]
    for p in candidates:
        if p.is_file():
            with open(p, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            log.info("Loaded exclusions.json from %s (%d keys)", p, len(data))
            return data

    log.warning("exclusions.json not found — checked %s", candidates)
    print(
        "WARNING: exclusions.json not found.  "
        "Place it next to this script for best results."
    )
    return {}


_EX = _load_exclusions()


def _lang_dict_to_sets(raw: dict) -> dict[str, set[str]]:
    """Convert a ``{lang: [words]}`` dict to ``{lang: {words}}``,
    silently skipping the ``_comment`` key."""
    return {
        k: set(v) for k, v in raw.items()
        if k != "_comment" and isinstance(v, list)
    }


def _extract_terms_list(raw) -> frozenset[str]:
    """Pull a flat term list from either ``[items]`` or
    ``{"_comment": ..., "terms": [items]}``."""
    if isinstance(raw, list):
        return frozenset(raw)
    if isinstance(raw, dict):
        return frozenset(raw.get("terms", []))
    return frozenset()


SECTION_HEADERS = _lang_dict_to_sets(_EX.get("section_headers", {}))
ACTION_VERBS_AND_FILLER = _lang_dict_to_sets(
    _EX.get("action_verbs_and_filler", {}),
)
DEGREE_TERMS = _lang_dict_to_sets(_EX.get("degree_terms", {}))
DATE_TERMS = _lang_dict_to_sets(_EX.get("date_terms", {}))
RESUME_METADATA = _lang_dict_to_sets(_EX.get("resume_metadata", {}))
BANNED_SUBSTRINGS = _lang_dict_to_sets(_EX.get("banned_substrings", {}))
NEVER_ALONE = _lang_dict_to_sets(_EX.get("never_alone", {}))


def _merge_lang(source: dict[str, set[str]], lang: str) -> set[str]:
    """Merge the 'en' set with the language-specific set."""
    result = set(source.get("en", set()))
    if lang != "en":
        result |= source.get(lang, set())
    return result


def _flatten_all(source: dict[str, set[str]]) -> frozenset[str]:
    """Merge ALL language sets into one for universal filtering."""
    result: set[str] = set()
    for k, v in source.items():
        result |= v
    return frozenset(result)


# Flat sets for filtering (all languages combined).
ALL_RESUME_METADATA = _flatten_all(RESUME_METADATA)
ALL_BANNED_SUBSTRINGS = _flatten_all(BANNED_SUBSTRINGS)
ALL_NEVER_ALONE = _flatten_all(NEVER_ALONE)


# ---------------------------------------------------------------------------
# Compound-term contaminant set
# ---------------------------------------------------------------------------
# Individual words that, when found as a COMPONENT of a multi-word term,
# indicate the term is a YAKE artefact (structural label glued to content)
# rather than a genuine skill.  Built by decomposing resume_metadata and
# date_terms into single words, plus known section-header words that are
# NEVER part of a real skill name.
#
# Example: "Overview_Python_GUI" → "overview" is a contaminant → REJECTED.
#          "Data_Science"        → neither word is a contaminant → PASS.

def _build_contaminants() -> frozenset[str]:
    """Build the set of words that poison compound terms."""
    words: set[str] = set()

    # All single words from resume metadata phrases.
    for phrase in ALL_RESUME_METADATA:
        for w in phrase.lower().split():
            if len(w) >= 3:  # Catch short structural words like "usa", "gpa"
                words.add(w)

    # All date terms (month names, "year", "annual", etc.).
    for lang_set in DATE_TERMS.values():
        for w in lang_set:
            if len(w) >= 3 and " " not in w:
                words.add(w.lower())

    # All degree terms ("bachelor", "diploma", "thesis", etc.).
    for lang_set in DEGREE_TERMS.values():
        for w in lang_set:
            if len(w) >= 3 and " " not in w:
                words.add(w.lower())

    # All never_alone terms are also compound contaminants.
    words.update(ALL_NEVER_ALONE)

    # Action verbs and filler — these poison compounds like
    # "Managed_Georgetown" or "Developed_Excel".
    for lang_set in ACTION_VERBS_AND_FILLER.values():
        for w in lang_set:
            if len(w) >= 3 and " " not in w:
                words.add(w.lower())

    # Section-header words that are purely structural.
    structural_headers = {
        "overview", "summary", "objective", "profile",
        "education", "experience", "employment", "history",
        "accomplishments", "achievements", "recognition",
        "interests", "hobbies", "activities",
        "references", "affiliations", "memberships",
        "volunteer", "volunteering",
        "coursework", "highlights",
        "experiencia", "formación", "educación",
        "expérience", "ausbildung", "erfahrung",
    }
    words.update(structural_headers)

    generic_structural = {
        "employer", "employee", "supervisor", "duties",
        "responsibilities", "obtained", "awarded", "conferred",
        "leaving", "authorization", "clearance", "sponsorship",
        "nationality", "citizenship", "relocation",
        "availability", "permitted", "allowed",
        "continued", "attached", "enclosed",
        "modality", "unpaid", "seasonal", "temporary",
        "professionals", "student", "students",
        "download", "upload",
        # URL / web artefacts that bleed into YAKE compounds.
        "bitly", "http", "https", "www", "html", "pdf",
    }
    words.update(generic_structural)

    # PROTECT legitimate skill-adjacent words — remove them from
    # the contaminant set even if they were pulled in above.
    # These words commonly appear in real skill names.
    protected = {
        "data", "analysis", "management", "design", "system",
        "systems", "development", "science", "research",
        "engineering", "learning", "network", "programming",
        "modeling", "testing", "security", "architecture",
        "analytics", "intelligence", "automation", "cloud",
        "database", "mobile", "digital", "strategic",
        "financial", "technical", "clinical", "statistical",
        "machine", "deep", "natural", "business", "project",
        "product", "process", "quality", "performance",
        "information", "communication", "operations",
        "marketing", "leadership", "training", "work",
        "python", "java", "excel", "power", "azure",
        "tableau", "linux", "docker", "react",
        "full", "stack", "front", "back", "devops",
        "agile", "scrum", "lean", "sigma",
        "teaching", "writing", "speaking", "consulting",
        "planning", "budgeting", "forecasting", "reporting",
        "visualization", "modelling", "simulation",
        "compliance", "audit", "governance", "risk",
        "applied", "advanced", "senior", "junior", "field",
    }
    words -= protected

    return frozenset(words)


COMPOUND_CONTAMINANTS: frozenset[str] = _build_contaminants()

log.info(
    "Filter stats: %d section-header langs, %d metadata terms, "
    "%d banned substrings, %d contaminants, %d never-alone",
    len(SECTION_HEADERS), len(RESUME_METADATA),
    len(BANNED_SUBSTRINGS), len(COMPOUND_CONTAMINANTS),
    len(NEVER_ALONE),
)


# ===================================================================== #
#                         BACKEND / NLP ENGINE                           #
# ===================================================================== #

def extract_text_from_file(file_path: str) -> str:
    """Read raw text from a PDF, DOCX, or TXT resume file."""
    ext = os.path.splitext(file_path)[1].lower()
    log.info("Reading file: %s (type: %s)", file_path, ext)
    if ext == ".pdf":
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            text = "\n".join(p.extract_text() or "" for p in pdf.pages)
            log.info("PDF: %d pages, %d chars", len(pdf.pages), len(text))
            return text
    if ext == ".docx":
        import docx
        doc = docx.Document(file_path)
        text = "\n".join(p.text for p in doc.paragraphs)
        log.info("DOCX: %d paragraphs, %d chars", len(doc.paragraphs), len(text))
        return text
    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as fh:
            text = fh.read()
        log.info("TXT: %d chars", len(text))
        return text
    raise RuntimeError(f"Unsupported file type: {ext}")


def validate_resume(text, lang=None):
    """Heuristically score whether *text* looks like a resume."""
    log.debug("validate_resume: %d chars, lang=%s", len(text or ""), lang)
    if not text or not text.strip():
        return False, 0.0, [], ["Document is empty."]
    if lang is None:
        try:
            lang = detect_language(text)
        except Exception:
            lang = "en"

    text_lower = text.lower()
    signals, warn = [], []
    score = 0.0

    # 1. Length.
    wc = len(text.split())
    if wc < 50:
        warn.append(f"Very short ({wc} words).")
    elif wc < 150:
        warn.append(f"Short ({wc} words).")
        score += 0.03
    elif wc <= 6000:
        signals.append(f"Reasonable length ({wc} words).")
        score += 0.10
    else:
        warn.append(f"Very long ({wc} words).")
        score += 0.03

    # 2. Section headers.
    all_h = set()
    for cl in {"universal", "en", lang}:
        all_h.update(SECTION_HEADERS.get(cl, set()))
    hits = set()
    for h in all_h:
        for p in [
            re.compile(r"(?:^|\n)\s*" + re.escape(h) + r"\s*[:\-|]?\s*(?:\n|$)", re.I | re.M),
            re.compile(r"(?:^|\n)\s*" + re.escape(h.upper()) + r"\s*(?:\n|$)", re.M),
        ]:
            if p.search(text):
                hits.add(h)
                break
    if len(hits) >= 4:
        signals.append(f"{len(hits)} section headers found.")
        score += 0.30
    elif len(hits) >= 2:
        signals.append(f"{len(hits)} section headers found.")
        score += 0.18
    elif len(hits) == 1:
        signals.append("1 section header found.")
        score += 0.08
    else:
        warn.append("No section headers found.")

    # 3. Contact info.
    ct = 0
    if re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text):
        signals.append("Email found."); ct += 1
    for p in [r"\+?\d[\d\-\.\s\(\)]{7,15}\d", r"\(\d{3}\)\s*\d{3}[\-\.]\d{4}"]:
        if re.search(p, text):
            signals.append("Phone found."); ct += 1; break
    if re.search(r"linkedin\.com/in/", text_lower):
        signals.append("LinkedIn URL."); ct += 1
    if ct >= 3: score += 0.20
    elif ct >= 2: score += 0.15
    elif ct >= 1: score += 0.08
    else: warn.append("No contact info.")

    # 4. Date ranges.
    dc = sum(len(re.findall(p, text_lower)) for p in [
        r"\b20\d{2}\s*[\-\u2013\u2014]\s*(?:20\d{2}|present|current|presente|actuellement)\b",
        r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*\d{4}\s*[\-\u2013\u2014]",
        r"(?:^|\n)\s*20[12]\d\s*(?:\n|$)",
    ])
    if dc >= 3: signals.append(f"{dc} date ranges."); score += 0.15
    elif dc >= 1: signals.append(f"{dc} date range(s)."); score += 0.08
    else: warn.append("No date ranges.")

    # 5. Skills.
    sk = {"python", "java", "javascript", "sql", "excel", "power bi",
          "tableau", "machine learning", "data analysis", "aws", "azure",
          "docker", "react", "tensorflow", "pandas", "git", "linux", "figma"}
    sh = sum(1 for s in sk if s in text_lower)
    if sh >= 5: signals.append(f"{sh} skill keywords."); score += 0.10
    elif sh >= 2: signals.append(f"{sh} skill keywords."); score += 0.06

    # 6. Bullets.
    bl = len(re.findall(r"(?:^|\n)\s*[\u2022\u25aa\u25b8\u25e6\u2023\-\*\u2013]\s+\w", text))
    if bl >= 5: signals.append(f"{bl} bullet points."); score += 0.07
    elif bl >= 2: score += 0.04

    # 7. Negative.
    for p, lb in [
        (r"\btable of contents\b", "Table of Contents"),
        (r"\bdear\s+(?:sir|madam|hiring|mr|ms)\b", "Cover letter"),
        (r"\bchapter\s+\d+\b", "Chapter headings"),
    ]:
        if re.search(p, text_lower): warn.append(lb + "."); score -= 0.08

    conf = max(0.0, min(1.0, score))
    log.info("validate_resume: conf=%.2f, pass=%s, signals=%d, warns=%d",
             conf, conf >= 0.40, len(signals), len(warn))
    return conf >= 0.40, conf, signals, warn


def detect_language(text: str) -> str:
    """Detect primary language of *text*."""
    try:
        from langdetect import DetectorFactory, detect
        DetectorFactory.seed = 0
        lang = detect(text[:5000])
        log.info("Language detected: %s", lang)
        return lang
    except ImportError:
        raise RuntimeError("pip install langdetect")
    except Exception as exc:
        log.warning("Language detection failed (%s), defaulting to 'en'", exc)
        return "en"


def build_stopwords(lang_code: str) -> set[str]:
    """Assemble stopword set for *lang_code*."""
    sw: set[str] = set()
    _map = {"en": "en", "es": "es", "fr": "fr", "de": "de", "pt": "pt",
            "it": "it", "nl": "nl", "ca": "ca", "ru": "ru",
            "zh-cn": "zh", "zh-tw": "zh", "ja": "ja", "ko": "ko",
            "ar": "ar", "hi": "hi", "pl": "pl", "sv": "sv"}
    try:
        sw.update(spacy.blank(_map.get(lang_code, lang_code)).Defaults.stop_words)
    except Exception:
        pass

    def _add(src):
        sw.update(src.get("universal", set()))
        sw.update(src.get(lang_code, set()))
        sw.update(src.get("en", set()))

    _add(SECTION_HEADERS)
    _add(ACTION_VERBS_AND_FILLER)
    _add(DEGREE_TERMS)
    _add(DATE_TERMS)
    log.info("Stopwords for '%s': %d total", lang_code, len(sw))
    return sw


def load_spacy_model(lang_code):
    """Load the best available spaCy model for *lang_code*."""
    prefs = {
        "en": ["en_core_web_sm"], "es": ["es_core_news_sm"],
        "fr": ["fr_core_news_sm"], "de": ["de_core_news_sm"],
        "pt": ["pt_core_news_sm"], "it": ["it_core_news_sm"],
        "nl": ["nl_core_news_sm"], "ca": ["ca_core_news_sm"],
        "zh-cn": ["zh_core_web_sm"], "ja": ["ja_core_news_sm"],
        "ko": ["ko_core_news_sm"], "ru": ["ru_core_news_sm"],
        "pl": ["pl_core_news_sm"],
    }
    for name in prefs.get(lang_code, []) + ["xx_ent_wiki_sm"]:
        try:
            nlp = spacy.load(name)
            log.info("Loaded spaCy model: %s", name)
            return nlp, name
        except OSError:
            log.debug("Model '%s' not installed, trying next", name)
            continue
    # Last resort — blank pipeline (no NER, but POS still works).
    log.warning("No trained model for '%s', using blank pipeline", lang_code)
    return spacy.blank("xx"), "xx (blank)"


def yake_extract(text, lang):
    """YAKE keyword extraction (lower score = more relevant)."""
    import yake
    ext = yake.KeywordExtractor(
        lan=lang.split("-")[0], n=2, dedupLim=0.7,
        dedupFunc="seqm", windowsSize=1, top=80,
    )
    return ext.extract_keywords(text)


def matches_filler_pattern(text):
    return any(p.match(text) for p in FILLER_PATTERNS)


def clean_term(term, lang="en", separator="_"):
    """Normalise a term for display, using *separator* between words."""
    cjk_punct = "\u3002\u3001\uff0c\uff1b\uff1a\uff01\uff1f"
    term = term.strip().strip(string.punctuation + cjk_punct).strip()
    if not term:
        return ""
    if lang in ("zh-cn", "zh-tw", "zh", "ja", "ko"):
        return term
    # Normalise internal whitespace to the separator.
    term = re.sub(r"\s+", "_", term)
    parts = []
    for p in term.split("_"):
        if not p:
            continue
        if p.upper() in PRESERVE_UPPER:
            parts.append(p.upper())
        elif p.isupper() and len(p) <= 5:
            parts.append(p)
        else:
            parts.append(p.capitalize())
    return separator.join(parts)


def is_valid_term(term, excluded, stopwords):
    """Decide whether *term* belongs on the banner."""
    if not term:
        return False
    raw = term.replace("_", " ").lower().strip()
    # Strip trailing possessives: "Professionals'" → "professionals"
    raw = re.sub(r"['\u2019]s?\b", "", raw).strip()
    if len(raw) < 2:
        return False
    if raw in stopwords or raw in excluded or raw in ALL_RESUME_METADATA:
        return False
    if re.match(r"^[\d\W]+$", raw, re.UNICODE):
        return False
    if matches_filler_pattern(raw):
        return False

    words = raw.split()

    # --- Single-word check ---
    if len(words) == 1:
        if raw in ALL_NEVER_ALONE:
            return False

    # --- Compound-term checks (2+ words only) ---
    if len(words) >= 2:
        # Banned substrings (exact phrase matches).
        for banned in ALL_BANNED_SUBSTRINGS:
            if banned in raw:
                return False

        # Component contamination — strip possessives from each word
        # before checking ("Professionals'" → "professionals").
        for w in words:
            w_clean = re.sub(r"['\u2019]s?$", "", w)
            if w_clean in COMPOUND_CONTAMINANTS:
                return False
            # Also reject if a component is a dynamically-excluded term
            # (e.g. person's name, city from header parsing or NER).
            if w_clean in excluded:
                return False

        # Repeated words ("Department_Department").
        if len(set(words)) == 1:
            return False

    # Reject 3+ word compounds (almost always YAKE artefacts).
    if len(words) > 2:
        return False

    # Allow single CJK characters; reject single Latin chars except R, C.
    if len(raw) == 1:
        if (unicodedata.category(raw[0]).startswith("L")
                and ord(raw[0]) < 0x2E80):
            return raw.upper() in ("R", "C")
    return True


def _extract_header_exclusions(text: str) -> set[str]:
    """Parse the resume header (first ~500 chars) and return terms to exclude.

    This catches the person's name, city, state, country, email fragments,
    GitHub usernames, LinkedIn slugs, and phone digits — all of which are
    personal info that should never appear on a word cloud.

    Runs before spaCy NER so personal info is excluded even if the
    trained model misses something.
    """
    excl: set[str] = set()
    header = text[:600]
    lines = [ln.strip() for ln in header.splitlines() if ln.strip()]
    if not lines:
        return excl

    # --- 1. First non-empty line is almost always the person's name ---
    name_line = lines[0]
    # Strip phone numbers and emails that might sit on the same line.
    name_line = re.sub(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "", name_line)
    name_line = re.sub(r"\+?\d[\d\-\.\s\(\)]{7,15}\d", "", name_line)
    name_line = re.sub(r"https?://\S+", "", name_line)
    name_line = re.sub(r"github\.com/\S+", "", name_line)
    name_line = re.sub(r"linkedin\.com/in/\S+", "", name_line)
    # Whatever remains is likely the name.
    for word in name_line.split():
        w = word.strip(string.punctuation).lower()
        if len(w) >= 2:
            excl.add(w)

    # --- 2. Scan the full header for contact / location artefacts ---
    header_lower = header.lower()

    # Email local parts (before @).
    for m in re.finditer(r"([a-zA-Z0-9._%+-]+)@", header):
        for part in re.split(r"[._]", m.group(1)):
            if len(part) >= 2:
                excl.add(part.lower())

    # GitHub / LinkedIn usernames.
    for m in re.finditer(r"github\.com/([a-zA-Z0-9_-]+)", header, re.I):
        excl.add(m.group(1).lower())
    for m in re.finditer(r"linkedin\.com/in/([a-zA-Z0-9_-]+)", header, re.I):
        excl.add(m.group(1).lower())
    # Short form: "in/pbachas" without the full linkedin.com domain.
    for m in re.finditer(r"(?:^|\s)in/([a-zA-Z0-9_-]+)", header):
        excl.add(m.group(1).lower())

    # US states (full names and abbreviations).
    us_states = {
        "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
        "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
        "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
        "maine", "maryland", "massachusetts", "michigan", "minnesota",
        "mississippi", "missouri", "montana", "nebraska", "nevada",
        "new hampshire", "new jersey", "new mexico", "new york",
        "north carolina", "north dakota", "ohio", "oklahoma", "oregon",
        "pennsylvania", "rhode island", "south carolina", "south dakota",
        "tennessee", "texas", "utah", "vermont", "virginia", "washington",
        "west virginia", "wisconsin", "wyoming", "district of columbia",
    }
    for st in us_states:
        if st in header_lower:
            excl.add(st)
            for w in st.split():
                excl.add(w)

    # Country names commonly found on resumes.
    countries = {
        "united states", "united kingdom", "canada", "australia", "india",
        "germany", "france", "spain", "italy", "brazil", "mexico",
        "argentina", "colombia", "chile", "peru",
    }
    for c in countries:
        if c in header_lower:
            excl.add(c)
            for w in c.split():
                if len(w) >= 3:
                    excl.add(w)

    # City + State patterns like "Miami, FL" or "Coral Gables, Florida".
    city_state = re.findall(
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*,\s*"
        r"([A-Z]{2}|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        header,
    )
    for city, state in city_state:
        for w in city.lower().split():
            excl.add(w)
        for w in state.lower().split():
            excl.add(w)

    log.info("Header exclusions: %d terms — %s",
             len(excl), sorted(excl)[:20])
    return excl


def extract_terms(resume_text, progress_callback=None,
                  lang_override=None, separator="_"):
    """Extract weighted professional terms using spaCy + YAKE.

    Pipeline:
        1. Header parsing   → personal-info exclusions
        2. spaCy NER        → entity exclusions (names, orgs, locations, dates)
        3. spaCy noun chunks → candidate skill phrases
        4. spaCy POS tokens  → candidate single-word skills
        5. YAKE keywords     → statistically important terms
        6. Merge, weight, rank → top MAX_TERMS + runner-ups
    """
    def _ui(msg):
        if progress_callback:
            progress_callback(msg)

    log.info("extract_terms: %d chars, lang_override=%s, sep='%s'",
             len(resume_text or ""), lang_override, separator)

    # -- Language detection --
    if lang_override:
        lang = lang_override
        _ui(f"Language (manual): {lang}")
    else:
        _ui("Detecting language \u2026")
        lang = detect_language(resume_text)
        _ui(f"Detected language: {lang}")

    stopwords = build_stopwords(lang)

    # -- 1. Header parsing (always runs) --
    _ui("Parsing header for personal info \u2026")
    excluded = _extract_header_exclusions(resume_text)

    # -- 2–4. spaCy: NER, noun phrases, POS tokens --
    _ui("Loading NLP model \u2026")
    nlp, model_name = load_spacy_model(lang)
    _ui(f"Model: {model_name}")

    _ui("Parsing resume \u2026")
    doc = nlp(resume_text[:100_000])
    log.debug("spaCy doc: %d tokens, %d ents", len(doc), len(doc.ents or []))

    # NER exclusions.
    excl_labels = {
        "PERSON", "PER", "ORG", "GPE", "LOC", "DATE", "TIME",
        "MONEY", "CARDINAL", "ORDINAL", "FAC", "NORP", "EVENT",
        "QUANTITY", "PERCENT",
    }
    for ent in (doc.ents or []):
        if ent.label_ in excl_labels:
            excluded.add(ent.text.lower().strip())
            for w in ent.text.lower().split():
                if len(w) > 2:
                    excluded.add(w)
    log.debug("Total exclusions: %d", len(excluded))

    # Noun phrases.
    _ui("Extracting noun phrases \u2026")
    noun_phrases = []
    skip_pos = {"DET", "PRON", "PUNCT", "SPACE", "NUM", "SYM",
                "ADP", "CCONJ", "SCONJ", "AUX", "PART"}
    if doc.has_annotation("DEP"):
        for chunk in doc.noun_chunks:
            toks = [t for t in chunk
                    if t.pos_ not in skip_pos
                    and t.text.lower() not in stopwords
                    and len(t.text) > 1 and not t.is_stop]
            if toks:
                phrase = " ".join(t.text for t in toks).strip()
                if len(phrase) > 2:
                    noun_phrases.append(phrase)
    log.debug("Noun phrases: %d", len(noun_phrases))

    # POS-filtered tokens.
    _ui("Collecting tokens \u2026")
    tech_tokens = []
    has_pos = doc.has_annotation("TAG") or doc.has_annotation("POS")
    for token in doc:
        txt = token.text.strip()
        if not txt or len(txt) < 3:
            continue
        if txt.lower() in stopwords or txt.lower() in excluded:
            continue
        if token.like_num or token.like_email or token.like_url:
            continue
        if all(c in string.punctuation for c in txt):
            continue
        if has_pos:
            if token.pos_ not in ("NOUN", "PROPN", "ADJ"):
                continue
            if token.is_stop:
                continue
        tech_tokens.append(txt)
    log.debug("Tech tokens: %d", len(tech_tokens))

    # -- 5. YAKE keywords --
    _ui("Running YAKE \u2026")
    yake_results = yake_extract(resume_text, lang)
    log.info("YAKE returned %d candidates", len(yake_results))

    # -- 6. Merge, weight, rank --
    _ui("Scoring terms \u2026")
    all_terms = []
    for phrase in noun_phrases:
        c = clean_term(phrase, lang, separator)
        if is_valid_term(c, excluded, stopwords):
            all_terms.extend([c] * NOUN_PHRASE_MULTIPLIER)
    for term, score in yake_results:
        c = clean_term(term, lang, separator)
        if is_valid_term(c, excluded, stopwords):
            reps = max(1, min(YAKE_MULTIPLIER_CAP, int(6 / (score + 0.1))))
            all_terms.extend([c] * reps)
    for tok in tech_tokens:
        c = clean_term(tok, lang, separator)
        if is_valid_term(c, excluded, stopwords):
            all_terms.extend([c] * TOKEN_MULTIPLIER)

    counts = Counter(all_terms)
    if not counts:
        log.warning("No valid terms after filtering — returning fallback")
        return {"Professional": 5000, "Skills": 5000}, {}, lang, model_name

    mx = max(counts.values())
    log_mx = math.log(mx + 1)
    all_ranked = counts.most_common()

    weighted = {}
    for term, count in all_ranked[:MAX_TERMS]:
        n = math.log(count + 1) / log_mx if log_mx > 0 else 1.0
        weighted[term] = int(WEIGHT_MIN + (WEIGHT_MAX - WEIGHT_MIN) * n)

    runner_ups: dict[str, int] = {}
    for term, count in all_ranked[MAX_TERMS : MAX_TERMS + 30]:
        n = math.log(count + 1) / log_mx if log_mx > 0 else 1.0
        runner_ups[term] = int(WEIGHT_MIN + (WEIGHT_MAX - WEIGHT_MIN) * n)

    log.info("Final terms: %d, runner-ups: %d", len(weighted), len(runner_ups))
    _ui(f"Done \u2014 {len(weighted)} terms.")
    return weighted, runner_ups, lang, model_name


def find_font(lang):
    """Find a system font for *lang* or None."""
    fm = {
        "cjk": ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/System/Library/Fonts/PingFang.ttc",
                "C:\\Windows\\Fonts\\msyh.ttc"],
        "arabic": ["/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",
                   "C:\\Windows\\Fonts\\arial.ttf"],
        "devanagari": ["/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
                       "C:\\Windows\\Fonts\\mangal.ttf"],
    }
    if lang in ("zh-cn", "zh-tw", "zh", "ja", "ko"):
        fam = "cjk"
    elif lang in ("ar", "fa", "ur", "he"):
        fam = "arabic"
    elif lang in ("hi", "mr", "ne"):
        fam = "devanagari"
    else:
        return None
    for p in fm[fam]:
        if os.path.isfile(p):
            return p
    return None


def generate_wordcloud(terms, background_color="white", colormap="turbo",
                       output_path="linkedin_banner.png", lang="en"):
    """Render and save a LinkedIn-banner word cloud."""
    log.info("generate_wordcloud: %d terms, bg=%s, cmap=%s, lang=%s",
             len(terms), background_color, colormap, lang)
    rand = {}
    for t, w in terms.items():
        j = int(w * 0.10)
        rand[t] = random.randint(max(100, w - j), min(WEIGHT_MAX, w + j))

    kw = dict(width=BANNER_WIDTH, height=BANNER_HEIGHT,
              background_color=background_color, colormap=colormap,
              prefer_horizontal=0.7, min_font_size=8, max_words=200,
              collocations=False, regexp=r"[\w\-\+\#']+")
    fp = find_font(lang)
    if fp:
        kw["font_path"] = fp
        log.debug("Using font: %s", fp)

    wc = WordCloud(**kw).generate_from_frequencies(rand)
    fig, ax = plt.subplots(figsize=(15.84, 3.96))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(output_path, format="png", dpi=300,
                bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    # PNG is lossless — the 'quality' param is ignored for PNG.
    # If the file exceeds the limit, reduce colours (quantise).
    img = Image.open(output_path)
    fsize = os.path.getsize(output_path) / (1024 * 1024)
    if fsize > MAX_FILE_SIZE_MB:
        log.info("PNG too large (%.2f MB), quantising to 256 colours", fsize)
        img = img.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
        img.save(output_path, format="png", optimize=True)
        fsize = os.path.getsize(output_path) / (1024 * 1024)
    log.info("Saved: %s (%.2f MB)", output_path, fsize)
    return output_path


# ===================================================================== #
#                          TKINTER GUI                                   #
# ===================================================================== #

# -- High-contrast palette (tested on macOS, Linux, Windows) ------------
CLR_BG = "#e8ecf1"            # main window background
CLR_FRAME = "#ffffff"          # card/panel background
CLR_PRIMARY = "#1d4ed8"        # blue — buttons, accents
CLR_PRIMARY_HV = "#1e40af"     # blue hover
CLR_DANGER = "#b91c1c"         # red — remove button
CLR_DANGER_HV = "#991b1b"      # red hover
CLR_SUCCESS = "#15803d"        # green — analyse / add
CLR_SUCCESS_HV = "#166534"     # green hover
CLR_TEXT = "#0f172a"           # near-black body text
CLR_TEXT_LIGHT = "#475569"     # secondary text (status, hints)
CLR_TREE_BG = "#ffffff"        # treeview background
CLR_TREE_FG = "#0f172a"        # treeview text
CLR_TREE_SEL = "#bfdbfe"       # treeview selection highlight
CLR_TREE_HEAD_BG = "#cbd5e1"   # treeview header background
CLR_TREE_HEAD_FG = "#0f172a"   # treeview header text

# Cross-platform font — pick the first available.
_FONT_FAMILY = "Segoe UI"     # Windows
if sys.platform == "darwin":
    _FONT_FAMILY = "Helvetica Neue"
elif sys.platform.startswith("linux"):
    _FONT_FAMILY = "DejaVu Sans"


def _downloads_folder() -> Path:
    """Return the best output folder: /data (Docker) → ~/Downloads → ~."""
    docker_data = Path("/data")
    if docker_data.is_dir():
        return docker_data
    dl = Path.home() / "Downloads"
    if dl.is_dir():
        return dl
    return Path.home()


class WordCloudApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("LinkedIn Banner Word Cloud Generator")
        self.configure(bg=CLR_BG)
        self.minsize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        self.update_idletasks()
        # Size to 80% of screen so all controls are visible on launch.
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w = max(WINDOW_MIN_WIDTH, int(sw * 0.80))
        h = max(WINDOW_MIN_HEIGHT, int(sh * 0.85))
        x = (sw - w) // 2
        y = max(0, (sh - h) // 2 - 30)   # nudge up slightly for taskbar
        self.geometry(f"{w}x{h}+{x}+{y}")

        self.terms = {}
        self.runner_ups = {}
        self.user_excluded = set()       # terms the user explicitly removed
        self._lock = threading.Lock()    # guards terms/runner_ups/user_excluded
        self.detected_lang = "en"
        self.spacy_model = ""
        self.resume_text = ""
        self.preview_image = None
        self._sort_col = "weight"          # current sort column
        self._sort_reverse = True           # descending by default
        self.bg_var = tk.StringVar(value="white")
        self.palette_var = tk.StringVar(value="turbo")
        self.separator_var = tk.StringVar(value="_")
        self.lang_var = tk.StringVar(value="auto")
        self._build_ui()

    # -- UI --

    def _build_ui(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        # -- Global font shorthand --
        _f = _FONT_FAMILY
        _fn = (_f, 10)
        _fb = (_f, 10, "bold")
        _fs = (_f, 9)
        _fsi = (_f, 9, "italic")
        _fh = (_f, 16, "bold")

        # -- Card frames --
        style.configure("Card.TLabelframe",
                        background=CLR_FRAME, borderwidth=1, relief="solid")
        style.configure("Card.TLabelframe.Label",
                        background=CLR_FRAME, foreground=CLR_TEXT, font=_fb)

        # -- Labels --
        style.configure("TLabel",
                        background=CLR_BG, foreground=CLR_TEXT, font=_fn)
        style.configure("Header.TLabel",
                        background=CLR_BG, foreground=CLR_TEXT, font=_fh)
        style.configure("Sub.TLabel",
                        background=CLR_BG, foreground=CLR_TEXT_LIGHT, font=_fs)
        style.configure("Status.TLabel",
                        background=CLR_BG, foreground=CLR_TEXT_LIGHT, font=_fsi)

        # -- Radio buttons --
        style.configure("TRadiobutton",
                        background=CLR_FRAME, foreground=CLR_TEXT, font=_fn)

        # -- Combobox --
        style.configure("TCombobox", font=_fn, padding=4)

        # -- Entry --
        style.configure("TEntry", font=_fn)

        # -- Treeview (high contrast) --
        style.configure("Treeview",
                        background=CLR_TREE_BG,
                        foreground=CLR_TREE_FG,
                        fieldbackground=CLR_TREE_BG,
                        font=_fn,
                        rowheight=24)
        style.configure("Treeview.Heading",
                        background=CLR_TREE_HEAD_BG,
                        foreground=CLR_TREE_HEAD_FG,
                        font=_fb,
                        relief="flat",
                        padding=4)
        style.map("Treeview.Heading",
                  background=[("active", "#94a3b8")])
        style.map("Treeview",
                  background=[("selected", CLR_TREE_SEL)],
                  foreground=[("selected", CLR_TREE_FG)])

        # -- ttk.Button styles (work on macOS unlike tk.Button) --
        for name, bg, bg_hv, fg in [
            ("Primary.TButton", CLR_PRIMARY, CLR_PRIMARY_HV, "#ffffff"),
            ("Success.TButton", CLR_SUCCESS, CLR_SUCCESS_HV, "#ffffff"),
            ("Danger.TButton",  CLR_DANGER,  CLR_DANGER_HV,  "#ffffff"),
        ]:
            style.configure(name,
                            background=bg, foreground=fg,
                            font=(_f, 9, "bold"),
                            padding=(14, 5), relief="flat", borderwidth=0)
            style.map(name,
                      background=[("active", bg_hv), ("pressed", bg_hv)],
                      foreground=[("active", fg), ("pressed", fg)])

        outer = tk.Frame(self, bg=CLR_BG)
        outer.pack(fill="both", expand=True)
        canvas = tk.Canvas(outer, bg=CLR_BG, highlightthickness=0)
        sb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        self.scroll_frame = tk.Frame(canvas, bg=CLR_BG)
        self.scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        win_id = canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        # Stretch inner frame to fill canvas width when window is resized.
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(win_id, width=e.width))
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        # Cross-platform scroll: macOS sends delta in different units.
        def _on_scroll(event):
            if sys.platform == "darwin":
                canvas.yview_scroll(-event.delta, "units")
            else:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_scroll)

        c = tk.Frame(self.scroll_frame, bg=CLR_BG)
        c.pack(fill="both", expand=True, padx=24, pady=12)

        ttk.Label(c, text="LinkedIn Banner Word Cloud Generator", style="Header.TLabel").pack(anchor="w", pady=(0, 2))
        ttk.Label(c, text="Multilingual \u00b7 Offline \u00b7 NLP-Powered", style="Sub.TLabel").pack(anchor="w", pady=(0, 16))

        # File section.
        f = tk.Frame(c, bg=CLR_BG)
        f.pack(fill="x", pady=(0, 4))
        self.file_var = tk.StringVar(value="No file selected")
        ttk.Label(f, textvariable=self.file_var).pack(side="left", fill="x", expand=True)
        self._btn(f, "Browse \u2026", self._browse_file, CLR_PRIMARY).pack(side="right", padx=(8, 0))
        self._btn(f, "Analyse Resume", self._run_extraction, CLR_SUCCESS).pack(side="right", padx=(8, 0))

        self.status_var = tk.StringVar(value="Ready \u2014 select a resume file.")
        ttk.Label(c, textvariable=self.status_var, style="Status.TLabel").pack(anchor="w", pady=(4, 8))

        # Term table.
        lf = ttk.LabelFrame(c, text="  Extracted Terms  ", style="Card.TLabelframe")
        lf.pack(fill="both", expand=True, pady=(0, 8))
        cols = ("term", "weight", "bar")
        self.tree = ttk.Treeview(lf, columns=cols, show="headings", height=18)
        self.tree.heading("term", text="Term ▽", command=lambda: self._sort_by("term"))
        self.tree.heading("weight", text="▼ Weight", command=lambda: self._sort_by("weight"))
        self.tree.heading("bar", text="Relative")
        self.tree.column("term", width=280, anchor="w"); self.tree.column("weight", width=80, anchor="e"); self.tree.column("bar", width=200, anchor="w")
        vsb = ttk.Scrollbar(lf, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
        vsb.pack(side="right", fill="y", padx=(0, 8), pady=8)

        # Edit section.
        ef = tk.Frame(c, bg=CLR_BG)
        ef.pack(fill="x", pady=(0, 8))
        # Row 0: Remove selected + Add term.
        self._btn(ef, "\u2212 Remove Selected", self._remove_selected, CLR_DANGER).grid(row=0, column=0, padx=(0, 16))
        tk.Label(ef, text="Add term:", bg=CLR_BG, font=(_FONT_FAMILY, 10), fg=CLR_TEXT).grid(row=0, column=1, sticky="w", padx=(0, 6))
        self.add_entry = ttk.Entry(ef, width=26)
        self.add_entry.grid(row=0, column=2, padx=(0, 6))
        self._btn(ef, "+ Add", self._add_term, CLR_SUCCESS).grid(row=0, column=3)
        # Row 1: Runner-up promotion.
        tk.Label(ef, text="Runner-ups:", bg=CLR_BG, font=(_FONT_FAMILY, 10), fg=CLR_TEXT).grid(row=1, column=0, sticky="w", padx=(0, 6), pady=(8, 0))
        self.runner_combo = ttk.Combobox(ef, state="readonly", width=30)
        self.runner_combo.grid(row=1, column=1, columnspan=2, sticky="w", padx=(0, 6), pady=(8, 0))
        self._btn(ef, "\u2191 Promote", self._promote_runner, CLR_PRIMARY).grid(row=1, column=3, pady=(8, 0))

        # Appearance.
        af = ttk.LabelFrame(c, text="  Appearance  ", style="Card.TLabelframe")
        af.pack(fill="x", pady=(0, 8))
        inner = tk.Frame(af, bg=CLR_FRAME)
        inner.pack(fill="x", padx=12, pady=8)

        # Row 0: Background + Palette.
        tk.Label(inner, text="Background:", bg=CLR_FRAME, font=(_FONT_FAMILY, 10), fg=CLR_TEXT).grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Radiobutton(inner, text="Light", variable=self.bg_var, value="white").grid(row=0, column=1, padx=(0, 8))
        ttk.Radiobutton(inner, text="Dark", variable=self.bg_var, value="black").grid(row=0, column=2, padx=(0, 24))
        tk.Label(inner, text="Palette:", bg=CLR_FRAME, font=(_FONT_FAMILY, 10), fg=CLR_TEXT).grid(row=0, column=3, sticky="w", padx=(0, 8))
        pals = ["turbo \u2014 Vibrant", "gray \u2014 Mono", "ocean \u2014 Ocean", "hot \u2014 Hot",
                "rainbow \u2014 Rainbow", "viridis \u2014 Viridis", "plasma \u2014 Plasma", "inferno \u2014 Inferno"]
        combo_pal = ttk.Combobox(inner, values=pals, state="readonly", width=20)
        combo_pal.current(0)
        combo_pal.grid(row=0, column=4)
        combo_pal.bind("<<ComboboxSelected>>", lambda e: self.palette_var.set(combo_pal.get().split(" \u2014 ")[0].strip()))

        # Row 1: Separator + Language.
        tk.Label(inner, text="Separator:", bg=CLR_FRAME, font=(_FONT_FAMILY, 10), fg=CLR_TEXT).grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(8, 0))
        ttk.Radiobutton(inner, text="Underscore (Data_Science)", variable=self.separator_var, value="_").grid(row=1, column=1, columnspan=2, sticky="w", padx=(0, 8), pady=(8, 0))
        ttk.Radiobutton(inner, text="Space (Data Science)", variable=self.separator_var, value=" ").grid(row=1, column=3, sticky="w", padx=(0, 24), pady=(8, 0))

        tk.Label(inner, text="Language:", bg=CLR_FRAME, font=(_FONT_FAMILY, 10), fg=CLR_TEXT).grid(row=1, column=4, sticky="e", padx=(16, 8), pady=(8, 0))
        langs = [
            "auto \u2014 Detect automatically",
            "en \u2014 English", "es \u2014 Espa\u00f1ol", "fr \u2014 Fran\u00e7ais",
            "de \u2014 Deutsch", "pt \u2014 Portugu\u00eas", "it \u2014 Italiano",
            "nl \u2014 Nederlands", "ca \u2014 Catal\u00e0",
            "ru \u2014 \u0420\u0443\u0441\u0441\u043a\u0438\u0439",
            "pl \u2014 Polski", "sv \u2014 Svenska", "tr \u2014 T\u00fcrk\u00e7e",
            "zh-cn \u2014 \u4e2d\u6587 (simplified)", "zh-tw \u2014 \u4e2d\u6587 (traditional)",
            "ja \u2014 \u65e5\u672c\u8a9e", "ko \u2014 \ud55c\uad6d\uc5b4",
            "ar \u2014 \u0627\u0644\u0639\u0631\u0628\u064a\u0629",
            "hi \u2014 \u0939\u093f\u0928\u094d\u0926\u0940",
        ]
        combo_lang = ttk.Combobox(inner, values=langs, state="readonly", width=24)
        combo_lang.current(0)
        combo_lang.grid(row=1, column=5, sticky="w", pady=(8, 0))
        combo_lang.bind("<<ComboboxSelected>>", lambda e: self.lang_var.set(combo_lang.get().split(" \u2014 ")[0].strip()))
        inner.columnconfigure(5, weight=1)

        # Actions.
        bf = tk.Frame(c, bg=CLR_BG)
        bf.pack(fill="x", pady=(0, 8))
        self._btn(bf, "Generate Word Cloud", self._run_generate, CLR_PRIMARY, width=22).pack(side="left", padx=(0, 8))
        self._btn(bf, "Save As \u2026", self._save_as, CLR_PRIMARY, width=14).pack(side="left", padx=(0, 8))
        self._btn(bf, "Export Terms (.txt)", self._export_terms, CLR_PRIMARY, width=18).pack(side="left")

        # Preview.
        pf = ttk.LabelFrame(c, text="  Preview  ", style="Card.TLabelframe")
        pf.pack(fill="both", expand=True, pady=(0, 16))
        self.preview_label = tk.Label(pf, bg=CLR_FRAME, text="Word cloud will appear here.", fg=CLR_TEXT_LIGHT, font=(_FONT_FAMILY, 10, "italic"))
        self.preview_label.pack(fill="both", expand=True, padx=8, pady=8)

    @staticmethod
    def _btn(parent, text, command, colour, width=None):
        """Create a ttk.Button with the matching colour style."""
        style_map = {
            CLR_PRIMARY: "Primary.TButton",
            CLR_DANGER:  "Danger.TButton",
            CLR_SUCCESS: "Success.TButton",
        }
        sty = style_map.get(colour, "Primary.TButton")
        kw = dict(text=text, command=command, style=sty, cursor="hand2")
        if width:
            kw["width"] = width
        return ttk.Button(parent, **kw)

    # -- Actions --

    def _browse_file(self):
        path = filedialog.askopenfilename(
            title="Select Resume / CV",
            initialdir=str(_downloads_folder()),
            filetypes=[("All supported", "*.pdf *.docx *.txt"), ("PDF", "*.pdf"), ("Word", "*.docx"), ("Text", "*.txt")])
        if path:
            self.file_var.set(path)

    def _set_status(self, msg):
        self.status_var.set(msg)
        self.update_idletasks()

    def _populate_tree(self):
        for row in self.tree.get_children():
            self.tree.delete(row)
        if not self.terms:
            return
        mx = max(self.terms.values())
        # Sort according to current sort state.
        if self._sort_col == "term":
            ordered = sorted(self.terms.items(),
                             key=lambda x: x[0].lower(),
                             reverse=self._sort_reverse)
        else:  # weight
            ordered = sorted(self.terms.items(),
                             key=lambda x: x[1],
                             reverse=self._sort_reverse)
        for t, w in ordered:
            bar = "\u2588" * (int((w / mx) * 20) if mx else 0)
            self.tree.insert("", "end", values=(t, w, bar))

    def _sort_by(self, col):
        """Toggle sort column/direction and update heading indicators."""
        if self._sort_col == col:
            self._sort_reverse = not self._sort_reverse
        else:
            self._sort_col = col
            self._sort_reverse = col == "weight"   # weight defaults desc, term defaults asc
        # Update heading text with arrow indicators.
        arrow_dn, arrow_up = "\u25bc", "\u25b2"   # ▼ ▲
        no_arrow = "\u25bd"                        # ▽ (inactive)
        if self._sort_col == "term":
            t_arrow = arrow_dn if self._sort_reverse else arrow_up
            self.tree.heading("term", text=f"{t_arrow} Term")
            self.tree.heading("weight", text=f"Weight {no_arrow}")
        else:
            w_arrow = arrow_dn if self._sort_reverse else arrow_up
            self.tree.heading("term", text=f"Term {no_arrow}")
            self.tree.heading("weight", text=f"{w_arrow} Weight")
        self._populate_tree()

    def _run_extraction(self):
        path = self.file_var.get()
        if not path or path == "No file selected":
            messagebox.showwarning("No file", "Select a resume first.")
            return
        log.info("--- Extraction started: %s ---", path)
        self._set_status("Reading file \u2026")

        def _worker():
            try:
                self.resume_text = extract_text_from_file(path)
                self._set_status(f"{len(self.resume_text):,} chars.  Validating \u2026")

                ok, conf, sigs, wrns = validate_resume(self.resume_text)
                pct = int(conf * 100)

                if not ok:
                    log.info("Validation failed (%d%%), asking user", pct)
                    rpt = self._fmt_report(sigs, wrns)
                    proceed = tk.BooleanVar(value=False)
                    evt = threading.Event()
                    def _ask():
                        proceed.set(messagebox.askyesno(
                            "Document Validation",
                            f"Doesn\u2019t look like a resume ({pct}%).\n\n{rpt}\n\nProceed anyway?"))
                        evt.set()
                    self.after(0, _ask); evt.wait()
                    if not proceed.get():
                        log.info("User cancelled after validation failure")
                        self.after(0, self._set_status, "Cancelled."); return
                elif conf < 0.65:
                    self.after(0, self._set_status, f"Probably a resume ({pct}%).  Proceeding \u2026")
                else:
                    self.after(0, self._set_status, f"Validated ({pct}%).  Running NLP \u2026")

                lang_choice = self.lang_var.get()
                sep = self.separator_var.get()
                log.info("Settings: lang=%s, separator='%s'", lang_choice, sep)
                terms, runner_ups, lang, model = extract_terms(
                    self.resume_text,
                    progress_callback=lambda m: self.after(0, self._set_status, m),
                    lang_override=None if lang_choice == "auto" else lang_choice,
                    separator=sep,
                )
                with self._lock:
                    self.terms, self.runner_ups = terms, runner_ups
                    self.user_excluded = set()  # fresh extraction → reset
                self.detected_lang, self.spacy_model = lang, model
                self.after(0, self._populate_tree)
                self.after(0, self._populate_runners)
                self.after(0, self._set_status, f"Done \u2014 {len(terms)} terms, {len(runner_ups)} runner-ups (lang: {lang}, model: {model})")
            except Exception as exc:
                log.error("Extraction failed: %s", exc, exc_info=True)
                self.after(0, lambda: messagebox.showerror("Error", str(exc)))
                self.after(0, self._set_status, "Error.")
        threading.Thread(target=_worker, daemon=True).start()

    @staticmethod
    def _fmt_report(signals, warnings_list):
        parts = []
        if signals:
            parts.append("Positive signals:")
            parts.extend(f"  \u2713 {s}" for s in signals)
        if warnings_list:
            if parts: parts.append("")
            parts.append("Concerns:")
            parts.extend(f"  \u26a0 {w}" for w in warnings_list)
        return "\n".join(parts)

    def _add_term(self):
        raw = self.add_entry.get().strip()
        if not raw: return
        sep = self.separator_var.get()
        with self._lock:
            for p in raw.split(","):
                c = clean_term(p.strip(), self.detected_lang, sep)
                if c:
                    self.terms[c] = MANUAL_ADD_WEIGHT
                    self.user_excluded.discard(c)
                    log.debug("Manual add: '%s' (weight=%d)", c, MANUAL_ADD_WEIGHT)
        self.add_entry.delete(0, "end")
        self._populate_tree()

    def _remove_selected(self):
        """Remove highlighted rows; auto-backfill from runner-ups."""
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("Nothing selected", "Click a term in the table first.")
            return
        with self._lock:
            for iid in sel:
                term = self.tree.item(iid, "values")[0]
                self.terms.pop(term, None)
                self.user_excluded.add(term)
                log.debug("Removed → excluded: '%s'", term)

            # Auto-backfill: pull top runner-ups to keep count at MAX_TERMS.
            while len(self.terms) < MAX_TERMS and self.runner_ups:
                # Find highest-weight runner-up not in user_excluded.
                best_term, best_w = None, -1
                for t, w in self.runner_ups.items():
                    if t not in self.user_excluded and w > best_w:
                        best_term, best_w = t, w
                if best_term is None:
                    break
                self.terms[best_term] = best_w
                del self.runner_ups[best_term]
                log.debug("Auto-backfilled: '%s' (%d)", best_term, best_w)
        self._populate_tree()
        self._populate_runners()

    def _promote_runner(self):
        """Move the selected runner-up into the main term list."""
        sel = self.runner_combo.get()
        if not sel:
            return
        # Parse "Term (weight)" format.
        match = re.match(r"^(.+?)\s*\((\d+)\)$", sel)
        if match:
            term, weight = match.group(1), int(match.group(2))
        else:
            term, weight = sel, MANUAL_ADD_WEIGHT
        with self._lock:
            self.terms[term] = weight
            self.runner_ups.pop(term, None)
            self.user_excluded.discard(term)   # un-exclude if re-promoted
        log.debug("Promoted runner-up: '%s' (%d)", term, weight)
        self._populate_tree()
        self._populate_runners()

    def _populate_runners(self):
        """Refresh the runner-up combobox, excluding user-removed terms."""
        items = [f"{t} ({w})" for t, w in
                 sorted(self.runner_ups.items(), key=lambda x: x[1], reverse=True)
                 if t not in self.user_excluded]
        self.runner_combo["values"] = items
        if items:
            self.runner_combo.current(0)
        else:
            self.runner_combo.set("")

    def _run_generate(self):
        if not self.terms:
            messagebox.showwarning("No terms", "Analyse a resume first.")
            return
        log.info("--- Generation started: %d terms, bg=%s, palette=%s ---",
                 len(self.terms), self.bg_var.get(), self.palette_var.get())
        self._set_status("Generating \u2026")
        def _worker():
            try:
                out = generate_wordcloud(
                    self.terms, background_color=self.bg_var.get(),
                    colormap=self.palette_var.get(),
                    output_path=str(_downloads_folder() / "linkedin_banner.png"),
                    lang=self.detected_lang)
                mb = os.path.getsize(out) / (1024 * 1024)
                self.after(0, self._show_preview, out)
                self.after(0, self._set_status, f"Saved: {out}  ({mb:.2f} MB)")
            except Exception as exc:
                log.error("Generation failed: %s", exc, exc_info=True)
                self.after(0, lambda: messagebox.showerror("Error", str(exc)))
                self.after(0, self._set_status, "Error.")
        threading.Thread(target=_worker, daemon=True).start()

    def _show_preview(self, path):
        img = Image.open(path)
        mw = max(600, self.winfo_width() - 80)
        img = img.resize((mw, int(img.height * mw / img.width)), Image.LANCZOS)
        self.preview_image = ImageTk.PhotoImage(img)
        self.preview_label.configure(image=self.preview_image, text="")

    def _save_as(self):
        src = str(_downloads_folder() / "linkedin_banner.png")
        if not os.path.isfile(src):
            messagebox.showwarning("No banner", "Generate one first."); return
        dest = filedialog.asksaveasfilename(
            defaultextension=".png", filetypes=[("PNG", "*.png")],
            initialfile="linkedin_banner.png",
            initialdir=str(_downloads_folder()))
        if dest:
            import shutil; shutil.copy2(src, dest)
            log.info("Saved copy: %s", dest)
            self._set_status(f"Saved to: {dest}")

    def _export_terms(self):
        if not self.terms:
            messagebox.showwarning("No terms", "Nothing to export."); return
        dest = filedialog.asksaveasfilename(
            defaultextension=".txt", filetypes=[("Text", "*.txt")],
            initialfile="extracted_terms.txt",
            initialdir=str(_downloads_folder()))
        if dest:
            with open(dest, "w", encoding="utf-8") as fh:
                fh.write(f"# Main Terms ({len(self.terms)})\n")
                for t, w in sorted(self.terms.items(), key=lambda x: x[1], reverse=True):
                    fh.write(f"{t}\t{w}\n")
                # Include runner-ups (excluding user-removed terms).
                visible = {t: w for t, w in self.runner_ups.items()
                           if t not in self.user_excluded}
                if visible:
                    fh.write(f"\n# Runner-Ups ({len(visible)})\n")
                    for t, w in sorted(visible.items(), key=lambda x: x[1], reverse=True):
                        fh.write(f"{t}\t{w}\n")
                # Note excluded terms if any.
                if self.user_excluded:
                    fh.write(f"\n# Excluded ({len(self.user_excluded)})\n")
                    for t in sorted(self.user_excluded):
                        fh.write(f"{t}\n")
            total = len(self.terms) + len(visible)
            log.info("Exported %d terms + %d runner-ups to: %s",
                     len(self.terms), len(visible), dest)
            self._set_status(f"Exported to: {dest}")


def main():
    log.info("Log file: %s", LOG_FILE)
    WordCloudApp().mainloop()


if __name__ == "__main__":
    main()
