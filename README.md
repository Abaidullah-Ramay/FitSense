# FitSense — Semantic Fitness & Meal Plan Recommender

FitSense is an AI-powered fitness recommendation app that understands plain English. Describe your situation — your goals, limitations, lifestyle — and FitSense finds the most semantically relevant workout and meal plans from a curated dataset. No forms to fill out. No rigid filters. Just talk to it.

---

## How It Works

FitSense uses **OpenAI embeddings + ChromaDB vector search** instead of traditional recommendation algorithms. Each fitness plan's natural-language description is embedded into a vector database at startup. When you submit a query, your text is embedded the same way and compared against every plan by cosine similarity — the closest matches surface first.

Optional dropdowns (Gender, Fitness Goal, Dietary Preference) apply structured filters *after* semantic search, so you get plans that match both your words and your demographic constraints.

```
Your query
    │
    ▼
OpenAI text-embedding-3-small
    │
    ▼
ChromaDB similarity_search (top 50 candidates)
    │
    ▼
pandas filter  →  Gender / Fitness Goal / Dietary Preference
    │
    ▼
Top 3 plans  →  Gradio UI
```

---

## Features

- Natural-language query — describe your needs in any words
- Semantic matching — finds plans by meaning, not keyword overlap
- Optional structured filters — Gender, Fitness Goal, Dietary Preference
- Persistent vector DB — embeddings built once, loaded instantly on restart
- Top-3 results — Plan ID, Exercise Schedule, Meal Plan, Nutritional Facts, Calories Burned
- Gradio web UI — runs locally in the browser, no frontend code needed

---

## Tech Stack

| Layer | Technology |
|---|---|
| Embedding model | OpenAI `text-embedding-3-small` |
| Vector database | ChromaDB (via `langchain-chroma`) |
| Document pipeline | `TextLoader` + `CharacterTextSplitter` |
| Data handling | `pandas`, `numpy` |
| Web UI | Gradio 6.x (`gr.Blocks`, Glass theme) |
| Environment | Python 3.12, `uv` package manager |
| Secrets | `python-dotenv` |

---

## Project Structure

```
FitSense/
├── app.py                  # Full application — env setup, vector DB, logic, UI
├── dataset.csv             # 200 fitness & meal plans (source of truth)
├── requirements.txt        # Python dependencies
├── .env                    # Your OpenAI API key (never committed)
├── .env.example            # Safe template to copy from
├── .gitignore
├── CLAUDE.md               # Project spec for Claude Code sessions
├── PROJECT_REPORT.md       # Detailed technical report
├── PROJECT_REPORT.pdf      # PDF version of the report
└── chroma_db/              # Auto-created — persisted vector embeddings
```

---

## Quickstart

### 1. Clone and enter the project

```bash
git clone <your-repo-url>
cd FitSense
```

### 2. Create a virtual environment

```bash
uv venv betterme-app --python 3.12
source betterme-app/bin/activate        # macOS / Linux
# betterme-app\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
uv pip install -r requirements.txt
```

### 4. Add your OpenAI API key

```bash
cp .env.example .env
```

Open `.env` and fill in your key:

```
OPENAI_API_KEY=sk-...
```

### 5. Run

```bash
python app.py
```

Open **http://127.0.0.1:7860** in your browser.

> **First run only:** FitSense calls the OpenAI Embeddings API to build the `chroma_db/` vector store. This takes ~10–20 seconds and costs a small amount of API credit. All subsequent runs load from disk instantly.

---

## Usage

1. Type your fitness situation in the **Your Query** box.
   - *"I'm a 30-year-old male looking to lose weight with a keto diet"*
   - *"Low-impact cardio for someone recovering from a knee injury"*
   - *"I travel a lot and need a bodyweight muscle-building routine"*

2. Optionally narrow results with the three dropdowns.

3. Click **Find My Plans**.

FitSense returns up to three plans, each showing:

```
Plan ID: PLAN_042
─────────────────────────────────────────────
🏃 Exercise Schedule:
   Mon: 30-min jog + core. Wed: HIIT 20 min ...

🥗 Meal Plan:
   Breakfast: Oats with berries ...

📊 Nutritional Facts:
   2,200 kcal/day | 160g protein | 220g carbs | 70g fat

🔥 Est. Calories Burned: 2,450 kcal/week
```

---

## Dataset

The dataset contains **200 fitness plans** covering a wide range of:

- **Gender:** Male, Female, Non-binary
- **Age Group:** 18-25, 26-35, 36-45, 46+
- **BMI Category:** Underweight, Normal, Overweight, Obese
- **Fitness Goals:** Weight Loss, Muscle Gain, Endurance, Flexibility, General Fitness
- **Activity Levels:** Sedentary, Light, Moderate, Active
- **Dietary Preferences:** Vegan, Keto, Halal, Balanced, and more
- **Medical Conditions & Allergies:** captured in Semantic_Description for NLP matching

---

## Configuration

Key hyperparameters in `app.py`:

| Parameter | Value | Why |
|---|---|---|
| Embedding model | `text-embedding-3-small` | Fast, cheap, high-quality |
| Similarity search pool | `k=50` | Survives multi-filter narrowing |
| Results shown | `3` | Enough choice, not overwhelming |
| Chunk size | `500` chars | Larger than longest description (226 chars) |
| Chroma persist dir | `chroma_db/` | Skip re-embedding on restart |

---

## Requirements

- Python 3.12+
- An [OpenAI API key](https://platform.openai.com/api-keys)
- Internet connection (for the OpenAI Embeddings API on first run)

---

## License

MIT
