# Semantic Fitness & Meal Plan Recommendation System
## Project Report

---

## 1. Project Description

This is a single-file Python web application that recommends personalized fitness and meal plans from a structured CSV dataset. Unlike traditional recommendation systems (collaborative filtering, matrix factorization), this system uses **pure NLP similarity search**: plan descriptions are embedded into a vector database, and a user's natural-language query is matched against them by semantic closeness.

A user types something like:

> *"I want a low-impact workout for someone with bad knees and a vegan diet"*

The system returns the three most semantically relevant plans from the dataset, optionally narrowed by Gender, Fitness Goal, and Dietary Preference dropdowns.

No model training is required. The system is stateless per-query and deterministic given the same inputs.

---

## 2. Tech Stack

| Layer               | Technology                                          | Version / Notes                        |
|---------------------|-----------------------------------------------------|----------------------------------------|
| Language            | Python                                              | 3.12                                   |
| Package manager     | `uv`                                                | Replaces pip                           |
| Environment secrets | `python-dotenv`                                     | Reads `.env` at startup                |
| Data handling       | `pandas`, `numpy`                                   | DataFrame ops + numeric casting        |
| Embedding model     | OpenAI `text-embedding-3-small`                     | Via `langchain-openai`                 |
| Vector database     | ChromaDB                                            | Via `langchain-chroma`                 |
| Document loading    | `langchain-community` → `TextLoader`                | Reads temp `.txt` file                 |
| Text splitting      | `langchain-text-splitters` → `CharacterTextSplitter`| One chunk per plan                     |
| Web UI              | Gradio 6.x (`gr.Blocks`)                            | Theme: `gr.themes.Glass()`             |
| API provider        | OpenAI                                              | Key loaded from `.env`                 |

---

## 3. File Structure

```
Betterme_Recommendation_system/
├── app.py                   # Entire application — all 4 steps
├── dataset.csv              # 200-row fitness plan dataset (source of truth)
├── requirements.txt         # Python dependencies
├── .env                     # OPENAI_API_KEY (never committed)
├── .env.example             # Safe empty template (committed)
├── .gitignore               # Ignores .env, chroma_db/, betterme-app/, etc.
├── CLAUDE.md                # Project spec read by Claude Code each session
├── TASKS.md                 # Feature checklist
├── PROMPTS.md               # Ordered prompts per phase
├── PROJECT_REPORT.md        # This document
└── chroma_db/               # Persisted Chroma vector store (auto-created at runtime)
```

---

## 4. Dataset Schema

The dataset (`dataset.csv`) has 200 rows and 14 columns:

| Column                   | Type    | Description                                               |
|--------------------------|---------|-----------------------------------------------------------|
| `Plan_ID`                | String  | Unique ID, e.g. `PLAN_001`                                |
| `Gender`                 | String  | `Male` / `Female` / `Non-binary`                          |
| `Age_Group`              | String  | `18-25`, `26-35`, `36-45`, `46+`                          |
| `BMI_Category`           | String  | `Underweight` / `Normal` / `Overweight` / `Obese`         |
| `Fitness_Goal`           | String  | `Weight Loss` / `Muscle Gain` / `Endurance` / etc.        |
| `Activity_Level`         | String  | `Sedentary` / `Light` / `Moderate` / `Active`             |
| `Dietary_Preference`     | String  | `Vegan` / `Keto` / `Halal` / `Balanced` / etc.           |
| `Medical_Conditions`     | String  | Free text or `None`                                       |
| `Allergies_Intolerances` | String  | Free text or `None`                                       |
| `Exercise_Schedule`      | Text    | Multi-sentence exercise plan                              |
| `Meal_Plan`              | Text    | Multi-sentence meal plan                                  |
| `Nutritional_Facts`      | String  | Macro summary (calories, protein, carbs, fat)             |
| `Est_Calories_Burned`    | Integer | Estimated kcal burned per week through exercise           |
| `Semantic_Description`   | Text    | Rich NL summary — **the only column embedded into Chroma**|

---

## 5. Architecture

### 5.1 Two-Phase Design

The system runs in two distinct phases:

**Phase 1 — Startup (runs once)**

```
.env ──────────────────► load_dotenv() ──► OPENAI_API_KEY validated
dataset.csv ───────────► pd.read_csv()  ──► df (200 rows × 14 cols)
                                            │
                              validate schema + cast Est_Calories_Burned
                                            │
                     ┌──── chroma_db/ exists?
                     │
                  YES │ ──► Chroma.load(persist_directory)  ──► vectordb
                     │
                  NO  └──► Semantic_Description column
                                    │
                           write to temp_descriptions.txt (one line per row)
                                    │
                           TextLoader.load() ──► raw Document
                                    │
                           CharacterTextSplitter(chunk_size=500, separator="\n")
                                    │
                           200 Document chunks
                                    │
                           OpenAI text-embedding-3-small (API call, once)
                                    │
                           Chroma.from_documents() ──► vectordb
                                    └──► persisted to chroma_db/
```

**Phase 2 — Per Query**

```
User query (text)
+ Gender / Fitness Goal / Dietary Preference (dropdowns)
        │
        ▼
vectordb.similarity_search(query, k=50)
        │
        ▼
50 Document chunks (ranked by cosine similarity)
        │
        ▼
Extract Plan_ID from first token of each doc.page_content
        │
        ▼
df[df["Plan_ID"].isin(retrieved_ids)]          (slice DataFrame)
        │
        ▼
pd.Categorical(categories=retrieved_ids)       (preserve similarity rank)
        │
        ▼
Apply structured filters (if value != "Any"):
  ├── Gender == target_gender
  ├── Fitness_Goal == target_goal
  └── Dietary_Preference == target_diet
        │
        ▼
filtered.head(3)  ──► list of 3 result dicts
        │
        ▼
format_plan() × 3  ──► 3 Gradio Textbox strings
```

---

### 5.2 Component Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│                         app.py                                      │
│                                                                      │
│  ┌─────────────┐   ┌────────────────────┐   ┌──────────────────┐  │
│  │  STEP 1     │   │  STEP 2            │   │  STEP 3          │  │
│  │  Env + Data │──►│  Vector DB Setup   │──►│  Recommendation  │  │
│  │             │   │                    │   │  Logic           │  │
│  │ load_dotenv │   │ initialize_        │   │                  │  │
│  │ pd.read_csv │   │ vector_db()        │   │ get_recommenda-  │  │
│  │ schema check│   │                    │   │ tions()          │  │
│  └─────────────┘   │ ┌──────────────┐  │   │                  │  │
│                     │ │ TextLoader   │  │   │ similarity_      │  │
│  ┌──────────┐       │ │ Splitter     │  │   │ search(k=50)     │  │
│  │dataset   │──────►│ │ OpenAI Embed│  │   │ + pandas filter  │  │
│  │.csv      │       │ │ Chroma store│  │   └──────────────────┘  │
│  └──────────┘       │ └──────────────┘  │           │             │
│                     └────────────────────┘           │             │
│                              │                       │             │
│                     ┌────────▼──────┐                │             │
│                     │  chroma_db/   │                │             │
│                     │  (persisted)  │                │             │
│                     └───────────────┘                ▼             │
│                                            ┌──────────────────┐   │
│                                            │  STEP 4          │   │
│                                            │  Gradio UI       │   │
│                                            │                  │   │
│                                            │  gr.Blocks       │   │
│                                            │  Query Textbox   │   │
│                                            │  3 Dropdowns     │   │
│                                            │  Search Button   │   │
│                                            │  3 Output Boxes  │   │
│                                            └──────────────────┘   │
└────────────────────────────────────────────────────────────────────┘

External Dependencies:
  OpenAI API ◄──── text-embedding-3-small (at startup + per query)
  .env        ──── OPENAI_API_KEY
```

---

## 6. Hyperparameters & Configuration Values

| Parameter                          | Value                        | Location            | Rationale                                                                                   |
|------------------------------------|------------------------------|---------------------|---------------------------------------------------------------------------------------------|
| Embedding model                    | `text-embedding-3-small`     | `initialize_vector_db()` | Cheap, fast, high-quality for short-to-medium text                                     |
| Similarity search candidate pool   | `k=50`                       | `get_recommendations()` | Large enough that 3 plans survive after multi-filter narrowing (gender + goal + diet)  |
| Top results returned to UI         | `3`                          | `filtered.head(3)` | Three plans give enough choice without overwhelming the user                                 |
| Chunk size (text splitter)         | `500` characters             | `CharacterTextSplitter` | Larger than the longest `Semantic_Description` (226 chars) so no line is split mid-text |
| Chunk overlap                      | `0`                          | `CharacterTextSplitter` | Each plan is independent — no benefit from overlapping context windows               |
| Splitter separator                 | `"\n"`                       | `CharacterTextSplitter` | One description per line; split on newline gives one chunk per plan                   |
| Chroma persist directory           | `"chroma_db"`                | `CHROMA_DIR`        | Standard relative path; excluded from git via `.gitignore`                                  |
| Temp file path                     | `"temp_descriptions.txt"`    | `TEMP_FILE`         | Intermediate file for TextLoader; excluded from git                                         |
| Gradio UI theme                    | `gr.themes.Glass()`          | `demo.launch()`     | Clean glass-morphism aesthetic; theme passed to `launch()` per Gradio 6.0 API change        |
| Gender choices (hardcoded)         | `Any, Male, Female, Non-binary` | `GENDER_CHOICES` | Fixed set matching dataset values                                                           |
| Goal / Diet choices (dynamic)      | Loaded from `df.unique()`    | `GOAL_CHOICES`, `DIET_CHOICES` | Always reflects actual dataset values; deduplicates the dataset "Any" value    |
| `Est_Calories_Burned` fill value   | `0`                          | `pd.to_numeric(...).fillna(0)` | Coerce invalid strings to 0 rather than NaN for safe int formatting               |

---

## 7. Key Design Decisions

### Semantic search, not structured search
Structured columns (`Gender`, `Fitness_Goal`, `Dietary_Preference`) are intentionally excluded from embeddings. Embedding them would tie retrieval to exact-match semantics. Instead, they act as post-retrieval filters applied in pandas — letting the vector search surface the 50 most semantically relevant plans, then trimming to the user's demographic constraints.

### Chroma persistence
Embedding 200 rows via the OpenAI API costs time and money. On the first run the full pipeline runs (write file → load → split → embed → persist). All subsequent runs skip to `Chroma(persist_directory=...)` and load in under a second.

### Rank preservation via `pd.Categorical`
After pandas slicing, the default sort order is CSV row order, not similarity rank. Using `pd.Categorical` with `retrieved_ids` as the ordered category list re-sorts filtered results back into Chroma's similarity-ranked order before `head(3)` is applied.

### "Any" sentinel pattern
The dataset itself contains `Dietary_Preference == "Any"` for 58 rows. The UI dropdown prepends its own "Any" sentinel to mean "skip this filter". These collide unless the dataset's "Any" values are excluded from `DIET_CHOICES` construction — which they are.

### Error surfacing over crashing
`get_recommendations` returns either a `list[dict]` or a plain `str` (error/warning message). The Gradio handler checks `isinstance(results, str)` to route errors to the UI rather than raising exceptions and killing the server.

---

## 8. Running the Application

```bash
# 1. Create and activate the virtual environment (uv)
uv venv betterme-app --python 3.12
source betterme-app/bin/activate

# 2. Install dependencies
uv pip install -r requirements.txt

# 3. Set your API key
cp .env.example .env
# Edit .env and add: OPENAI_API_KEY=sk-...

# 4. Run
python app.py
# Open http://127.0.0.1:7860 in a browser
```

On first run, the system calls the OpenAI Embeddings API to build `chroma_db/`. Subsequent runs load from disk immediately.

---

## 9. Limitations & Potential Improvements

| Limitation                                              | Possible Fix                                                      |
|---------------------------------------------------------|-------------------------------------------------------------------|
| Re-embedding required if dataset changes                | Add a hash/checksum of `dataset.csv` to detect changes automatically |
| Plan_ID extraction relies on description format         | Store Plan_ID as Chroma document metadata for robust retrieval    |
| No real-time streaming of results                       | Use Gradio streaming or SSE for long queries                      |
| k=50 is fixed regardless of filter selectivity          | Dynamically increase k when multiple strict filters are active    |
| UI outputs are plain text                               | Use `gr.Markdown` or `gr.HTML` for richer card-style layouts      |
| No query history or session state                       | Add Gradio `State` component to track past queries                |
