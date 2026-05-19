"""
app.py — Semantic Fitness & Meal Plan Recommendation System
============================================================

Overview
--------
A single-file Gradio application that recommends personalised fitness and meal
plans from a CSV dataset using OpenAI embeddings and a Chroma vector store.
No traditional ML (no sklearn, no collaborative filtering) — pure NLP similarity
search with Chroma metadata pre-filtering for guaranteed filter compliance.

Pipeline (executed once at startup, then per query)
----------------------------------------------------
Startup:
  1. Load .env → validate OPENAI_API_KEY.
  2. Read dataset.csv into a pandas DataFrame; validate schema.
  3. Build (or load from disk) a Chroma vector store from the
     Semantic_Description column using text-embedding-3-small.

Per query:
  4. Build a Chroma metadata filter from non-"Any" dropdown values.
  5. Embed the query and run similarity_search(k=10, filter=...) within
     the already-constrained subset — every result satisfies the filters.
  6. Extract Plan_IDs from document metadata (no text parsing needed).
  7. Slice the DataFrame to matched rows; return top-3 as Gradio text blocks.

Inputs (Gradio UI)
------------------
- Your Query      : free-text natural-language description of fitness needs
- Gender          : dropdown — "Any" | "Male" | "Female" | "Non-binary"
- Fitness Goal    : dropdown — "Any" | values from dataset
- Dietary Preference : dropdown — "Any" | values from dataset

Outputs (Gradio UI)
-------------------
- Plan 1 / Plan 2 / Plan 3 : formatted text blocks, each showing:
    Plan_ID, Exercise_Schedule, Meal_Plan, Nutritional_Facts, Est_Calories_Burned

Run
---
    source betterme-app/bin/activate
    python app.py

Requirements
------------
- dataset.csv must be in the same working directory
- OPENAI_API_KEY must be set in .env
- All packages in requirements.txt installed via: uv pip install -r requirements.txt
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# Standard library
# ─────────────────────────────────────────────────────────────────────────────
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

# Gradio UI
import gradio as gr


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — ENVIRONMENT SETUP & DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

# Load all variables defined in the .env file into os.environ.
# This must run before any os.getenv() call so the key is available.
load_dotenv()

# Read the API key and strip accidental whitespace.
# os.getenv returns "" (not None) when the key is absent, thanks to the default.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# Fail fast: if the key is missing or blank, raise a clear error rather than
# letting the program crash later with a cryptic OpenAI auth exception.
if not OPENAI_API_KEY:
    raise EnvironmentError(
        "OPENAI_API_KEY is not set or is empty.\n"
        "Add it to a .env file in the project root:\n"
        "  OPENAI_API_KEY=sk-..."
    )

# ── Load the dataset ──────────────────────────────────────────────────────────
print("Loading dataset.csv ...")
df = pd.read_csv("dataset.csv")

# Print shape so we can immediately see row/column counts on startup.
print(f"  Shape      : {df.shape}")

# Print column names to confirm the CSV schema matches expectations.
print(f"  Columns    : {df.columns.tolist()}")

# Print the first 2 rows for a quick sanity-check of the actual data.
print("  First 2 rows:")
print(df.head(2).to_string(index=False))
print()

# ── Validate required columns ────────────────────────────────────────────────
# Both columns are critical: Plan_ID links results back to the DataFrame,
# and Semantic_Description is the text we embed into Chroma.
for col in ["Plan_ID", "Semantic_Description"]:
    if col not in df.columns:
        raise ValueError(
            f"Required column '{col}' not found in dataset.csv.\n"
            f"Available columns: {df.columns.tolist()}"
        )

# ── Cast Est_Calories_Burned to integer ──────────────────────────────────────
# pd.to_numeric with errors="coerce" turns non-numeric strings into NaN
# so fillna(0) catches them before the final astype(int) cast.
df["Est_Calories_Burned"] = (
    pd.to_numeric(df["Est_Calories_Burned"], errors="coerce")
    .fillna(0)
    .astype(int)
)

print("Dataset loaded and validated successfully.")
print(f"  {len(df)} plans ready.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — CHROMA VECTOR DATABASE SETUP
# ─────────────────────────────────────────────────────────────────────────────

# Paths used by the vector DB pipeline.
CHROMA_DIR = "chroma_db"  # persisted vector store lives here


def initialize_vector_db(dataframe: pd.DataFrame) -> Chroma:
    """
    Build a Chroma vector store from Semantic_Description with structured
    metadata, or load it from disk if already built on a previous run.

    Each document stores Gender, Fitness_Goal, and Dietary_Preference as
    metadata so that similarity_search can pre-filter by those dimensions
    before scoring — guaranteeing results always match the selected filters.

    Args:
        dataframe: The validated fitness-plans DataFrame from STEP 1.

    Returns:
        A LangChain Chroma object ready for similarity_search().
    """

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # ── Load existing DB if present ──────────────────────────────────────────
    if os.path.isdir(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        print(f"Found existing Chroma DB at '{CHROMA_DIR}'. Loading from disk ...")
        vectordb = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
        )
        count = vectordb._collection.count()
        print(f"  {count} documents loaded — skipping re-embedding.")
        return vectordb

    # ── First run: build Document objects with structured metadata ───────────
    # Structured columns are stored as metadata (not embedded into the vector).
    # This lets Chroma filter by Gender / Fitness_Goal / Dietary_Preference
    # before similarity scoring, so every returned doc satisfies the filters.
    print(f"Building {len(dataframe)} documents with metadata ...")
    docs = []
    for _, row in dataframe.iterrows():
        docs.append(Document(
            page_content=str(row["Semantic_Description"]).strip().replace("\n", " "),
            metadata={
                "Plan_ID":            str(row["Plan_ID"]),
                "Gender":             str(row.get("Gender", "")),
                "Fitness_Goal":       str(row.get("Fitness_Goal", "")),
                "Dietary_Preference": str(row.get("Dietary_Preference", "")),
            },
        ))

    # ── Embed and persist to Chroma ──────────────────────────────────────────
    print("Embedding documents with OpenAI — this may take a moment ...")
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    count = vectordb._collection.count()
    print(f"  Done. {count} documents embedded and saved to '{CHROMA_DIR}'.")
    return vectordb


# Build or load the vector DB at module level so it is ready before the UI
# starts. This runs once on startup; subsequent requests reuse `vectordb`.
print("\nInitialising vector database ...")
vectordb = initialize_vector_db(df)
print("Vector DB ready.\n")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — RETRIEVAL & RECOMMENDATION LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def get_recommendations(
    user_query: str,
    target_gender: str,
    target_goal: str,
    target_diet: str,
) -> list[dict] | str:
    """
    Retrieve the top fitness/meal plan recommendations for a natural-language query.

    Pipeline:
      1. Build a Chroma metadata filter from non-"Any" dropdown values.
      2. Run similarity_search(k=10, filter=...) — Chroma pre-filters to the
         matching demographic subset before ranking by cosine similarity.
         Every returned document is guaranteed to satisfy the active filters.
      3. Extract Plan_IDs from document metadata (no text parsing).
      4. Slice df to those rows, preserving similarity-ranked order.
      5. Return up to 3 results as a list of plain dicts, or an error string.

    Args:
        user_query:     Natural-language description of the user's fitness needs.
        target_gender:  Value from the Gender dropdown, or "Any" to skip.
        target_goal:    Value from the Fitness Goal dropdown, or "Any" to skip.
        target_diet:    Value from the Dietary Preference dropdown, or "Any" to skip.

    Returns:
        List of up to 3 dicts on success, or an error/warning string on failure.
    """

    # ── 1. Validate input ────────────────────────────────────────────────────
    if not user_query or not user_query.strip():
        return "⚠️ Please enter a query describing your fitness needs."

    # ── 2. Build Chroma metadata filter ─────────────────────────────────────
    # Non-"Any" values are passed as a `where` clause to Chroma so the vector
    # search operates only on the matching subset — no post-retrieval pruning.
    # ChromaDB requires $and when filtering on more than one field simultaneously.
    raw_filters: dict[str, str] = {}
    if target_gender and target_gender != "Any":
        raw_filters["Gender"] = target_gender
    if target_goal and target_goal != "Any":
        raw_filters["Fitness_Goal"] = target_goal
    if target_diet and target_diet != "Any":
        raw_filters["Dietary_Preference"] = target_diet

    if not raw_filters:
        where_clause = None
    elif len(raw_filters) == 1:
        key, val = next(iter(raw_filters.items()))
        where_clause = {key: val}
    else:
        where_clause = {"$and": [{k: v} for k, v in raw_filters.items()]}

    # ── 3. Semantic similarity search ────────────────────────────────────────
    # k=10 is enough — Chroma pre-filters to the matching subset first, then
    # returns the 10 most semantically similar docs within that subset.
    try:
        similar_docs = vectordb.similarity_search(
            user_query.strip(),
            k=10,
            filter=where_clause,
        )
    except Exception as exc:
        return (
            f"❌ Vector search failed: {exc}\n"
            "Check your OPENAI_API_KEY and internet connection."
        )

    # ── 4. Extract Plan_IDs from document metadata ───────────────────────────
    # Metadata is reliable — no regex parsing of page_content needed.
    retrieved_ids: list[str] = []
    for doc in similar_docs:
        plan_id = doc.metadata.get("Plan_ID", "")
        if plan_id:
            retrieved_ids.append(plan_id)

    if not retrieved_ids:
        return (
            "⚠️ No matching plans found. "
            "Try broadening your filters or rephrasing your query."
        )

    # ── 5. Slice df and preserve Chroma's similarity rank order ─────────────
    filtered = df[df["Plan_ID"].isin(retrieved_ids)].copy()
    filtered["_rank"] = pd.Categorical(
        filtered["Plan_ID"], categories=retrieved_ids, ordered=True
    )
    filtered = filtered.sort_values("_rank").drop(columns="_rank")

    if filtered.empty:
        return (
            "⚠️ No matching plans found. "
            "Try broadening your filters or rephrasing your query."
        )

    # ── 6. Build and return the top-3 result list ────────────────────────────
    results: list[dict] = []
    for _, row in filtered.head(3).iterrows():
        results.append({
            "Plan_ID":            str(row["Plan_ID"]),
            "Exercise_Schedule":  str(row.get("Exercise_Schedule", "N/A")),
            "Meal_Plan":          str(row.get("Meal_Plan", "N/A")),
            "Nutritional_Facts":  str(row.get("Nutritional_Facts", "N/A")),
            "Est_Calories_Burned": int(row["Est_Calories_Burned"]),
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — GRADIO DASHBOARD UI
# ─────────────────────────────────────────────────────────────────────────────
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Dropdown choices (form profiling) ────────────────────────────────────────
AGE_CHOICES      = ["18-25", "26-35", "36-45", "46+"]
GENDER_CHOICES   = ["Male", "Female", "Non-binary"]
BMI_CHOICES      = ["Underweight", "Normal", "Overweight", "Obese"]
ACTIVITY_CHOICES = ["Sedentary", "Light", "Moderate", "Active"]
GOAL_CHOICES     = sorted(df["Fitness_Goal"].dropna().unique().tolist())
DIET_CHOICES     = sorted(v for v in df["Dietary_Preference"].dropna().unique() if v != "Any")


# ── UI helper functions ───────────────────────────────────────────────────────

def _parse_daily_kcal(nutritional_facts: str) -> int:
    """Extract daily calorie intake from a Nutritional_Facts string."""
    m = re.search(r"([\d,]+)\s*kcal", str(nutritional_facts))
    return int(m.group(1).replace(",", "")) if m else 2000


def _plan_card_html(plan: dict, rank: int) -> str:
    """Render one plan as a styled HTML card."""
    medals    = ["🥇", "🥈", "🥉"]
    medal     = medals[rank] if rank < 3 else ""
    daily_cal = _parse_daily_kcal(plan["Nutritional_Facts"])
    return f"""
<div class="fit-card">
  <div class="fit-card-hdr">
    <span class="fit-medal">{medal}</span>
    <span class="fit-plan-id">{plan["Plan_ID"]}</span>
    <span class="fit-cal-chip">🔥 {plan["Est_Calories_Burned"]:,} kcal / wk burned</span>
  </div>
  <div class="fit-sec fit-exercise">
    <div class="fit-sec-lbl">🏃 Exercise Schedule</div>
    <div class="fit-sec-body">{plan["Exercise_Schedule"]}</div>
  </div>
  <div class="fit-sec fit-meal">
    <div class="fit-sec-lbl">🥗 Meal Plan</div>
    <div class="fit-sec-body">{plan["Meal_Plan"]}</div>
  </div>
  <div class="fit-sec fit-nutrition">
    <div class="fit-sec-lbl">📊 Nutritional Facts</div>
    <div class="fit-sec-body">{plan["Nutritional_Facts"]}</div>
    <div class="fit-intake-note">≈ {daily_cal:,} kcal / day &nbsp;·&nbsp; {daily_cal * 7:,} kcal / week food intake</div>
  </div>
</div>"""


def _build_chart(plans: list[dict]) -> plt.Figure:
    """Grouped bar chart: calories burned (workout) vs consumed (food) per plan."""
    labels   = [p["Plan_ID"] for p in plans]
    burned   = [p["Est_Calories_Burned"] for p in plans]
    consumed = [_parse_daily_kcal(p["Nutritional_Facts"]) * 7 for p in plans]

    n, w = len(labels), 0.35
    x    = list(range(n))

    fig, ax = plt.subplots(figsize=(max(6, n * 3.5), 5))
    fig.patch.set_facecolor("#1e293b")
    ax.set_facecolor("#0f172a")

    b1 = ax.bar([i - w / 2 for i in x], burned,   w, color="#ef4444", label="🔥 Burned  (workout)", zorder=3)
    b2 = ax.bar([i + w / 2 for i in x], consumed, w, color="#3b82f6", label="🍽️  Consumed (food)",   zorder=3)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 50, f"{int(h):,}",
                ha="center", va="bottom", color="white", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="#e2e8f0", fontsize=10)
    ax.set_ylabel("kcal / week", color="#94a3b8", fontsize=10)
    ax.set_title("Weekly Calorie Balance — Burned vs. Consumed", color="#f1f5f9", fontsize=13, pad=14)
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.yaxis.grid(True, color="#334155", linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=9)
    fig.tight_layout()
    return fig


# ── Event handlers ────────────────────────────────────────────────────────────

def on_submit(name, age, gender, bmi, goal, activity, diet, query):
    """Form submit: run recommendation, switch to dashboard view."""
    # Auto-build query from profile if the user left the text box blank
    effective_q = (query or "").strip() or (
        f"{goal or ''} {diet or ''} plan for a {bmi or 'normal'} "
        f"{activity or 'moderate'} {gender or 'person'} aged {age or ''}".strip()
    )
    results = get_recommendations(effective_q, gender or "Any", goal or "Any", diet or "Any")

    if isinstance(results, str):
        # Stay on form, surface error message
        return (
            gr.update(visible=True), gr.update(visible=False), [],
            "", "", "", "",
            gr.update(choices=[], value=[]),
            gr.update(visible=False),
            f'<p class="fit-error">{results}</p>',
        )

    user     = (name or "").strip() or "there"
    goal_txt = goal if goal else "your goal"
    diet_txt = f" · {diet}" if diet else ""
    choices  = [f"{['🥇','🥈','🥉'][i]} {r['Plan_ID']}" for i, r in enumerate(results)]

    welcome = f"""
<div class="fit-welcome">
  <h2>Welcome, {user}! 👋</h2>
  <p>Top <strong>{len(results)}</strong> plans matched for <em>{goal_txt}</em>{diet_txt}</p>
</div>"""

    cards = [_plan_card_html(r, i) for i, r in enumerate(results)]
    while len(cards) < 3:
        cards.append('<div class="fit-card fit-card-empty"><p>No additional plan found.</p></div>')

    return (
        gr.update(visible=False), gr.update(visible=True), results,
        welcome, cards[0], cards[1], cards[2],
        gr.update(choices=choices, value=[]),
        gr.update(visible=False),
        "",
    )


def on_chart_change(selected, plans_data):
    """Checkbox change: regenerate the calorie comparison chart."""
    if not selected or not plans_data:
        return gr.update(visible=False), None
    lookup = {p["Plan_ID"]: p for p in plans_data}
    chosen = [lookup[lbl.split(" ", 1)[-1]] for lbl in selected if lbl.split(" ", 1)[-1] in lookup]
    if not chosen:
        return gr.update(visible=False), None
    return gr.update(visible=True), _build_chart(chosen)


def on_back():
    """Back button: return to profile form."""
    return (
        gr.update(visible=True), gr.update(visible=False), [],
        "", "", "", "",
        gr.update(choices=[], value=[]),
        gr.update(visible=False),
        "",
    )


# ── Custom CSS ────────────────────────────────────────────────────────────────
CSS = """
.gradio-container { max-width:1280px !important; margin:0 auto !important;
    font-family:'Inter',system-ui,sans-serif !important; }

/* ── Brand ── */
.fit-brand { text-align:center; padding:44px 16px 28px; }
.fit-brand h1 { font-size:2.8rem; font-weight:800; margin-bottom:8px;
    background:linear-gradient(90deg,#6366f1,#38bdf8);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.fit-brand p { color:#94a3b8; font-size:1.05rem; }

/* ── Form card ── */
#form-card { background:rgba(30,41,59,.88) !important;
    border:1px solid rgba(99,102,241,.28) !important;
    border-radius:20px !important; padding:32px !important;
    max-width:880px; margin:0 auto;
    box-shadow:0 24px 56px rgba(0,0,0,.5) !important; }

/* ── Buttons ── */
#submit-btn { background:linear-gradient(135deg,#6366f1,#3b82f6) !important;
    border:none !important; border-radius:12px !important;
    font-size:1rem !important; font-weight:700 !important;
    width:100% !important; margin-top:14px !important;
    box-shadow:0 4px 18px rgba(99,102,241,.4) !important;
    transition:all .2s !important; }
#submit-btn:hover { transform:translateY(-2px) !important;
    box-shadow:0 8px 28px rgba(99,102,241,.55) !important; }
#back-btn { background:rgba(30,41,59,.7) !important;
    border:1px solid rgba(148,163,184,.25) !important;
    color:#94a3b8 !important; border-radius:8px !important; font-size:.85rem !important; }

/* ── Dashboard ── */
#dashboard-wrap { padding:20px 4px !important; }

/* ── Welcome strip ── */
.fit-welcome { padding:4px 0 18px; }
.fit-welcome h2 { font-size:1.75rem; font-weight:700; color:#f1f5f9; margin-bottom:4px; }
.fit-welcome p  { color:#94a3b8; font-size:.95rem; }

/* ── Section headings ── */
.fit-heading { font-size:1.15rem; font-weight:700; color:#e2e8f0;
    margin:28px 0 10px; padding-bottom:8px;
    border-bottom:1px solid rgba(99,102,241,.3); }
.fit-hint { color:#64748b; font-size:.85rem; margin:0 0 10px; }

/* ── Plan cards ── */
.fit-card { background:rgba(15,23,42,.92) !important;
    border:1px solid rgba(99,102,241,.2) !important;
    border-radius:16px !important; overflow:hidden !important;
    transition:transform .2s,box-shadow .2s; height:100%; }
.fit-card:hover { transform:translateY(-4px);
    box-shadow:0 14px 36px rgba(99,102,241,.28); }

.fit-card-hdr { display:flex; align-items:center; gap:10px; padding:14px 18px;
    background:rgba(99,102,241,.1); border-bottom:1px solid rgba(99,102,241,.18); }
.fit-medal  { font-size:1.4rem; }
.fit-plan-id { font-size:.9rem; font-weight:700; color:#a5b4fc;
    flex:1; letter-spacing:.05em; }
.fit-cal-chip { font-size:.72rem; font-weight:600;
    background:rgba(239,68,68,.15); color:#fca5a5;
    border:1px solid rgba(239,68,68,.28); border-radius:20px;
    padding:3px 10px; white-space:nowrap; }

.fit-sec { padding:13px 18px; border-bottom:1px solid rgba(255,255,255,.05); }
.fit-sec:last-child { border-bottom:none; }
.fit-sec-lbl { font-size:.7rem; font-weight:700; letter-spacing:.08em;
    text-transform:uppercase; margin-bottom:7px; }
.fit-exercise .fit-sec-lbl { color:#34d399; }
.fit-meal     .fit-sec-lbl { color:#fbbf24; }
.fit-nutrition .fit-sec-lbl { color:#a78bfa; }
.fit-sec-body { font-size:.84rem; color:#cbd5e1; line-height:1.7; }
.fit-intake-note { margin-top:7px; font-size:.74rem; color:#a78bfa; font-weight:600; }
.fit-card-empty { display:flex; align-items:center; justify-content:center;
    min-height:200px; color:#475569; font-size:.9rem; }

/* ── Checkbox group ── */
#plan-selector { background:rgba(15,23,42,.65) !important;
    border:1px solid rgba(99,102,241,.22) !important;
    border-radius:12px !important; padding:16px 20px !important; margin:6px 0 !important; }

/* ── Chart wrapper ── */
#chart-wrap { background:rgba(15,23,42,.75) !important;
    border:1px solid rgba(99,102,241,.18) !important;
    border-radius:16px !important; padding:20px !important; margin-top:8px !important; }

/* ── Error ── */
.fit-error { background:rgba(239,68,68,.1); border:1px solid rgba(239,68,68,.3);
    color:#fca5a5; border-radius:8px; padding:10px 14px;
    font-size:.875rem; margin:8px 0; }
"""

# ── Layout ────────────────────────────────────────────────────────────────────
with gr.Blocks(title="FitSense") as demo:

    plans_state = gr.State([])

    # ══════════════════════════════════════════
    # SCREEN 1 — USER PROFILE FORM
    # ══════════════════════════════════════════
    with gr.Column(visible=True, elem_id="form-wrap") as form_col:

        gr.HTML("""
<div class="fit-brand">
  <h1>🏋️ FitSense</h1>
  <p>Complete your fitness profile and we'll match the perfect plan for you.</p>
</div>""")

        with gr.Group(elem_id="form-card"):
            name_inp = gr.Textbox(label="Your Name", placeholder="e.g. Alex")
            with gr.Row():
                age_dd      = gr.Dropdown(AGE_CHOICES,      label="Age Group",          value=None)
                gender_dd   = gr.Dropdown(GENDER_CHOICES,   label="Gender",             value=None)
                bmi_dd      = gr.Dropdown(BMI_CHOICES,      label="BMI Category",       value=None)
            with gr.Row():
                goal_dd     = gr.Dropdown(GOAL_CHOICES,     label="Fitness Goal",       value=None)
                activity_dd = gr.Dropdown(ACTIVITY_CHOICES, label="Activity Level",     value=None)
                diet_dd     = gr.Dropdown(DIET_CHOICES,     label="Dietary Preference", value=None)
            query_box = gr.Textbox(
                label="Describe your needs in your own words (optional)",
                placeholder='"e.g. low-impact cardio for bad knees, prefer morning sessions"',
                lines=2,
            )
            error_html = gr.HTML("")
            submit_btn = gr.Button("Find My Plans →", variant="primary", elem_id="submit-btn")

    # ══════════════════════════════════════════
    # SCREEN 2 — RECOMMENDATIONS DASHBOARD
    # ══════════════════════════════════════════
    with gr.Column(visible=False, elem_id="dashboard-wrap") as dashboard_col:

        with gr.Row():
            welcome_html = gr.HTML("", scale=5)
            back_btn     = gr.Button("← New Search", scale=1, size="sm", elem_id="back-btn")

        gr.HTML('<div class="fit-heading">🏆 Your Recommended Plans</div>')

        with gr.Row(equal_height=True):
            plan1_html = gr.HTML("")
            plan2_html = gr.HTML("")
            plan3_html = gr.HTML("")

        gr.HTML('<div class="fit-heading">📊 Calorie Comparison</div>')
        gr.HTML('<p class="fit-hint">Tick a plan to see its weekly calories burned vs. food intake side-by-side.</p>')

        plan_selector = gr.CheckboxGroup(
            choices=[], value=[], label="", elem_id="plan-selector",
        )

        with gr.Column(visible=False, elem_id="chart-wrap") as chart_col:
            chart_plot = gr.Plot(show_label=False)

    # ── Event wiring ──────────────────────────────────────────────────────────
    _submit_outputs = [
        form_col, dashboard_col, plans_state,
        welcome_html, plan1_html, plan2_html, plan3_html,
        plan_selector, chart_col, error_html,
    ]

    submit_btn.click(
        fn=on_submit,
        inputs=[name_inp, age_dd, gender_dd, bmi_dd, goal_dd, activity_dd, diet_dd, query_box],
        outputs=_submit_outputs,
    )

    plan_selector.change(
        fn=on_chart_change,
        inputs=[plan_selector, plans_state],
        outputs=[chart_col, chart_plot],
    )

    back_btn.click(
        fn=on_back,
        outputs=_submit_outputs,
    )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Glass(), css=CSS)
