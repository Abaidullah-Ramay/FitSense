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
# STEP 4 — GRADIO UI
# ─────────────────────────────────────────────────────────────────────────────

# Build dropdown choices dynamically from the dataset so they always reflect
# whatever values are actually present in the CSV.
GENDER_CHOICES = ["Any", "Male", "Female", "Non-binary"]
GOAL_CHOICES   = ["Any"] + sorted(df["Fitness_Goal"].dropna().unique().tolist())
# Exclude the dataset's own "Any" value to avoid a duplicate "Any" in the dropdown.
# Plans with Dietary_Preference=="Any" in the CSV still appear when the user picks
# "Any" from the UI (no filter applied), so they are never lost.
DIET_CHOICES   = ["Any"] + sorted(
    v for v in df["Dietary_Preference"].dropna().unique() if v != "Any"
)

DIVIDER = "─" * 45


def format_plan(plan: dict) -> str:
    """Format a single result dict into a human-readable block for a Textbox."""
    return (
        f"Plan ID: {plan['Plan_ID']}\n"
        f"{DIVIDER}\n"
        f"🏃 Exercise Schedule:\n{plan['Exercise_Schedule']}\n\n"
        f"🥗 Meal Plan:\n{plan['Meal_Plan']}\n\n"
        f"📊 Nutritional Facts:\n{plan['Nutritional_Facts']}\n\n"
        f"🔥 Est. Calories Burned: {plan['Est_Calories_Burned']:,} kcal/week"
    )


def run_search(
    query: str,
    gender: str,
    goal: str,
    diet: str,
) -> tuple[str, str, str]:
    """
    Gradio event handler — bridges the UI and get_recommendations().

    Calls get_recommendations, formats each result dict into a display string,
    and pads missing slots to always return exactly 3 strings (one per output
    Textbox). Error strings from get_recommendations go into Plan 1; the other
    two boxes are left blank so the UI doesn't show stale content.

    Args:
        query:  Raw text from the query Textbox.
        gender: Selected value from the Gender dropdown.
        goal:   Selected value from the Fitness Goal dropdown.
        diet:   Selected value from the Dietary Preference dropdown.

    Returns:
        A 3-tuple of strings (plan1_text, plan2_text, plan3_text).
    """
    results = get_recommendations(query, gender, goal, diet)

    # get_recommendations returns a string on any error or warning condition.
    if isinstance(results, str):
        return results, "", ""

    # Format each plan dict and pad to exactly 3 entries.
    formatted = [format_plan(r) for r in results]
    while len(formatted) < 3:
        formatted.append("No additional plans found.")

    return formatted[0], formatted[1], formatted[2]


# ── Build the interface ───────────────────────────────────────────────────────
with gr.Blocks(title="Semantic Fitness Recommender") as demo:

    # Header
    gr.Markdown("# 🏋️ Semantic Fitness & Meal Plan Recommender")
    gr.Markdown(
        "Describe your fitness situation in plain English and get AI-matched plans."
    )

    # Query input
    with gr.Row():
        query_box = gr.Textbox(
            label="Your Query",
            placeholder=(
                "e.g. I want a safe low-impact workout for someone with bad knees "
                "and a vegan diet"
            ),
            lines=3,
        )

    # Filter dropdowns — side by side
    with gr.Row():
        gender_dd = gr.Dropdown(
            choices=GENDER_CHOICES,
            value="Any",
            label="Gender",
        )
        goal_dd = gr.Dropdown(
            choices=GOAL_CHOICES,
            value="Any",
            label="Fitness Goal",
        )
        diet_dd = gr.Dropdown(
            choices=DIET_CHOICES,
            value="Any",
            label="Dietary Preference",
        )

    # Search trigger
    search_btn = gr.Button("🔍 Find My Plans", variant="primary")

    gr.Markdown("---")
    gr.Markdown("## 🥇 Top Recommendations")

    # One output box per recommendation slot
    with gr.Row():
        out1 = gr.Textbox(label="Plan 1", lines=10, interactive=False)
        out2 = gr.Textbox(label="Plan 2", lines=10, interactive=False)
        out3 = gr.Textbox(label="Plan 3", lines=10, interactive=False)

    # Wire button → wrapper → output boxes
    search_btn.click(
        fn=run_search,
        inputs=[query_box, gender_dd, goal_dd, diet_dd],
        outputs=[out1, out2, out3],
    )


# ── Launch ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # theme moved to launch() in Gradio 6.0
    demo.launch(theme=gr.themes.Glass())
