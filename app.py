"""
app.py — Semantic Fitness & Meal Plan Recommendation System
============================================================

Overview
--------
A single-file Gradio application that recommends personalised fitness and meal
plans from a CSV dataset using OpenAI embeddings and a Chroma vector store.
No traditional ML (no sklearn, no collaborative filtering) — pure NLP similarity
search followed by structured pandas filtering.

Pipeline (executed once at startup, then per query)
----------------------------------------------------
Startup:
  1. Load .env → validate OPENAI_API_KEY.
  2. Read dataset.csv into a pandas DataFrame; validate schema.
  3. Build (or load from disk) a Chroma vector store from the
     Semantic_Description column using text-embedding-3-small.

Per query:
  4. Embed the user's natural-language query and run similarity_search(k=20).
  5. Extract Plan_IDs from the returned document text.
  6. Filter the DataFrame to matched rows, honouring optional dropdowns
     (Gender / Fitness Goal / Dietary Preference). Dropdown value "Any" skips
     that filter entirely.
  7. Return the top-3 rows as formatted text blocks in the Gradio UI.

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
import re

# Data handling — numpy is a project dependency (requirements.txt) reserved
# for any numeric operations added in future; pandas drives all current logic.
import numpy as np
import pandas as pd

# Environment variable loader — reads .env into os.environ
from dotenv import load_dotenv

# LangChain: vector store, document loader, embeddings, text splitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

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
CHROMA_DIR = "chroma_db"          # persisted vector store lives here
TEMP_FILE  = "temp_descriptions.txt"  # intermediate text file for TextLoader


def initialize_vector_db(dataframe: pd.DataFrame) -> Chroma:
    """
    Build a Chroma vector store from Semantic_Description, or load it from
    disk if it was already built on a previous run.

    Why persist? Embedding 1 000+ rows via the OpenAI API takes time and money.
    On restart we skip re-embedding entirely by loading the saved collection.

    Args:
        dataframe: The validated fitness-plans DataFrame from STEP 1.

    Returns:
        A LangChain Chroma object ready for similarity_search().
    """

    # Create the embedding model once — used for both building and loading.
    # text-embedding-3-small is fast, cheap, and accurate for this use case.
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # ── Check for an existing persisted DB ───────────────────────────────────
    # os.listdir returns [] for an empty dir, so `and os.listdir(...)` guards
    # against a leftover empty chroma_db/ folder triggering a false positive.
    if os.path.isdir(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        print(f"Found existing Chroma DB at '{CHROMA_DIR}'. Loading from disk ...")
        vectordb = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
        )
        count = vectordb._collection.count()
        print(f"  {count} documents loaded — skipping re-embedding.")
        return vectordb

    # ── First run: write descriptions to a temp text file ────────────────────
    # TextLoader expects a plain text file; one description per line lets
    # CharacterTextSplitter treat each line as a single document chunk.
    print(f"Writing {len(dataframe)} descriptions to '{TEMP_FILE}' ...")
    with open(TEMP_FILE, "w", encoding="utf-8") as f:
        for desc in dataframe["Semantic_Description"].astype(str):
            # Flatten any embedded newlines so each row stays on one line.
            f.write(desc.strip().replace("\n", " ") + "\n")

    # ── Load the file with TextLoader ────────────────────────────────────────
    # TextLoader reads the whole file as a single Document; splitting happens
    # in the next step.
    loader  = TextLoader(TEMP_FILE, encoding="utf-8")
    raw_docs = loader.load()

    # ── Split into one chunk per plan description ────────────────────────────
    # chunk_size=500 is larger than the longest description (226 chars) so no
    # line is ever split mid-text; splitting happens purely on the "\n" separator.
    # chunk_overlap=0 → each plan is independent, no overlapping windows needed.
    splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=0,
        separator="\n",
    )
    docs = splitter.split_documents(raw_docs)

    print(f"  {len(dataframe)} dataset rows → {len(docs)} document chunks")
    if len(docs) != len(dataframe):
        # Mismatch means some descriptions contained literal newlines; the
        # replace() above should prevent this, but flag it just in case.
        print("  WARNING: chunk count != row count. Check Semantic_Description for newlines.")

    # ── Embed and persist to Chroma ──────────────────────────────────────────
    # Chroma.from_documents calls the OpenAI Embeddings API for every chunk,
    # then writes the resulting vectors + metadata to CHROMA_DIR on disk.
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
      1. Embed the query and run semantic similarity search (top 20 candidates).
      2. Extract Plan_IDs from the returned document text.
      3. Slice df to those rows, preserving the similarity-ranked order.
      4. Apply optional structured filters (gender / goal / diet).
      5. Return up to 3 results as a list of plain dicts, or an error string.

    Returning a string (not raising) on error keeps the Gradio UI alive — the
    caller can check `isinstance(result, str)` to detect and display the message.

    Args:
        user_query:     Natural-language description of the user's fitness needs.
        target_gender:  Value from the Gender dropdown, or "Any" to skip.
        target_goal:    Value from the Fitness Goal dropdown, or "Any" to skip.
        target_diet:    Value from the Dietary Preference dropdown, or "Any" to skip.

    Returns:
        List of up to 3 dicts on success, or an error/warning string on failure.
    """

    # ── 1. Validate input ────────────────────────────────────────────────────
    # Guard against empty submissions before touching the API.
    if not user_query or not user_query.strip():
        return "⚠️ Please enter a query describing your fitness needs."

    # ── 2. Semantic similarity search ────────────────────────────────────────
    # k=50 gives a large enough candidate pool so that after dietary/gender/goal
    # filters are applied (each independently reduces results by ~50-75%), at
    # least 3 plans survive. k=20 was too small when multiple filters combined.
    try:
        similar_docs = vectordb.similarity_search(user_query.strip(), k=50)
    except Exception as exc:
        return (
            f"❌ Vector search failed: {exc}\n"
            "Check your OPENAI_API_KEY and internet connection."
        )

    # ── 3. Extract Plan_IDs from document text ───────────────────────────────
    # Each document's page_content begins with its Plan_ID
    # (e.g. "PLAN_042 This plan is designed for ...").
    # We take the first whitespace-separated token and strip any stray
    # punctuation that could prevent a match against df["Plan_ID"].
    retrieved_ids: list[str] = []
    for doc in similar_docs:
        first_token = doc.page_content.strip().split()[0]
        plan_id = re.sub(r"[^\w\-]", "", first_token)   # keep alphanum + _ + -
        if plan_id:
            retrieved_ids.append(plan_id)

    if not retrieved_ids:
        return (
            "⚠️ Could not extract Plan IDs from search results.\n"
            "Check that Semantic_Description values begin with the Plan_ID."
        )

    # ── 4. Slice df to retrieved rows ────────────────────────────────────────
    # Using pd.Categorical with retrieved_ids as categories preserves the
    # similarity-ranked order returned by Chroma instead of the CSV row order.
    filtered = df[df["Plan_ID"].isin(retrieved_ids)].copy()
    filtered["_rank"] = pd.Categorical(
        filtered["Plan_ID"], categories=retrieved_ids, ordered=True
    )
    filtered = filtered.sort_values("_rank").drop(columns="_rank")

    # ── 5. Apply optional structured filters ────────────────────────────────
    # "Any" means the user doesn't care about that dimension — skip the filter.
    if target_gender and target_gender != "Any":
        filtered = filtered[filtered["Gender"] == target_gender]

    if target_goal and target_goal != "Any":
        filtered = filtered[filtered["Fitness_Goal"] == target_goal]

    if target_diet and target_diet != "Any":
        filtered = filtered[filtered["Dietary_Preference"] == target_diet]

    # ── 6. Handle zero results ───────────────────────────────────────────────
    if filtered.empty:
        return (
            "⚠️ No matching plans found. "
            "Try broadening your filters or rephrasing your query."
        )

    # ── 7 & 8. Build and return the top-3 result list ────────────────────────
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
