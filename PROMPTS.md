# PROMPTS.md — Claude Code Prompt Sequence

> Send each prompt to Claude Code **in order**, one phase at a time.
> Wait for Claude Code to finish and confirm before sending the next prompt.
> Each prompt is self-contained and references `CLAUDE.md` for conventions.

---

## ▶ PHASE 1 — Project Scaffolding & Environment

```
Read CLAUDE.md carefully. Then scaffold the project:

1. Create a `.gitignore` that ignores: `.env`, `chroma_db/`, `temp_descriptions.txt`, `__pycache__/`, `*.pyc`, `.DS_Store`
2. Create `.env` with a single line: OPENAI_API_KEY=your-key-here
3. Create `.env.example` with the same line but no real value: OPENAI_API_KEY=
4. Create `requirements.txt` with these exact packages (do not pin versions yet, use latest stable):
   - python-dotenv
   - pandas
   - numpy
   - langchain-community
   - langchain-openai
   - langchain-chroma
   - langchain-text-splitters
   - gradio
   - chromadb
5. Create an empty `app.py` with a top-of-file docstring describing the project.

Confirm each file is created and show me the directory listing.
```

---

## ▶ PHASE 2 — Data Loading & Environment Setup

```
Read CLAUDE.md. Open `app.py` and implement STEP 1 only:

- At the top of app.py, import all required packages.
- Load environment variables using `load_dotenv()` from `python-dotenv`.
- Read `OPENAI_API_KEY` from the environment. If it is missing or empty, raise an
  EnvironmentError with a helpful message telling the user to set it in `.env`.
- Load `dataset.csv` using pandas into a DataFrame called `df`.
- Print the shape, column names, and first 2 rows to the console on startup.
- Validate that columns `Semantic_Description` and `Plan_ID` exist; raise a clear
  ValueError if either is missing.
- Cast `Est_Calories_Burned` to integer, coercing errors.

Add clear inline comments to every block. Do not build the Chroma DB yet.
Show me the full updated app.py when done.
```

---

## ▶ PHASE 3 — Vector Database Initialization

```
Read CLAUDE.md. Open `app.py` and implement STEP 2 — the Chroma vector DB setup:

- Write a function `initialize_vector_db(df)` that:
  1. Writes the `Semantic_Description` column to `temp_descriptions.txt`, one description per line.
  2. Loads the file using LangChain's `TextLoader`.
  3. Splits documents with `CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")`.
  4. Checks if `chroma_db/` directory already exists and is non-empty. If yes, load the
     existing Chroma collection instead of re-embedding (saves API cost on restart).
  5. If no existing DB, creates a new `Chroma` vector store using `OpenAIEmbeddings()`
     with `persist_directory="chroma_db"`.
  6. Prints how many documents are in the collection.
  7. Returns the `Chroma` vector store object.

- Call `initialize_vector_db(df)` at module level and store the result in a variable `vectordb`.

Add clear inline comments. Show me the full updated app.py.
```

---

## ▶ PHASE 4 — Retrieval & Filtering Logic

```
Read CLAUDE.md. Open `app.py` and implement STEP 3 — the recommendation function:

Implement `get_recommendations(user_query, target_gender, target_goal, target_diet)`:

1. Validate that `user_query` is not empty; return an error string if it is.
2. Run `vectordb.similarity_search(user_query, k=20)` to get the top 20 semantically
   similar documents.
3. For each returned document, extract the Plan_ID by taking the first whitespace-separated
   token from `doc.page_content`. Strip any punctuation.
4. Filter `df` to only rows whose `Plan_ID` is in the extracted list.
5. Apply optional filters:
   - If `target_gender` is not "Any", filter df where `Gender == target_gender`
   - If `target_goal` is not "Any", filter df where `Fitness_Goal == target_goal`
   - If `target_diet` is not "Any", filter df where `Dietary_Preference == target_diet`
6. If zero rows remain after filtering, return the string:
   "⚠️ No matching plans found. Try broadening your filters or rephrasing your query."
7. Take the top 3 rows (or fewer if less than 3 exist).
8. Return a list of dicts, each containing:
   { "Plan_ID", "Exercise_Schedule", "Meal_Plan", "Nutritional_Facts", "Est_Calories_Burned" }

Wrap the similarity search in a try/except and return a helpful error string on failure.
Show me the full updated app.py.
```

---

## ▶ PHASE 5 — Gradio UI Dashboard

```
Read CLAUDE.md. Open `app.py` and implement STEP 4 — the full Gradio UI:

Build the UI using `gr.Blocks(theme=gr.themes.Glass())`:

1. Add a `gr.Markdown` header: "# 🏋️ Semantic Fitness & Meal Plan Recommender"
   and a subtitle: "Describe your fitness situation in plain English and get AI-matched plans."

2. In a `gr.Row`, add:
   - `gr.Textbox(label="Your Query", placeholder="e.g. I want a safe low-impact workout for someone with bad knees and a vegan diet", lines=3)`

3. In the next `gr.Row`, add three dropdowns side by side:
   - Gender: choices = ["Any", "Male", "Female", "Non-binary"], value = "Any"
   - Fitness Goal: choices = ["Any"] + sorted(df["Fitness_Goal"].dropna().unique().tolist()), value = "Any"
   - Dietary Preference: choices = ["Any"] + sorted(df["Dietary_Preference"].dropna().unique().tolist()), value = "Any"

4. Add a `gr.Button("🔍 Find My Plans", variant="primary")`.

5. Add a `gr.Markdown("---")` separator.

6. Add `gr.Markdown("## 🥇 Top Recommendations")`.

7. Add 3 `gr.Textbox` output components (label="Plan 1", "Plan 2", "Plan 3"), each with
   `lines=10` and `interactive=False`.

8. Wire the button click to a wrapper function `run_search(query, gender, goal, diet)`
   that calls `get_recommendations` and formats each result as:
   ```
   Plan ID: {Plan_ID}
   ─────────────────────
   🏃 Exercise Schedule:
   {Exercise_Schedule}

   🥗 Meal Plan:
   {Meal_Plan}

   📊 Nutritional Facts:
   {Nutritional_Facts}

   🔥 Est. Calories Burned: {Est_Calories_Burned} kcal/week
   ```
   If fewer than 3 results, fill missing output boxes with "No additional plans found."
   If `get_recommendations` returns an error string, put it in Plan 1 output, others blank.

9. End with `demo.launch()`.

Show me the final complete app.py with all 4 steps integrated.
```

---

## ▶ PHASE 6 — Polish, Testing & Final Review

```
Read CLAUDE.md. Do a full review and polish pass on `app.py`:

1. Add a module-level docstring at the top explaining what the app does, its inputs/outputs,
   and how to run it.
2. Add docstrings to `initialize_vector_db`, `get_recommendations`, and `run_search`.
3. Ensure all inline comments are present and accurate.
4. Confirm there are no hardcoded API keys, no `print` debug statements left over
   (startup prints are fine), and no TODO comments.
5. Confirm error handling covers: missing API key, empty query, zero results, Chroma failure.
6. Confirm Chroma persistence logic works: if `chroma_db/` exists and is non-empty,
   skip re-embedding.
7. Confirm dropdown "Any" logic skips the pandas filter correctly.
8. Run a quick syntax check with `python -m py_compile app.py` and fix any errors.

Show me the final, clean, production-ready `app.py`.
```

---

## ▶ PHASE 7 (Optional) — README

```
Read CLAUDE.md. Create a `README.md` for this project with:

1. Project title and one-paragraph description.
2. Prerequisites: Python 3.10+, uv, an OpenAI API key.
3. Setup instructions using `uv`:
   uv venv
   source .venv/bin/activate   # or .venv\Scripts\activate on Windows
   uv pip install -r requirements.txt
4. Configuration: copy `.env.example` to `.env` and fill in the OpenAI key.
5. How to run: `python app.py`
6. How to reset the vector DB: delete the `chroma_db/` folder and rerun.
7. A note that `dataset.csv` must be present in the project root.
8. Tech stack table (same as in CLAUDE.md).
```
