# TASKS.md — Semantic Fitness Recommendation System

> Tick off each item as Claude Code completes it.
> Format: `- [x]` = done, `- [ ]` = pending.

---

## Phase 1 — Project Scaffolding & Environment

- [ ] Create project folder structure as defined in `CLAUDE.md`
- [ ] Create `.env` with `OPENAI_API_KEY` placeholder
- [ ] Create `.env.example` as safe commit template
- [ ] Create `requirements.txt` with all pinned dependencies
- [ ] Verify `uv` is the package manager (no pip references in README)
- [ ] Add `chroma_db/` to `.gitignore`
- [ ] Add `.env` to `.gitignore`
- [ ] Add `temp_descriptions.txt` to `.gitignore`

---

## Phase 2 — Data Loading & Inspection

- [ ] Load `dataset.csv` with `pandas` in `app.py`
- [ ] Print shape and column names on startup for sanity check
- [ ] Validate that `Semantic_Description` column is present and non-null
- [ ] Validate that `Plan_ID` column exists and values are unique
- [ ] Validate that `Est_Calories_Burned` is integer-castable
- [ ] Load `OPENAI_API_KEY` from `.env` via `python-dotenv`
- [ ] Raise a clear `EnvironmentError` if `OPENAI_API_KEY` is missing or empty

---

## Phase 3 — Vector Database Initialization

- [ ] Write `Semantic_Description` values to `temp_descriptions.txt` (one per line)
- [ ] Load file with `langchain_community.document_loaders.TextLoader`
- [ ] Split with `CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")`
- [ ] Confirm chunk count equals row count in dataset
- [ ] Initialize `OpenAIEmbeddings()` (model: `text-embedding-3-small`)
- [ ] Create `Chroma` vector store with `persist_directory="chroma_db"`
- [ ] Add logic to **skip re-embedding** if `chroma_db/` already exists and is non-empty
- [ ] Print confirmation message: how many documents are in the Chroma collection

---

## Phase 4 — Retrieval & Filtering Logic

- [ ] Implement `get_recommendations(user_query, target_gender, target_goal, target_diet)` function
- [ ] Run `similarity_search(user_query, k=20)` against the Chroma DB
- [ ] Extract `Plan_ID` from each returned document's `page_content` (first token)
- [ ] Filter original DataFrame to rows whose `Plan_ID` is in the retrieved set
- [ ] Apply conditional Gender filter (skip if value is `"Any"`)
- [ ] Apply conditional Fitness_Goal filter (skip if value is `"Any"`)
- [ ] Apply conditional Dietary_Preference filter (skip if value is `"Any"`)
- [ ] Return top 3 rows from filtered DataFrame
- [ ] Return columns: `Plan_ID`, `Exercise_Schedule`, `Meal_Plan`, `Nutritional_Facts`, `Est_Calories_Burned`
- [ ] Handle edge case: fewer than 3 results after filtering (return however many exist)
- [ ] Handle edge case: zero results → return a user-friendly message string

---

## Phase 5 — Gradio UI Dashboard

- [ ] Initialize `gr.Blocks(theme=gr.themes.Glass())` as the app shell
- [ ] Add app title and subtitle via `gr.Markdown`
- [ ] Add `gr.Textbox` for the natural language user query (with placeholder example)
- [ ] Add `gr.Dropdown` for Gender (`Any`, `Male`, `Female`, `Non-binary`)
- [ ] Add `gr.Dropdown` for Fitness_Goal (dynamic from dataset unique values + `Any`)
- [ ] Add `gr.Dropdown` for Dietary_Preference (dynamic from dataset unique values + `Any`)
- [ ] Add `gr.Button` labeled "🔍 Find My Plans" to trigger search
- [ ] Add 3 output blocks (e.g., `gr.Markdown` or `gr.Textbox`) for each recommendation
- [ ] Each output block shows: Plan ID, Exercise Schedule, Meal Plan, Nutritional Facts, Calories
- [ ] Wire button click to `get_recommendations` function
- [ ] Show a loading spinner / status while search is running
- [ ] Handle and display errors gracefully in the UI (no raw tracebacks)
- [ ] Launch app with `demo.launch()`

---

## Phase 6 — Polish & Robustness

- [ ] Add docstrings to all functions
- [ ] Add inline comments to all major code blocks
- [ ] Ensure `temp_descriptions.txt` is cleaned up after Chroma is built (optional)
- [ ] Test with at least 3 different natural language queries manually
- [ ] Test all "Any" dropdown combinations
- [ ] Test with a query that returns zero results after filtering
- [ ] Confirm app restarts without re-embedding (Chroma persistence works)
- [ ] Confirm API key error is surfaced cleanly without crashing Gradio
- [ ] Final review: no hardcoded secrets, no TODO comments, no debug prints

---

## Phase 7 — Documentation (Optional)

- [ ] Write a `README.md` with setup instructions using `uv`
- [ ] Add example screenshots to README
- [ ] Document how to regenerate the Chroma DB (delete `chroma_db/` folder and rerun)
