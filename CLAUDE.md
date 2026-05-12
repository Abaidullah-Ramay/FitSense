# CLAUDE.md — Semantic Fitness Recommendation System

> This file is automatically read by Claude Code at the start of every session.
> Do NOT delete or rename it.

---

## 📌 Project Overview & Purpose

A **Semantic Recommendation System** built with NLP and LLMs that recommends personalized
fitness and meal plans from a CSV dataset. Instead of traditional collaborative filtering,
we embed rich natural-language plan descriptions into a **Chroma vector database** and use
**OpenAI embeddings + similarity search** to surface the most relevant plans for a user query.

The user interacts through a **Gradio UI** where they type a natural language query
(e.g. "I want a safe low-impact workout for bad knees") and optionally filter by gender,
fitness goal, and dietary preference.

---

## 🛠 Tech Stack & Packages

| Layer            | Library / Tool                                      |
|------------------|-----------------------------------------------------|
| Package manager  | `uv` (not pip)                                      |
| Environment vars | `python-dotenv`                                     |
| Data handling    | `pandas`, `numpy`                                   |
| LLM provider     | `langchain-openai` → `OpenAIEmbeddings`             |
| Vector DB        | `langchain-chroma` → `Chroma`                       |
| Document loaders | `langchain-community` → `TextLoader`                |
| Text splitting   | `langchain-text-splitters` → `CharacterTextSplitter`|
| UI               | `gradio`                                            |
| Runtime model    | `gpt-4o` / `text-embedding-3-small`                 |

All packages are pinned in `requirements.txt`.

---

## 📁 Folder & File Structure

```
project-root/
├── app.py                  # Main application (single-file, fully commented)
├── dataset.csv             # Source dataset — DO NOT modify schema
├── CLAUDE.md               # This file — Claude Code reads it every session
├── TASKS.md                # Feature checklist, updated as tasks complete
├── PROMPTS.md              # Ordered prompts to send to Claude Code per phase
├── requirements.txt        # All Python dependencies
├── .env                    # Secret keys — NEVER commit
├── .env.example            # Safe template — commit this
└── chroma_db/              # Persisted Chroma vector store (auto-created at runtime)
```

---

## 🗂 Dataset Schema (`dataset.csv`)

| Column                  | Type    | Notes                                              |
|-------------------------|---------|----------------------------------------------------|
| `Plan_ID`               | String  | Unique identifier, e.g. `PLAN_001`                |
| `Gender`                | String  | `Male` / `Female` / `Non-binary`                  |
| `Age_Group`             | String  | e.g. `18-25`, `26-35`, `36-45`, `46+`             |
| `BMI_Category`          | String  | `Underweight` / `Normal` / `Overweight` / `Obese` |
| `Fitness_Goal`          | String  | `Weight Loss` / `Muscle Gain` / `Endurance` / etc.|
| `Activity_Level`        | String  | `Sedentary` / `Light` / `Moderate` / `Active`     |
| `Dietary_Preference`    | String  | `Vegan` / `Vegetarian` / `Keto` / `Balanced` etc. |
| `Medical_Conditions`    | String  | Free text, may be `None`                          |
| `Allergies_Intolerances`| String  | Free text, may be `None`                          |
| `Exercise_Schedule`     | Text    | Multi-sentence exercise plan                      |
| `Meal_Plan`             | Text    | Multi-sentence meal plan                          |
| `Nutritional_Facts`     | String  | Macro summary string                              |
| `Est_Calories_Burned`   | Integer | Estimated calories burned per week                |
| `Semantic_Description`  | Text    | Rich NL summary — **this is the embedding source**|

---

## 🔑 Key Conventions

1. **Environment variables**: All secrets loaded via `python-dotenv` from `.env`.
   Never hardcode API keys. Always check `OPENAI_API_KEY` is set before making calls.

2. **Embedding source**: Only `Semantic_Description` is embedded into Chroma.
   Structured columns (`Gender`, `Fitness_Goal`, `Dietary_Preference`) are used for
   **post-retrieval pandas filtering**, not for embedding.

3. **RAG pipeline approach**:
   - Write `Semantic_Description` values to `temp_descriptions.txt` (one per line).
   - Load with `TextLoader` → split with `CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")`.
   - Each chunk = one plan description.
   - Store in `Chroma` with `persist_directory="chroma_db"` so the DB is not rebuilt every run.
   - At query time: `similarity_search(query, k=20)` → extract Plan_IDs → pandas filter → return top 3.

4. **Plan_ID extraction**: The `Semantic_Description` field begins with the Plan_ID string.
   Use `split()[0]` or a regex on `doc.page_content` to extract it reliably.

5. **Dropdown "Any" / "All" convention**: When a Gradio dropdown value is `"Any"` or `"All"`,
   skip filtering on that column entirely. Only filter when a specific value is selected.

6. **Single-file app**: All logic lives in `app.py`. No separate modules unless explicitly
   requested. Keep functions small and well-commented.

7. **Gradio theme**: Use `gr.themes.Glass()` for the UI theme.

8. **Error handling**: Always wrap LLM/embedding calls in try/except. Surface clear error
   messages in the Gradio UI rather than crashing silently.

9. **No traditional ML models**: Do NOT use sklearn, collaborative filtering, matrix
    factorization, or any tabular ML. Pure NLP + vector similarity only.

10. **Chroma persistence**: Use `persist_directory="chroma_db"`. On subsequent runs,
    load the existing DB instead of re-embedding (check if the directory exists and is non-empty).
