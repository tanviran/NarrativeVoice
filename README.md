# NarrativeVoice prototype: evidence-grounded narrative analysis

Turns interview transcripts into themes where every finding links to a participant and line in the original
transcript, and a researcher reviews and corrects the result. The demo uses **100 synthetic interviews**
(no real data). You can also upload your own non-sensitive transcripts.

## Quick start

Requires Python 3.12 (tested) and an internet connection the first time, to download two small models (about 300 MB in total).

**1. Get the code and create an environment**
```
git clone https://github.com/tanviran/narrativevoice.git
cd narrativevoice

# Option A: conda
conda create -n narrativevoice python=3.12 -y
conda activate narrativevoice

# Option B: venv (no conda needed)
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

**2. Install the dependencies**
```
pip install -r requirements.txt
```

**3. Start the dashboard**
```
streamlit run app.py               # then open http://localhost:8501
```
Results for the 100 synthetic interviews are already included in `results/`, so the dashboard works straight away.
The first search on the "Map and search" page downloads the embedding model (all-MiniLM-L6-v2), so allow a minute.

**4. (Optional) Rebuild everything from scratch**
```
python src/generate_data.py        # 100 synthetic interviews, with interviewer questions
python src/pipeline.py             # embeddings, tone, 2D map (a few minutes the first time)
```

**5. (Optional) Enable "Draft insight" with a local LLM**

The AI-drafted insight on the "Explore a theme" page runs on your own machine through [Ollama](https://ollama.com),
so no interview text is sent to an outside service.
```
# 1. Install Ollama from https://ollama.com/download (macOS, Windows, Linux)
# 2. Download the model used here (about 2 GB):
ollama pull llama3.2:3b
# 3. Make sure Ollama is running. The desktop app starts it automatically;
#    otherwise run this in a separate terminal and leave it open:
ollama serve
# 4. Quick check that the model answers:
ollama run llama3.2:3b "Say hello in five words."
```
Then restart the dashboard. If Ollama is not running, the page shows a clear message and the rest of the app still works.
The model name and address are set at the top of `src/insights.py`.

**6. (Optional) Evaluation and tests**
```
python src/evaluate.py             # compares models and methods, writes docs/model_comparison.md
pip install -r requirements-dev.txt
python -m pytest tests -q          # 7 fast tests
```
Offline test mode (no model downloads): `NV_EMBEDDER=tfidf python src/pipeline.py`.

**Troubleshooting**
- *"No results yet"* on start: run steps 4's two commands.
- *"Could not reach the local model"*: Ollama is not running, or the model is not pulled (step 5).
- *Slow first run*: model downloads only happen once and are cached by Hugging Face.

## The dashboard, in the order a researcher works
| Page | Purpose |
|---|---|
| Start here | What it does and does not do, a health check, and where to go next |
| Overview | How many different people raised each theme, tone, which questions produced which themes |
| Explore a theme | Quotes (one per participant, verified), the whole interview around a quote, optional AI-drafted insight |
| Map and search | Every answer as a point (select to read), and search by meaning |
| Patterns | Which themes go together (lift and significance), and whether groups differ |
| Quality checks | Live evidence check, voices that do not fit, contrast cases, privacy scan |
| Review and report | Accept, rename or reject themes, audit log, downloadable report |
| Methods and choices | What was chosen, alternatives, and measured comparisons |
| Upload transcripts | Format check, then the same analysis on your own files |

## Project layout
```
app.py            entry point: wires the pages together
ui/               one module per dashboard page, plus shared helpers and the sidebar
src/              the analysis: loader, embed, tone, themes, relations, evidence, quality, privacy,
                  insights (optional LLM), pipeline, evaluate, generate_data
tests/            unit tests for the parts that must not silently break
docs/             decisions.md (why), architecture.md (diagrams), model_comparison.md (after evaluate.py)
```

## Why the choices were made
See `docs/decisions.md`: each decision with the alternatives, the reason, the cost, and when to revisit it.

## Limitations
Synthetic template-based data, approximate tone, exploratory subgroup comparisons, a pattern-based privacy scan,
and no fairness audit of a trained classifier yet. Full list in `docs/decisions.md`.
