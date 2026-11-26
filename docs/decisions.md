# Technologies and design decisions

Short notes on what I chose for the prototype and why. Measured comparisons are in `model_comparison.md`, and the production design is in section 8 of `technical_documentation.md`.

## 1. Synthetic data from templates

The brief asks for synthetic or public non-sensitive data, and I preferred to use synthetic data for this prototype. I wrote a generator that builds 100 interviews from templates, with a fixed seed. Because I control the topics, I know what the pipeline should find, so I can score it. LLM-written interviews would read more naturally, but I could not control or check what patterns they contain. The cost is that template answers are repetitive, so results look cleaner than real interviews would.

## 2. One answer as the unit

An interview covers many topics, so one label per interview says little, and single sentences are often too short to carry meaning. I treat each participant answer (one transcript line) as the unit. It is easy to quote and it has a line number, which the quote check needs. The interviewer's question is stored next to each answer but is not mixed into the text, because embedding it would make answers group by the wording of the question. Follow-up probes such as "Can you say more?" keep the previous main question. A long answer that covers two topics gets only one theme, which would matter with real data.

## 3. Sentence embeddings

I use all-MiniLM-L6-v2. It matches meaning ("moved homes" and "changed placements") where keyword counts do not, it is small enough for a laptop, and it runs locally. A hosted embedding API would send participant text to a third party, which I did not want for vulnerable participants. I also ran MPNet, a larger model, and TF-IDF as a baseline. With k-means, MPNet scored lower than MiniLM on this data, so the larger model did not earn its extra time.

## 4. K-means with k chosen by silhouette

K-means gives every answer a theme, the cluster centre gives an objective "most typical" answer, and it is easy to explain and adjust, so the dashboard lets the researcher change k live. I compared agglomerative clustering and HDBSCAN. HDBSCAN scored higher on topic recovery (0.65 against 0.37 for k-means with MiniLM) but left 55% of answers without any theme, so I did not use it. Silhouette picked 15 themes for the 8 planted topics, and stability across random starts is only 0.42. The dashboard shows both numbers and the researcher can override k. I did not ask an LLM to read everything and name the themes, because that cannot be checked.

## 5. Theme labels from distinctive words

Labels are the words most distinctive to each theme, computed from the data, so they cannot add framing the answers do not support. They are rough, and the researcher renames them on the Review page. An LLM could write nicer names, but I did not want it to introduce interpretation at this step.

## 6. Quotes: typical, one per participant, verified

For each theme I show the answers nearest the cluster centre, at most one per participant, so a single talkative person does not look like a pattern. Every quote is checked against the original file and must appear word for word at that participant and line. On the demo data 992 of 992 answers pass. Typical answers can hide unusual voices, so the Quality checks page lists the ones that do not fit.

## 7. Tone shown separately

Embeddings group by topic, so "I stayed in one placement for years" and "I was moved seven times" can land in the same theme. A small sentiment model (DistilBERT) labels each answer positive, negative or uncertain so the split is visible. It was trained on film reviews, so it agrees with the planted stance 79.9% of the time (the word-list fallback gets 64.0%), and it is weak on mixed answers. I treat tone as a prompt to read, not a result.

## 8. Links between themes: lift and Fisher's test

Raw shared-participant counts make big themes look related to everything. Lift asks whether two themes appear in the same people more often than their sizes predict, and Fisher's exact test with a Bonferroni correction says how surprising that is across the many pairs. Co-occurrence is not causation, and with 100 people many pairs are noisy, which the page says. I did not try to extract cause and effect from the text because it cannot be verified.

## 9. Optional local LLM for insights

The brief asks for AI-generated insights, so I added one, but it is optional. It runs through Ollama (llama3.2:3b) on the same machine, sees only verified quotes for one theme, and must cite evidence numbers for every claim. A draft is rejected if it cites evidence that does not exist, names a participant ID, or contains a number that is not in the cited quotes. A hosted API would be stronger but sends data out. The checks cannot catch a claim that overreaches while citing real quotes, so a researcher reads every draft.

## 10. Human review with an audit log

The AI proposes and the researcher decides. Each theme can be accepted, renamed, marked as needing edits or rejected, with notes. Every change is logged with who, when and the old and new value, and the report contains accepted themes only. With several reviewers this would need accounts, roles and an agreement measure between coders.

## 11. Quality checks and a privacy scan

Summaries smooth over untidy voices, so the dashboard lists answers that sit far from their theme and answers whose tone is the opposite of the theme majority. It also has a live test of the quote checker. A pattern-based scan flags emails, phone numbers, postcodes and names for a person to check. It misses nicknames and school names, so it supports a human check and does not replace it.

## 12. Upload with a strict format

Transcripts are `INTERVIEWER:` and `PARTICIPANT:` lines, one file per participant, and the Upload page lists any line it cannot read. A strict format keeps the speaker and line number reliable, which verification depends on. Guessing the structure of free text with an LLM would put errors at the start of the pipeline that nobody could check. Real transcripts would need converting first, and a production version would add importers and speech to text.

## 13. Streamlit, plain files, everything local

Streamlit was the quickest way to build live filters, a selectable map and a review step in Python. Results are stored as CSV, NPY and JSON files that are easy to inspect, with nothing to install. Everything runs on the machine because the narratives are sensitive, and a local pipeline needs no data sharing agreement with an API vendor. The cost is smaller models, one process and no user accounts. The production path (React, Django, PostgreSQL with pgvector, queued workers) is in section 8 of `technical_documentation.md`.

## How I compared the options

Because the demo data has planted topics, I could score each method on whether it recovers them. I compared three embedders (MiniLM, MPNet, TF-IDF), four clustering approaches (k-means with automatic k, k-means with k=8, agglomerative, HDBSCAN) and two tone methods. The measures were adjusted Rand index, normalised mutual information, the share of answers given a theme, stability across random starts, and run time. I report the share assigned because a method can score well by leaving most answers unassigned. The data is template-built and easier than real interviews, so the scores rank methods and do not estimate real accuracy. Run `python src/evaluate.py` to reproduce them. The results are in `model_comparison.md` and on the Methods page of the dashboard.
