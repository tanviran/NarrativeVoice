# Design decisions: what we chose, what else we could have done, and why not

## Why keep this document at all?

1. **Defensibility.** Anyone reviewing the work (a supervisor, an ethics panel, a commissioner) can ask
   "why did you do it this way?" A recorded reason is stronger than a memory.
2. **Reproducibility.** Someone else can rerun the analysis and know which choices drove the results.
3. **Honesty about trade-offs.** Every method has weaknesses. Writing them down stops them being hidden.
4. **Handover.** The people who build a system are often not the people who maintain it.
5. **Governance.** Work with vulnerable participants needs an audit trail of how their words were processed.

Format of each record: the decision, options considered, what we chose, why, the cost of the choice,
and what would make us change it.

---

## D1. Synthetic data, generated from templates
- **Options:** (a) real interviews, (b) public datasets, (c) LLM-generated interviews, (d) template-generated interviews.
- **Chosen:** (d). The brief requires synthetic or public non-sensitive data.
- **Why:** it is safe, it lets us plant known patterns (which topics co-occur, a regional effect) so we can TEST
  whether the pipeline finds them, and it is fully reproducible (fixed random seed).
- **Why not the others:** real data is out of scope and unsafe. Public datasets rarely have care-experienced
  interviews with questions and metadata. LLM-generated interviews would be more natural, but we could not control or
  verify what patterns they contain.
- **Cost:** templates are repetitive, so results look cleaner than real interviews. Scores from evaluate.py show
  how methods compare, not how accurate the system would be on real data.
- **Revisit when:** real, consented data is available.

## D2. Keep the interviewer's question with every answer
- **Options:** (a) answers only, (b) answers plus question as metadata, (c) embed question and answer together.
- **Chosen:** (b). Themes are found from the answer text. The question is stored and shown.
- **Why:** a short answer such as "Yes, mostly" means nothing without its question. Keeping the question lets the
  dashboard show which questions produced which themes. Embedding question plus answer (c) would make answers to the
  same question cluster together because of the question wording, not what people said.
- **Follow-up probes** ("Can you say more?") carry no topic, so those answers keep the previous main question.
- **Cost:** answers that raise a topic unprompted, under a different question, stay in their own theme but are
  displayed with the question they were given.
- **Revisit when:** a study asks very open questions where the question adds little.

## D3. One answer (one transcript line) as the unit of analysis
- **Options:** (a) whole interview, (b) sentence, (c) answer.
- **Chosen:** (c).
- **Why:** an interview covers many topics, so one label per interview is meaningless. Sentences are often too short
  to carry meaning on their own. An answer is a natural, quotable unit and maps to a line number for verification.
- **Cost:** a long answer covering two topics gets one theme. Our synthetic answers are mostly single-topic.
- **Revisit when:** real answers are long and multi-topic: split into sentences or paragraphs.

## D4. Sentence embeddings (all-MiniLM-L6-v2) to represent meaning
- **Options:** (a) keyword counts/TF-IDF, (b) topic model such as LDA, (c) small embedding model,
  (d) larger embedding model (all-mpnet-base-v2), (e) a hosted embedding API.
- **Chosen:** (c), with (a) and (d) available for comparison (NV_EMBEDDER=tfidf or mpnet).
- **Why:** embeddings match meaning ("moved homes" and "changed placements"), which keyword methods miss.
  MiniLM is small, fast on a laptop and runs locally, so no participant text leaves the machine.
- **Why not (a)/(b):** short answers share few exact words, so counts are sparse. In our offline test the TF-IDF
  baseline recovered the planted topics poorly (adjusted Rand index about 0.2) and gave unstable themes. Run
  `python src/evaluate.py minilm mpnet tfidf` to see all three on your machine.
- **Why not (e):** sends participant text to a third party, which is hard to justify for vulnerable participants.
- **Cost:** small models are weaker than large ones, and they capture topic better than tone.
- **Revisit when:** evaluate.py shows mpnet is clearly better and the extra time is acceptable.

## D5. K-means clustering, with k chosen by silhouette score
- **Options:** (a) k-means, (b) hierarchical (agglomerative) clustering, (c) HDBSCAN (density-based),
  (d) BERTopic, (e) an LLM that reads everything and names themes, (f) manual coding only.
- **Chosen:** (a).
- **Why:** simple to explain, fast, every answer gets a theme, the cluster centre gives an objective "most
  typical answer", and the user can change k live in the dashboard.
- **Why not (c)/(d):** HDBSCAN leaves many points unassigned as noise and needs tuning, and BERTopic stacks several
  tools that are harder to explain in a panel. Both are good next steps.
- **Why not (e):** an LLM reading everything cannot be checked and may invent themes. We use an LLM only after
  themes exist, on verified quotes.
- **Why not (f):** the point is to help researchers cope with volume.
- **Cost:** k-means assumes round, similar-sized groups and forces every answer into a theme. Silhouette can favour
  too many clusters, so the dashboard reports it, shows stability, and lets the researcher override k.
- **Quality checks built in:** stability across random starts (shown in the sidebar) and comparison with planted
  topics (evaluate.py).

## D6. Theme labels from distinctive words (TF-IDF), not generated text
- **Options:** (a) top words, (b) LLM-generated names, (c) researcher names only.
- **Chosen:** (a) as a starting label, (c) as the final label in the Review tab.
- **Why:** the labels are computed from the data, so they cannot invent anything. An LLM name would read better but
  could introduce framing the data does not support.
- **Cost:** labels are rough keyword lists. The researcher is expected to rename them.

## D7. Evidence: representative quotes, one per participant, each verified
- **Options:** (a) random quotes, (b) the answers nearest the cluster centre, (c) LLM-selected quotes.
- **Chosen:** (b), at most one per participant, and every displayed quote is re-checked against the source file.
- **Why:** nearest-the-centre is objective and repeatable. One per participant stops one talkative person looking like
  a pattern. Verification (the quote must appear word for word at that participant and line) is a deterministic
  defence against fabricated or altered evidence.
- **Cost:** "most typical" can hide unusual but important voices. The dashboard lets the researcher change the order
  and show more quotes.

## D8. Tone shown separately from theme
- **Options:** (a) ignore tone, (b) a small sentiment model, (c) ask an LLM, (d) manual only.
- **Chosen:** (b) with a crude word-list fallback, clearly flagged.
- **Why:** embeddings group by topic, so "I stayed in one placement for years" and "I was moved seven times" can land in
  the same theme. Without tone a researcher could read that as a single negative finding.
- **Cost:** the sentiment model was trained on film reviews, not care narratives, and tone is not the same as
  stance. It is unreliable on "mixed" answers. evaluate.py reports its agreement with the planted stance, and
  researchers should treat tone as a prompt for reading, not a result.

## D9. Theme relationships: lift and Fisher's exact test
- **Options:** (a) raw shared-participant counts, (b) Jaccard overlap, (c) lift with a significance test,
  (d) an LLM or knowledge graph that extracts "X causes Y" relations.
- **Chosen:** (c), with counts and Jaccard also available.
- **Why:** raw counts make big themes look related to everything. Lift asks whether two themes co-occur more than
  their sizes alone would predict, and the test (with Bonferroni correction for the many pairs) says how
  surprising that is with 100 people.
- **Why not (d):** extracting causes from text is a much harder and less verifiable problem.
- **Cost:** co-occurrence is not causation, and with 100 participants many tests are noisy. The dashboard says so
  next to the chart.

## D10. Optional local LLM for insight drafting, behind hard checks
- **Options:** (a) no LLM, (b) hosted LLM API, (c) local open-source LLM via Ollama.
- **Chosen:** (c), optional. Quotes only, every claim cites evidence numbers, and claims with unknown evidence,
  participant IDs, or numbers not found in the cited evidence are rejected.
- **Why:** it meets the "AI-generated insight" requirement while keeping data on the machine and keeping the
  strongest defence against unsupported claims. (a) is safest but has no generated insight. (b) sends data out.
- **Cost:** a 3B model is weak. The checks catch some failures (numbers, IDs, bad references), not all: a claim
  can still overreach while citing real quotes, so a researcher must read every draft.

## D11. Human validation with an audit log
- **Chosen:** accept, reject, rename and annotate each theme. Every change is logged with who, when, old value, new value.
  The final report contains accepted themes only.
- **Why:** AI output is a proposal. Interpretation of people's lived experience belongs to researchers (and ideally
  co-researchers). The log makes the process auditable.
- **Revisit when:** several reviewers are involved: add accounts, roles and inter-coder agreement measures.

## D12. Streamlit dashboard
- **Options:** (a) Streamlit, (b) Dash, (c) Jupyter notebook, (d) React frontend plus Django REST API.
- **Chosen:** (a).
- **Why:** quick to build and change in Python, which is what this exercise is judging. Live filters, selectable map, and
  search are possible without writing front-end code.
- **Why not (d) now:** it is the right production path (see architecture.md) but too much for a small prototype.
- **Cost:** limited layout control, no real user accounts, one process.

## D13. Files and CSV instead of a database
- **Chosen:** transcripts as text files, results as CSV/NPY/JSON.
- **Why:** transparent, easy to inspect, nothing to install.
- **Revisit when:** many users or many interviews: PostgreSQL (with pgvector for similarity search).

## D14. Everything runs locally
- **Why:** participant narratives are sensitive. A local pipeline needs no data-sharing agreement with an API vendor.
- **Cost:** smaller models, slower on a laptop.
- **For production:** controlled infrastructure with encryption, role-based access, and a data management plan under UK GDPR.

---

## D15. Upload with a format check, then the same pipeline
- **Options:** (a) only a fixed folder, (b) upload and analyse, (c) free-form documents parsed by an LLM.
- **Chosen:** (b) with a strict, simple format (`INTERVIEWER:` / `PARTICIPANT:` lines, one file per participant).
- **Why:** the brief asks for the ability to "upload or process" transcripts. A strict format makes problems visible
  (the page lists lines it cannot read) and keeps speaker and line number reliable, which verification depends on.
- **Why not (c):** guessing structure from free text would put unverifiable errors at the very start of the pipeline.
- **Cost:** real transcripts need converting to this format first. A production version would add importers for
  common transcript formats and speech-to-text.

## D16. Quality checks that question the AI
- **Chosen:** (1) live evidence check, (2) voices that do not fit (answers far from their theme centre),
  (3) contrast cases (tone opposite to the theme majority), (4) a privacy scan for identifiers.
- **Why:** summaries smooth over unusual voices, and looking for negative or deviant cases is standard good practice
  in qualitative work. The privacy scan supports the data-protection duty, because identifiers must be found before analysis.
- **Cost:** the privacy scan is pattern matching, so it misses nicknames, school names and unusual spellings. It
  supports a human check and does not replace it. Contrast cases depend on tone, which is approximate.

## D17. Dashboard structure and design
- **Options:** (a) one long page, (b) tabs for everything, (c) sidebar navigation with one page per task.
- **Chosen:** (c), ordered as a researcher works: see patterns, read evidence, question the AI, decide. Each page is a
  module (`ui/page_*.py`), and `app.py` only wires them together.
- **Design choices:** colour-blind-safe palettes (so tone never relies on red versus green), plain-language labels,
  explanations beside each chart, and help text for statistical terms. Filters apply across pages, with a notice when active.
- **Cost:** Streamlit limits layout control and has no real user accounts. A production system would use the
  React and Django design in `architecture.md`.

## D18. How the comparison was done
- **What is compared:** three embedding models (MiniLM, MPNet, TF-IDF), four clustering approaches (k-means with
  automatic k, k-means with planted k, agglomerative, HDBSCAN), and two tone methods.
- **Measures:** agreement with the planted topics (adjusted Rand index and normalised mutual information), share of
  answers given a theme, stability across random starts, and run time. Tone is measured as agreement with the
  planted positive or negative stance.
- **Why these measures:** because the synthetic data has known topics, we can measure whether a method recovers them.
  Share assigned matters because a method can score well by leaving most answers unassigned.
- **Caveat:** the data is template-built and therefore easier than real interviews. The numbers rank methods and
  do not estimate real accuracy. Reproduce them with `python src/evaluate.py`. Results are in
  `docs/model_comparison.md` and on the Methods page of the dashboard.

## Known limitations (state these openly)
- Synthetic, template-based data: results are cleaner than real interviews.
- Tone is approximate, and the fallback word list is crude.
- Subgroup comparisons (region, age and so on) are exploratory. Small groups produce noise.
- No fairness audit of a trained classifier yet: the pipeline is unsupervised, so there is nothing trained to audit. A
  fairness audit (for example with AI Fairness 360) belongs with a supervised classifier built on a hand-coded gold set.
- The LLM layer drafts text but cannot guarantee that a claim is fully supported.
