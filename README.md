# The Unofficial Guide — Project 1

> **How to use this template:**
> Complete each section *after* you've built and tested the corresponding part of your system.
> Do not write placeholder text — if a section isn't done yet, leave it blank and come back.
> Every section below is required for submission. One-liners will not receive full credit.

---

## Domain

Students' reviews of programs and professors at George Washington University. The project is useful because official school pages describe programs and opportunities, but they do not capture the lived experience of students: how hard classes are, which professors are worth taking, whether internships are easy to find, or whether the cost feels justified.
---

## Document Sources

<!-- List every source you collected documents from.
     Be specific: include URLs, subreddit names, forum thread titles, or file names.
     Aim for variety — sources that together cover different subtopics or perspectives. -->

| # | Source | Type | URL or file path |
|---|--------|------|-----------------|
| 1 | RateMyProfessors | review site | https://www.ratemyprofessors.com/search/professors/353?q=* |
| 2 | GW Engineering | official school site | https://engineering.gwu.edu/ |
| 3 | GW Employment | official employment site | https://gwu-studentemployment.peopleadmin.com/postings/search?utf8=%E2%9C%93&query=&query_v0_posted_at_date=&1387%5B%5D=5&435=&query_organizational_tier_3_id%5B%5D=any&commit=Search |
| 4 | College Confidential | forum | https://talk.collegeconfidential.com/c/colleges-and-universities/the-george-washington-university/230/l/latest?ascending=false&order=activity |
| 5 | Unigo | review site | https://www.unigo.com/colleges/george-washington-university |

I also attempted to collect Reddit, US News, Niche, and Glassdoor, but those sources were blocked or timed out in this environment, so they were not included in the final indexed corpus.

---

## Chunking Strategy

<!-- Describe your chunking approach with enough specificity that someone else could reproduce it.
     Include:
     - Chunk size (characters or tokens) and why that size fits your documents
     - Overlap size and why (or why not) you used overlap
     - Any preprocessing you did before chunking (e.g., stripping HTML, removing headers)
     - What your final chunk count was across all documents -->

**Chunk size:**

250 tokens

**Overlap:**

50 tokens

**Why these choices fit your documents:**

The corpus is review-heavy and opinion-dense, so a 250-token window is large enough to keep a full student thought together while still staying within the embedding model's comfort zone. A 50-token overlap preserves context across boundaries so a key sentence is less likely to be cut off. Before chunking, the ingestion script strips HTML, removes obvious page chrome and boilerplate, normalizes whitespace, and keeps the remaining text in token-based sliding windows.

**Final chunk count:**

12 chunks across the current indexed corpus
---

## Embedding Model

<!-- Name the embedding model you used and explain your choice.
     Then answer: if you were deploying this system for real users and cost wasn't a constraint,
     what tradeoffs would you weigh in choosing a different model?
     Consider: context length limits, multilingual support, accuracy on domain-specific text,
     latency, and local vs. API-hosted. -->

**Model used:**

all-MiniLM-L6-v2 via sentence-transformers

**Production tradeoff reflection:**

I chose MiniLM because it is fast, local, and sufficient for short review chunks. If cost were not a constraint, I would consider a stronger hosted embedding model such as text-embedding-3-large for better semantic matching on nuanced review language, or a longer-context local model if I wanted larger chunks with less boundary loss. The tradeoff is accuracy versus latency, cost, and operational simplicity.

## Embedding and Retrieval

If your system `python3` does not have the project dependencies installed, activate the repo virtualenv first:

```bash
source .venv/bin/activate
```

Build the ChromaDB index from the chunked corpus:

```bash
python3 embedding_retrieval.py build --chunks-file chunks.jsonl --persist-dir chroma_db
```

Run a retrieval query against the stored chunks:

```bash
python3 embedding_retrieval.py query "Which CS professors are worth taking?" --top-k 5 --persist-dir chroma_db
```

---

## Grounded Generation

<!-- Explain how your system enforces grounding — how does it prevent the LLM from answering
     beyond the retrieved documents?
     Describe both your system prompt (what instruction you gave the model) and any structural
     choices (e.g., how you formatted the context, whether you filtered low-relevance chunks).
     Do not just say "I told it to use the documents" — show the actual instruction or explain
     the mechanism. -->

**System prompt grounding instruction:**

The generation layer uses a strict prompt: answer only from the provided context, do not use outside knowledge, and if the context is insufficient reply exactly with `I don't have enough information on that.` It also tells the model to keep the answer concise, use bracket citations like `[1]`, and not invent sources. The CLI post-processes the model output so the final response stays in a predictable format.

**How source attribution is surfaced in the response:**

The retrieved context is numbered before it is sent to the LLM. The response is printed as two sections: `Answers` and `Source list`. Each source entry includes the source name, source URI, chunk index, token span, and similarity score so the answer can be traced back to the original document.

## Query Interface

The project includes a small Gradio web UI that calls the same grounded generation pipeline.

```bash
source .venv/bin/activate
python3 app.py
```

Open `http://127.0.0.1:7860` in your browser after launching the app.
---

## Evaluation Report

<!-- Run your 5 test questions from planning.md through your system and record the results.
     Be honest — a partially accurate or inaccurate result that you explain well is more
     valuable than a suspiciously perfect result. -->

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | Is GWU considered worth the cost by its students? | Reviews discussing tuition, financial aid, and value relative to D.C. opportunities | Some students say GWU is worth it because of strong academics and real-world opportunities, while others worry about cost and debt. | Relevant | Accurate |
| 2 | Which CS professors are worth taking? | Named professors from RateMyProfessors with comments about class quality | The system identified Joe Goldfrank from RateMyProfessors and reported his quality rating and how often students would take the class again. | Relevant | Accurate |
| 3 | What do students say about class sizes in the engineering school? | References to small or large classes, access to professors, and discussion quality | The system returned the exact fallback: `I don't have enough information on that.` | Partially relevant | Accurate |
| 4 | Is GW a good school? | Reviews discussing GW reputation and overall student sentiment | The system said GWU is considered a good school because of its academics, D.C. location, internships, and research opportunities, while noting cost concerns. | Relevant | Accurate |
| 5 | Research/job opportunities | Students' reviews on research and jobs for GW graduates | The system summarized internships, research assistant positions, tutoring jobs, and D.C.-based networking opportunities. | Relevant | Accurate |

**Retrieval quality:** Relevant / Partially relevant / Off-target  
**Response accuracy:** Accurate / Partially accurate / Inaccurate

---

## Failure Case Analysis

<!-- Identify at least one question where retrieval or generation did not work as expected.
     Write a specific explanation of *why* it failed, tied to a part of the pipeline.

     "The answer was wrong" is not an explanation.

     "The relevant information was split across a chunk boundary, so retrieval returned
     only half the context — the model didn't have enough to answer correctly" is an explanation.

     "The embedding model treated the professor's nickname as out-of-vocabulary and returned
     results from an unrelated review" is an explanation. -->

**Question that failed:**

What do students say about class sizes in the engineering school?

**What the system returned:**

I don't have enough information on that.

**Root cause (tied to a specific pipeline stage):**

The retrieval corpus does not contain an explicit source that discusses engineering class sizes in enough detail. The retriever returned broad GWU review and engineering-page chunks, but they did not include a direct answer, so the generation layer correctly abstained instead of guessing. This is a retrieval coverage issue, not a model hallucination issue.

**What you would change to fix it:**

Add a more targeted source that discusses engineering course sizes directly, or collect a better engineering forum/review page. I would also consider a small query router or reranker for questions about class size, because that wording should favor documents that mention section size, lecture size, or professor access.
---

## Spec Reflection

<!-- Reflect on how planning.md shaped your implementation.
     Answer both questions with at least 2–3 sentences each. -->

**One way the spec helped you during implementation:**

The planning document gave a clear target for the chunking strategy, embedding model, top-k retrieval size, and the kinds of sources the corpus should include. That made it straightforward to keep the ingestion, retrieval, and generation layers aligned instead of inventing a separate architecture midstream.

**One way your implementation diverged from the spec, and why:**

The spec assumed all ten planned sources would be available, but several sites blocked scraping or returned errors in this environment. To keep the project working, I used best-effort ingestion, stored only the sources I could actually collect, and added a confidence gate so the generator could refuse low-signal queries rather than hallucinating.
---

## AI Usage

<!-- Describe at least 2 specific instances where you used an AI tool during this project.
     For each: what did you give the AI as input, what did it produce, and what did you
     change, override, or direct differently?

     "I used Claude to help me code" is not sufficient.
     "I gave Claude my Chunking Strategy section from planning.md and asked it to implement
     chunk_text(). It returned a function using a fixed character split. I overrode the
     chunk size from 500 to 200 because my documents are short reviews, not long guides." -->

**Instance 1**

- *What I gave the AI:*

     The chunking strategy and source list from `planning.md`, plus the requirement to load documents, clean them, and split them into ~250-token windows with 50-token overlap.
- *What it produced:*

     A token-based ingestion pipeline with cleaning, planning-file URL parsing, and JSONL chunk output.
- *What I changed or overrode:*

     I tightened the boilerplate filters, added site-specific extraction for the noisier sources, and added preview/output helpers so the chunk quality could be checked directly.

**Instance 2**

- *What I gave the AI:*

     The retrieval and grounding requirements from `planning.md`, including `all-MiniLM-L6-v2`, `top-k = 5`, ChromaDB storage, and grounded answers with source attribution.
- *What it produced:*

     A retrieval module and a generation wrapper that used the retrieved chunks as context for an LLM.
- *What I changed or overrode:*

     I added a confidence threshold, enforced the exact fallback sentence when the corpus did not support an answer, and wrapped the generation output so the response format stayed `Answers` plus `Source list`.
