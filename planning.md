# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

students' reviews of programs at George Washington University

---

## Documents

<!-- List your specific sources: URLs, subreddit names, forum threads, or file descriptions.
     Aim for at least 10 sources that together cover different subtopics or perspectives within your domain. -->

| # | Source | Description | URL or location |
|---|--------|-------------|-----------------|
| 1 | ratemyprofessors | students' reviews of professors | https://www.ratemyprofessors.com/search/professors/353?q=* |
| 2 | gwu reddit | The unofficial subreddit of The George Washington University | https://www.reddit.com/r/gwu/ |
| 3 | us news | the us news page for gwu| https://www.usnews.com/best-colleges/george-washington-university-1444 |
| 4 | gw engineering| gw official website for the school of engineering | https://engineering.gwu.edu/ |
| 5 | niche | aggregates student reviews on academics, campus life, safety, value, etc. | https://www.niche.com/colleges/george-washington-university/reviews/ |
| 6 | gw employment | employment opportunities at gw | https://gwu-studentemployment.peopleadmin.com/postings/search?utf8=%E2%9C%93&query=&query_v0_posted_at_date=&1387%5B%5D=5&435=&query_organizational_tier_3_id%5B%5D=any&commit=Search |
| 7 | college confidential | forums with candid student/parent discussions | https://talk.collegeconfidential.com/c/colleges-and-universities/the-george-washington-university/230/l/latest?ascending=false&order=activity |
| 8 | unigo | student-written reviews and Q&As | https://www.unigo.com/colleges/george-washington-university |
| 9 | Glassdoor | student workers & TA perspectives on GW | https://www.glassdoor.com/Reviews/George-Washington-University-Reviews-E3733.html |
| 10 | subreddit | GWU subreddit wiki/ top posts | https://www.reddit.com/r/gwu/search/?q=cs+professor&sort=top |

---

## Chunking Strategy

<!-- How will you split documents into chunks?
     State your chunk size (in tokens or characters), overlap size, and explain why those
     numbers fit the structure of your documents.
     A review-heavy corpus warrants different chunking than a long FAQ. -->

**Chunk size:**

 ~200–300 tokens

**Overlap:**

50 tokens

**Reasoning:**

Reviews are short, opinion-dense, and self-contained. A 200-token chunk captures one person's full thought without mixing opinions across reviewers. Overlap ensures a sentence that straddles a boundary isn't lost. Reddit threads need special care — chunk by comment, not by post, since each comment is its own opinion.

---

## Retrieval Approach

<!-- Which embedding model are you using (e.g., all-MiniLM-L6-v2 via sentence-transformers)?
     How many chunks will you retrieve per query (top-k)?
     If you were deploying this for real users and cost wasn't a constraint, what tradeoffs
     would you weigh in choosing a different embedding model — context length, multilingual
     support, accuracy on domain-specific text, latency? -->

**Embedding model:**

all-MiniLM-L6-v2 (via sentence-transformers) 

**Top-k:**

k = 5

**Production tradeoff reflection:**

use text-embedding-3-large (OpenAI) — higher accuracy but paid and slower

Context length: MiniLM handles up to 256 tokens, which fits chunk size — but a model with longer context (like nomic-embed-text) gives more headroom if I want bigger chunks

---

## Evaluation Plan

<!-- List your 5 test questions with their expected correct answers.
     Questions should be specific enough that you can judge whether the system's response
     is right or wrong. "What are good dining halls?" is too vague.
     "What do students say about wait times at [dining hall name] during lunch?" is testable. -->

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | Is GWU considered worth the cost by its students? | Reviews discussing tuition, financial aid, value relative to DC location/networking/job opportunities/reputation |
| 2 | Which CS professors are worth taking? | Named professors from RateMyProfessors with comments about class quality |
| 3 | What do students say about class sizes in the engineering school? | References to small/large classes, access to professors, discussion quality |
| 4 | Is GW a good school? | Reviews dicussing gw reputation |
| 5 | Research/job oppotunires | Students' reviews on research oppotunities and job for students who graduate from gw |

---

## Anticipated Challenges

<!-- What could go wrong? Name at least two specific risks with reasoning.
     Consider: noisy or inconsistent documents, missing source attribution, off-topic
     retrieval, chunks that split key information across boundaries. -->

1. Noisy, opinion-mixed chunks from Reddit. A single Reddit thread might contain 40 replies ranging from housing advice to memes. If chunk by character rather than by comment, easily get semantically incoherent chunks that hurt retrieval precision. Fix: parse Reddit threads by comment before chunking.

2. Source attribution loss — When a chunk is retrieved, need to know it came from RateMyProfessors vs. Reddit vs. Niche, because those have very different credibility levels. If we don't store metadata alongside embeddings, generated answers can't cite sources meaningfully. Fix: store source, url, and date as metadata fields in vector store from the start.

---

## Architecture

<!-- Draw a diagram of your pipeline showing the five stages:
     Document Ingestion → Chunking → Embedding + Vector Store → Retrieval → Generation
     Label each stage with the tool or library you're using.
     You can use ASCII art, a Mermaid diagram, or embed a sketch as an image.
     You'll use this diagram as context when prompting AI tools to implement each stage. -->

     BeautifulSoup / PRAW /requests -> Chunking (LangChain, RecursiveCharacterTextSplitter) ~250 tokens, 50 overlap -> Embedding (sentence-transformers, all-MiniLM-L6-v2) -> Vector Store (Chroma or FAISS with metadata) -> Retrieval (Top-k=5 similarity search) -> Generation (Claude / GPT-4with retrieved chunks as context)

---

## AI Tool Plan

<!-- For each part of the pipeline below, describe:
     - Which AI tool you plan to use (Claude, Copilot, ChatGPT, etc.)
     - What you'll give it as input (which sections of this planning.md, which requirements)
     - What you expect it to produce
     - How you'll verify the output matches your spec

     "I'll use AI to help me code" is not a plan.
     "I'll give Claude my Chunking Strategy section and ask it to implement chunk_text()
     with my specified chunk size and overlap" is a plan. -->

**Milestone 3 — Ingestion and chunking:**

**Milestone 4 — Embedding and retrieval:**

**Milestone 5 — Generation and interface:**
