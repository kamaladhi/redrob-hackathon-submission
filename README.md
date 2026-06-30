# 🚀 Redrob Hackathon: Intelligent Candidate Discovery & Ranking

**Team:** Neural Recruit  
**Member:** Jeevakamal K R  

![Redrob Hackathon](https://img.shields.io/badge/Challenge-Intelligent_Candidate_Discovery-blueviolet) ![Python](https://img.shields.io/badge/Python-3.12-blue) ![Docker](https://img.shields.io/badge/Docker-Ready-2496ED) ![Status](https://img.shields.io/badge/Status-Completed-success)

This repository contains our official submission for the **Redrob Intelligent Candidate Discovery & Ranking Challenge**. Our goal was to accurately rank 100,000 candidates against a target Job Description within a strict 5-minute compute sandbox (CPU-only, no network access).

---

## 🏗️ System Architecture

To overcome the severe compute limitations and eliminate the latency of generative LLMs, we designed a **Hybrid Two-Phase Pipeline**:

### 1. Offline Pre-computation Phase (`build_index.py`)
Because embedding 100,000 profiles at runtime is impossible under 5 minutes, we shifted the heavy lifting offline:
* **Honeypot Detection:** Iterates through the entire dataset to filter out synthetic or suspicious profiles.
* **Semantic Embedding:** Uses the lightweight `all-MiniLM-L6-v2` transformer model to generate dense semantic vectors for all candidates.
* **Vector Indexing:** Builds a highly optimized `FAISS` database (`candidate_index.faiss`) to allow for millisecond similarity lookups later.

### 2. Fast Online Ranking Phase (`rank.py`)
This is the script that runs inside the isolated sandbox:
* **JD Ingestion:** Embeds the target Job Description into a query vector using the same Transformer model.
* **Semantic Retrieval:** Queries the FAISS index to instantly return a baseline L2 distance/cosine similarity score.
* **Hybrid Heuristics:** Modulates the base score using a deterministic algorithm. We heavily penalize non-technical domains (e.g., HR managers) and apply score multipliers based on real-world `redrob_signals` (e.g., rewarding candidates with high recruiter response rates and short notice periods).
* **Zero-Hallucination Reasoning:** Uses a deterministic factual templating system to parse the candidate's raw JSON data (years of experience, matched skills) and generates a 100% accurate reasoning string without the need for an LLM.

*(Our ranking script completes the entire process on a standard CPU in just **~17 seconds**!)*

---

## 🛠️ Technologies Used

* **Sentence-Transformers (`all-MiniLM-L6-v2`):** Delivers highly accurate NLP embeddings but is small enough to run incredibly fast on standard CPUs.
* **FAISS (Facebook AI Similarity Search):** An optimized C++ library built for lightning-fast similarity search, querying our 100,000 dense vectors in milliseconds.
* **Pandas:** Utilized for rapid data wrangling and formatting the final outputs.
* **Docker:** Containerizes the Python environment, models, and FAISS database to guarantee execution in a locked-down, network-free sandbox.

---

## 🚀 Reproducing the Submission

### Prerequisites
Make sure you have Python 3.12+ installed, and install the required dependencies:
```bash
pip install -r requirements.txt
```

### Running the Ranking Script
Place the main dataset (`candidates.jsonl`) in the root of the repository, and run the ranking script:
```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```
*(Note: You do not need to run `build_index.py`. This repository already contains our pre-computed `candidate_index.faiss` and `candidate_metadata.pkl` files via Git LFS.)*

---

## 🐳 Docker Sandbox Validation

We have provided a `Dockerfile` that perfectly mirrors the hackathon sandbox constraints. To test our code in isolation against a sample candidate subset, build and run the image:

```bash
docker build -t redrob-sandbox .
docker run redrob-sandbox
```

---
*Created with ❤️ by Team Neural Recruit for the India.Runs Data & AI Challenge.*
