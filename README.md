# 🎬 Hybrid Movie Recommendation System with LLMs

A **production-style hybrid recommendation engine** that combines **Sentence-BERT embeddings for fast retrieval** with **LLM-based re-ranking for intelligent semantic reasoning**.

Instead of recommending movies only by genre or keywords, this system understands **themes, tone, storytelling style, and deeper meaning** by delivering recommendations that feel much more human.

Built end-to-end with **PyTorch, Sentence-BERT, GPT-based reasoning, and Streamlit**, with a strong focus on **speed, scalability, and cost-efficiency**.

---

# 🚀 Features

### 🔍 Hybrid AI Architecture
Two-stage recommendation pipeline:
- Embeddings → fast candidate retrieval
- LLM → intelligent re-ranking + explanations

### 🧠 Semantic Recommendations
Understands meaning, themes, and storytelling — not just genres

### ⚡ Real-Time Performance
- ~50ms embedding search
- ~2–3s LLM reasoning
- ~3s total response time

### 💬 Explainable AI
Each recommendation includes a natural language explanation of *why* it was suggested

### 💸 Cost Optimized
~$0.01 per search using GPT-4o-mini

### 🎨 Modern UI
Clean dark-themed interface built with Streamlit + custom CSS

### 🎥 Rich Movie Details
- Genre
- Cast
- Director
- Duration
- Release year
- Posters
- Overview

### 📊 Filters
- Genre
- Type
- Year
- Duration

### ⚙️ Backend Efficiency
- Precomputed embeddings
- Cosine similarity
- Cached API responses

---

# 🧠 Tech Stack

## Embeddings
- Sentence-BERT (MiniLM-L6-v2)
- PyTorch
- 384-dimension vectors
- Cosine similarity (scikit-learn)

## LLM Layer
- GPT-4o-mini
- Re-ranking + explanation generation
- Temperature-controlled outputs
- Caching for cost reduction

## Backend
- Python
- Pandas
- NumPy
- Pickle / FAISS-ready embeddings

## Frontend
- Streamlit
- Custom CSS dark theme

## APIs
- OpenAI API (LLM reasoning)
- OMDb API (movie posters)

---

# 🌐 Live Demo

👉 https://movie-recommendation-sys-with-llms.streamlit.app/

---

# 🧪 How It Works

## Step 1 - Offline Embeddings
Generate embeddings for:
- genre
- description
- cast
- director
- title

~44,000 embeddings are precomputed and stored locally for instant loading.

---

## Step 2 - Fast Retrieval (Recall)
When a user selects a movie:
- compute cosine similarity
- search across all movies
- retrieve top-10 candidates in ~50ms

---

## Step 3 - LLM Re-ranking (Precision)
Pass only 10 candidates to GPT-4o-mini:
- analyze themes & tone
- understand deeper semantic relationships
- re-rank to top-5
- generate explanations

---

## Step 4 - UI Display
Show:
- similarity scores
- explanations
- posters
- metadata

---

# ⚖️ Key Product Decisions

## Why Hybrid?

| Approach | Issue |
|-----------|-----------------------------|
| Embeddings only | Fast but shallow |
| LLM only | Smart but slow & expensive |
| Hybrid | Fast + intelligent ✅ |

---

## Engineering Tradeoffs
- Precomputed embeddings → 13 min ➜ 1 sec startup
- GPT-4o-mini → 60x cheaper than GPT-4
- scikit-learn instead of FAISS for small scale
- Caching reduces repeated API costs

---

## Performance Metrics

| Metric | Value |
|-----------|-----------|
| Retrieval time | ~50ms |
| LLM time | ~2–3s |
| Total latency | ~3s |
| Cost per search | ~$0.01 |


