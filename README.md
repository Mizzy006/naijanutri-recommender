# NaijaNutri Pro 🇳🇬🍽️
### DSN × BCT LLM Agent Challenge | Hackathon 3.0

A context-aware, cross-domain recommendation agent powered by Hybrid ML/LLM architectures, explicitly designed for Nigerian consumer behavior and dietary constraints.

NaijaNutri Pro moves beyond generic preference vectors by treating users as dynamic, culturally-situated agents. It features a custom Nigerian Persona Layer, robust TDEE mathematical scaling, and strict Domain Isolation across three modalities (Food, Appliances, and Books) to deliver hyper-personalized, culturally authentic recommendations without cross-domain hallucinations.

## 🚀 Key Features

* **Culturally Grounded Personas:** Dynamically maps users to 6 Nigerian archetypes (e.g., Sapa Student, Tech Bro, Omo Landlord) based on vocabulary signals, applying archetype-specific rating biases and Pidgin language adaptation.
* **Strict Domain Isolation:** Separate LLM context pipelines for Food, Appliances, and Books. Ensures dietary constraints (e.g., allergies, calorie deficits) never bleed into appliance or literature recommendations.
* **TDEE & Dietary Hard-Filtering:** Integrates Mifflin-St Jeor TDEE calculations. Strict dietary constraints (halal, allergens, glycemic index) are enforced at the vector database level before LLM evaluation.
* **Embedded Vector Storage:** Uses ChromaDB's `PersistentClient` for local, lightning-fast semantic retrieval with zero external API dependencies.

## 🌟 Original Contribution: Nigerian Nutritional Dataset

Standard recommendation benchmarks (like Yelp) lack the metadata required for strict dietary and localized budget constraints. To solve this, NaijaNutri Pro introduces a custom, meticulously engineered Nigerian Food Dataset (60 items).

Unlike generic LLM knowledge, this dataset allows for hard mathematical filtering prior to semantic retrieval using the following metadata variables:
* `exact_naira_cost`: Enables precise budget constraints (no approximation via $ price tiers).
* `protein_level` & `glycemic_index`: Anchors the TDEE math for weight loss/gain goals.
* `allergens`: Enforces strict zero-shot filtering for health safety.
* `prep_time`: Enables meal-type classification (quick home cook vs. full prep).

## 🛠️ Tech Stack

* **Large Language Model:** Llama 3.1-8b-instant (via Groq LPU for 10-30x faster batch inference)
* **Embedding Model:** sentence-transformers (all-MiniLM-L6-v2)
* **Vector Database:** ChromaDB (Embedded)
* **Backend:** FastAPI, Python, Pandas
* **Frontend:** Streamlit
* **Deployment:** Fully Dockerized

## 📊 Evaluation Metrics

The system was benchmarked using a rigorous leave-one-out evaluation protocol, handling sparsity and cold-start scenarios gracefully.

**Task A (User Modeling - Rating & Review Prediction):**
* **BERTScore F1:** 0.8345 (Proves high semantic alignment despite generative lexical novelty)
* **Rating RMSE:** 0.5654
* **Rating MAE:** 0.4821

**Task B (Cross-Domain Recommendation):**
* **NDCG@10:** 0.0726
* **Hit@10:** 0.08
* **MRR:** 0.0712
* **Cold-Start NDCG@10:** 0.0625

> *(Note: Complete evaluation scripts and ablation study logs are available in the `/evaluation` directory).*

## 💻 How to Run Locally

This application is fully containerized for zero-configuration reproducibility.

### Prerequisites
* Docker & Docker Compose
* A Groq API Key

### Step 1: Environment Setup
Clone the repository and create a `.env` file in the root directory:

```bash
git clone [https://github.com/YourUsername/naijanutri-recommender.git](https://github.com/YourUsername/naijanutri-recommender.git)
cd naijanutri-recommender
echo "GROQ_API_KEY=your_actual_key_here" > .env
```

### Step 2: Launch with Docker
Run the unified Docker deployment:

```bash
docker compose up --build
```

Frontend UI: http://localhost:8501

FastAPI Backend: http://localhost:8000

Alternative
If testing locally on Windows without Docker installed,
simply double-click the run_demo.bat script at the project root to instantly launch both the Uvicorn backend and Streamlit frontend in parallel.

## 📁 Repository Structure
├── backend/                  # FastAPI server, LLM agents, and ChromaDB core

├── data/                     # Ingestion scripts and clean parquet files

├── evaluation/               # Traditional ML metrics, BERTScore/ROUGE scripts

├── frontend/                 # Streamlit UI and session state management

├── Dockerfile                # Unified environment blueprint

├── docker-compose.yml        # Multi-container orchestration

└── run_demo.bat              # Windows rapid-launch script
