# 🚨 IT Project Deadline Miss Prediction (with GenAI Explanations)

A machine learning system that predicts whether a software project is likely
to miss its deadline, based on planning-stage metrics (team experience,
project complexity, scope). Combines traditional ML, explainable AI (SHAP),
and Generative AI (Groq/LLM) to give Project Managers a clear, human-readable
risk assessment — not just a number.

🔗 **Live Demo**: [[Live working]((https://mayur93-02-deadline-miss-prediction-genai-app-qakzb1.streamlit.app/))]
📓 **Notebook**: `notebooks/deadline_prediction.ipynb`

---

## 📌 Problem Statement

IT/software companies frequently miss project deadlines, damaging client
trust and revenue. This project predicts deadline-miss risk using
**planning-stage data only** (available before the project starts), so
Project Managers can proactively flag high-risk projects and take early
corrective action — explained in plain language, not just a probability
score.

---

## 📊 Dataset

| | |
|---|---|
| **Base (Real) Data** | [Desharnais Dataset](https://zenodo.org/records/1119795) — 81 real software projects from Canadian software houses (1989), sourced from the PROMISE Software Engineering Repository |
| **Augmentation** | 900 additional synthetic rows generated using a **Gaussian Copula** method that preserves the real data's feature correlations and statistical distributions |
| **Final Size** | 981 rows, 24 engineered features |
| **Target Variable** | `deadline_missed` (engineered — see below, no direct real-world label existed) |
| **Transparency** | Every row is tagged `real` or `synthetic` in the raw dataset |

> ⚠️ **Honesty note**: This is not a fully real-world dataset. Public,
> labeled "IT project deadline miss" datasets don't exist due to data
> privacy. The real 81-project academic dataset was used as a statistical
> foundation, with synthetic rows added to reach a trainable size — a
> standard technique (data augmentation) used in both research and industry
> when real data is scarce.

### How the target (`deadline_missed`) was engineered
Built from a weighted combination of realistic risk factors:
- **40%** — Project complexity (Function Points, Entities, Transactions)
- **30%** — Team & manager experience (inverse relationship)
- **15%** — Planned project length
- **15%** — Actual effort (bounded contribution, not the sole driver)
- Plus randomness, reflecting real unpredictable factors (client behavior,
  external dependencies)

---

## 🧹 Data Cleaning

- **Missing value placeholders**: `TeamExp` and `ManagerExp` contained `-1`
  values — a known placeholder for missing data in the original 1989
  dataset. Fixed via median imputation.
- **Redundant columns** (`id`, `Project`) removed.
- **Categorical encoding**: `Language` (1/2/3) one-hot encoded.

## 🔧 Feature Engineering

| Feature | Description |
|---|---|
| `Complexity_Score` | Combined size/complexity metric (PointsAjust, Entities, Transactions) |
| `Total_Experience`, `Experience_Gap`, `Experience_Ratio` | Team vs. manager experience relationships |
| `Transactions_per_Length`, `Entities_per_Length`, `Transactions_per_Entity` | Complexity-to-timeline ratios |
| `Adjustment_Ratio` | Technical difficulty relative to raw size |
| `Point_size_*` (Small/Medium/Large/Very Large) | Project size bucket, one-hot encoded |

## ⚠️ Data Leakage — Identified & Resolved

`Effort` (actual work done) was excluded from the model's input features,
since the target variable was partially derived from it. Including it would
have produced an artificially high, misleading accuracy — a real project's
actual effort isn't known in advance (that's precisely what we're trying to
predict). All Effort-derived engineered columns were excluded for the same
reason.

---

## 🤖 Models Compared

| Model | Accuracy |
|---|---|
| Logistic Regression | ~57% |
| Random Forest | ~57–61% |
| SVM | ~57% |
| **K-Nearest Neighbors (Final)** | **~57–60%** |
| XGBoost | ~57% |

### Key Finding
Five algorithm families (linear, tree-based, distance-based, kernel-based,
boosting) converged on a **consistent 55–61% accuracy range**. This
confirms — rather than being a modeling shortfall — that planning-stage
data alone has a genuine ceiling on how predictable project delays are.
Real projects are also affected by factors unknowable at planning time
(client behavior, sudden scope changes, external dependencies).

---

## 🔍 Explainability (SHAP)

Each prediction is accompanied by a **SHAP-based explanation** showing which
specific factors pushed that project's risk up or down — not a single
global importance ranking, but a per-project, per-prediction breakdown,
computed against a real training-data background sample for accuracy.

**Global finding**: `Complexity_Score` and `PointsAjust` (project complexity
metrics) are the strongest predictors of deadline risk — more influential
than individual team/manager experience metrics.

---

## 🤖 Generative AI Layer (New)

Raw SHAP output (feature names + numeric impact values) is useful for a data
scientist, but not immediately readable for a non-technical Project Manager.
This project adds a **GenAI layer** on top of SHAP:

1. The top 5 SHAP-ranked factors for a specific prediction are formatted
   into a structured prompt.
2. The prompt is sent to an LLM (**Groq API**, running Llama 3.3 70B) which
   returns a short, plain-language paragraph explaining:
   - The project's overall risk status
   - The 2–3 biggest contributing factors, in plain words
   - One practical, actionable recommendation
3. This combines the **statistical rigor of SHAP** (grounded in the actual
   trained model, not a hallucinated guess) with the **readability of an
   LLM** — the AI explains real numbers, it doesn't invent them.

> This mirrors a growing real-world pattern: using GenAI as a
> **communication/reporting layer** on top of traditional ML outputs, rather
> than replacing the ML model itself.

---

## 💼 Business Recommendations

1. Prioritize **scope/complexity management** over pure resource allocation
   for high-risk projects.
2. Collect **execution-stage data** (weekly progress %, blocker counts,
   scope-change frequency) for meaningfully better future predictions —
   planning-stage metrics alone have limited predictive power.
3. Allocate extra time/resource buffers to high-complexity projects from
   the outset.

---

## 🖥️ Interactive Dashboard

Built with Streamlit — enter a project's planning details and get:
1. **Risk prediction** (Low / High) with probability
2. **"Why" breakdown** — top factors driving that specific prediction (SHAP)
3. **Actionable suggestions** — concrete steps to reduce identified risks
4. **AI-generated summary** — a plain-language paragraph explaining the
   result (Groq/Llama 3.3)

```bash
streamlit run app.py
```

---

## 🛠️ Tech Stack

Python · Pandas · NumPy · Scikit-learn · SHAP · Streamlit · Matplotlib ·
Groq API (Llama 3.3 70B) · python-dotenv

## 📁 Project Structure

```
project-deadline-prediction/
├── data/
│   ├── real_desharnais.csv              # Original 81 real projects
│   ├── augmented_project_dataset_v2.csv # Final 981-row dataset
│   └── background_sample.csv            # Sample used for SHAP explainer
├── notebooks/
│   └── deadline_prediction.ipynb        # Full analysis: EDA → model → SHAP
├── models/
│   └── deadline_miss_model.pkl          # Trained KNN model
├── app.py                               # Streamlit dashboard (+ GenAI layer)
├── .env                                 # API key (NOT committed to git)
├── .gitignore
├── requirements.txt
└── README.md
```

## 🚀 How to Run

```bash
# Clone the repo
git clone <your-repo-url>
cd project-deadline-prediction

# Set up environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Add your Groq API key
# Create a file named .env in the project root with:
# GROQ_API_KEY=your_key_here
# (Get a free key at https://console.groq.com)

# Explore the notebook
jupyter notebook notebooks/deadline_prediction.ipynb

# Or launch the dashboard directly
streamlit run app.py
```

## 📄 requirements.txt

```
pandas
numpy
scikit-learn
shap
streamlit
matplotlib
joblib
groq
python-dotenv
```

## 🔐 Security Note

The `.env` file (containing the Groq API key) is excluded via `.gitignore`
and must **never** be committed to version control. Anyone running this
project needs to supply their own free API key.

---

## 🙏 Acknowledgements

Base real dataset: Desharnais, J.M. (1989), hosted via the
[PROMISE Software Engineering Repository](http://promise.site.uottawa.ca/SERepository/).

LLM inference: [Groq](https://groq.com) (Llama 3.3 70B).

---

## 📬 Contact

[Mayur Mahajan] · [https://www.linkedin.com/in/mayur-mahajan-9m3/] · [mahajanmayur930250@gamil.com]
