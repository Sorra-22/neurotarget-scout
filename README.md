# 🧠 NeuroTarget Scout
### AI-assisted drug target prioritization for Schizophrenia & Major Depressive Disorder

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?style=flat&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Open Targets](https://img.shields.io/badge/Data-Open%20Targets%20Platform-3B9C3D?style=flat)](https://platform.opentargets.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

> **NeuroTarget Scout** is an end-to-end drug target prioritization tool that combines live biological evidence from the [Open Targets Platform](https://platform.opentargets.org/) with a Machine Learning scoring model to systematically rank CNS drug targets — mirroring real workflows used in early-stage pharmaceutical drug discovery.



## 🎯 Motivation

Early drug discovery teams at pharma and biotech companies spend significant time manually curating evidence for hundreds of potential targets before deciding which ones to advance. This process — called **target identification and prioritization** — is time-consuming, inconsistent, and hard to scale.

NeuroTarget Scout automates this by:
- Pulling **real-time, multi-modal evidence** from Open Targets via GraphQL
- Applying a **Random Forest classifier** to learn non-linear evidence weightings
- Delivering a ranked, exportable target list through an **interactive dashboard**

Built with a focus on **schizophrenia and major depressive disorder** — two of the most underserved and complex CNS indications in drug development.

---

## ✨ Features

- 🔴 **Live API Integration** — Real-time GraphQL queries to Open Targets Platform (no static files)
- 🤖 **ML Scoring** — Random Forest classifier trained on 5 evidence dimensions with self-supervised labeling
- 📊 **Interactive Dashboard** — 4-tab Streamlit UI with ranked tables, scatter plots, evidence profiles
- 🔍 **Target Deep-Dive** — Per-gene evidence breakdown with direct link to Open Targets page
- ⬇️ **Export** — Download scored target list as CSV for downstream analysis

---

## 🧬 Evidence Types

| Evidence | Source | Description |
|---|---|---|
| 🧬 Genetic | GWAS, gene burden, credible sets | Variants statistically associated with disease |
| 📖 Literature | Europe PMC text mining | Co-mentions with disease in publications |
| 🐭 Animal Model | IMPC, PhenoDigm | Phenotypic evidence from model organisms |
| 💊 Known Drugs | ChEMBL | Approved or clinical-stage compounds against target |
| 🔬 Somatic | COSMIC, IntOGen | Somatic mutation burden in disease tissue |

---

## 🤖 ML Approach

Standard drug target datasets lack clean binary labels (good vs. bad target). This project uses a **self-supervised labeling strategy**:

```
Top 33% by OT association score  →  labeled High Priority (1)
Bottom 33%                        →  labeled Low Priority (0)
Middle 33%                        →  excluded (ambiguous signal)
```

A **Random Forest (200 trees)** is then trained on the 5 evidence features and predicts a **priority probability (0–1)** for all targets. This captures non-linear interactions between evidence types that a simple weighted sum cannot.

The model also outputs **feature importances**, revealing which evidence dimensions drive target prioritization for each disease.

---

## 🗂️ Project Structure

```
neurotarget-scout/
├── app.py              # Streamlit dashboard — 4 tabs, KPIs, plots
├── data_fetcher.py     # Open Targets GraphQL API integration
├── ml_model.py         # Random Forest training, scoring, feature importance
├── requirements.txt    # Python dependencies
└── README.md
```

---

## ⚙️ Installation & Usage

```bash
# 1. Clone the repository
git clone https://github.com/Sorra-22neurotarget-scout.git
cd neurotarget-scout

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the dashboard
streamlit run app.py
```

> **No API key required.** Open Targets Platform is completely free and open access.

---

## 🖥️ How to Use

1. Select **Schizophrenia** or **Major Depressive Disorder** from the sidebar
2. Set the number of targets to analyze (20–100)
3. Set a minimum association score filter
4. Click **🚀 Fetch & Score Targets**
5. Explore the 4 dashboard tabs:
   - **🏆 Ranked Targets** — color-coded priority table, CSV export
   - **📊 Score Analysis** — distributions, scatter plots, evidence profiles
   - **🔬 Feature Importance** — what evidence drives the RF model
   - **🔍 Target Deep-Dive** — per-gene breakdown + Open Targets link

---

## 📊 Example Output

| Rank | Gene | RF Priority Score | Genetic | Literature | Known Drugs |
|------|------|:-----------------:|:-------:|:----------:|:-----------:|
| 1 | DRD2 | 0.94 | 0.88 | 0.76 | 0.95 |
| 2 | HTR2A | 0.91 | 0.71 | 0.81 | 0.89 |
| 3 | COMT | 0.87 | 0.79 | 0.65 | 0.42 |
| 4 | CACNA1C | 0.81 | 0.74 | 0.58 | 0.31 |
| 5 | DISC1 | 0.76 | 0.69 | 0.71 | 0.00 |

> Targets with **high genetic score but zero known_drugs_score** are the most interesting — well-validated by biology but not yet drugged.

---

## 🔧 Tech Stack

| Layer | Technology |
|---|---|
| Data | Open Targets Platform GraphQL API |
| ML | scikit-learn · RandomForestClassifier |
| Dashboard | Streamlit |
| Visualization | Plotly |
| Language | Python 3.10+ |

---

## 🚀 Future Extensions

- [ ] SHAP explainability for individual target scores
- [ ] LLM-powered target summaries from PubMed abstracts
- [ ] Protein druggability scoring via AlphaFold pocket prediction
- [ ] Multi-disease comparison mode (SCZ vs. MDD side-by-side)
- [ ] Docker containerization for reproducible deployment

---

## 👤 Author

**Sarvesh** · MS Bioinformatics, Northeastern University  
Bridging pharmaceutical sciences and computational drug discovery

[![GitHub](https://img.shields.io/badge/GitHub-Sorra-22-181717?style=flat&logo=github)](https://github.com/Sorra-22)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin)](www.linkedin.com/in/sarveshsunilkurup)

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

---

*Data sourced from the [Open Targets Platform](https://platform.opentargets.org/) · Built for computational drug discovery portfolio*