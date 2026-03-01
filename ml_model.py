"""
Random Forest target prioritization model.

Since Open Targets doesn't provide labeled "good/bad target" ground truth,
we use a self-supervised approach:
  - Targets with high overall_score are labeled as "high priority" (1)
  - Targets with low overall_score are labeled as "low priority" (0)
  - We then train RF on the evidence features and predict a priority score

This gives us:
  1. A model-learned weighting of evidence types vs. raw linear combination
  2. Feature importances showing WHICH evidence matters most
  3. A probability score (0-1) that serves as the final priority ranking
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score

FEATURE_COLS = [
    "genetic_score",
    "literature_score",
    "animal_model_score",
    "known_drugs_score",
    "somatic_score",
]

FEATURE_LABELS = {
    "genetic_score":       "Genetic Association",
    "literature_score":    "Literature Evidence",
    "animal_model_score":  "Animal Model",
    "known_drugs_score":   "Known Drugs",
    "somatic_score":       "Somatic Mutation",
}


def train_and_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Train a Random Forest on evidence features and return df with
    rf_priority_score column added.

    Args:
        df: DataFrame from fetch_targets()

    Returns:
        df with 'rf_priority_score' column added
    """
    df = df.copy()

    # ── Feature matrix ────────────────────────────────────────────────────────
    X = df[FEATURE_COLS].fillna(0).values

    # Scale features to [0, 1]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Self-supervised labeling ───────────────────────────────────────────────
    # Label top 33% as "high priority" (1), bottom 33% as "low priority" (0)
    # Middle 33% excluded to give the model a cleaner signal
    overall = df["overall_score"].values
    threshold_high = np.percentile(overall, 67)
    threshold_low  = np.percentile(overall, 33)

    labels = []
    idx_keep = []
    for i, s in enumerate(overall):
        if s >= threshold_high:
            labels.append(1)
            idx_keep.append(i)
        elif s <= threshold_low:
            labels.append(0)
            idx_keep.append(i)

    X_train = X_scaled[idx_keep]
    y_train = np.array(labels)

    # ── Train Random Forest ───────────────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=4,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced"
    )
    rf.fit(X_train, y_train)

    # Cross-val score for display
    cv_scores = cross_val_score(rf, X_train, y_train, cv=min(5, len(idx_keep) // 2), scoring="roc_auc")

    # ── Score ALL targets ─────────────────────────────────────────────────────
    proba = rf.predict_proba(X_scaled)
    # Column 1 = probability of being "high priority"
    df["rf_priority_score"] = proba[:, 1]

    # ── Store feature importances in session state for UI ─────────────────────
    importances = rf.feature_importances_
    st.session_state["feature_importances"] = {
        FEATURE_LABELS[col]: float(imp)
        for col, imp in zip(FEATURE_COLS, importances)
    }
    st.session_state["cv_auc"] = float(cv_scores.mean())

    return df.sort_values("rf_priority_score", ascending=False).reset_index(drop=True)