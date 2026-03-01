"""
NeuroTarget Scout
Drug target prioritization for Schizophrenia & Depression
Uses Open Targets API + Random Forest ML scoring
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from data_fetcher import fetch_targets
from ml_model import train_and_score

st.set_page_config(
    page_title="NeuroTarget Scout",
    page_icon="🧠",
    layout="wide"
)

# ── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: #f0f4ff;
        border-radius: 10px;
        padding: 16px 20px;
        border-left: 4px solid #4f46e5;
    }
    h1 { color: #1e1b4b; }
    .stDataFrame { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🧠 NeuroTarget Scout")
st.caption("AI-assisted drug target prioritization for CNS diseases · Powered by Open Targets + Random Forest")

st.divider()

# ── Sidebar Controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    disease = st.selectbox(
        "Disease",
        ["Schizophrenia", "Major Depressive Disorder"],
        index=0
    )
    top_n = st.slider("Targets to analyze", min_value=20, max_value=100, value=50, step=10)
    min_score = st.slider("Min. Overall Association Score", 0.0, 1.0, 0.1, 0.05)

    st.divider()
    run_btn = st.button("🚀 Fetch & Score Targets", use_container_width=True, type="primary")

    st.divider()
    st.markdown("**Evidence types used:**")
    st.markdown("- 🧬 Genetic associations (GWAS)")
    st.markdown("- 📖 Literature (text mining)")
    st.markdown("- 🧪 Animal models")
    st.markdown("- 💊 Known drugs")
    st.markdown("- 🔬 Somatic mutations")

    st.divider()
    st.info("Data sourced live from [Open Targets Platform](https://platform.opentargets.org/) via GraphQL API.")

# ── Main Logic ────────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner(f"Fetching top {top_n} targets for **{disease}** from Open Targets..."):
        raw_df = fetch_targets(disease, top_n)

    if raw_df is None or raw_df.empty:
        st.error("Failed to fetch data. Check your internet connection or try again.")
        st.stop()

    with st.spinner("Training Random Forest model and scoring targets..."):
        scored_df = train_and_score(raw_df)

    # Filter
    filtered_df = scored_df[scored_df["overall_score"] >= min_score].copy()

    st.success(f"✅ Found **{len(filtered_df)}** targets after filtering (from {len(scored_df)} fetched)")

    # ── KPI Row ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Targets", len(filtered_df))
    c2.metric("Avg RF Priority Score", f"{filtered_df['rf_priority_score'].mean():.3f}")
    c3.metric("High-Priority Targets", int((filtered_df["rf_priority_score"] > 0.7).sum()))
    c4.metric("Targets w/ Known Drugs", int((filtered_df["known_drugs_score"] > 0).sum()))

    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["🏆 Ranked Targets", "📊 Score Analysis", "🔬 Feature Importance", "🔍 Target Deep-Dive"])

    # ── Tab 1: Ranked Table ───────────────────────────────────────────────────
    with tab1:
        st.subheader(f"Top Prioritized Targets — {disease}")
        display_cols = [
            "gene_symbol", "target_name", "rf_priority_score",
            "overall_score", "genetic_score", "literature_score",
            "animal_model_score", "known_drugs_score", "somatic_score"
        ]
        display_df = filtered_df[display_cols].sort_values("rf_priority_score", ascending=False).reset_index(drop=True)
        display_df.index += 1

        # Color-code priority score
        def color_score(val):
            if val >= 0.7:
                return "background-color: #bbf7d0; color: #14532d"
            elif val >= 0.4:
                return "background-color: #fef9c3; color: #713f12"
            else:
                return "background-color: #fee2e2; color: #7f1d1d"

        styled = display_df.style.applymap(color_score, subset=["rf_priority_score"]).format({
            col: "{:.3f}" for col in display_df.columns if "score" in col
        })
        st.dataframe(styled, use_container_width=True, height=480)

        csv = display_df.to_csv(index=False)
        st.download_button("⬇️ Download CSV", csv, file_name=f"neuro_targets_{disease.lower().replace(' ','_')}.csv", mime="text/csv")

    # ── Tab 2: Score Analysis ─────────────────────────────────────────────────
    with tab2:
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("RF Priority Score Distribution")
            fig1 = px.histogram(
                filtered_df, x="rf_priority_score", nbins=20,
                color_discrete_sequence=["#4f46e5"],
                labels={"rf_priority_score": "RF Priority Score"},
            )
            fig1.update_layout(showlegend=False, height=340)
            st.plotly_chart(fig1, use_container_width=True)

        with col_b:
            st.subheader("Genetic vs Literature Evidence")
            fig2 = px.scatter(
                filtered_df, x="genetic_score", y="literature_score",
                color="rf_priority_score", size="overall_score",
                hover_name="gene_symbol",
                color_continuous_scale="Viridis",
                labels={
                    "genetic_score": "Genetic Evidence Score",
                    "literature_score": "Literature Evidence Score",
                    "rf_priority_score": "RF Score"
                }
            )
            fig2.update_layout(height=340)
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Evidence Profile — Top 15 Targets")
        top15 = filtered_df.sort_values("rf_priority_score", ascending=False).head(15)
        evidence_cols = ["genetic_score", "literature_score", "animal_model_score", "known_drugs_score", "somatic_score"]
        fig3 = px.bar(
            top15.melt(id_vars="gene_symbol", value_vars=evidence_cols, var_name="Evidence Type", value_name="Score"),
            x="gene_symbol", y="Score", color="Evidence Type",
            barmode="group",
            color_discrete_sequence=px.colors.qualitative.Bold,
            labels={"gene_symbol": "Gene Symbol"}
        )
        fig3.update_layout(height=380, xaxis_tickangle=-35)
        st.plotly_chart(fig3, use_container_width=True)

    # ── Tab 3: Feature Importance ─────────────────────────────────────────────
    with tab3:
        st.subheader("Random Forest Feature Importance")
        st.caption("How much each evidence type contributes to the RF priority score")

        if "feature_importances" in st.session_state:
            fi = st.session_state["feature_importances"]
            fi_df = pd.DataFrame({"Feature": list(fi.keys()), "Importance": list(fi.values())})
            fi_df = fi_df.sort_values("Importance", ascending=True)

            fig4 = px.bar(
                fi_df, x="Importance", y="Feature", orientation="h",
                color="Importance", color_continuous_scale="Blues",
                labels={"Importance": "Feature Importance (Gini)"}
            )
            fig4.update_layout(height=340, showlegend=False)
            st.plotly_chart(fig4, use_container_width=True)

            st.markdown("**Interpretation:**")
            top_feat = fi_df.iloc[-1]["Feature"]
            st.markdown(f"- The most predictive feature is **{top_feat}**, meaning it drives the most variance in target prioritization.")
            st.markdown("- Features with low importance contribute less to distinguishing high-priority targets.")
        else:
            st.info("Run the model to see feature importances.")

    # ── Tab 4: Target Deep-Dive ───────────────────────────────────────────────
    with tab4:
        st.subheader("Single Target Inspector")
        gene_list = filtered_df.sort_values("rf_priority_score", ascending=False)["gene_symbol"].tolist()
        selected = st.selectbox("Select a target gene", gene_list)

        row = filtered_df[filtered_df["gene_symbol"] == selected].iloc[0]

        dc1, dc2 = st.columns([1, 2])
        with dc1:
            st.markdown(f"### {row['gene_symbol']}")
            st.caption(row.get("target_name", ""))
            st.metric("RF Priority Score", f"{row['rf_priority_score']:.3f}")
            st.metric("Overall OT Score", f"{row['overall_score']:.3f}")

            priority_label = "🟢 High Priority" if row["rf_priority_score"] > 0.7 else ("🟡 Medium" if row["rf_priority_score"] > 0.4 else "🔴 Low")
            st.markdown(f"**Priority tier:** {priority_label}")

        with dc2:
            evidence_types = {
                "Genetic": row["genetic_score"],
                "Literature": row["literature_score"],
                "Animal Model": row["animal_model_score"],
                "Known Drugs": row["known_drugs_score"],
                "Somatic": row["somatic_score"],
            }
            fig5 = go.Figure(go.Bar(
                x=list(evidence_types.values()),
                y=list(evidence_types.keys()),
                orientation="h",
                marker_color=["#4f46e5","#7c3aed","#db2777","#ea580c","#16a34a"]
            ))
            fig5.update_layout(
                title="Evidence Breakdown",
                xaxis_title="Score (0–1)",
                height=280,
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig5, use_container_width=True)

        ot_url = f"https://platform.opentargets.org/target/{row.get('target_id','')}"
        st.markdown(f"🔗 [View {row['gene_symbol']} on Open Targets Platform]({ot_url})")

else:
    # Landing state
    st.markdown("""
    ### Welcome to NeuroTarget Scout 👋

    This tool helps you identify and prioritize drug targets for **Schizophrenia** and **Major Depressive Disorder**
    using real evidence data from Open Targets and a Machine Learning scoring model.

    **How it works:**
    1. Select a disease and parameters in the sidebar
    2. Click **Fetch & Score Targets** to pull live data from the Open Targets API
    3. A Random Forest model scores each target across 5 evidence dimensions
    4. Explore ranked targets, visualizations, and individual deep-dives

    ---
    *Built with Open Targets Platform API · scikit-learn · Streamlit · Plotly*
    """)