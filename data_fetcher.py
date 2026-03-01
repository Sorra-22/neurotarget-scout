"""
Open Targets GraphQL API fetcher
Pulls target-disease associations for schizophrenia / MDD

Fixed for current Open Targets Platform API schema (2024/2025).
Key changes from old version:
  - associatedTargets now uses `aggregations` + `rows` differently
  - datatypeScores is now `datasourceScores` inside each row
  - Score field names updated to match current schema
"""

import requests
import pandas as pd
import streamlit as st

OT_API = "https://api.platform.opentargets.org/api/v4/graphql"

DISEASE_IDS = {
    "Schizophrenia":              "MONDO_0005090",
    "Major Depressive Disorder":  "EFO_0003761",
}

# Updated query using current schema
QUERY = """
query TargetsForDisease($diseaseId: String!, $size: Int!) {
  disease(efoId: $diseaseId) {
    name
    associatedTargets(
      page: { index: 0, size: $size }
    ) {
      rows {
        target {
          id
          approvedSymbol
          approvedName
        }
        score
        datasourceScores {
          id
          score
        }
      }
    }
  }
}
"""

# Maps Open Targets datasource IDs → readable column names
# These IDs reflect the current OT datasource naming
DATASOURCE_MAP = {
    # Genetic
    "ot_genetics_portal":    "genetic_score",
    "gwas_credible_sets":    "genetic_score",
    "gene_burden":           "genetic_score",
    "eva":                   "genetic_score",
    "eva_somatic":           "somatic_score",
    "cancer_gene_census":    "somatic_score",
    "intogen":               "somatic_score",
    # Literature
    "europepmc":             "literature_score",
    # Animal models
    "phenodigm":             "animal_model_score",
    "impc":                  "animal_model_score",
    # Known drugs
    "chembl":                "known_drugs_score",
    # Somatic / other
    "uniprot_variants":      "genetic_score",
    "clinvar":               "genetic_score",
    "crispr_screen":         "genetic_score",
}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_targets(disease: str, top_n: int = 50) -> pd.DataFrame | None:
    """
    Fetch target-disease associations from Open Targets Platform.

    Args:
        disease: 'Schizophrenia' or 'Major Depressive Disorder'
        top_n:   Number of top targets to retrieve (sorted by OT score)

    Returns:
        DataFrame with target info and aggregated evidence scores, or None on failure.
    """
    disease_id = DISEASE_IDS.get(disease)
    if not disease_id:
        st.error(f"Unknown disease: {disease}")
        return None

    payload = {
        "query": QUERY,
        "variables": {"diseaseId": disease_id, "size": top_n}
    }

    try:
        resp = requests.post(
            OT_API,
            json=payload,
            timeout=30,
            headers={"Content-Type": "application/json"}
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.HTTPError as e:
        # Show the actual API error message for easier debugging
        st.error(f"API HTTP error: {e}")
        try:
            st.code(resp.text[:800], language="json")
        except Exception:
            pass
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        return None

    # Check for GraphQL-level errors
    if "errors" in data:
        st.error("GraphQL query error:")
        st.json(data["errors"])
        return None

    rows = (
        data.get("data", {})
            .get("disease", {})
            .get("associatedTargets", {})
            .get("rows", [])
    )

    if not rows:
        st.warning("No rows returned from Open Targets. The disease ID may have changed.")
        return None

    records = []
    for row in rows:
        tgt = row["target"]

        # Initialize all score buckets to 0
        record = {
            "target_id":          tgt["id"],
            "gene_symbol":        tgt["approvedSymbol"],
            "target_name":        tgt["approvedName"],
            "overall_score":      float(row.get("score", 0) or 0),
            "genetic_score":      0.0,
            "literature_score":   0.0,
            "animal_model_score": 0.0,
            "known_drugs_score":  0.0,
            "somatic_score":      0.0,
        }

        # Aggregate datasource scores into the 5 evidence buckets
        # We take the MAX across datasources that map to the same bucket
        bucket_scores: dict[str, list[float]] = {
            "genetic_score":      [],
            "literature_score":   [],
            "animal_model_score": [],
            "known_drugs_score":  [],
            "somatic_score":      [],
        }

        for ds in row.get("datasourceScores", []):
            bucket = DATASOURCE_MAP.get(ds["id"])
            if bucket:
                bucket_scores[bucket].append(float(ds["score"] or 0))

        for bucket, scores in bucket_scores.items():
            record[bucket] = max(scores) if scores else 0.0

        records.append(record)

    df = pd.DataFrame(records)
    return df