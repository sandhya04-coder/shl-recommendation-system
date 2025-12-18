import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import plotly.express as px

# ======================================================
# CONFIG
# ======================================================
st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="🚀",
    layout="wide"
)

# ======================================================
# JOB ROLE PRESETS
# ======================================================
JOB_ROLE_PRESETS = {
    "Data Analyst": "data analysis, SQL, statistics, numerical reasoning, critical thinking",
    "Software Engineer": "problem solving, logical reasoning, algorithms, coding, debugging",
    "Product Manager": "decision making, stakeholder management, communication, strategy",
    "HR Executive": "behavioral assessment, situational judgement, personality traits",
    "Management Trainee": "general aptitude, logical reasoning, leadership potential"
}

# ======================================================
# LOAD SHL ASSESSMENT CATALOG
# ======================================================
@st.cache_data
def load_shl_catalog():
    data = {
        "name": [
            "Numerical Reasoning Test", "Verbal Reasoning Test", "Inductive Reasoning",
            "Deductive Reasoning", "Situational Judgement Test", "Personality Questionnaire",
            "Abstract Reasoning", "Critical Thinking", "OPQ32", "Verify Interactive",
            "Talent Q Dimensions", "Universal Competency Assessment",
            "General Ability Test", "Work Style Questionnaire", "Motivation Questionnaire"
        ],
        "skills": [
            "numerical analysis, data interpretation, percentages, ratios",
            "reading comprehension, logical inference, vocabulary",
            "pattern recognition, sequences, spatial reasoning",
            "logical deduction, syllogisms, conditional reasoning",
            "workplace scenarios, decision making, ethics",
            "big five personality, work preferences",
            "non-verbal reasoning, matrices, analogies",
            "argument analysis, evidence evaluation",
            "occupational personality, 32 dimensions",
            "ability and personality combined",
            "talent q behavioral styles",
            "competency-based assessment",
            "general cognitive ability",
            "work style preferences",
            "career motivation drivers"
        ],
        "difficulty": [
            "Medium", "Medium", "High", "High", "Medium",
            "Low", "High", "High", "Medium", "High",
            "Medium", "Medium", "Medium", "Low", "Medium"
        ],
        "duration": [25, 20, 30, 25, 35, 20, 28, 30, 40, 45, 30, 35, 25, 20, 25],
        "category": [
            "Cognitive", "Cognitive", "Cognitive", "Cognitive", "Behavioral",
            "Personality", "Cognitive", "Cognitive", "Personality", "Combined",
            "Personality", "Behavioral", "Cognitive", "Personality", "Personality"
        ]
    }
    return pd.DataFrame(data)

# ======================================================
# LOAD MODEL
# ======================================================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# ======================================================
# FEEDBACK STORAGE (LEARNING COMPONENT)
# ======================================================
@st.cache_data
def init_feedback():
    return pd.DataFrame(columns=["skills", "assessment", "rating"])

# ======================================================
# APPLY FEEDBACK BOOST (ML LEARNING)
# ======================================================
def apply_feedback_boost(catalog, feedback_df):
    if feedback_df.empty:
        catalog["final_score"] = catalog["score"]
        return catalog

    feedback_weight = feedback_df.groupby("assessment")["rating"].mean() / 5
    catalog["feedback_boost"] = catalog["name"].map(feedback_weight).fillna(0)
    catalog["final_score"] = 0.8 * catalog["score"] + 0.2 * catalog["feedback_boost"]
    return catalog.sort_values("final_score", ascending=False)

# ======================================================
# MAIN UI
# ======================================================
st.title("🚀 SHL Assessment Recommendation Engine")
st.markdown("AI-powered assessment recommendations using **semantic ML + learning feedback**")

model = load_model()
catalog = load_shl_catalog()
feedback_db = init_feedback()

# ------------------------------------------------------
# USER INPUT
# ------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    selected_role = st.selectbox(
        "Select job role (optional)",
        ["None"] + list(JOB_ROLE_PRESETS.keys())
    )

with col2:
    difficulty_filter = st.multiselect(
        "Preferred difficulty",
        ["Low", "Medium", "High"],
        default=["Medium", "High"]
    )

user_input = st.text_area(
    "Your skills / experience",
    placeholder="Python, SQL, leadership, problem solving"
)

n_recommendations = st.slider("Number of recommendations", 3, 10, 5)

# Auto-fill from role
if selected_role != "None":
    user_input = JOB_ROLE_PRESETS[selected_role]

# ------------------------------------------------------
# RECOMMENDATION ENGINE
# ------------------------------------------------------
if st.button("🔍 Get Recommendations", type="primary") and user_input:
    with st.spinner("Running ML recommender..."):

        # Filter catalog
        filtered_catalog = catalog[catalog["difficulty"].isin(difficulty_filter)].copy()

        # Embeddings
        user_embedding = model.encode([user_input])
        catalog_embeddings = model.encode(filtered_catalog["skills"].tolist())

        # Similarity
        similarities = cosine_similarity(user_embedding, catalog_embeddings)[0]
        filtered_catalog["score"] = similarities

        # Apply learning boost
        ranked_catalog = apply_feedback_boost(filtered_catalog, feedback_db)

        # Top N
        top_recs = ranked_catalog.head(n_recommendations)

        # --------------------------------------------------
        # DISPLAY RESULTS
        # --------------------------------------------------
        st.success(f"✅ Top {len(top_recs)} recommendations")

        for _, rec in top_recs.iterrows():
            with st.container(border=True):
                c1, c2, c3 = st.columns([3, 1, 1])
                c1.markdown(f"### {rec['name']}")
                c1.caption(f"{rec['category']} • {rec['difficulty']} • {rec['duration']} min")
                c2.metric("Match", f"{rec['final_score']:.1%}")
                c3.info(rec["skills"])

                # Feedback
                rating = st.slider(
                    f"Rate relevance",
                    1, 5, 3,
                    key=f"rate_{rec['name']}"
                )

                if st.button(f"Submit feedback", key=f"fb_{rec['name']}"):
                    feedback_db.loc[len(feedback_db)] = [
                        user_input, rec["name"], rating
                    ]
                    st.toast("Feedback saved ✔")

        # --------------------------------------------------
        # VISUALIZATION
        # --------------------------------------------------
        fig = px.bar(
            top_recs,
            x="name",
            y="final_score",
            color="category",
            title="Recommendation Confidence"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "🧠 **How it works:** Semantic similarity + difficulty filtering + "
            "feedback-driven ML re-ranking"
        )

else:
    st.info("👆 Select a role or enter skills to begin")

