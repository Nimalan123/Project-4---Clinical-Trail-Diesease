
import os
import re
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from collections import Counter

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Clinical Trial Intelligence",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Clinical Trial Disease Classification")
st.caption(
    "NLP • Machine Learning • Clinical Trial Analytics"
)

# --------------------------------------------------
# 2. FILE PATHS
# --------------------------------------------------

DATA_PATH = "clinical_trials_raw_patient2trial_conditions.csv"
MODEL_PATH = "models/linear_svm_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

# --------------------------------------------------
# 3. LOAD DATA AND MODEL
# --------------------------------------------------

@st.cache_data
def load_data(path):
    data = pd.read_csv(path)

    required = ["brief_summary", "source_condition_query"]
    missing = [col for col in required if col not in data.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = data.dropna(subset=required).copy()

    data["brief_summary"] = (
        data["brief_summary"].astype(str).str.strip()
    )
    data["source_condition_query"] = (
        data["source_condition_query"].astype(str).str.strip()
    )

    data = data[
        data["brief_summary"].ne("") &
        data["source_condition_query"].ne("")
    ]

    return data


@st.cache_resource
def load_model(path):
    return joblib.load(path)


if not os.path.exists(DATA_PATH):
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

if not os.path.exists(MODEL_PATH):
    st.error(f"Model not found: {MODEL_PATH}")
    st.stop()

if not os.path.exists(VECTORIZER_PATH):
    st.error(f"Vectorizer not found: {VECTORIZER_PATH}")
    st.stop()

try:
    df = load_data(DATA_PATH)
    model = load_model(MODEL_PATH)
    vectorizer = load_model(VECTORIZER_PATH)
except Exception as exc:
    st.error(f"Unable to load project files: {exc}")
    st.stop()

# --------------------------------------------------
# 4. SIDEBAR
# --------------------------------------------------

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Disease Distribution",
        "Medical Terms",
        "Text Patterns",
        "Model Performance",
        "Disease Prediction"
    ]
)

st.sidebar.divider()
st.sidebar.metric("Clinical Trial Records", f"{len(df):,}")
st.sidebar.metric(
    "Disease Categories",
    df["source_condition_query"].nunique()
)

# --------------------------------------------------
# 5. DASHBOARD
# --------------------------------------------------

if page == "Dashboard":

    st.header("Clinical Trial Insights Dashboard")

    total_records = len(df)
    total_categories = df["source_condition_query"].nunique()
    avg_words = df["brief_summary"].str.split().str.len().mean()

    category_counts = df["source_condition_query"].value_counts()

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Trials", f"{total_records:,}")
    col2.metric("Disease Categories", total_categories)
    col3.metric("Average Summary Words", f"{avg_words:.1f}")

    st.subheader("Disease Category Distribution")

    fig, ax = plt.subplots(figsize=(11, 5))

    category_counts.sort_values().plot(
        kind="barh",
        ax=ax
    )

    ax.set_xlabel("Number of Clinical Trials")
    ax.set_ylabel("Disease Category")
    ax.set_title("Clinical Trials by Disease Category")

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Dataset Preview")

    st.dataframe(
        df[["brief_summary", "source_condition_query"]].head(10),
        use_container_width=True
    )

# --------------------------------------------------
# 6. DISEASE CATEGORY DISTRIBUTION
# --------------------------------------------------

elif page == "Disease Distribution":

    st.header("Disease Category Distribution")

    counts = df["source_condition_query"].value_counts()

    distribution = counts.rename_axis(
        "Disease Category"
    ).reset_index(name="Trial Count")

    distribution["Percentage"] = (
        distribution["Trial Count"] / len(df) * 100
    )

    st.dataframe(
        distribution.style.format(
            {"Percentage": "{:.2f}%"}
        ),
        use_container_width=True
    )

    fig, ax = plt.subplots(figsize=(11, 6))

    counts.sort_values().plot(
        kind="barh",
        ax=ax
    )

    ax.set_title("Clinical Trials by Disease Category")
    ax.set_xlabel("Number of Trials")
    ax.set_ylabel("Disease Category")

    st.pyplot(fig)
    plt.close(fig)

    st.download_button(
        "Download Disease Distribution CSV",
        distribution.to_csv(index=False).encode("utf-8"),
        file_name="disease_distribution.csv",
        mime="text/csv"
    )

# --------------------------------------------------
# 7. FREQUENTLY OCCURRING MEDICAL TERMS
# --------------------------------------------------

elif page == "Medical Terms":

    st.header("Frequently Occurring Medical Terms")

    @st.cache_data
    def calculate_terms(texts):
        # Basic word frequency analysis.
        # Stop words and generic terms are excluded.
        stop_words = set("""
            the a an and or but is are was were be been being
            to of in on for with by from at as that this these
            those it its their they them we our study studies
            trial patients patient participants participant
            group groups will may can have has had using use
            used assess evaluate evaluation include including
            treatment clinical safety efficacy effect effects
            compared compare determine determine whether
        """.split())

        counter = Counter()

        for text in texts:
            words = re.findall(
                r"\b[a-zA-Z]{3,}\b",
                str(text).lower()
            )

            counter.update(
                word for word in words
                if word not in stop_words
            )

        return counter.most_common(20)

    terms = calculate_terms(
        df["brief_summary"].tolist()
    )

    terms_df = pd.DataFrame(
        terms,
        columns=["Term", "Frequency"]
    )

    fig, ax = plt.subplots(figsize=(11, 7))

    terms_df.sort_values("Frequency").plot(
        kind="barh",
        x="Term",
        y="Frequency",
        ax=ax,
        legend=False
    )

    ax.set_title("Top 20 Frequently Occurring Terms")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Term")

    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(terms_df, use_container_width=True)

    st.caption(
        "These are frequent words in the summaries, not a "
        "validated medical terminology extraction. Generic "
        "terms may remain in the results."
    )

# --------------------------------------------------
# 8. CLINICAL TRIAL TEXT PATTERNS
# --------------------------------------------------

elif page == "Text Patterns":

    st.header("Clinical Trial Text Patterns and Trends")

    text_df = df.copy()

    text_df["Word Count"] = (
        text_df["brief_summary"].str.split().str.len()
    )

    text_df["Character Count"] = (
        text_df["brief_summary"].str.len()
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Average Words",
        f"{text_df['Word Count'].mean():.1f}"
    )

    c2.metric(
        "Median Words",
        f"{text_df['Word Count'].median():.0f}"
    )

    c3.metric(
        "Maximum Words",
        f"{text_df['Word Count'].max():.0f}"
    )

    st.subheader("Distribution of Summary Lengths")

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.histplot(
        data=text_df,
        x="Word Count",
        bins=50,
        ax=ax
    )

    ax.set_title("Clinical Trial Summary Word Counts")
    ax.set_xlabel("Number of Words")
    ax.set_ylabel("Number of Trials")

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Average Summary Length by Disease")

    avg_length = (
        text_df.groupby("source_condition_query")["Word Count"]
        .mean()
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(11, 6))

    avg_length.plot(kind="barh", ax=ax)

    ax.set_xlabel("Average Number of Words")
    ax.set_ylabel("Disease Category")
    ax.set_title("Average Summary Length by Disease Category")

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Summary Length Statistics")

    st.dataframe(
        text_df.groupby("source_condition_query")[
            ["Word Count", "Character Count"]
        ].agg(["mean", "median", "min", "max"]).round(2),
        use_container_width=True
    )

# --------------------------------------------------
# 9. MODEL PERFORMANCE
# --------------------------------------------------

elif page == "Model Performance":

    st.header("Model Performance and Prediction Accuracy")

    st.info(
        "The metrics below must be calculated on held-out test "
        "data, not on the entire dataset."
    )

    # Load the same held-out test set used during model evaluation.
    # Create test_results.csv in the training notebook as described below.
    RESULTS_PATH = "test_results.csv"

    if not os.path.exists(RESULTS_PATH):

        st.warning(
            "Test evaluation results are not available yet. "
            "Generate test_results.csv from your training notebook "
            "using the instructions below."
        )

    else:
        results_df = pd.read_csv(RESULTS_PATH)

        required_cols = ["actual", "predicted"]

        if not all(col in results_df.columns for col in required_cols):
            st.error(
                "test_results.csv must contain actual and predicted columns."
            )
        else:
            y_true = results_df["actual"]
            y_pred = results_df["predicted"]

            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(
                y_true, y_pred,
                average="macro",
                zero_division=0
            )
            recall = recall_score(
                y_true, y_pred,
                average="macro",
                zero_division=0
            )
            f1 = f1_score(
                y_true, y_pred,
                average="macro",
                zero_division=0
            )

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Accuracy", f"{accuracy:.2%}")
            c2.metric("Macro Precision", f"{precision:.2%}")
            c3.metric("Macro Recall", f"{recall:.2%}")
            c4.metric("Macro F1-score", f"{f1:.2%}")

            st.subheader("Classification Report")

            report = classification_report(
                y_true,
                y_pred,
                output_dict=True,
                zero_division=0
            )

            st.dataframe(
                pd.DataFrame(report).transpose().round(3),
                use_container_width=True
            )

            st.subheader("Confusion Matrix")

            labels = sorted(
                set(y_true) | set(y_pred)
            )

            cm = confusion_matrix(
                y_true,
                y_pred,
                labels=labels
            )

            fig, ax = plt.subplots(figsize=(11, 8))

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=labels,
                yticklabels=labels,
                ax=ax
            )

            ax.set_xlabel("Predicted Category")
            ax.set_ylabel("Actual Category")
            ax.set_title("Linear SVM Confusion Matrix")

            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)

            st.pyplot(fig)
            plt.close(fig)

# --------------------------------------------------
# 10. DISEASE PREDICTION
# --------------------------------------------------

elif page == "Disease Prediction":

    st.header("Predict Disease Category")

    summary = st.text_area(
        "Enter the clinical trial brief summary",
        height=250,
        placeholder="Paste a clinical trial brief summary..."
    )

    if st.button("Predict Disease Category", type="primary"):

        if not summary.strip():
            st.warning("Please enter a summary.")

        else:
            features = vectorizer.transform([summary])
            prediction = model.predict(features)[0]

            st.success(f"Predicted Category: {prediction}")

            st.caption(
                "This prediction supports clinical-trial text "
                "classification research. It is not a medical diagnosis."
            )

# --------------------------------------------------
# 11. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Clinical Trial Disease Category Classification | "
    "NLP and Machine Learning Project"
)
