import streamlit as st
import pandas as pd
from transformers import pipeline
import re

# Load NLP Pipelines
@st.cache_resource
def load_pipelines():
    ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", grouped_entities=True)
    zero_shot_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    return ner_pipeline, zero_shot_pipeline

ner_pipeline, zero_shot_pipeline = load_pipelines()

# Define standard industry labels
INDUSTRY_LABELS = ["Healthcare", "Technology", "Finance", "Energy", "Consumer Goods", "Real Estate", "Utilities", "Industrials", "Telecommunications"]

# Extract company names using NER
def extract_companies(text, ner_model):
    entities = ner_model(text)
    companies = [ent['word'] for ent in entities if ent['entity_group'] == 'ORG']
    return list(set(companies))

# Classify article by industry using zero-shot classification
def classify_industry(text, zero_shot_model):
    classification = zero_shot_model(text, candidate_labels=INDUSTRY_LABELS)
    return classification['labels'][0]  # Top industry label

# Process news articles
def process_articles(articles):
    processed_data = []
    for article in articles:
        companies = extract_companies(article, ner_pipeline)
        industry = classify_industry(article, zero_shot_pipeline)
        processed_data.append({"article": article, "companies": companies, "industry": industry})
    return processed_data

# Streamlit app
st.title("Investment Intelligence System")

# Upload dataset
uploaded_file = st.file_uploader("ft.lk", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if "article" not in df.columns:
        st.error("The uploaded file must contain an 'article' column.")
    else:
        articles = df["Article Title"].dropna().tolist()

        # Process articles
        with st.spinner("Processing articles..."):
            results = process_articles(articles)

        # Convert results to DataFrame
        results_df = pd.DataFrame(results)

        # Display filters
        selected_company = st.selectbox("Filter by Company", ["All"] + list(set(sum(results_df["companies"].tolist(), []))))
        selected_industry = st.selectbox("Filter by Industry", ["All"] + INDUSTRY_LABELS)

        # Filter results
        filtered_df = results_df.copy()
        if selected_company != "All":
            filtered_df = filtered_df[filtered_df["companies"].apply(lambda x: selected_company in x)]
        if selected_industry != "All":
            filtered_df = filtered_df[filtered_df["industry"] == selected_industry]

        # Display results
        st.subheader("Filtered News Articles")
        for _, row in filtered_df.iterrows():
            st.write(f"**Article:** {row['article']}")
            st.write(f"- **Companies:** {', '.join(row['companies'])}")
            st.write(f"- **Industry:** {row['industry']}")
            st.markdown("---")
