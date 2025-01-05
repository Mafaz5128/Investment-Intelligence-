import streamlit as st
from transformers import pipeline

# Load the entailment model
entailment_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)

# List of companies
companies = ['Hemas', 'John Keells', 'Dialog']

# Function to classify companies using zero-shot classification
def classify_companies(sentence, companies):
    """
    Perform zero-shot classification to detect which companies are mentioned.

    Args:
        sentence (str): The input text to classify.
        companies (list): List of company names to check.

    Returns:
        dict: A dictionary mapping company names to entailment scores.
    """
    result = entailment_model(
        sequences=sentence,
        candidate_labels=companies,
        hypothesis_template="This text is about {}."
    )
    results = {label: score for label, score in zip(result["labels"], result["scores"])}
    return results

# Streamlit UI
st.title("News Categorization by Companies")

st.write("""
    Paste news articles (one per line) and categorize them based on the companies mentioned.
""")

# Text input for articles
text_input = st.text_area("Enter news articles here (one per line):")

if st.button("Categorize"):
    # Split input into individual sentences (articles)
    text_dataset = text_input.split("\n")
    
    categorized_articles = {}

    # Process each article
    for sentence in text_dataset:
        detected_companies = classify_companies(sentence, companies)

        # Categorize articles based on companies
        for company, score in detected_companies.items():
            if score > 0.5:  # Threshold for "entailment"
                if company not in categorized_articles:
                    categorized_articles[company] = []
                categorized_articles[company].append(sentence)

    if categorized_articles:
        st.subheader("Results")
        for company, articles in categorized_articles.items():
            st.write(f"### {company}")
            for article in articles:
                st.write(f"- {article}")
    else:
        st.write("No companies found in the articles.")

