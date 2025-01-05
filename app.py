import streamlit as st
from transformers import pipeline

# Load the entailment model
entailment_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)

# List of companies
companies = ['Hemas', 'John Keells', 'Dialog']

# Function to classify companies using zero-shot classification
def classify_companies(sentence, companies):
    result = entailment_model(
        sequences=sentence,
        candidate_labels=companies,
        hypothesis_template="This text is about {}."
    )
    results = {label: score for label, score in zip(result["labels"], result["scores"])}
    return results

# Streamlit UI
st.title("Investment Intelligence System")

# Sidebar for selecting company
selected_company = st.sidebar.selectbox("Select a company", companies)

st.write("""
    Paste news articles (one per line) and categorize them based on the companies mentioned.
""")

# Text input for articles
text_input = st.text_area("Enter news articles here (one per line):")

# Button to trigger categorization
if st.button("Categorize"):
    if text_input.strip() == "":
        st.error("Please enter some articles!")
    else:
        # Split input into individual sentences (articles) and clean the text
        text_dataset = [sentence.strip() for sentence in text_input.split("\n") if sentence.strip()]

        # Initialize dictionary for categorized articles
        categorized_articles = {}

        # Process each article
        with st.spinner("Categorizing articles..."):
            for sentence in text_dataset:
                detected_companies = classify_companies(sentence, companies)

                # Categorize articles based on companies
                for company, score in detected_companies.items():
                    if score > 0.5:  # Threshold for "entailment"
                        if company not in categorized_articles:
                            categorized_articles[company] = []
                        categorized_articles[company].append(sentence)

        # Display results for the selected company
        if selected_company in categorized_articles:
            st.subheader(f"News related to {selected_company}")
            for article in categorized_articles[selected_company]:
                st.write(f"- {article}")
        else:
            st.write(f"No news found for {selected_company}.")
