import requests
from bs4 import BeautifulSoup
import os
import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Hugging Face API URL and headers for NER
API_URL = "https://api-inference.huggingface.co/models/FacebookAI/xlm-roberta-large-finetuned-conll03-english"
headers = {"Authorization": "Bearer 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"}

# Function to query the Hugging Face model for NER
def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# Function to extract organizations from NER output using the Hugging Face API
def extract_organizations(title):
    output = query({"inputs": title})
    if isinstance(output, list) and len(output) > 0:
        orgs = [item['word'] for item in output if item['entity_group'] == 'ORG']
        return orgs
    else:
        return []

# Function to highlight organizations in the title
def highlight_org_entities(title):
    orgs = extract_organizations(title)  # Get organizations from the title
    highlighted_title = title
    for org in orgs:
        highlighted_title = highlighted_title.replace(org, f"<b>{org}</b>")  # Highlight orgs in title
    return highlighted_title

# Load the entailment model for news category classification
entailment_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)

# List of companies, industries, and news categories for filtering
companies = ['Hemas', 'John Keells', 'Dialog', 'CSE']
industries = [
    "Energy", "Materials", "Industrials", "Consumer Discretionary",
    "Consumer Staples", "Health Care", "Financials", "Information Technology",
    "Communication Services", "Utilities", "Real Estate"
]
news_categories = [
    "Politics", "Business", "Technology", "Health", "Entertainment", "Sports"
]

# Function to classify content based on selected categories (using entailment model)
def classify_content(sentence, categories, hypothesis_template):
    result = entailment_model(
        sequences=sentence,
        candidate_labels=categories,
        hypothesis_template=hypothesis_template
    )
    return {label: score for label, score in zip(result["labels"], result["scores"])}

# Function to scrape articles from the website
def crawl_website(base_url, start_page, end_page, step):
    articles = []
    try:
        for page_num in range(start_page, end_page, step):
            page_url = f"{base_url}/{page_num}"
            response = requests.get(page_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            article_divs = soup.find_all('div', class_='col-md-6 lineg')

            for div in article_divs:
                link = div.find('a', href=True)
                if link:
                    article_url = link['href']
                    if not article_url.startswith("http"):
                        article_url = base_url + article_url

                    article_response = requests.get(article_url)
                    article_response.raise_for_status()
                    article_soup = BeautifulSoup(article_response.text, 'html.parser')

                    title = article_soup.find('h1').get_text(strip=True) if article_soup.find('h1') else "No Title"
                    content_div = article_soup.find('header', class_='inner-content')
                    content = "\n".join([p.get_text(strip=True) for p in content_div.find_all('p')]) if content_div else "No content found."

                    highlighted_title = highlight_org_entities(title)  # Use the new approach for highlighting orgs
                    articles.append({'title': highlighted_title, 'url': article_url, 'content': content})

    except Exception as e:
        print(f"Error: {e}")

    return articles

# Streamlit UI
st.title("Investment Intelligence System")

# Initialize session state to store articles
if "scraped_articles" not in st.session_state:
    st.session_state["scraped_articles"] = []

# Step 1: Scrape Articles
base_url_input = st.text_input("Enter the website URL:")
if st.button("Scrape Data"):
    if base_url_input.strip() == "":
        st.error("Please enter a website URL!")
    else:
        with st.spinner("Scraping website..."):
            articles = crawl_website(base_url_input, start_page=30, end_page=100, step=30)
            st.session_state["scraped_articles"] = articles
        st.success(f"Scraped {len(articles)} articles!") if articles else st.warning("No articles found.")

# Step 2: Filter Articles
if st.session_state["scraped_articles"]:
    st.write("## Filter Articles")
    
    # Sidebar filters
    st.sidebar.subheader("Filter by Company")
    company_selected = [company for company in companies if st.sidebar.checkbox(company, value=False)]
    
    st.sidebar.subheader("Filter by Industry")
    industry_selected = [industry for industry in industries if st.sidebar.checkbox(industry, value=False)]
    
    st.sidebar.subheader("Filter by News Classification")
    news_selected = [category for category in news_categories if st.sidebar.checkbox(category, value=False)]
    
    # Combine the selected categories
    selected_options = company_selected + industry_selected + news_selected
    hypothesis_template = "This text is about {}."
    
    filtered_articles = []
    if selected_options:
        with st.spinner("Classifying articles..."):
            for article in st.session_state["scraped_articles"]:
                detected_categories = classify_content(article['title'], selected_options, hypothesis_template)
                for category, score in detected_categories.items():
                    if score > 0.5:
                        filtered_articles.append(article)
                        break  # Stop checking further categories once a match is found
    else:
        filtered_articles = st.session_state["scraped_articles"]  # Show all if no filter is selected

    # Display filtered articles
    st.write("### Filtered Articles:")
    for article in filtered_articles:
        st.markdown(f"**{article['title']}**")  # Highlighted titles
        st.write(f"[Read more]({article['url']})")
        st.write(f"{article['content'][:300]}...")  # Show first 300 characters

else:
    st.info("No articles scraped yet. Please scrape data first.")
