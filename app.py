import requests
from bs4 import BeautifulSoup
import streamlit as st
from transformers import pipeline
from collections import Counter

# Hugging Face API URL and headers for NER
API_URL = "https://api-inference.huggingface.co/models/FacebookAI/xlm-roberta-large-finetuned-conll03-english"
headers = {"Authorization": "Bearer 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"}

# Function to query the Hugging Face model for NER
def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# Function to extract organizations from NER output using the Hugging Face API
def extract_organizations(text):
    output = query({"inputs": text})
    if isinstance(output, list) and len(output) > 0:
        orgs = [item['word'] for item in output if item['entity_group'] == 'ORG']
        return orgs
    else:
        return []

# Function to highlight organizations in the title with red color
def highlight_org_entities(title, orgs):
    highlighted_title = title
    for org in orgs:
        highlighted_title = highlighted_title.replace(org, f'<span style="color: red;">{org}</span>')
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
    org_counter = Counter()  # Track frequency of organizations
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

                    orgs_title = extract_organizations(title)
                    orgs_content = extract_organizations(content)
                    org_counter.update(orgs_title + orgs_content)

                    highlighted_title = highlight_org_entities(title, orgs_title)
                    articles.append({'title': highlighted_title, 'url': article_url, 'content': content, 'orgs': orgs_title + orgs_content})

    except Exception as e:
        print(f"Error: {e}")

    return articles, org_counter

# CSS Styling for Streamlit App
st.markdown("""
    <style>
    body {
        background-color: #f5f5f5;
        font-family: 'Arial', sans-serif;
    }
    .stTitle {
        color: #1a73e8;
        font-size: 2.5em;
        text-align: center;
        margin-bottom: 1em;
    }
    .stSidebar {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    }
    .stCheckbox, .stButton {
        margin: 5px 0;
    }
    .stButton button {
        background-color: #1a73e8;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 15px;
        font-size: 1em;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: #135ba1;
    }
    .article {
        background-color: white;
        padding: 20px;
        margin-bottom: 15px;
        border-radius: 10px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
    }
    .article h3 {
        font-size: 1.2em;
        color: #1a73e8;
        margin-bottom: 10px;
    }
    .article p {
        font-size: 1em;
        color: #555;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.title("Investment Intelligence System")

if "scraped_articles" not in st.session_state:
    st.session_state["scraped_articles"] = []
if "org_counter" not in st.session_state:
    st.session_state["org_counter"] = Counter()

# Scrape Articles Section
base_url_input = st.text_input("Enter the website URL:")
if st.button("Scrape Data"):
    if not base_url_input.strip():
        st.error("Please enter a website URL!")
    else:
        with st.spinner("Scraping website..."):
            articles, org_counter = crawl_website(base_url_input, start_page=30, end_page=100, step=30)
            st.session_state["scraped_articles"] = articles
            st.session_state["org_counter"] = org_counter
        if articles:
            st.success(f"Scraped {len(articles)} articles!")
        else:
            st.warning("No articles found.")

if st.session_state["scraped_articles"]:
    with st.sidebar:
        st.write("### Trending Organizations")
        trending_orgs = [org for org, _ in st.session_state["org_counter"].most_common(10)]
        for org in trending_orgs:
            if st.button(org):
                filtered_articles = [article for article in st.session_state["scraped_articles"] if org in article['orgs']]
                st.write(f"### Articles related to **{org}**:")
                for article in filtered_articles:
                    st.markdown(f"""
                        <div class="article">
                            <h3>{article['title']}</h3>
                            <p>{article['content'][:300]}...</p>
                            <a href="{article['url']}" target="_blank">Read More</a>
                        </div>
                    """, unsafe_allow_html=True)

    st.write("## Filter Articles")
    # Filters Section
    company_selected = [company for company in companies if st.sidebar.checkbox(company, key=f"company_{company}")]
    industry_selected = [industry for industry in industries if st.sidebar.checkbox(industry, key=f"industry_{industry}")]
    news_selected = [category for category in news_categories if st.sidebar.checkbox(category, key=f"category_{category}")]

    selected_options = company_selected + industry_selected + news_selected
    hypothesis_template = "This text is about {}."

    filtered_articles = []
    if selected_options:
        with st.spinner("Classifying articles..."):
            for article in st.session_state["scraped_articles"]:
                detected_categories = classify_content(article['title'], selected_options, hypothesis_template)
                if any(score > 0.5 for score in detected_categories.values()):
                    filtered_articles.append(article)
    else:
        filtered_articles = st.session_state["scraped_articles"]

    st.write("### Filtered Articles:")
    for article in filtered_articles:
        st.markdown(f"""
            <div class="article">
                <h3>{article['title']}</h3>
                <p>{article['content'][:300]}...</p>
                <a href="{article['url']}" target="_blank">Read More</a>
            </div>
        """, unsafe_allow_html=True)
