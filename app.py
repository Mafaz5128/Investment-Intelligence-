import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import os
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large-finetuned-conll03-english")
model = AutoModelForTokenClassification.from_pretrained("xlm-roberta-large-finetuned-conll03-english")
ner_pipe = pipeline("ner", model=model, tokenizer=tokenizer)

# Load the entailment model
entailment_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)


# List of companies, industries, and news categories
companies = ['Hemas', 'John Keells', 'Dialog', 'CSE']
industries = [
    "Energy", "Materials", "Industrials", "Consumer Discretionary",
    "Consumer Staples", "Health Care", "Financials", "Information Technology",
    "Communication Services", "Utilities", "Real Estate"
]
news_categories = [
    "Politics", "Business", "Technology", "Health", "Entertainment", "Sports"
]

# Function to classify content
def classify_content(sentence, categories, hypothesis_template):
    result = entailment_model(
        sequences=sentence,
        candidate_labels=categories,
        hypothesis_template=hypothesis_template
    )
    results = {label: score for label, score in zip(result["labels"], result["scores"])}
    return results

# Function to highlight entities in the text
def highlight_org_entities(title):
    ner_results = ner_pipe(title)  # Get NER results
    highlighted_title = title

    # Loop through the NER results and highlight organizations (ORG)
    for entity in ner_results:
        if entity['entity'] == 'B-ORG' or entity['entity'] == 'I-ORG':  # Organization entities
            # Replace organization entities in the title with highlighted version
            highlighted_title = highlighted_title.replace(entity['word'], f"<b>{entity['word']}</b>")

    return highlighted_title

# Scraping function for extracting articles from the website
def crawl_website(base_url, start_page, end_page, step, output_dir):
    articles = []
    try:
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        for page_num in range(start_page, end_page, step):
            # Update the URL format as per your pattern (base_url + '44/{page_num}')
            page_url = f"{base_url}/{44}/{page_num}"  # Adjust the URL structure based on your site
            print(f"Accessing: {page_url}")

            try:
                # Fetch website content
                response = requests.get(page_url)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')

                # Find all article links within <div class="col-md-6 lineg">
                article_divs = soup.find_all('div', class_='col-md-6 lineg')

                for div in article_divs:
                    link = div.find('a', href=True)  # Extract the anchor tag with href
                    if link:
                        article_url = link['href']
                        # Ensure the URL is valid (absolute URL)
                        if not article_url.startswith("http"):
                            article_url = base_url + article_url

                        # Access the article page to scrape content
                        try:
                            article_response = requests.get(article_url)
                            article_response.raise_for_status()
                            article_soup = BeautifulSoup(article_response.text, 'html.parser')

                            # Extract the article title
                            title = article_soup.find('h1').get_text(strip=True) if article_soup.find('h1') else "No Title"

                            # Extract the article content (focus on <p> tags inside <header class="inner-content">)
                            content_div = article_soup.find('header', class_='inner-content')
                            if content_div:
                                paragraphs = content_div.find_all('p')
                                content = "\n".join([p.get_text(strip=True) for p in paragraphs])
                            else:
                                content = "No content found."

                            # Highlight ORG entities in the title
                            highlighted_title = highlight_org_entities(title)

                            # Store article details
                            articles.append({'title': highlighted_title, 'url': article_url, 'content': content})
                            print(f"Scraped article: {title}")

                        except requests.exceptions.RequestException as e:
                            print(f"Error accessing article: {article_url}: {e}")

            except requests.exceptions.RequestException as e:
                print(f"Error accessing page URL {page_url}: {e}")

    except Exception as e:
        print(f"An error occurred: {e}")

    return articles

# Streamlit UI
st.title("Investment Intelligence System")

# Initialize global variable for storing articles
if "scraped_articles" not in st.session_state:
    st.session_state["scraped_articles"] = []

# Step 1: Scrape articles
st.write("""Paste the website URL from which you want to scrape news articles. The articles will be displayed first, and you can filter them by company, industry, or news category.""")

base_url_input = st.text_input("Enter the website URL:")

if st.button("Scrape Data"):
    if base_url_input.strip() == "":
        st.error("Please enter a website URL!")
    else:
        with st.spinner("Scraping website..."):
            articles = crawl_website(base_url_input, start_page=30, end_page=100, step=30, output_dir="scraped_data")
            st.session_state["scraped_articles"] = articles  # Store scraped articles in session state

        if articles:
            st.success(f"Scraped {len(articles)} articles!")
        else:
            st.warning("No articles found.")

# Step 2: Filter articles with checkboxes
if st.session_state["scraped_articles"]:
    st.write("## Filter Articles")

    # Create a sidebar with checkboxes for Companies and Industries
    st.sidebar.subheader("Filter by Company")
    company_selected = [company for company in companies if st.sidebar.checkbox(company, value=False)]

    st.sidebar.subheader("Filter by Industry")
    industry_selected = [industry for industry in industries if st.sidebar.checkbox(industry, value=False)]

    st.sidebar.subheader("Filter by News Classification")
    news_selected = [category for category in news_categories if st.sidebar.checkbox(category, value=False)]

    # Combine the selected companies, industries, and news categories
    selected_options = company_selected + industry_selected + news_selected
    hypothesis_template = "This text is about {}."

    # Show all articles if no filter is selected, otherwise show filtered articles
    filtered_articles = []

    if selected_options:
        with st.spinner("Classifying articles..."):
            for article in st.session_state["scraped_articles"]:
                detected_categories = classify_content(article['title'], selected_options, hypothesis_template)
                for category, score in detected_categories.items():
                    if score > 0.5:  # Threshold for "entailment"
                        filtered_articles.append(article)
                        break  # No need to check further categories for this article
    else:
        filtered_articles = st.session_state["scraped_articles"]  # Show all articles if no filter is selected

    # Display filtered results
    st.write("### Filtered Articles:")
    for article in filtered_articles:
        st.markdown(f"**{article['title']}**")  # Use st.markdown for HTML rendering
        st.write(f"[Read more]({article['url']})")
        st.write(f"{article['content'][:300]}...")  # Show the first 300 characters
else:
    st.info("No articles scraped yet. Please scrape data first.")
