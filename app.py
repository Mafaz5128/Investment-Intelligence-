import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import os

# Load the entailment model
entailment_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)

# List of companies and industries
companies = ['Hemas', 'John Keells', 'Dialog']
industries = [
    "Energy", "Materials", "Industrials", "Consumer Discretionary",
    "Consumer Staples", "Health Care", "Financials", "Information Technology",
    "Communication Services", "Utilities", "Real Estate"
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

# Scraping function for extracting articles from the website
def crawl_website(base_url, start_page, end_page, step, output_dir):
    articles = []
    try:
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Extract the category name from the URL
        category = base_url.split("/")[-2] if base_url.endswith("/") else base_url.split("/")[-1]

        for page_num in range(start_page, end_page, step):
            page_url = f"{base_url}/{page_num}"  # Construct the page URL
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

                            # Store article details
                            articles.append({'title': title, 'url': article_url, 'content': content})
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

# User choice: Classify by company or industry
classification_type = st.sidebar.radio("Classify by:", ["Company", "Industry"])

# Sidebar selection based on classification type
if classification_type == "Company":
    selected_option = st.sidebar.selectbox("Select a company", companies)
    categories = companies
    hypothesis_template = "This text is about {}."
else:
    selected_option = st.sidebar.selectbox("Select an industry", industries)
    categories = industries
    hypothesis_template = "This text is about the {} industry."

st.write("""
    Paste the website URL from which you want to scrape news articles, and it will categorize the articles based on your selection.
""")

# Text input for website URL
base_url_input = st.text_input("Enter the website URL:")

# Button to trigger crawling and categorization
if st.button("Scrape and Categorize"):
    if base_url_input.strip() == "":
        st.error("Please enter a website URL!")
    else:
        with st.spinner("Scraping website..."):
            articles = crawl_website(base_url_input, start_page=1, end_page=10, step=1, output_dir="scraped_data")

            if not articles:
                st.write("No articles found.")
            else:
                categorized_articles = {category: [] for category in categories}

                # Classify each article
                for article in articles:
                    detected_categories = classify_content(article['content'], categories, hypothesis_template)

                    for category, score in detected_categories.items():
                        if score > 0.5:  # Threshold for "entailment"
                            categorized_articles[category].append(article)

                # Display results for the selected category
                if categorized_articles[selected_option]:
                    st.subheader(f"News related to {selected_option}")
                    for article in categorized_articles[selected_option]:
                        st.write(f"**{article['title']}**")
                        st.write(f"[Read more]({article['url']})")
                        st.write(f"{article['content'][:500]}...")  # Show the first 500 chars
                else:
                    st.write(f"No news found for {selected_option}.")
