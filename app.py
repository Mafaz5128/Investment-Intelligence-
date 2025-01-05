import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import os

# Load the entailment model
entailment_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)

# List of companies and industries
companies = ['Hemas', 'John Keells', 'Dialog', 'CSE']
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

# Initialize global variable for storing articles
if "scraped_articles" not in st.session_state:
    st.session_state["scraped_articles"] = []

# Step 1: Scrape articles
st.write("""
    Paste the website URL from which you want to scrape news articles. The articles will be displayed first, and you can filter them by company or industry.
""")
base_url_input = st.text_input("Enter the website URL:")

if st.button("Scrape Data"):
    if base_url_input.strip() == "":
        st.error("Please enter a website URL!")
    else:
        with st.spinner("Scraping website..."):
            articles = crawl_website(base_url_input, start_page=1, end_page=10, step=1, output_dir="scraped_data")
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

    # Combine the selected companies and industries
    selected_options = company_selected + industry_selected
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
        st.write(f"**{article['title']}**")
        st.write(f"[Read more]({article['url']})")
        st.write(f"{article['content'][:300]}...")  # Show the first 300 characters
else:
    st.info("No articles scraped yet. Please scrape data first.")
