import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import validators
import math

# Load the model and tokenizer directly
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")

# Device selection based on availability (GPU or CPU)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# List of companies and industries
companies = ['Hemas', 'John Keells', 'Dialog', 'CSE']
industries = [
    "Energy", "Materials", "Industrials", "Consumer Discretionary",
    "Consumer Staples", "Health Care", "Financials", "Information Technology",
    "Communication Services", "Utilities", "Real Estate"
]

# Function to classify content
def classify_content(sentence, categories, hypothesis_template):
    max_length = 512  # Truncate to model's token limit
    sentence = sentence[:max_length]

    # Prepare the inputs for the model
    encoded_inputs = tokenizer(
        sentence, 
        padding=True, 
        truncation=True, 
        return_tensors="pt", 
        max_length=max_length
    ).to(device)

    # Create a list of hypotheses (e.g., "This text is about [category]")
    hypotheses = [hypothesis_template.format(category) for category in categories]
    encoded_hypotheses = tokenizer(
        hypotheses, 
        padding=True, 
        truncation=True, 
        return_tensors="pt", 
        max_length=max_length
    ).to(device)

    # Perform zero-shot classification using the model's forward method
    with torch.no_grad():  # Disable gradient computation for inference
        outputs = model(
            input_ids=encoded_inputs["input_ids"], 
            attention_mask=encoded_inputs["attention_mask"], 
            decoder_input_ids=encoded_hypotheses["input_ids"]
        )

    # Extract the logits (scores) and convert them to a dictionary with category labels
    logits = outputs.logits.squeeze().cpu().numpy()
    results = {label: score for label, score in zip(categories, logits)}

    return results

# Scraping function to extract articles from the website
def crawl_website(base_url, start_page, end_page, step, output_dir):
    articles = []
    try:
        os.makedirs(output_dir, exist_ok=True)

        for page_num in range(start_page, end_page, step):
            page_url = f"{base_url}/{page_num}"
            try:
                response = requests.get(page_url, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')

                # Extracting articles from specific div
                article_divs = soup.find_all('div', class_='col-md-6 lineg')
                for div in article_divs:
                    link = div.find('a', href=True)
                    if link:
                        article_url = link['href']
                        if not article_url.startswith("http"):
                            article_url = base_url + article_url
                        try:
                            article_response = requests.get(article_url, timeout=10)
                            article_response.raise_for_status()
                            article_soup = BeautifulSoup(article_response.text, 'html.parser')

                            # Extracting article title and content
                            title = article_soup.find('h1').get_text(strip=True) if article_soup.find('h1') else "No Title"
                            content_div = article_soup.find('header', class_='inner-content')
                            content = "\n".join([p.get_text(strip=True) for p in content_div.find_all('p')]) if content_div else "No content found."
                            
                            # Storing article details
                            articles.append({'title': title, 'url': article_url, 'content': content})
                        except Exception as e:
                            print(f"Error scraping article {article_url}: {e}")
            except Exception as e:
                print(f"Error accessing page {page_url}: {e}")
    except Exception as e:
        print(f"An error occurred during scraping: {e}")

    return articles

# Streamlit UI
st.title("Investment Intelligence System")

# Initialize global variable for storing articles in session state
if "scraped_articles" not in st.session_state:
    st.session_state["scraped_articles"] = []

# Step 1: Scrape articles from the URL
st.write("""Paste the website URL from which you want to scrape news articles. The articles will be stored, and you can filter them by company or industry.""")

base_url_input = st.text_input("Enter the website URL:")

if st.button("Scrape Data"):
    if not validators.url(base_url_input):
        st.error("Invalid URL. Please enter a valid website URL!")
    else:
        with st.spinner("Scraping website..."):
            articles = crawl_website(base_url_input, start_page=1, end_page=10, step=1, output_dir="scraped_data")
            st.session_state["scraped_articles"] = articles  # Store articles in session state

        if articles:
            st.success(f"Scraped {len(articles)} articles!")
        else:
            st.warning("No articles found.")

# Step 2: Apply Filters
if st.session_state["scraped_articles"]:
    st.write("## Filter Articles")
    classification_type = st.radio("Filter by:", ["Company", "Industry"])

    if classification_type == "Company":
        selected_option = st.selectbox("Select a company", companies)
        categories = companies
        hypothesis_template = "This text is about {}."
    else:
        selected_option = st.selectbox("Select an industry", industries)
        categories = industries
        hypothesis_template = "This text is about the {} industry."

    # Filtering functionality
    if st.button("Apply Filter"):
        if f"classified_{selected_option}" not in st.session_state:
            with st.spinner("Classifying articles..."):
                categorized_articles = {category: [] for category in categories}

                # Classifying articles
                for article in st.session_state["scraped_articles"]:
                    detected_categories = classify_content(article['content'], categories, hypothesis_template)
                    for category, score in detected_categories.items():
                        if score > 0.5:  # Threshold for "entailment"
                            categorized_articles[category].append(article)

                st.session_state[f"classified_{selected_option}"] = categorized_articles[selected_option]
        else:
            categorized_articles = st.session_state[f"classified_{selected_option}"]

        # Paginate results
        if categorized_articles:
            num_articles = len(categorized_articles)
            articles_per_page = 5
            num_pages = math.ceil(num_articles / articles_per_page)

            page = st.number_input("Page", 1, num_pages, step=1)
            start_idx = (page - 1) * articles_per_page
            end_idx = start_idx + articles_per_page

            for article in categorized_articles[start_idx:end_idx]:
                st.write(f"**{article['title']}**")
                st.write(f"[Read more]({article['url']})")
                st.write(f"{article['content'][:500]}...")  # Show first 500 characters

        else:
            st.write(f"No news found for {selected_option}.")
else:
    st.info("No articles scraped yet. Please scrape data first.")
