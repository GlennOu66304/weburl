import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import pipeline



# Function to fetch HTML content from a URL
def fetch_html_content(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        st.error(f"Failed to fetch HTML content: {response.status_code}")
        return None

# Function to extract text content from HTML
def extract_text(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    # Extract text from relevant HTML elements
    text = soup.get_text()
    return text

# Function to summarize text
def summarize_text(text):
    summarizer = pipeline("summarization")
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
    return summary

# Function to answer questions using a pre-trained language model
def answer_question(question, context):
    qa_pipeline = pipeline("question-answering")
    answer = qa_pipeline(question=question, context=context)
    return answer['answer']

def main():
    st.title("Web Content Summarizer & QA")

    # Input URL
    url = st.text_input("Enter URL:")
    if url:
        html_content = fetch_html_content(url)
        if html_content:
            text_content = extract_text(html_content)
            st.write("Text content extracted:")
            st.write(text_content)

            # Summarize text
            st.write("Summary:")
            # summary = summarize_text(text_content)
            # st.write(summary)

            # Question answering
            question = st.text_input("Ask a question about the content:")
            if question:
                answer = answer_question(question, text_content)
                st.write("Answer:")
                st.write(answer)
        else:
            st.warning("Please enter a valid URL.")

if __name__ == "__main__":
    main()
