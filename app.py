import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader
from dotenv import load_dotenv
import validators

# Load environment variables from .env file
load_dotenv()

# Streamlit configuration
st.set_page_config(page_title="QuickSummarizeIt", layout="centered")

st.title("QuickSummarizeIt")
st.subheader("Effortlessly summarize content from YouTube using advanced AI techniques with Langchain.")

# Input for the YouTube URL
youtube_url = st.text_input("Enter the YouTube URL to summarize", value='')

def summarize_content(youtube_url):
    if not youtube_url.strip():
        st.error("Please provide a YouTube URL to get started")
    elif not validators.url(youtube_url):
        st.error("Invalid URL")
    else:
        try:
            with st.spinner("Summarizing..."):
                loader = YoutubeLoader.from_youtube_url(youtube_url, add_video_info=True)
                data = loader.load()

                # Initialize the language model
                llm = ChatGroq(model="Gemma-7b-It", groq_api_key=st.secrets["GROQ_API_KEY"])

                # Create the prompt template
                prompt_template = PromptTemplate(
                    input_variables=["text"],
                    template='''Provide a summary of the following content in 300 words:
                    Content: {text}'''
                )

                # Create and run the summarization chain
                chain = load_summarize_chain(llm=llm, chain_type="stuff", prompt=prompt_template)
                summary = chain.run({"input_documents": data})
                st.success(summary)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Display button and handle click
if st.button("Summarize the content"):
    summarize_content(youtube_url)

# Footer
st.markdown(
    """
    <div class="footer">
        Maintained and developed by <a href="https://www.linkedin.com/in/kshitiz-garg-898403207/" target="_blank" style="color: white;">Kshitiz Garg</a>
    </div>
    """,
    unsafe_allow_html=True
)
