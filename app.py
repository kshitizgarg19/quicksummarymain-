import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import validators

# Load environment variables from .env file
load_dotenv()

# Streamlit configuration
st.set_page_config(page_title="QuickSummarizeIt", layout="centered")

st.title("QuickSummarizeIt")
st.subheader("Effortlessly summarize content from YouTube or PDFs using advanced AI techniques with Langchain.")

# Option to choose between YouTube and PDF
option = st.selectbox("Choose input type:", ["YouTube", "PDF"])

# Input fields based on chosen option
if option == "YouTube":
    youtube_url = st.text_input("Enter the YouTube URL to summarize", value='')
    def summarize_content_youtube(youtube_url):
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
                    llm = ChatGroq(model="Gemma-7b-It", groq_api_key="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")

                    # Create the prompt template
                    prompt_template = PromptTemplate(
                        input_variables=["text"],
                        template='''Provide a summary of the following content in 300 words:
                        Content: {text}'''
                    )

                    # Create and run the summarization chain
                    chain = load_summarize_chain(llm=llm, chain_type="stuff", prompt=prompt_template)
                    summary = chain.run({"input_documents": data})
                    
                    # Display summary and copy button
                    st.markdown(f"**Summary:**")
                    st.write(summary)
                    st.markdown(
                        """
                        <button onclick="copyToClipboard()">Copy Summary</button>
                        <script>
                        function copyToClipboard() {
                            var summary = document.querySelector("pre").innerText;
                            navigator.clipboard.writeText(summary).then(function() {
                                alert('Summary copied to clipboard!');
                            }, function(err) {
                                console.error('Error copying text: ', err);
                            });
                        }
                        </script>
                        """,
                        unsafe_allow_html=True
                    )
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    
    if st.button("Summarize the content"):
        summarize_content_youtube(youtube_url)

elif option == "PDF":
    uploaded_files = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temp_pdf = f"./temp.pdf"
            with open(temp_pdf, "wb") as file:
                file.write(uploaded_file.getvalue())
                
            loader = PyPDFLoader(temp_pdf)
            docs = loader.load()
            documents.extend(docs)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()    

        system_prompt = (
            "You are an assistant for summarizing tasks. "
            "Use the following pieces of retrieved context to provide a summary of the text. "
            "If you don't know the summary, say that you don't know. Use three sentences maximum and keep the summary concise."
            "\n\n"
            "{context}"
        )
        qa_prompt = PromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )
        
        chain = load_summarize_chain(llm=llm, chain_type="stuff", prompt=qa_prompt)

        def summarize_content_pdf():
            with st.spinner("Summarizing..."):
                summary = chain.run({"input_documents": documents})
                
                # Display summary and copy button
                st.markdown(f"**Summary:**")
                st.write(summary)
                st.markdown(
                    """
                    <button onclick="copyToClipboard()">Copy Summary</button>
                    <script>
                    function copyToClipboard() {
                        var summary = document.querySelector("pre").innerText;
                        navigator.clipboard.writeText(summary).then(function() {
                            alert('Summary copied to clipboard!');
                        }, function(err) {
                            console.error('Error copying text: ', err);
                        });
                    }
                    </script>
                    """,
                    unsafe_allow_html=True
                )
    
    if st.button("Summarize the content"):
        summarize_content_pdf()

# Footer
st.markdown(
    """
    <div class="footer">
        Maintained and developed by <a href="https://www.linkedin.com/in/kshitiz-garg-898403207/" target="_blank" style="color: white;">Kshitiz Garg</a>
    </div>
    """,
    unsafe_allow_html=True
)
