import os
import validators
import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

# Load environment variables from .env (local development)
load_dotenv()

# Get API key from environment (works both locally and in deployment)
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("❌ Groq API key is not set. Please set GROQ_API_KEY in backend environment.")
    st.stop()

# Streamlit app setup
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader("Summarize from a URL (YouTube or website)")

# URL input
generic_url = st.text_input("Enter YouTube or Website URL")

# Prompt template
prompt_template = """
Provide a summary of the following content in 600 words:
Content:{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

if st.button("Summarize"):
    if not generic_url.strip():
        st.error("Please enter a URL.")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL.")
    else:
        try:
            with st.spinner("Fetching content..."):
                # Load data from YouTube or Website
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0"}
                    )
                docs = loader.load()

                # Initialize Groq LLM (updated model)
                llm = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)

                # Summarization chain
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)

                st.success("✅ Summary generated successfully!")
                st.write(output_summary)

        except Exception as e:
            st.exception(f"Exception: {e}")
