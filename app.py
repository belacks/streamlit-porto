import streamlit as st
from PIL import Image
import os # Untuk memeriksa keberadaan file gambar

# --- Page Configuration (Optional but recommended) ---
st.set_page_config(
    page_title="Muhammad Ali Ashari Portfolio", # Changed to English
    page_icon="📊",
    layout="wide"
)

# --- Project Data ---
# Replace with your actual project details.
# Create a dictionary for each project.
# Make sure the image paths are correct and the image files exist in the specified folder (e.g., 'images/')
# ==============================================================================
# PLACE YOUR PROJECT DATA HERE
# ==============================================================================
projects = [
    {
        "title": "Indonesian Hoax News Detection",
        "demo_link": "https://detecthoax.streamlit.app/",
        "colab_link": "https://colab.research.google.com/drive/1jI7xyYF4qogBrShcqPD8hy4r6G1NFTtn?usp=sharing",
        "code_image_path": "images/project1_code.png",
        "result_image_path": "images/project1_result.png",
        "background": """
        **Problem:** The spread of hoax news on social media and online platforms is increasingly concerning
        and can negatively impact society. An automated way to identify potential hoaxes is needed
        to help users filter information.
        """,
        "description": """
        This project aims to build a text classification model using machine learning/deep learning
        to detect whether an Indonesian news article is a hoax or not.
        The model is trained on a labeled news dataset.
        """,
        "methodology_tools": """
        * **Methodology:** Ensemble Stacking (combining SVM, Random Forest, IndoBERT+MLP), Data Balancing Technique (SMOTE+Tomek), Cross-Validation (Stratified 5-Fold).
        * **Features:** TF-IDF, Fixed IndoBERT Embeddings, Linguistic/Stylistic Features (if applicable).
        * **Tools:** Python, Scikit-learn, Pandas, NumPy, NLTK/Sastrawi (for linguistic features if applicable), Transformers (Hugging Face), Streamlit (for demo).
        """,
        "tasks": """
        * Data Collection and Cleaning.
        * Feature Engineering (TF-IDF, Embedding Extraction, Linguistic Features).
        * Implementation and Training of Base Models (SVM, RF, MLP).
        * Implementation of Data Balancing Techniques.
        * Implementation of Stacking Ensemble with Meta-learner.
        * Model Evaluation using relevant metrics (Accuracy, Precision, Recall, F1-Score, AUC).
        * Development of a Simple Demo Application (Streamlit).
        """,
        "benefits": """
        * Provides a tool to assist in the automatic identification of potential hoaxes.
        * Enhances users' digital literacy by providing an indicator of news credibility.
        * Demonstrates the application of ensemble learning and NLP techniques for real-world text classification problems.
        """
    },
    {
        "title": "Sentiment Analysis on IMDB Movie Reviews",
        "demo_link": "https://anantiment.streamlit.app/",
        "colab_link": "https://colab.research.google.com/drive/1Ay7z0fmJIVPWQGXri4tE0HwZSYafsKgH?usp=sharing",
        "code_image_path": "images/project2_code.png",
        "result_image_path": "images/project2_result.png",
        "background": """
        **Problem:** E-commerce platforms have numerous product reviews from users. Manually analyzing the sentiment
        of these reviews is time-consuming. An automated method is needed to understand customer opinions
        (positive/negative/neutral) towards products based on their reviews.
        """,
        "description": """
        Building a classification model to analyze the sentiment of text from product reviews (originally Indonesian, adapted here for IMDB example).
        The model classifies reviews into positive, negative, or potentially neutral categories.
        *(Note: The title references IMDB, but the original description mentioned Indonesian e-commerce. Adjusted for clarity based on title)*
        """,
        "methodology_tools": """
        * **Methodology:** Fine-tuning a pre-trained language model (like BERT or IndoBERT if Indonesian) for text classification, TF-IDF + Logistic Regression (as a baseline).
        * **Tools:** Python, Pandas, Scikit-learn, Transformers (Hugging Face), Matplotlib/Seaborn (for visualization).
        """,
        "tasks": """
        * Data loading and preprocessing (cleaning, normalization).
        * Training and fine-tuning the language model.
        * Training the baseline model.
        * Result analysis and model comparison.
        * Visualization of sentiment distribution.
        """,
        "benefits": """
        * Provides quick insights to sellers/platforms regarding product reception.
        * Helps potential buyers understand the general opinion about a product.
        * Automates the process of customer feedback analysis.
        """
    },
    {
        "title": "Canadian Amazon Product Information Chatbot",
        "demo_link": None,
        "colab_link": "https://colab.research.google.com/drive/1S7j4htNk3lZ6MmYD74-P8ZeCZCLDs6R-?usp=sharing",
        "code_image_path": "images/project3_code.png",
        "result_image_path": "images/project3_result.png",
        # --- UPDATED SECTIONS START HERE ---
        "background": """
        **Problem:** Finding specific details about products on large e-commerce sites like Amazon.ca can involve navigating multiple pages or sifting through reviews.
        Users often need quick answers to specific questions about product features, specifications, or compatibility.
        An interactive chatbot could provide a more efficient way to access this information.
        """,
        "description": """
        This project involves developing a conversational AI (chatbot) capable of answering user questions about products available on Amazon.ca.
        It leverages a dataset of Canadian Amazon product information (potentially scraped or using an API if available) and utilizes Natural Language Processing (NLP) techniques,
        likely Retrieval-Augmented Generation (RAG), to understand questions and retrieve relevant information from the product data.
        """,
        "methodology_tools": """
        * **Methodology:** Retrieval-Augmented Generation (RAG), Vector Embeddings (e.g., Sentence-BERT, OpenAI Ada), Vector Database (e.g., FAISS, ChromaDB), Large Language Model (LLM) Integration (e.g., GPT API, open-source LLMs). Data scraping/collection might be involved.
        * **Tools:** Python, LangChain or LlamaIndex (for RAG framework), Pandas (for data handling), Vector Database library, LLM API library (e.g., OpenAI), potentially BeautifulSoup/Scrapy (for scraping).
        """,
        "tasks": """
        * Data Acquisition: Gathering product information from Amazon.ca (e.g., scraping product pages, descriptions, specifications).
        * Data Cleaning and Structuring: Preparing the text data for processing and embedding.
        * Embedding Generation: Converting product information into vector embeddings.
        * Vector Store Setup: Indexing embeddings in a vector database for efficient retrieval.
        * RAG Pipeline Implementation: Integrating the retriever (vector store) with an LLM to generate answers based on retrieved context.
        * Chat Interface Logic: Developing the conversational flow and handling user queries.
        * Testing and Evaluation: Assessing the chatbot's ability to understand questions and provide accurate, relevant answers.
        """,
        "benefits": """
        * Provides users with quick and direct answers to specific product questions on Amazon.ca.
        * Enhances the user experience by reducing the need for manual searching through product pages.
        * Demonstrates the practical application of RAG and LLMs in an e-commerce context.
        * Can potentially summarize key product features or compare aspects based on user queries.
        """
        # --- UPDATED SECTIONS END HERE ---
    },
    # --- ADD OTHER PROJECT DICTIONARIES HERE ---
]
# ==============================================================================

# --- Main Header ---
# Updated title and welcome message to English
st.title("Project Portfolio - Muhammad Ali Ashari") # <-- Ganti [Nama Anda]
st.write("""
Welcome to my portfolio! Below are some of the projects I've worked on
in the fields of Data Science, Machine Learning, and Data Analysis.
(Feel free to add a brief introduction about yourself here).
""")
st.divider()

# --- Display Each Project ---
for project in projects:
    st.header(project["title"])

    col1, col2 = st.columns(2) # Create two columns for links

    with col1:
        if project["demo_link"]:
            # Updated button text to English
            st.link_button("🔗 View Live Demo (Streamlit)", project["demo_link"])
    with col2:
        if project["colab_link"]:
            # Updated button text to English
            st.link_button("📓 View Code/Notebook (Google Colab)", project["colab_link"])

    st.markdown("---") # Separator line

    # Project Explanation in Expander
    # Updated expander label to English
    with st.expander("🔍 View Project Details"):
        # Updated labels to English
        st.markdown(f"**Project Background (Problem):** {project['background']}")
        st.markdown(f"**Project Description:** {project['description']}")
        st.markdown(f"**Methodology and Tools:** {project['methodology_tools']}")
        st.markdown(f"**My Tasks:** {project['tasks']}") # Changed label slightly for clarity
        st.markdown(f"**Expected Benefits:** {project['benefits']}")

    st.markdown("---") # Separator line

    # Display Images (Code and Results)
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        # Updated subheader to English
        st.subheader("Code Snippet")
        # Check if the image file exists before displaying
        if os.path.exists(project["code_image_path"]):
            try:
                code_image = Image.open(project["code_image_path"])
                # Updated caption to English
                st.image(code_image, caption=f"Code Screenshot for {project['title']}", use_column_width=True)
            except Exception as e:
                st.warning(f"Failed to load code image: {project['code_image_path']}. Error: {e}")
        else:
            st.warning(f"Image file not found: {project['code_image_path']}")

    with col_img2:
        # Updated subheader to English
        st.subheader("Result Example")
         # Check if the image file exists
        if os.path.exists(project["result_image_path"]):
            try:
                result_image = Image.open(project["result_image_path"])
                # Updated caption to English
                st.image(result_image, caption=f"Result Screenshot for {project['title']}", use_column_width=True)
            except Exception as e:
                 st.warning(f"Failed to load result image: {project['result_image_path']}. Error: {e}")
        else:
             st.warning(f"Image file not found: {project['result_image_path']}")

    st.divider() # Separator between projects

# --- Footer (Optional) ---
st.markdown("---")
# Updated footer text to English
st.write("Thank you for visiting my portfolio.")
# Add links to LinkedIn, GitHub, etc. if desired
# st.write("[Your LinkedIn](https://linkedin.com/in/...) | [Your GitHub](https://github.com/...)")
