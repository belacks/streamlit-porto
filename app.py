import streamlit as st
from dotenv import load_dotenv
import smtplib
import ssl
from PIL import Image
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Muhammad Ali Ashari Portfolio",
    page_icon="📊",
    layout="wide"
)

# --- Custom CSS for better styling ---
def load_css():
    st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .project-card {
        background-color: #1f242e;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        border-left: 5px solid #4682B4;
    }
    .project-title {
        color: #010c1f;
        margin-bottom: 1rem;
    }
    .section-divider {
        margin: 2rem 0;
        border-top: 1px solid #e0e0e0;
    }
    .highlight-box {
        background-color: #f0f7ff;
        padding: 1.2rem;
        border-radius: 5px;
        margin-bottom: 1rem;
        border-left: 3px solid #4682B4;
    }
    .profile-header {
        display: flex;
        align-items: center;
        gap: 20px;
    }
    .profile-pic {
        background-color: #4682B4;
        height: 200px;
        width: 200px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 54px;
    }
    .social-links {
        display: flex;
        gap: 20px;
    }
    .social-link {
        background-color: #f0f0f0;
        padding: 8px 15px;
        border-radius: 20px;
        text-decoration: none;
        color: #333;
        font-weight: 500;
        transition: background-color 0.3s;
    }
    .social-link:hover {
        background-color: #4682B4;
        color: white;
    }
    .badge {
        display: inline-block;
        padding: 5px 10px;
        background-color: #e7f0ff;
        color: #4682B4;
        border-radius: 15px;
        margin-right: 8px;
        margin-bottom: 8px;
        font-size: 0.85rem;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Project Data with Added Dataset Sources ---
projects = [
        {
            "title": "Indonesian News Claim Verification Agent",
          "demo_link": None, # Atau None jika tidak live, atau link video demo
          "github_link": "https://github.com/belacks/news-verif-agent", # Ganti dengan URL repo Anda yang benar
          "colab_link": "https://github.com/belacks/news-verif-agent/blob/master/agent_app.py", # Atau link ke Colab spesifik jika ada bagian di sana
          "code_image_path": "images/project4_code.png", # Sesuaikan path gambar Anda
          "result_image_path": "images/project4_result.png", # Sesuaikan path gambar Anda
          "background": """
          **Problem:** The rapid spread of misinformation and hoaxes online, particularly within specific regions like Indonesia, presents a significant challenge.
          Manually verifying news claims by cross-referencing multiple trusted sources is time-consuming and difficult for the average user.
          An automated assistant could significantly speed up the process of checking claims against reliable news and fact-checking websites.
          """,
          "description": """
          This project features a conversational AI agent designed to assist users in verifying Indonesian news headlines or article URLs.
          The agent accepts user input, identifies the core claim, and utilizes LangChain agents and tools to perform targeted web searches against a predefined list of trusted Indonesian news portals and fact-checking sites.
          It leverages Google's Gemini Pro LLM to analyze the findings, synthesize information, and generate a structured verification report, including a classification (Hoax/Valid/Uncertain) and confidence score.
          """,
          "methodology_tools": """
          * **Methodology:** LLM-powered Agent (ReAct framework), Tool Usage (Web Search, URL Scraping), Natural Language Understanding, Information Synthesis.
          * **Tools:** Python, LangChain, Google Gemini Pro (via `langchain-google-genai` and Vertex AI API), Tavily Search API (for targeted web search), Requests & BeautifulSoup4 (for input URL scraping), Streamlit (for UI), Google Cloud Compute Engine (GCE for hosting), Python Virtual Environments (`venv`).
          """,
          "tasks": """
          * Environment Setup: Configuring a GCE VM, setting up Python virtual environment, installing dependencies.
          * GCP Authentication: Setting up Service Account permissions (IAM Roles, API Scopes) and authentication methods (ADC/Key File) for Vertex AI API access.
          * Tool Implementation: Developing LangChain tools for targeted web search (using Tavily wrapper) and initial URL content scraping.
          * Agent & Prompt Engineering: Designing and implementing a LangChain ReAct agent with detailed prompts to guide its reasoning, tool usage, and interaction with the LLM.
          * LLM Integration: Integrating the Google Gemini Pro model via the `langchain-google-genai` library.
          * Web Interface Development: Building an interactive user interface using Streamlit.
          * Testing & Iteration: Evaluating the agent's performance on various news claims (hoaxes and valid news, titles and URLs) and refining prompts and logic based on results. Troubleshooting authentication and deployment issues.
          """,
          "benefits": """
          * Provides users with a quick way to check Indonesian news claims against a curated list of trusted sources.
          * Assists in combating the spread of misinformation by leveraging reliable news portals and fact-checking sites.
          * Demonstrates a practical application of LLMs and LangChain agents for automated research and verification tasks.
          * Offers a structured summary and confidence assessment, helping users evaluate news credibility more effectively.
          """,
          "dataset_source": """
          **Information Sources:** Live Web Search Results & Scraped Input URL Content
        
          **Description:** Unlike projects relying on static datasets, this agent gathers information dynamically based on user input. It uses:
          1.  **Targeted Web Search:** Employs the Tavily Search API to query specific, predefined Indonesian domains known for credible news or fact-checking.
          2.  **Input URL Scraping:** Attempts to scrape the main text content directly from the news article URL provided by the user (if input is a URL).
        
          **Targeted Domains for Search:**
          * `kompas.com`
          * `tempo.co`
          * `cnnindonesia.com`
          * `turnbackhoax.id`
          * `cekfakta.com`
        
          **Note:** The agent processes information retrieved *live* from these sources via API results (snippets, links) and limited scraping, rather than operating on a pre-existing, fixed dataset.
          """
        },
        {
          "title": "Real-time Audio Noise Suppression with GRUUNet2 and Streamlit",
          "demo_link": "https://audiodenoise.streamlit.app",
          "github_link": "https://github.com/belacks/audio-denoising",
          "colab_link": "https://github.com/belacks/audio-denoising",
          "code_image_path": "images/project6_code.png",
          "result_image_path": "images/project6_result.png",
          "background": """
          **Problem:** Background noise in real-time audio communication (e.g., online meetings, voice chats, streaming) significantly degrades user experience and comprehension.
          Traditional noise suppression methods may not always be effective against diverse or non-stationary noises, or they might introduce undesirable audio artifacts.
          There's a need for intelligent, real-time noise reduction that can adapt to various noise types while preserving the quality of the primary audio (speech).
          """,
          "description": """
          This project implements a real-time audio noise suppression system using a deep learning model, GRUUNet2, integrated into an interactive web application built with Streamlit.
          The application captures audio input from the user's microphone via WebRTC, processes it frame by frame to remove background noise, and plays back the cleaned audio in real-time.
          The core denoising is performed by the GRUUNet2 model (a U-Net architecture with GRU cells) which operates on Mel Spectrogram representations of the audio.
          The system aims to provide a user-friendly tool for cleaner audio, similar to noise suppression features found in popular communication platforms.
          """,
          "methodology_tools": """
          * **Methodology:** Deep Learning-based Denoising, Supervised Learning (model predicts noise or clean signal components), Real-time Signal Processing (STFT, Mel Spectrogram, Inverse STFT/Griffin-Lim, Overlap-Add).
          * **Model:** GRUUNet2 (PyTorch implementation)[cite: 1, 2].
          * **Tools & Libraries:**
            * **Python:** Core programming language.
            * **PyTorch:** Deep learning framework for model building and inference[cite: 1, 2].
            * **Streamlit:** For creating the interactive web user interface.
            * **streamlit-webrtc:** For handling real-time audio streaming (microphone input, audio output) in the browser.
            * **Librosa & Torchaudio:** For audio processing tasks like loading, resampling, STFT, Mel Spectrogram computation, and inverse transforms[cite: 1, 2, 3].
            * **NumPy:** For numerical operations.
            * **AV (PyAV):** For audio frame manipulation within `streamlit-webrtc`.
          """,
          "tasks": """
          * **Model Implementation & Training:** Defining the GRUUNet2 architecture in PyTorch, preparing a dataset (not detailed here, but assumed to be noisy/clean audio pairs), and training the model to learn denoising patterns (details in `main.ipynb`)[cite: 1, 2].
          * **Real-time Audio Pipeline Development:**
            * Implementing STFT and Mel Spectrogram conversion for incoming audio frames.
            * Integrating the trained GRUUNet2 model for real-time inference on these spectrogram frames.
            * Maintaining the GRU hidden state (`hx`) across consecutive frames for temporal context.
            * Implementing inverse Mel-scale transformation and Griffin-Lim algorithm for reconstructing the time-domain audio signal from the processed spectrogram.
            * Implementing an Overlap-Add (OLA) mechanism for smooth audio output from framed processing.
          * **Streamlit Web Application Development:**
            * Building an interactive UI with start/stop controls for the denoising process.
            * Integrating `streamlit-webrtc` to capture microphone audio and play back processed audio.
            * Creating an `AudioProcessorBase` class to encapsulate the real-time audio processing logic.
            * Managing application state and user interaction.
          * **UI/UX Enhancements:** Styling the application with custom CSS for a more aesthetic and user-friendly experience.
          * **Configuration & Debugging:** Setting up model configurations, STFT parameters, and troubleshooting issues related to real-time processing, model loading, and UI behavior.
          """,
          "benefits": """
          * Provides users with an instant, real-time method to reduce background noise from their microphone input.
          * Improves audio clarity for online communication, streaming, or recording directly from the browser.
          * Demonstrates a practical application of a U-Net based model with recurrent (GRU) components for time-series signal processing.
          * Offers an interactive and user-friendly interface built with Streamlit, making advanced denoising accessible.
          * Showcases the integration of PyTorch models into a real-time web application using `streamlit-webrtc`.
          """,
          "dataset_source": """
          **Input Data:** Real-time audio stream from the user's microphone.
        
          **Training Data (Assumed based on `main.ipynb` and general practice):**
          The GRUUNet2 model was likely trained on a dataset consisting of pairs of noisy audio and corresponding clean (target) audio. The `main.ipynb` suggests processing of audio files, potentially mixing clean speech with various noise types to create these training pairs. The model learns to transform or separate the noise components from the speech components in the Mel Spectrogram domain[cite: 1]. The specific noise and clean speech datasets used for training are not detailed here but would have been part of the model development phase.
          """
    },
    {
        "title": "Jabodetabek Food Price Forecasting [Work In Progress - v0.1]",
        "demo_link": "https://prediksipangan.streamlit.app/", # 
        "colab_link": "https://colab.research.google.com/drive/1WHgKAEDsI8eeOmamdgDeNu0hD4LNKQtf?usp=sharing", # 
        "code_image_path": "images/project5_code.png", # 
        "result_image_path": "images/project5_result.png", # 
        "background": """
        **Problem:** Food price stability is crucial for economic well-being and inflation control, especially in major urban centers. The Jabodetabek metropolitan area (Jakarta, Bogor, Depok, Tangerang, Bekasi) is Indonesia's primary economic hub, and understanding its food price dynamics is vital for consumers, businesses, and policymakers. Volatility and unpredictable price changes require tools for better monitoring and anticipation.
    
        **NOTE:** This project is currently a work in progress (v0.1). The current version focuses on building and evaluating individual forecasting models. Deployment and potential enhancements (like multi-step forecasting) are planned future steps.
        """,
        "description": """
        This project focuses on developing a system to analyze and forecast daily prices of key food commodities across selected traditional markets in the Jabodetabek region.
        This initial version implements **individual Univariate LSTM (Long Short-Term Memory) deep learning models** for *each specific commodity-location pair*.
        The goal is to provide near-term price predictions based on historical trends derived from publicly available data.
        """,
        "methodology_tools": """
        *   **Methodology:** Univariate Time Series Forecasting, Deep Learning (LSTM), Extensive Data Cleaning & Preprocessing (Handling inconsistencies, missing values '-', comma separators), Data Reshaping (Wide Format Transformation), Data Imputation (Forward Fill & Backward Fill for missing dates/values), Feature Scaling (MinMaxScaler), Time Series Splitting (Train/Validation/Test - Chronological), Iterative Model Training & Evaluation (per time series).
        *   **Tools:** Python, Pandas, NumPy, Scikit-learn (MinMaxScaler, metrics), TensorFlow/Keras (LSTM model building, training, evaluation), Plotly (for visualization), Joblib (for saving/loading scalers), Gdown (for accessing models/scalers from Google Drive during deployment), Streamlit (Target platform for deployment).
        """,
        "tasks": """
        *   **Data Acquisition:** Collecting multi-year daily price data for multiple strategic food commodities across 5 key cities/regencies in Jabodetabek from PIHPS Nasional.
        *   **Data Cleaning & Preprocessing:** Handling diverse data formats, inconsistent headers, missing value markers ('-'), comma separators in numbers, ensuring daily frequency, and aligning time series across locations.
        *   **Data Imputation:** Implementing ffill and bfill strategies to handle missing data points after alignment.
        *   **Data Reshaping:** Transforming data from raw format to a 'wide' format suitable for individual time series processing.
        *   **Iterative Univariate Model Training:** Developing a pipeline to automatically train, evaluate, and save separate LSTM models and scalers for each individual commodity-location time series (~155 models trained in this version).
        *   **LSTM Implementation:** Building and compiling a standard LSTM architecture.
        *   **Model Evaluation:** Calculating MAE, RMSE, and MAPE metrics for each model on a held-out test set.
        *   **Model & Scaler Persistence:** Saving trained models (.keras) and scalers (.gz) for later use.
        *   **Streamlit App Development:** *[In Progress/Planned]* Designing and building the user interface and backend logic for deployment.
        """,
        "benefits": """
        *   Demonstrates the application of deep learning (LSTM) for complex, real-world time series forecasting problems at scale (handling hundreds of series).
        *   Highlights advanced data wrangling and preprocessing techniques necessary for messy, publicly sourced time series data.
        *   Provides a foundational system for monitoring price trends of essential food items in a major metropolitan area.
        *   Showcases an automated pipeline for training and evaluating multiple time series models.
        *   Establishes a base for future work on more advanced forecasting techniques (e.g., multivariate, multi-output models) or deployment features.
        """,
        "dataset_source": """
        **Dataset:** Pusat Informasi Harga Pangan Strategis (PIHPS) Nasional - Processed Data
    
        **Description:** Daily average prices for ~31 strategic food commodities collected from selected traditional markets in 5 key locations within the Jabodetabek region (Jakarta Pusat, Kota Bogor, Kota Depok, Kota Bekasi, Kota Tangerang). Raw data was downloaded in yearly chunks per location.
    
        **Time Range Used:** January 2018 - April 2025 (Note: Data for 2017 was found to be largely unavailable).
    
        **Size:** The final processed 'wide' dataset (`df_wide_imputed`) contains approximately 2300+ daily entries (rows) and ~155 columns (one for each unique Location-Commodity pair), resulting from cleaning and imputing the original source data.
    
        **Features (Processed Wide Format):** DatetimeIndex (Daily), Columns representing 'Location_CommodityName', Values are imputed daily prices in Indonesian Rupiah (Rp).
    
        **Source:** [PIHPS Nasional - Bank Indonesia](https://hargapangan.id/)
        """
    },
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
        """,
        "dataset_source": """
        **Dataset:** Indonesian Hoax News Dataset
        
        **Description:** A collection of labeled news articles from various Indonesian news sources and fact-checking platforms, containing both legitimate and hoax content.
        
        **Size:** Approximately 29,000 articles (Around 9500 hoax and 19500 non-hoax news)
        
        **Features:** title, Article text, publication date, source URL, category, and binary label (hoax/legitimate)
        
        **Source:** Kaggle - Indonesian Fact and Hoax Political News
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
        """,
        "dataset_source": """
        **Dataset:** IMDB Movie Reviews Dataset
        
        **Description:** A widely-used benchmark collection for sentiment analysis in natural language processing.
        
        **Size:** 50,000 movie reviews (25,000 positive and 25,000 negative)
        
        **Features:** Review text and sentiment label (positive/negative)
        
        **Source:** Kaggle - IMDB Dataset of 50K Movie Reviews
        """
    },
    {
        "title": "Canadian Amazon Product Information Chatbot",
        "demo_link": None,
        "colab_link": "https://colab.research.google.com/drive/1S7j4htNk3lZ6MmYD74-P8ZeCZCLDs6R-?usp=sharing",
        "code_image_path": "images/project3_code.png",
        "result_image_path": "images/project3_result.png",
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
        """,
        "dataset_source": """
        **Dataset:** Amazon.ca Product Information Dataset
        
        **Description:** With over 2.1 million unique products, this dataset offers a comprehensive view of the products available on Amazon.ca, one of Canada's leading online retailers. Collected through a web scraping process in 2023, this dataset provides valuable insights into product titles, pricing, ratings, and more.
        
        **Size:** Approximately 2.1 million products across multiple categories
        
        **Features:** Product ID, descriptions, imgUrl, productURL, reviews, pricing, rating, listPrice, categoryName, and isBestSeller
        
        **Source:** Kaggle - Amazon Canada Products 2023 (2.1M Products)
        """
    },
]

# --- Functions for Different Sections ---

def header_section():
    """Display the profile header section"""
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Profile picture placeholder with initials
        st.markdown('<div class="profile-pic">MA</div>', unsafe_allow_html=True)
    
    with col2:
        st.title("Muhammad Ali Ashari")
        st.subheader("Data Science & Machine Learning Portfolio")
        
        # Professional summary
        st.markdown("""
        Data Science undergraduate at Telkom University combining strong technical skills with exceptional leadership abilities. 
        Experienced in leading cross-functional teams, managing complex projects, and implementing data-driven solutions.
        """)
        
        # Skills badges
        st.markdown("<div style='margin-top: 15px;'>", unsafe_allow_html=True)
        for skill in ["Machine Learning", "NLP", "Data Analysis", "Python", "Leadership", "Project Management"]:
            st.markdown(f"<span class='badge'>{skill}</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Social links
        st.markdown("""
        <div class="social-links" style="margin-top: 15px;">
            <a href="https://www.linkedin.com/in/ali-ashari/" target="_blank" class="social-link">
                <i class="fab fa-linkedin"></i> LinkedIn
            </a>
            <a href="https://github.com/belacks" target="_blank" class="social-link">
                <i class="fab fa-github"></i> GitHub
            </a>
            <a href="mailto:aliashari0304@gmail.com" class="social-link">
                <i class="fas fa-envelope"></i> Email
            </a>
        </div>
        """, unsafe_allow_html=True)

def about_section():
    """Display the about me section"""
    
    st.header("About Me")
    
    st.markdown("""
        I'm a Data Science student passionate about leveraging data and AI to build impactful solutions. 
        I thrive on tackling complex challenges, from understanding human language with NLP to forecasting real-world trends with time series analysis.
        My experience extends beyond academics, having led significant student organization initiatives and managed event logistics, honing my leadership, project management, and collaborative skills. 
        I'm constantly exploring new techniques in ML, DL, and cloud technologies.
    """)

def skills_section():
    """Display the skills section"""
    
    st.header("Skills & Technologies")
    
    # Create three columns for different skill categories
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.subheader("Programming")
        st.markdown("""
        - Python (Advanced)
        - SQL (Intermediate)
        - R (Intermediate) 
        - HTML/CSS (Basic)
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.subheader("Data Science")
        st.markdown("""
        - Machine Learning
        - Natural Language Processing
        - Computer Vision
        - Statistical Analysis
        - Data Visualization
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col3:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.subheader("Tools & Frameworks")
        st.markdown("""
        - Scikit-learn, TensorFlow, PyTorch
        - Pandas, NumPy, Matplotlib
        - NLTK, Transformers (Hugging Face)
        - Git/GitHub
        - Streamlit, Flask
        """)
        st.markdown('</div>', unsafe_allow_html=True)

def project_section():
    """Display the projects section"""
    
    st.header("Projects")
    st.markdown("""
    Below are some of the key projects I've developed in the fields of Machine Learning, Natural Language Processing, and Data Science. 
    Each project demonstrates different technical skills and problem-solving approaches.
    """)
    
    # Display each project in a card-like format
    for i, project in enumerate(projects):
        # Project header with styling
        st.markdown(f"""
        <div class="project-card">
            <h3 class="project-title">{project['title']}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Project links
        col1, col2 = st.columns(2)
        with col1:
            if project["demo_link"]:
                st.link_button("🚀 View Live Demo", project["demo_link"], use_container_width=True)
            else:
                st.info("🔹 Demo not available yet")
        
        with col2:
            if project["colab_link"]:
                st.link_button("💻 View Code/Notebook", project["colab_link"], use_container_width=True)
        
        # Project details using tabs for better organization
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Overview", "🔬 Methodology", "💾 Dataset", "📊 Results"])
        
        with tab1:
            st.markdown("### 🎯 Project Background")
            st.markdown(project['background'])
            
            st.markdown("### 📝 Project Description")
            st.markdown(project['description'])
            
            st.markdown("### 💡 Benefits")
            st.markdown(project['benefits'])
            
        with tab2:
            st.markdown("### 🛠️ Methodology and Tools")
            st.markdown(project['methodology_tools'])
            
            st.markdown("### 📋 Tasks Performed")
            st.markdown(project['tasks'])
            
        with tab3:
            st.markdown("### 📊 Dataset Information")
            st.markdown(project['dataset_source'])
            
        with tab4:
            col_img1, col_img2 = st.columns(2)
            
            with col_img1:
                st.markdown("### 💻 Code Implementation")
                if os.path.exists(project["code_image_path"]):
                    try:
                        code_image = Image.open(project["code_image_path"])
                        st.image(code_image, caption=f"Code Sample - {project['title']}", use_container_width=True)
                    except Exception as e:
                        st.warning(f"Failed to load code image: {e}")
                else:
                    st.markdown("""
                    <div style="background-color:#f0f0f0; height:200px; display:flex; align-items:center; 
                    justify-content:center; border-radius:5px; border:1px dashed #ccc;">
                        <span style="color:#888;">Code Image Not Available</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col_img2:
                st.markdown("### 📈 Results Visualization")
                if os.path.exists(project["result_image_path"]):
                    try:
                        result_image = Image.open(project["result_image_path"])
                        st.image(result_image, caption=f"Results - {project['title']}", use_container_width=True)
                    except Exception as e:
                        st.warning(f"Failed to load result image: {e}")
                else:
                    st.markdown("""
                    <div style="background-color:#f0f0f0; height:200px; display:flex; align-items:center; 
                    justify-content:center; border-radius:5px; border:1px dashed #ccc;">
                        <span style="color:#888;">Result Image Not Available</span>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

def education_section():
    """Display education section"""
    
    st.header("Education")
    
    st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
    st.markdown("""
    ### Telkom University
    **B.Sc. Data Science** | 2022 - 2026 (Expected)
     GPA: 3.62/4.00
     
    - Relevant Coursework: Machine Learning, Statistical Learning, Big Data Analytics, Data Mining, Deep Learning, Artifical Intelligence
    - Academic Projects: Developed and optimized a hybrid neural network architecture using CNN, RNN, and MLP, achieving significant improvements in audio noise reduction and enhanced overall sound quality.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

def contact_section():
    """Menampilkan bagian kontak dan menangani pengiriman email."""
    
    st.header("Contact Me")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Contact Information")
        st.markdown("""
        - 📧 **Email:** aliashari0304@gmail.com 
        - 📱 **Phone:** +62 851-5755-8843
        - 📍 **Location:** Bandung, Indonesia
        
        Feel free to reach out for collaborations, job opportunities, or just to say hello!
        """)
    
    with col2:
        st.subheader("Send a Message")
        # Menggunakan st.form untuk mengumpulkan semua input sebelum submit
        with st.form(key="contact_form", clear_on_submit=True):
            user_name = st.text_input("Your Name", key="user_name")
            user_email = st.text_input("Your Email", key="user_email")
            message_subject_from_user = st.text_input("Subject", key="message_subject") # Opsional, bisa Anda tambahkan
            message_body = st.text_area("Message", key="message_body")
            
            submit_button = st.form_submit_button("Send Message", type="primary")

            if submit_button:
                if not user_name or not user_email or not message_body:
                    st.warning("⚠️ Please fill in all required fields (Name, Your Email, Message).")
                elif "@" not in user_email or "." not in user_email: # Validasi email sederhana
                    st.warning("⚠️ Please enter a valid email address.")
                else:
                    try:
                        # --- Mengambil Kredensial ---
                        # Coba ambil dari Streamlit Secrets (untuk deployment)
                        try:
                            SENDER_EMAIL = st.secrets["EMAIL_ADDRESS"]
                            SENDER_PASSWORD = st.secrets["EMAIL_PASSWORD"]
                            SMTP_SERVER_ADDRESS = st.secrets["SMTP_SERVER"]
                            SMTP_PORT_NUMBER = int(st.secrets["SMTP_PORT"]) # Pastikan port adalah integer
                            RECEIVER_EMAIL = st.secrets["RECEIVER_EMAIL_ADDRESS"]
                            using_secrets = True
                        except (FileNotFoundError, KeyError): 
                            # Jika gagal (misalnya saat pengembangan lokal tanpa secrets didefinisikan di cloud)
                            # coba ambil dari environment variables (file .env)
                            load_dotenv() # Memuat variabel dari .env
                            SENDER_EMAIL = os.getenv("EMAIL_ADDRESS")
                            SENDER_PASSWORD = os.getenv("EMAIL_PASSWORD")
                            SMTP_SERVER_ADDRESS = os.getenv("SMTP_SERVER")
                            SMTP_PORT_STR = os.getenv("SMTP_PORT")
                            RECEIVER_EMAIL = os.getenv("RECEIVER_EMAIL_ADDRESS")
                            using_secrets = False

                            if not all([SENDER_EMAIL, SENDER_PASSWORD, SMTP_SERVER_ADDRESS, SMTP_PORT_STR, RECEIVER_EMAIL]):
                                st.error("🚨 Konfigurasi email tidak lengkap. Pastikan file .env (lokal) atau Streamlit Secrets (deployment) sudah benar.")
                                return # Hentikan eksekusi jika konfigurasi tidak lengkap
                            SMTP_PORT_NUMBER = int(SMTP_PORT_STR)

                        # --- Membuat Konten Email ---
                        email_subject = f"New Portfolio Contact from: {user_name}"
                        if message_subject_from_user: # Jika pengguna mengisi subjek
                             email_subject += f" - Subject: {message_subject_from_user}"

                        email_body_content = f"""
                        You have received a new message from your Streamlit Portfolio contact form:

                        Name: {user_name}
                        Email: {user_email}
                        Subject: {message_subject_from_user if message_subject_from_user else "N/A"}

                        Message:
                        {message_body}
                        """
                        
                        # Menggunakan f-string untuk header email yang benar
                        full_email_message = f"From: {SENDER_EMAIL}\nTo: {RECEIVER_EMAIL}\nSubject: {email_subject}\n\n{email_body_content}"

                        # --- Mengirim Email ---
                        context = ssl.create_default_context() # Membuat konteks SSL default
                        
                        if SMTP_PORT_NUMBER == 465: # Umumnya SSL
                            with smtplib.SMTP_SSL(SMTP_SERVER_ADDRESS, SMTP_PORT_NUMBER, context=context) as server:
                                server.login(SENDER_EMAIL, SENDER_PASSWORD)
                                server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, full_email_message.encode('utf-8'))
                        elif SMTP_PORT_NUMBER == 587: # Umumnya TLS
                            with smtplib.SMTP(SMTP_SERVER_ADDRESS, SMTP_PORT_NUMBER) as server:
                                server.starttls(context=context) # Mengamankan koneksi dengan TLS
                                server.login(SENDER_EMAIL, SENDER_PASSWORD)
                                server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, full_email_message.encode('utf-8'))
                        else:
                            st.error(f"🚨 Port SMTP tidak didukung: {SMTP_PORT_NUMBER}. Gunakan 465 (SSL) atau 587 (TLS).")
                            return

                        st.success("✅ Thank you for your message! It has been sent successfully.")
                        if not using_secrets:
                            st.caption("_Email sent using local .env configuration._")

                    except smtplib.SMTPAuthenticationError:
                        st.error("🚨 SMTP Authentication Error: Username atau password email pengirim salah. Jika menggunakan Gmail, pastikan Anda menggunakan App Password yang benar.")
                    except smtplib.SMTPServerDisconnected:
                        st.error("🚨 SMTP Server Disconnected: Gagal terhubung ke server. Cek alamat server dan port.")
                    except smtplib.SMTPException as e_smtp:
                        st.error(f"🚨 SMTP Error: {e_smtp}")
                    except ConnectionRefusedError:
                        st.error("🚨 Connection Refused: Pastikan server SMTP dan port sudah benar dan tidak diblokir firewall.")
                    except ssl.SSLError as e_ssl:
                        st.error(f"🚨 SSL Error: {e_ssl}. Mungkin ada masalah dengan konfigurasi SSL/TLS server.")
                    except Exception as e:
                        st.error(f"🚨 An unexpected error occurred: {e}")

# --- Main Function ---
def main():
    # Apply custom CSS
    load_css()
    
    # Add a navigation menu in the sidebar
    st.sidebar.title("Navigation")
    options = ["Home", "About", "Skills", "Projects", "Education", "Contact"]
    selection = st.sidebar.radio("Go to", options)
    
    # Dark mode toggle (simplistic approach)
    st.sidebar.markdown("---")
    if st.sidebar.checkbox("Dark Mode (Experimental)"):
        st.markdown("""
        <style>
        .stApp {
            background-color: #121212;
            color: #f0f0f0;
        }
        .project-card {
            background-color: #1f242e;
            border-left: 5px solid #64b5f6;
        }
        .highlight-box {
            background-color: #1e1e1e;
            border-left: 3px solid #64b5f6;
        }
        .project-title {
            color: #010c1f;
        }
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            color: #f0f0f0;
        }
        </style>
        """, unsafe_allow_html=True)
    
    # Display the selected section
    if selection == "Home":
        header_section()
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        about_section()
    elif selection == "About":
        about_section()
    elif selection == "Skills":
        skills_section()
    elif selection == "Projects":
        project_section()
    elif selection == "Education":
        education_section()
    elif selection == "Contact":
        contact_section()
    
    # Footer
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center;">
        <p>© 2025 Muhammad Ali Ashari | Portfolio built with Streamlit</p>
        <p>
            <a href="https://www.linkedin.com/in/ali-ashari/" target="_blank">LinkedIn</a> •
            <a href="https://github.com/belacks" target="_blank">GitHub</a>
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
