import streamlit as st
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
    {
        "title": "Indonesian News Claim Verification Agent",
      "demo_link": "http://34.142.222.255:8080", # Atau None jika tidak live, atau link video demo
      "github_link": "https://github.com/belacks/news-verif-agent", # Ganti dengan URL repo Anda yang benar
      "colab_link": None, # Atau link ke Colab spesifik jika ada bagian di sana
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
    I am a Data Science undergraduate student at Telkom University with exceptional professional skills in leadership, 
    project management, and strategic communication. I've successfully balanced rigorous academics with significant 
    organizational responsibilities, including leading the most significant e-sports and sports event in my department.

    My role in the Data Science Student Association has refined my abilities in coordinating cross-functional initiatives 
    and streamlining operations. These experiences have strengthened my expertise in event coordination and logistics management, 
    establishing me as a proactive problem solver who thrives in fast-paced, collaborative environments.
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
    """Display contact section"""
    
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
        name = st.text_input("Name")
        email = st.text_input("Email")
        message = st.text_area("Message")
        submit_button = st.button("Send Message", type="primary")
        
        if submit_button:
            st.success("Thank you for your message! I'll get back to you soon.")

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
