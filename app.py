import streamlit as st
from PIL import Image
import os # Untuk memeriksa keberadaan file gambar

# --- Konfigurasi Halaman (Opsional tapi bagus) ---
st.set_page_config(
    page_title="Portofolio Muhammad Ali Ashari",
    page_icon="📊", # Anda bisa gunakan emoji lain atau path ke file favicon
    layout="wide" # 'centered' atau 'wide'
)

# --- Data Proyek ---
# Ganti dengan detail proyek Anda yang sebenarnya.
# Buat dictionary untuk setiap proyek.
# Pastikan path gambar benar dan file gambar ada di folder yang sama atau subfolder (misal 'images/')
# ==============================================================================
# TEMPATKAN DATA PROYEK ANDA DI SINI
# ==============================================================================
projects = [
    {
        "title": "Indonesian Hoax News Detection",
        "demo_link": "https://detecthoax.streamlit.app/", # Ganti dengan link asli atau None
        "colab_link": "https://colab.research.google.com/drive/1jI7xyYF4qogBrShcqPD8hy4r6G1NFTtn?usp=sharing", # Ganti dengan link asli atau None
        "code_image_path": "images/project1_code.png", # Ganti dengan path ke screenshot kode Anda
        "result_image_path": "images/project1_result.png", # Ganti dengan path ke screenshot hasil Anda
        "background": """
        **Masalah:** Penyebaran berita bohong (hoaks) di media sosial dan platform berita online semakin meresahkan
        dan dapat menimbulkan dampak negatif pada masyarakat. Dibutuhkan cara untuk mengidentifikasi potensi hoaks
        secara otomatis untuk membantu pengguna memfilter informasi.
        """,
        "description": """
        Proyek ini bertujuan untuk membangun model klasifikasi teks menggunakan machine learning/deep learning
        untuk mendeteksi apakah suatu artikel berita berbahasa Indonesia termasuk hoaks atau bukan.
        Model dilatih menggunakan dataset berita yang telah diberi label.
        """,
        "methodology_tools": """
        * **Metodologi:** Ensemble Stacking (menggabungkan SVM, Random Forest, IndoBERT+MLP), Teknik Balancing Data (SMOTE+Tomek), Cross-Validation (Stratified 5-Fold).
        * **Fitur:** TF-IDF, Fixed IndoBERT Embeddings, Fitur Linguistik/Stilistik (jika ada).
        * **Tools:** Python, Scikit-learn, Pandas, NumPy, NLTK/Sastrawi (untuk fitur linguistik jika ada), Transformers (Hugging Face), Streamlit (untuk demo).
        """,
        "tasks": """
        * Pengumpulan dan Pembersihan Data.
        * Rekayasa Fitur (TF-IDF, Embedding Extraction, Linguistic Features).
        * Implementasi dan Pelatihan Model Dasar (SVM, RF, MLP).
        * Implementasi Teknik Balancing Data.
        * Implementasi Stacking Ensemble dengan Meta-learner.
        * Evaluasi Model menggunakan metrik yang relevan (Accuracy, Precision, Recall, F1-Score, AUC).
        * Pembuatan Aplikasi Demo Sederhana (Streamlit).
        """,
        "benefits": """
        * Menyediakan alat bantu untuk identifikasi potensi hoaks secara otomatis.
        * Meningkatkan literasi digital pengguna dengan memberikan indikator kredibilitas berita.
        * Demonstrasi penerapan teknik ensemble learning dan NLP untuk masalah klasifikasi teks nyata.
        """
    },
    {
        "title": "Sentiment Analysis on IMDB Movie Reviews",
        "demo_link": "https://anantiment.streamlit.app/", # Tidak ada demo streamlit
        "colab_link": "https://colab.research.google.com/drive/1Ay7z0fmJIVPWQGXri4tE0HwZSYafsKgH?usp=sharing", # Ganti dengan link asli atau None
        "code_image_path": "images/project2_code.png", # Ganti path
        "result_image_path": "images/project2_result.png", # Ganti path
        "background": """
        **Masalah:** E-commerce memiliki banyak ulasan produk dari pengguna. Menganalisis sentimen ulasan ini secara manual
        memakan waktu. Dibutuhkan cara otomatis untuk memahami opini pelanggan (positif/negatif/netral)
        terhadap produk berdasarkan ulasan mereka.
        """,
        "description": """
        Membangun model klasifikasi untuk menganalisis sentimen dari teks ulasan produk berbahasa Indonesia.
        Model akan mengklasifikasikan ulasan ke dalam kategori positif, negatif, atau netral.
        """,
        "methodology_tools": """
        * **Metodologi:** Fine-tuning model IndoBERT untuk klasifikasi teks, TF-IDF + Logistic Regression (sebagai baseline).
        * **Tools:** Python, Pandas, Scikit-learn, Transformers (Hugging Face), Matplotlib/Seaborn (untuk visualisasi).
        """,
        "tasks": """
        * Web scraping data ulasan (jika diperlukan dan diizinkan).
        * Preprocessing teks (cleaning, normalization).
        * Pelatihan dan fine-tuning model IndoBERT.
        * Pelatihan model baseline.
        * Analisis hasil dan perbandingan model.
        * Visualisasi distribusi sentimen.
        """,
        "benefits": """
        * Memberikan insight cepat kepada penjual/platform mengenai penerimaan produk.
        * Membantu calon pembeli memahami opini umum tentang suatu produk.
        * Otomatisasi proses analisis feedback pelanggan.
        """
    },
    {
        "title": "Canadian Amazon Product Information Chatbot",
        "demo_link": None, # Tidak ada demo streamlit
        "colab_link": "https://colab.research.google.com/drive/1S7j4htNk3lZ6MmYD74-P8ZeCZCLDs6R-?usp=sharing", # Ganti dengan link asli atau None
        "code_image_path": "images/project3_code.png", # Ganti path
        "result_image_path": "images/project3_result.png", # Ganti path
        "background": """
        **Masalah:** E-commerce memiliki banyak ulasan produk dari pengguna. Menganalisis sentimen ulasan ini secara manual
        memakan waktu. Dibutuhkan cara otomatis untuk memahami opini pelanggan (positif/negatif/netral)
        terhadap produk berdasarkan ulasan mereka.
        """,
        "description": """
        Membangun model klasifikasi untuk menganalisis sentimen dari teks ulasan produk berbahasa Indonesia.
        Model akan mengklasifikasikan ulasan ke dalam kategori positif, negatif, atau netral.
        """,
        "methodology_tools": """
        * **Metodologi:** Fine-tuning model IndoBERT untuk klasifikasi teks, TF-IDF + Logistic Regression (sebagai baseline).
        * **Tools:** Python, Pandas, Scikit-learn, Transformers (Hugging Face), Matplotlib/Seaborn (untuk visualisasi).
        """,
        "tasks": """
        * Web scraping data ulasan (jika diperlukan dan diizinkan).
        * Preprocessing teks (cleaning, normalization).
        * Pelatihan dan fine-tuning model IndoBERT.
        * Pelatihan model baseline.
        * Analisis hasil dan perbandingan model.
        * Visualisasi distribusi sentimen.
        """,
        "benefits": """
        * Memberikan insight cepat kepada penjual/platform mengenai penerimaan produk.
        * Membantu calon pembeli memahami opini umum tentang suatu produk.
        * Otomatisasi proses analisis feedback pelanggan.
        """
    },
    # --- TAMBAHKAN DICTIONARY PROYEK LAIN DI SINI ---
]
# ==============================================================================

# --- Header Utama ---
st.title("Portofolio Proyek [Nama Anda]")
st.write("""
Selamat datang di portofolio saya! Berikut adalah beberapa proyek yang telah saya kerjakan
di bidang Data Science, Machine Learning, dan Analisis Data.
(Tambahkan sedikit perkenalan tentang diri Anda di sini jika mau).
""")
st.divider()

# --- Tampilkan Setiap Proyek ---
for project in projects:
    st.header(project["title"])

    col1, col2 = st.columns(2) # Buat dua kolom untuk link

    with col1:
        if project["demo_link"]:
            st.link_button("🔗 Lihat Demo Aplikasi (Streamlit)", project["demo_link"])
    with col2:
         if project["colab_link"]:
            st.link_button("📓 Lihat Kode/Notebook (Google Colab)", project["colab_link"])

    st.markdown("---") # Garis pemisah

    # Penjelasan Proyek dalam Expander
    with st.expander("🔍 Lihat Penjelasan Detail Proyek"):
        st.markdown(f"**Latar Belakang Proyek (Masalah):** {project['background']}")
        st.markdown(f"**Deskripsi Proyek:** {project['description']}")
        st.markdown(f"**Metodologi dan Tools:** {project['methodology_tools']}")
        st.markdown(f"**Deskripsi Tugas Saya:** {project['tasks']}")
        st.markdown(f"**Manfaat yang Diharapkan:** {project['benefits']}")

    st.markdown("---") # Garis pemisah

    # Tampilkan Gambar (Kode dan Hasil)
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.subheader("Contoh Kode")
        # Cek apakah file gambar ada sebelum menampilkannya
        if os.path.exists(project["code_image_path"]):
            try:
                code_image = Image.open(project["code_image_path"])
                st.image(code_image, caption=f"Screenshot Kode {project['title']}", use_column_width=True)
            except Exception as e:
                st.warning(f"Gagal memuat gambar kode: {project['code_image_path']}. Error: {e}")
        else:
            st.warning(f"File gambar tidak ditemukan: {project['code_image_path']}")

    with col_img2:
        st.subheader("Contoh Hasil")
         # Cek apakah file gambar ada
        if os.path.exists(project["result_image_path"]):
            try:
                result_image = Image.open(project["result_image_path"])
                st.image(result_image, caption=f"Screenshot Hasil {project['title']}", use_column_width=True)
            except Exception as e:
                 st.warning(f"Gagal memuat gambar hasil: {project['result_image_path']}. Error: {e}")
        else:
             st.warning(f"File gambar tidak ditemukan: {project['result_image_path']}")

    st.divider() # Pemisah antar proyek

# --- Footer (Opsional) ---
st.markdown("---")
st.write("Terima kasih telah mengunjungi portofolio saya.")
# Tambahkan link ke LinkedIn, GitHub, dll. jika mau
# st.write("[LinkedIn Anda](https://linkedin.com/in/...) | [GitHub Anda](https://github.com/...)")
