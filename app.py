import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from gensim.models import Word2Vec
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import joblib
import os
import base64
from io import StringIO

# --- (1) Konfigurasi Halaman dan Utilities ---
st.set_page_config(layout="wide", page_title="Analisis Berita Fake vs True", initial_sidebar_state="expanded")

# --- Fungsi-fungsi yang dibutuhkan ---

# Fungsi untuk mengunduh resource NLTK dengan aman
def download_nltk_resources():
    resources = {
        "stopwords": "corpora/stopwords",
        "punkt": "tokenizers/punkt",
        "wordnet": "corpora/wordnet",
        "punkt_tab": "tokenizers/punkt_tab"
    }
    for resource_name, resource_path in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            st.info(f"Mengunduh resource NLTK: {resource_name}...")
            nltk.download(resource_name)
            st.success(f"Resource {resource_name} berhasil diunduh.")

download_nltk_resources() # Panggil fungsi di awal

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+','', text)
    text = re.sub(r'<.*?>','', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 2]
    return " ".join(words)

# Fungsi untuk Word2Vec, dibutuhkan di beberapa tempat
def get_sentence_vector(tokens, model, vector_size=100):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if not vectors:
        return np.zeros(vector_size)
    return np.mean(vectors, axis=0)

@st.cache_data
def load_data(csv_file_path='dataset_gabungan (1).csv'):
    try:
        df = pd.read_csv(csv_file_path)
        if 'subject' in df.columns and 'date' in df.columns: df = df.drop(columns=['subject', 'date'])
        if 'hoax' in df.columns: df = df.rename(columns={'hoax': 'hoax or not'})
        df = df.drop_duplicates().reset_index(drop=True)
        if 'clean_text' not in df.columns and 'text' in df.columns:
             with st.spinner("Melakukan pra-pemrosesan data teks... Ini mungkin butuh beberapa menit."):
                df['clean_text'] = df['text'].apply(preprocess_text)
             st.success("Pra-pemrosesan selesai.")
        if 'hoax or not' in df.columns:
            df['target'] = df['hoax or not'].map({'Fake': 0, 'True': 1})
        else:
            st.error("Kolom 'hoax or not' tidak ditemukan.")
            return pd.DataFrame()
        return df
    except FileNotFoundError:
        st.error(f"File dataset '{csv_file_path}' tidak ditemukan."); return pd.DataFrame()
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat data: {e}"); return pd.DataFrame()

df_cleaned = load_data()

# --- (3) Sidebar Navigasi ---
st.sidebar.title("Menu Navigasi")
page = st.sidebar.radio(
    "Pilih Halaman:",
    ("📊 Dashboard Analisis & EDA", "📈 Pelatihan & Performa Model", "🔍 Prediksi Berita")
)

# --- (4) Konten Halaman ---

if page == "📊 Dashboard Analisis & EDA":
    st.title("📊 Dashboard Analisis & Eksplorasi Data Awal (EDA)")
    tab_ringkasan, tab_eda, tab_fitur = st.tabs(["Ringkasan Proyek", "Eksplorasi Data Visual", "Insight Fitur"])
    # ... (Konten untuk halaman ini sama seperti sebelumnya, tidak perlu diubah) ...
    with tab_ringkasan:
        st.header("Ringkasan Proyek")
        st.markdown("""
        Selamat datang di dashboard analisis berita! Proyek ini bertujuan untuk membangun model *Natural Language Processing* (NLP)
        yang mampu mengklasifikasikan apakah sebuah berita termasuk kategori "Fake" atau "True".
        **Anggota Kelompok:** Syafrizal Rabbanie, Moh. Zahidi, Ernaya Fitri.
        """)
        if not df_cleaned.empty:
            st.subheader("Sekilas Dataset")
            st.text(f"Jumlah Entri Setelah Hapus Duplikat: {len(df_cleaned)}")
        else: st.warning("Data tidak berhasil dimuat.")
    with tab_eda:
        st.header("Eksplorasi Data Visual")
        if not df_cleaned.empty:
            st.subheader("Distribusi Kelas Berita")
            fig, ax = plt.subplots(); sns.countplot(data=df_cleaned, x='hoax or not', ax=ax, palette=['#FF6347','#4682B4']); st.pyplot(fig)
            st.subheader("☁️ Word Cloud")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Berita Hoax")
                text = " ".join(df_cleaned[df_cleaned['hoax or not'] == 'Fake']['clean_text'].astype(str))
                if text.strip(): wc = WordCloud(width=400, height=300, background_color='white').generate(text); fig, ax = plt.subplots(); ax.imshow(wc, interpolation='bilinear'); ax.axis("off"); st.pyplot(fig)
            with col2:
                st.markdown("#### Berita Benar")
                text = " ".join(df_cleaned[df_cleaned['hoax or not'] == 'True']['clean_text'].astype(str))
                if text.strip(): wc = WordCloud(width=400, height=300, background_color='white').generate(text); fig, ax = plt.subplots(); ax.imshow(wc, interpolation='bilinear'); ax.axis("off"); st.pyplot(fig)
        else: st.warning("Data tidak berhasil dimuat.")
    with tab_fitur:
        st.header("Insight dari Ekstraksi Fitur")
        if not df_cleaned.empty:
            st.subheader("Top 10 Kata (TF-IDF)"); tfidf_vectorizer = TfidfVectorizer(max_features=5000); X_tfidf = tfidf_vectorizer.fit_transform(df_cleaned['clean_text'].astype(str)); sum_words = X_tfidf.sum(axis=0); words_freq = [(word, sum_words[0, idx]) for word, idx in tfidf_vectorizer.vocabulary_.items()]; sorted_words = sorted(words_freq, key=lambda x: x[1], reverse=True); st.table(pd.DataFrame(sorted_words[:10], columns=['Kata', 'Skor TF-IDF']).style.format({'Skor TF-IDF': "{:.2f}"}))
            st.subheader("Top 10 Kata (Frekuensi)"); bow_vectorizer = CountVectorizer(max_features=5000); X_bow = bow_vectorizer.fit_transform(df_cleaned['clean_text'].astype(str)); sum_words_bow = X_bow.sum(axis=0); words_freq_bow = [(word, sum_words_bow[0, idx]) for word, idx in bow_vectorizer.vocabulary_.items()]; sorted_words_bow = sorted(words_freq_bow, key=lambda x: x[1], reverse=True); st.table(pd.DataFrame(sorted_words_bow[:10], columns=['Kata', 'Frekuensi']))
        else: st.warning("Data tidak berhasil dimuat.")


elif page == "📈 Pelatihan & Performa Model":
    st.title("📈 Pelatihan & Performa Model")
    st.info("Di halaman ini, Anda bisa melatih dan menyimpan berbagai model. Model yang disimpan akan tersedia untuk digunakan di halaman Prediksi.")

    if not df_cleaned.empty and 'clean_text' in df_cleaned.columns and 'target' in df_cleaned.columns:
        X_text = df_cleaned['clean_text'].astype(str)
        y = df_cleaned['target']

        if y.isnull().any() or y.nunique() <= 1:
            st.error("Kolom target 'y' bermasalah.")
        else:
            X_train_text, X_test_text, y_train, y_test = train_test_split(X_text, y, test_size=0.2, random_state=42, stratify=y)
            
            def display_model_results(model_name, y_test_data, y_pred_data):
                st.markdown(f"#### {model_name}")
                acc = accuracy_score(y_test_data, y_pred_data)
                st.metric(label="Akurasi", value=f"{acc:.2%}")
                report = classification_report(y_test_data, y_pred_data, output_dict=True, zero_division=0)
                st.text("Laporan Klasifikasi:"); st.dataframe(pd.DataFrame(report).transpose().style.format("{:.2f}"))
                cm = confusion_matrix(y_test_data, y_pred_data)
                fig_cm, ax_cm = plt.subplots(figsize=(5,4)); sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax_cm, xticklabels=['Fake', 'True'], yticklabels=['Fake', 'True'])
                ax_cm.set_xlabel('Prediksi'); ax_cm.set_ylabel('Aktual'); ax_cm.set_title(f'Confusion Matrix'); st.pyplot(fig_cm)

            # === MODIFIKASI: Menambahkan tab Word2Vec ===
            tab_tfidf, tab_bow, tab_word2vec = st.tabs(["TF-IDF", "Bag of Words", "Word Embedding (Word2Vec)"])

            with tab_tfidf:
                # ... (Sama seperti sebelumnya, tapi kita bisa sederhanakan sedikit) ...
                st.header("Model dengan TF-IDF Vectorizer")
                with st.spinner("Melatih model dengan TF-IDF..."):
                    tfidf_vectorizer = TfidfVectorizer(max_features=5000); X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text); X_test_tfidf = tfidf_vectorizer.transform(X_test_text)
                    # Naive Bayes
                    nb_model_tfidf = MultinomialNB(); nb_model_tfidf.fit(X_train_tfidf, y_train); nb_pred_tfidf = nb_model_tfidf.predict(X_test_tfidf)
                    with st.expander("Naive Bayes (TF-IDF) Results", expanded=True):
                        display_model_results("Naive Bayes (TF-IDF)", y_test, nb_pred_tfidf); joblib.dump(nb_model_tfidf, 'nb_model_tfidf.pkl'); joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl'); st.success("Model Naive Bayes (TF-IDF) dan vectorizer disimpan.")
                    # Decision Tree
                    dt_model_tfidf = DecisionTreeClassifier(random_state=42); dt_model_tfidf.fit(X_train_tfidf, y_train); dt_pred_tfidf = dt_model_tfidf.predict(X_test_tfidf)
                    with st.expander("Decision Tree (TF-IDF) Results", expanded=True):
                        display_model_results("Decision Tree (TF-IDF)", y_test, dt_pred_tfidf); joblib.dump(dt_model_tfidf, 'dt_model_tfidf.pkl'); st.success("Model Decision Tree (TF-IDF) disimpan.")

            with tab_bow:
                # ... (Sama seperti sebelumnya) ...
                st.header("Model dengan Bag of Words")
                with st.spinner("Melatih model dengan Bag of Words..."):
                    bow_vectorizer = CountVectorizer(max_features=5000); X_train_bow = bow_vectorizer.fit_transform(X_train_text); X_test_bow = bow_vectorizer.transform(X_test_text)
                    # Naive Bayes
                    nb_model_bow = MultinomialNB(); nb_model_bow.fit(X_train_bow, y_train); nb_pred_bow = nb_model_bow.predict(X_test_bow)
                    with st.expander("Naive Bayes (BoW) Results", expanded=True):
                        display_model_results("Naive Bayes (BoW)", y_test, nb_pred_bow); joblib.dump(nb_model_bow, 'nb_model_bow.pkl'); joblib.dump(bow_vectorizer, 'bow_vectorizer.pkl'); st.success("Model Naive Bayes (BoW) dan vectorizer disimpan.")
                    # Decision Tree
                    dt_model_bow = DecisionTreeClassifier(random_state=42); dt_model_bow.fit(X_train_bow, y_train); dt_pred_bow = dt_model_bow.predict(X_test_bow)
                    with st.expander("Decision Tree (BoW) Results", expanded=True):
                        display_model_results("Decision Tree (BoW)", y_test, dt_pred_bow); joblib.dump(dt_model_bow, 'dt_model_bow.pkl'); st.success("Model Decision Tree (BoW) disimpan.")

            # === TAB BARU UNTUK WORD2VEC ===
            with tab_word2vec:
                st.header("Model dengan Word Embedding (Word2Vec)")
                st.info("Proses ini melatih model Word2Vec dari awal dan mungkin memakan waktu lebih lama.")
                with st.spinner("Melatih model Word2Vec dan classifier..."):
                    # Latih model Word2Vec pada seluruh data bersih
                    sentences_w2v = [word_tokenize(text.lower()) for text in df_cleaned['clean_text'].astype(str)]
                    word2vec_model = Word2Vec(sentences=sentences_w2v, vector_size=100, window=5, min_count=1, workers=4, sg=0)
                    joblib.dump(word2vec_model, 'word2vec_model.pkl')
                    st.success("Model Word2Vec utama disimpan sebagai 'word2vec_model.pkl'.")
                    
                    # Buat fitur vektor untuk seluruh dataset
                    X_word2vec_all = np.array([get_sentence_vector(tokens, word2vec_model) for tokens in sentences_w2v])
                    # Lakukan train-test split pada data vektor
                    X_train_w2v, X_test_w2v, y_train_w2v, y_test_w2v = train_test_split(X_word2vec_all, y, test_size=0.2, random_state=42, stratify=y)
                    
                    # Gaussian Naive Bayes
                    nb_model_w2v = GaussianNB()
                    nb_model_w2v.fit(X_train_w2v, y_train_w2v)
                    nb_pred_w2v = nb_model_w2v.predict(X_test_w2v)
                    with st.expander("Gaussian Naive Bayes (Word2Vec) Results", expanded=True):
                        display_model_results("Gaussian Naive Bayes (Word2Vec)", y_test_w2v, nb_pred_w2v)
                        joblib.dump(nb_model_w2v, 'nb_model_word2vec.pkl')
                        st.success("Model Naive Bayes (Word2Vec) disimpan.")
                    
                    # Decision Tree
                    dt_model_w2v = DecisionTreeClassifier(random_state=42)
                    dt_model_w2v.fit(X_train_w2v, y_train_w2v)
                    dt_pred_w2v = dt_model_w2v.predict(X_test_w2v)
                    with st.expander("Decision Tree (Word2Vec) Results", expanded=True):
                        display_model_results("Decision Tree (Word2Vec)", y_test_w2v, dt_pred_w2v)
                        joblib.dump(dt_model_w2v, 'dt_model_word2vec.pkl')
                        st.success("Model Decision Tree (Word2Vec) disimpan.")


elif page == "🔍 Prediksi Berita":
    st.title("🔍 Prediksi Berita Fake vs True")
    st.markdown("Gunakan model yang sudah dilatih untuk memprediksi berita baru.")

    # === MODIFIKASI: Menambahkan Word2Vec ke Pilihan ===
    st.subheader("1. Pilih Model dan Transformasi")
    col1, col2 = st.columns(2)
    with col1:
        feature_method = st.selectbox(
            "Pilih Metode Ekstraksi Fitur:",
            ("TF-IDF", "Bag of Words (BoW)", "Word Embedding (Word2Vec)")
        )
    with col2:
        # Pilihan Naive Bayes disesuaikan namanya untuk Word2Vec
        model_options = ("Decision Tree", "Naive Bayes") if feature_method != "Word Embedding (Word2Vec)" else ("Decision Tree", "Gaussian Naive Bayes")
        model_type = st.selectbox("Pilih Model Klasifikasi:", model_options)

    # --- Logika Memuat Model Sesuai Pilihan ---
    model, vectorizer = None, None
    model_name_map = {"Decision Tree": "dt", "Naive Bayes": "nb", "Gaussian Naive Bayes": "nb"}
    feature_name_map = {"TF-IDF": "tfidf", "Bag of Words (BoW)": "bow", "Word Embedding (Word2Vec)": "word2vec"}
    model_abbr = model_name_map[model_type]
    feature_abbr = feature_name_map[feature_method]
    
    # Logika file berbeda untuk Word2Vec
    if feature_method == "Word Embedding (Word2Vec)":
        model_filename = f"{model_abbr}_model_word2vec.pkl"
        vectorizer_filename = "word2vec_model.pkl" # 'vectorizer' untuk W2V adalah model W2V itu sendiri
    else:
        model_filename = f"{model_abbr}_model_{feature_abbr}.pkl"
        vectorizer_filename = f"{feature_abbr}_vectorizer.pkl"

    if os.path.exists(model_filename) and os.path.exists(vectorizer_filename):
        try:
            model = joblib.load(model_filename)
            vectorizer = joblib.load(vectorizer_filename) # Untuk W2V, ini adalah model Word2Vec
            st.success(f"Model '{model_type} ({feature_method})' dan komponennya berhasil dimuat.")
        except Exception as e: st.error(f"Gagal memuat file: {e}")
    else:
        st.warning(f"File '{model_filename}' atau '{vectorizer_filename}' tidak ditemukan.")
        st.info(f"Harap latih model yang sesuai terlebih dahulu di halaman 'Pelatihan & Performa Model'.")

    st.markdown("---"); st.subheader("2. Lakukan Prediksi")
    tab_single, tab_batch = st.tabs(["Prediksi Teks Tunggal", "Prediksi File Batch (CSV/JSON)"])

    # --- Logika Prediksi dengan Kondisi untuk Word2Vec ---
    with tab_single:
        user_input = st.text_area("Ketik atau tempel berita di sini...", height=150, key="single_text_input")
        if st.button("🔎 Prediksi Teks Tunggal"):
            if model and vectorizer:
                with st.spinner("Memproses dan memprediksi..."):
                    processed_input = preprocess_text(user_input)
                    
                    # Alur prediksi berbeda untuk Word2Vec
                    if feature_method == "Word Embedding (Word2Vec)":
                        tokens = word_tokenize(processed_input.lower())
                        vector = get_sentence_vector(tokens, vectorizer) # 'vectorizer' di sini adalah model W2V
                        vector_reshaped = np.array(vector).reshape(1, -1)
                        prediction = model.predict(vector_reshaped)
                        prediction_proba = model.predict_proba(vector_reshaped)
                    else: # Alur untuk TF-IDF dan BoW
                        vectorized_input = vectorizer.transform([processed_input])
                        prediction = model.predict(vectorized_input)
                        prediction_proba = model.predict_proba(vectorized_input)

                    st.markdown("##### Hasil Prediksi:")
                    if prediction[0] == 0: st.error(f"🚨 Berita ini kemungkinan **FAKE** (Probabilitas: {prediction_proba[0][0]:.2%})")
                    else: st.success(f"✅ Berita ini kemungkinan **TRUE** (Probabilitas: {prediction_proba[0][1]:.2%})")
            else: st.error("Model tidak siap. Harap pilih dan latih model terlebih dahulu.")

    with tab_batch:
        uploaded_file = st.file_uploader("Pilih file CSV atau JSON", type=['csv', 'json'], key="batch_upload")
        if uploaded_file:
            df_upload = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_json(uploaded_file)
            st.dataframe(df_upload.head())
            text_column = st.selectbox("Pilih kolom teks:", df_upload.columns)
            if st.button("🚀 Prediksi File Batch"):
                if model and vectorizer:
                    with st.spinner(f"Memproses {len(df_upload)} baris data..."):
                        processed_texts = df_upload[text_column].astype(str).apply(preprocess_text)
                        
                        # Alur prediksi batch berbeda untuk Word2Vec
                        if feature_method == "Word Embedding (Word2Vec)":
                            tokens_list = [word_tokenize(text.lower()) for text in processed_texts]
                            vectorized_texts = np.array([get_sentence_vector(tokens, vectorizer) for tokens in tokens_list])
                        else: # Alur untuk TF-IDF dan BoW
                            vectorized_texts = vectorizer.transform(processed_texts)

                        df_upload['prediksi'] = model.predict(vectorized_texts)
                        probas = model.predict_proba(vectorized_texts)
                        df_upload['probabilitas_fake'] = probas[:, 0]; df_upload['probabilitas_true'] = probas[:, 1]
                        df_upload['label_prediksi'] = df_upload['prediksi'].map({0: 'FAKE', 1: 'TRUE'})
                        
                        st.success("Prediksi batch selesai!")
                        st.dataframe(df_upload)
                        csv_results = df_upload.to_csv(index=False).encode('utf-8')
                        st.download_button(label="📥 Unduh Hasil Prediksi (CSV)", data=csv_results, file_name=f"hasil_prediksi_{uploaded_file.name}", mime='text/csv')
                else: st.error("Model tidak siap. Harap pilih dan latih model terlebih dahulu.")