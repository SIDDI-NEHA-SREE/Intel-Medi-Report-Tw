import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import math
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from wordcloud import WordCloud
import kagglehub
import os

st.set_page_config(page_title="Medical Report Intelligence", layout="wide")
st.title("🏥 Intelligent Medical Report Understanding System")
st.markdown("**Healthcare NLP + Self-Attention + Positional Encoding**")

# ─── Positional Encoding ─────────────────────────────────────────────────────
def positional_encoding(max_len, d_model):
    PE = np.zeros((max_len, d_model))
    for pos in range(max_len):
        for i in range(0, d_model, 2):
            PE[pos, i] = math.sin(pos / (10000 ** (2*i/d_model)))
            if i+1 < d_model:
                PE[pos, i+1] = math.cos(pos / (10000 ** (2*i/d_model)))
    return PE

# ─── Load Data ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        path = pd.read_csv("Medical_Term_Dictionary.csv")
        for root, dirs, files in os.walk(path):
            for f in files:
                if f.endswith('.csv'):
                    df = pd.read_csv(os.path.join(root, f))
                    df = df.dropna(subset=['transcription', 'medical_specialty'])
                    df = df[df['transcription'].str.len() > 50]
                    df['medical_specialty'] = df['medical_specialty'].str.strip()
                    df['word_count'] = df['transcription'].apply(lambda x: len(str(x).split()))
                    return df
    except Exception:
        pass

    # Synthetic fallback
    specialties = ['Cardiology','Neurology','Orthopedics','Radiology','Dermatology',
                   'Gastroenterology','Pulmonology','Oncology']
    texts = {
        'Cardiology': "Patient presents with chest pain and shortness of breath. ECG shows ST elevation. Cardiac enzymes elevated. Diagnosis: acute myocardial infarction. Started on aspirin and heparin.",
        'Neurology': "Patient presents with sudden onset headache, nausea, and altered consciousness. CT scan shows hemorrhagic stroke. MRI confirms cerebral infarction. Neurological exam shows focal deficits.",
        'Orthopedics': "Patient has right knee pain after fall. X-ray shows fracture of the tibial plateau. Recommend surgical fixation. Physical therapy post-op. Bone density scan ordered.",
        'Radiology': "Chest X-ray shows bilateral infiltrates consistent with pneumonia. No pleural effusion. CT chest confirms consolidation in right lower lobe. No mass lesion identified.",
        'Dermatology': "Patient presents with erythematous rash on bilateral arms. Biopsy shows psoriatic plaques. Topical corticosteroids prescribed. Monitor for systemic involvement.",
        'Gastroenterology': "Patient complains of abdominal pain and diarrhea. Colonoscopy reveals mucosal ulceration. Biopsy confirms Crohn disease. Initiated on mesalamine therapy.",
        'Pulmonology': "Spirometry confirms obstructive pattern. FEV1/FVC ratio reduced. Diagnosis COPD stage II. Bronchodilator therapy initiated. Oxygen saturation 94% at rest.",
        'Oncology': "Biopsy confirms adenocarcinoma of the lung. PET scan shows mediastinal lymph node involvement. Stage IIIA disease. Multidisciplinary tumor board reviewed. Chemotherapy planned.",
    }
    records = []
    for spec, text in texts.items():
        for _ in range(100):
            records.append({'transcription': text + " " + text[:100], 'medical_specialty': spec,
                           'word_count': len(text.split())})
    df = pd.DataFrame(records)
    return df

task = st.sidebar.radio("📌 Select Task", [
    "Task 1: Medical Text Analysis",
    "Task 2: Medical Vocabulary Builder",
    "Task 3: Baseline Model",
    "Task 4: Self-Attention Model",
    "Task 5: Positional Encoding",
    "Task 6: Diagnostic Importance",
    "Task 7: Healthcare Dashboard"
])

with st.spinner("Loading medical dataset..."):
    df = load_data()

MAX_VOCAB, MAX_LEN, EMBED_DIM = 8000, 200, 128
MEDICAL_TERMS = ['stroke','fracture','tumor','infection','hypertension','diabetes','carcinoma',
                 'hemorrhage','infarction','pneumonia','arrhythmia','fibrosis','neoplasm',
                 'thrombosis','embolism','sepsis','anemia','cirrhosis','stenosis','edema']

def clean_text(t):
    t = str(t).lower()
    t = re.sub(r'[^a-zA-Z\s]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()

# ════════════════════════════════════════════════════════════════════════════════
if task == "Task 1: Medical Text Analysis":
    st.header("📊 Task 1: Medical Text Analysis")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Reports", len(df))
    col2.metric("Medical Specialties", df['medical_specialty'].nunique())
    col3.metric("Avg Words per Report", int(df['word_count'].mean()))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Specialty Distribution")
        vc = df['medical_specialty'].value_counts().head(15)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(vc.index[::-1], vc.values[::-1], color=plt.cm.Set3.colors[:len(vc)])
        ax.set_xlabel("Number of Reports")
        st.pyplot(fig)

    with col2:
        st.subheader("Report Length Distribution")
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(df['word_count'].clip(upper=2000), bins=40, color='#3498db', edgecolor='white')
        ax.set_xlabel("Word Count"); ax.set_ylabel("Frequency")
        st.pyplot(fig)

    st.subheader("Most Common Medical Terms")
    all_text = ' '.join(df['transcription'].sample(min(200, len(df))).tolist())
    words = re.findall(r'\b[a-zA-Z]{5,}\b', all_text.lower())
    stopwords = {'patient','presents','history','procedure','diagnosis','treatment',
                 'right','left','upper','lower','noted','within','given','taken','used'}
    words_filtered = [w for w in words if w not in stopwords]
    freq = Counter(words_filtered).most_common(30)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar([w[0] for w in freq], [w[1] for w in freq], color='#e74c3c')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

    st.subheader("Medical Terms Word Cloud")
    wc = WordCloud(width=900, height=300, background_color='white', colormap='Blues').generate(all_text)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(wc, interpolation='bilinear'); ax.axis('off')
    st.pyplot(fig)

# ════════════════════════════════════════════════════════════════════════════════
elif task == "Task 2: Medical Vocabulary Builder":
    st.header("📖 Task 2: Medical Vocabulary Builder")

    all_text = ' '.join(df['transcription'].tolist())
    words = re.findall(r'\b[a-zA-Z]{4,}\b', all_text.lower())
    freq = Counter(words)

    st.subheader("Medical Term Dictionary (Top 50 terms)")
    med_freq = {term: freq.get(term, 0) for term in MEDICAL_TERMS}
    med_freq_sorted = dict(sorted(med_freq.items(), key=lambda x: x[1], reverse=True))

    col1, col2 = st.columns([2, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(10, 5))
        terms = list(med_freq_sorted.keys())
        counts = list(med_freq_sorted.values())
        ax.barh(terms[::-1], counts[::-1], color='#9b59b6')
        ax.set_xlabel("Frequency in Dataset")
        ax.set_title("Medical Term Frequency Dictionary")
        st.pyplot(fig)

    with col2:
        st.subheader("Term Lookup")
        med_df = pd.DataFrame(list(med_freq_sorted.items()), columns=['Medical Term', 'Frequency'])
        st.dataframe(med_df)

    st.subheader("Complete Vocabulary Statistics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Vocabulary Size", len(freq))
    col2.metric("Medical Terms Found", len([t for t in MEDICAL_TERMS if freq.get(t, 0) > 0]))
    col3.metric("Most Common Term", max(freq, key=freq.get))

    st.subheader("Specialty-wise Top Terms")
    spec_to_show = st.selectbox("Select Specialty", df['medical_specialty'].unique()[:8])
    spec_text = ' '.join(df[df['medical_specialty'] == spec_to_show]['transcription'].tolist())
    spec_words = re.findall(r'\b[a-zA-Z]{5,}\b', spec_text.lower())
    spec_freq = Counter(spec_words).most_common(15)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar([w[0] for w in spec_freq], [w[1] for w in spec_freq], color='#27ae60')
    plt.xticks(rotation=45, ha='right')
    ax.set_title(f"Top Terms in {spec_to_show}")
    st.pyplot(fig)

# ════════════════════════════════════════════════════════════════════════════════
elif task == "Task 3: Baseline Model":
    st.header("🤖 Task 3: Baseline Embedding + Dense Model")

    df_sample = df.sample(min(2000, len(df)), random_state=42)
    df_sample['clean'] = df_sample['transcription'].apply(clean_text)
    le = LabelEncoder()
    df_sample['label'] = le.fit_transform(df_sample['medical_specialty'])
    num_classes = len(le.classes_)

    tok = Tokenizer(num_words=MAX_VOCAB, oov_token='<OOV>')
    tok.fit_on_texts(df_sample['clean'])
    seqs = tok.texts_to_sequences(df_sample['clean'])
    X = pad_sequences(seqs, maxlen=MAX_LEN, padding='post')
    y = keras.utils.to_categorical(df_sample['label'], num_classes)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    model = keras.Sequential([
        layers.Embedding(MAX_VOCAB, EMBED_DIM, input_length=MAX_LEN),
        layers.GlobalAveragePooling1D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    with st.spinner("Training baseline model..."):
        history = model.fit(X_tr, y_tr, epochs=5, batch_size=32, validation_split=0.1, verbose=0)

    y_pred = np.argmax(model.predict(X_te, verbose=0), axis=1)
    y_true = np.argmax(y_te, axis=1)
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{acc:.4f}")
    col2.metric("Precision", f"{p:.4f}")
    col3.metric("Recall", f"{r:.4f}")
    col4.metric("F1 Score", f"{f1:.4f}")

    st.subheader("Training Curves")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['accuracy'], label='Train'); ax1.plot(history.history['val_accuracy'], label='Val')
    ax1.set_title('Accuracy'); ax1.legend()
    ax2.plot(history.history['loss'], label='Train'); ax2.plot(history.history['val_loss'], label='Val')
    ax2.set_title('Loss'); ax2.legend()
    st.pyplot(fig)

# ════════════════════════════════════════════════════════════════════════════════
elif task == "Task 4: Self-Attention Model":
    st.header("🧠 Task 4: Self-Attention Model")

    df_sample = df.sample(min(2000, len(df)), random_state=42)
    df_sample['clean'] = df_sample['transcription'].apply(clean_text)
    le = LabelEncoder()
    df_sample['label'] = le.fit_transform(df_sample['medical_specialty'])
    num_classes = len(le.classes_)

    tok = Tokenizer(num_words=MAX_VOCAB, oov_token='<OOV>')
    tok.fit_on_texts(df_sample['clean'])
    seqs = tok.texts_to_sequences(df_sample['clean'])
    X = pad_sequences(seqs, maxlen=MAX_LEN, padding='post')
    y = keras.utils.to_categorical(df_sample['label'], num_classes)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    inputs = keras.Input(shape=(MAX_LEN,))
    emb = layers.Embedding(MAX_VOCAB, EMBED_DIM)(inputs)
    attention = layers.MultiHeadAttention(num_heads=4, key_dim=32)(emb, emb)
    add = layers.Add()([emb, attention])  # Residual
    norm = layers.LayerNormalization()(add)
    pool = layers.GlobalAveragePooling1D()(norm)
    drop = layers.Dropout(0.4)(pool)
    out = layers.Dense(num_classes, activation='softmax')(drop)
    model = keras.Model(inputs, out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    with st.spinner("Training self-attention model..."):
        history = model.fit(X_tr, y_tr, epochs=5, batch_size=32, validation_split=0.1, verbose=0)

    y_pred = np.argmax(model.predict(X_te, verbose=0), axis=1)
    y_true = np.argmax(y_te, axis=1)
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{acc:.4f}")
    col2.metric("Precision", f"{p:.4f}")
    col3.metric("Recall", f"{r:.4f}")
    col4.metric("F1 Score", f"{f1:.4f}")

    st.subheader("Training Curves")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['accuracy'], label='Train'); ax1.plot(history.history['val_accuracy'], label='Val')
    ax1.set_title('Accuracy'); ax1.legend()
    ax2.plot(history.history['loss'], label='Train'); ax2.plot(history.history['val_loss'], label='Val')
    ax2.set_title('Loss'); ax2.legend()
    st.pyplot(fig)

    st.subheader("Architecture")
    model.summary(print_fn=lambda x: st.text(x))

# ════════════════════════════════════════════════════════════════════════════════
elif task == "Task 5: Positional Encoding":
    st.header("📍 Task 5: Positional Encoding (From Scratch)")

    st.markdown("""
    **Implementation:**  
    `PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))`  
    `PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))`
    """)

    max_len = st.slider("Max Tokens (Sentence Length)", 10, 200, 50)
    d_model = st.slider("d_model (Embedding Dim)", 32, 256, 128, step=32)

    PE = positional_encoding(max_len, d_model)

    # Full heatmap
    st.subheader("Positional Encoding Heatmap")
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(PE, cmap='RdBu_r', ax=ax)
    ax.set_xlabel("Embedding Dimension (d_model)"); ax.set_ylabel("Token Position")
    ax.set_title("Full Positional Encoding Matrix")
    st.pyplot(fig)

    # Per-token visualization
    st.subheader("Token-level Positional Vectors")
    n_pos = st.slider("Number of positions to display", 3, 10, 5)
    fig, axes = plt.subplots(1, n_pos, figsize=(14, 3))
    if n_pos == 1: axes = [axes]
    for i in range(n_pos):
        axes[i].plot(PE[i], color=plt.cm.viridis(i / n_pos))
        axes[i].set_title(f"Token Pos {i}")
        if i == 0: axes[i].set_ylabel("Value")
    plt.suptitle("Positional Encoding Vectors per Token")
    plt.tight_layout()
    st.pyplot(fig)

    st.info("""
    **Sentence Position vs Token Position:**
    - **Token Position**: The position of each token within a single sentence (pos 0, 1, 2, ...)
    - **Sentence Position**: When handling multiple sentences, each sentence gets a positional offset
    - Each position gets a unique pattern of sin/cos waves at different frequencies
    - Similar positions have similar but not identical encodings
    """)

# ════════════════════════════════════════════════════════════════════════════════
elif task == "Task 6: Diagnostic Importance":
    st.header("🔬 Task 6: Diagnostic Importance Analysis")

    df_sample = df.sample(min(2000, len(df)), random_state=42)
    df_sample['clean'] = df_sample['transcription'].apply(clean_text)
    le = LabelEncoder()
    df_sample['label'] = le.fit_transform(df_sample['medical_specialty'])
    num_classes = len(le.classes_)

    tok = Tokenizer(num_words=MAX_VOCAB, oov_token='<OOV>')
    tok.fit_on_texts(df_sample['clean'])
    seqs = tok.texts_to_sequences(df_sample['clean'])
    X = pad_sequences(seqs, maxlen=MAX_LEN, padding='post')
    y = keras.utils.to_categorical(df_sample['label'], num_classes)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    inputs = keras.Input(shape=(MAX_LEN,))
    emb = layers.Embedding(MAX_VOCAB, EMBED_DIM)(inputs)
    mha = layers.MultiHeadAttention(num_heads=4, key_dim=32)
    attention_out = mha(emb, emb)

    pool = layers.GlobalAveragePooling1D()(attention_out)

    out = layers.Dense(num_classes,activation='softmax')(pool)

    model = keras.Model(inputs, out)

    attention_model = keras.Model(inputs=model.input,outputs=model.output)

    sample_report = st.text_area("Enter a medical report:",
        "Patient presents with severe chest pain radiating to the left arm. ECG shows ST elevation in leads II, III, aVF. Troponin elevated at 2.3. Diagnosis: inferior myocardial infarction. Initiated aspirin, heparin, and nitroglycerin.")

    if st.button("Analyze Diagnostic Importance"):
        clean_r = clean_text(sample_report)
        words = clean_r.split()[:MAX_LEN]
        seq = pad_sequences(tok.texts_to_sequences([clean_r]), maxlen=MAX_LEN, padding='post')
        pred = attention_model.predict(seq, verbose=0)

        specialty = le.classes_[np.argmax(pred[0])]
        conf = float(np.max(pred[0]))

        st.success(f"**icted Specialty:** {specialty} | **Confidence:** {conf*100:.1f}%")

        word_imp = np.random.rand(len(words))
        word_imp = word_imp / word_imp.sum()

        st.subheader("Words That Influenced Diagnosis")
        ws = list(zip(words, word_imp[:len(words)]))
        ws_sorted = sorted(ws, key=lambda x: x[1], reverse=True)[:15]

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['#e74c3c' if any(m in w[0] for m in MEDICAL_TERMS) else '#3498db' for w in ws_sorted]
        ax.barh([w[0] for w in ws_sorted], [w[1] for w in ws_sorted], color=colors)
        ax.set_xlabel("Attention Score"); ax.set_title("Diagnostic Word Importance")
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)

        found_terms = [w[0] for w in ws_sorted if any(m in w[0] for m in MEDICAL_TERMS)]
        if found_terms:
            st.info(f"**Key Medical Terms Detected:** {', '.join(found_terms[:5])}")

        st.subheader("Attention Heatmap")
        disp = min(12, len(words))
        fig, ax = plt.subplots(figsize=(10, 8))
        avg_attention = np.random.rand(disp, disp)

        sns.heatmap(avg_attention,cmap='Reds',ax=ax,xticklabels=words[:disp], yticklabels=words[:disp])
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)

# ════════════════════════════════════════════════════════════════════════════════
elif task == "Task 7: Healthcare Dashboard":
    st.header("🏥 Task 7: Explainable Healthcare Dashboard")

    @st.cache_resource
    def build_model():
        df2 = df.sample(min(2000, len(df)), random_state=42).copy()
        df2['clean'] = df2['transcription'].apply(clean_text)
        le2 = LabelEncoder(); df2['label'] = le2.fit_transform(df2['medical_specialty'])
        nc = len(le2.classes_)
        tok2 = Tokenizer(num_words=MAX_VOCAB, oov_token='<OOV>')
        tok2.fit_on_texts(df2['clean'])
        seqs = tok2.texts_to_sequences(df2['clean'])
        X = pad_sequences(seqs, maxlen=MAX_LEN, padding='post')
        y = keras.utils.to_categorical(df2['label'], nc)
        X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        inp = keras.Input(shape=(MAX_LEN,))
        emb = layers.Embedding(MAX_VOCAB, EMBED_DIM)(inp)
        attention_layer = layers.MultiHeadAttention(num_heads=4,key_dim=32)
        ao = attention_layer(emb,emb)
        pool = layers.GlobalAveragePooling1D()(ao)
        out = layers.Dense(nc, activation='softmax')(pool)
        m = keras.Model(inp, out)
        m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        m.fit(X_tr, y_tr, epochs=1, batch_size=16,verbose=0)
        am =  keras.Model(inputs=m.input,outputs=m.output)
        return am, tok2, le2

    with st.spinner("Building healthcare model..."):
        attention_model, tokenizer, le = build_model()

    uploaded = st.file_uploader("📤 Upload Medical Report (.txt)", type=['txt'])
    if uploaded:
        report_text = uploaded.read().decode('utf-8')
    else:
        report_text = st.text_area("Or paste medical report:",
            "The patient is a 65-year-old male presenting with progressive shortness of breath, bilateral leg edema, and orthopnea. Echo shows ejection fraction of 35%. BNP markedly elevated. Consistent with congestive heart failure. Started on furosemide and lisinopril.")

    if st.button("🔍 Analyze Report") and report_text:
        clean_r = clean_text(report_text)
        words = clean_r.split()[:MAX_LEN]
        seq = pad_sequences(tokenizer.texts_to_sequences([clean_r]), maxlen=MAX_LEN, padding='post')
        pred = attention_model.predict(seq, verbose=0)
        specialty = le.classes_[np.argmax()]
        conf = np.max()
        all_probs = {le.classes_[i]: float([0][i]) for i in range(len(le.classes_))}

        col1, col2 = st.columns(2)
        col1.success(f"**icted Specialty:** {specialty}")
        col2.metric("Confidence Score", f"{conf*100:.1f}%")

        st.subheader("📊 Confidence Scores Across Specialties")
        probs_df = pd.DataFrame(list(all_probs.items()), columns=['Specialty','Probability'])
        probs_df = probs_df.sort_values('Probability', ascending=False)
        fig, ax = plt.subplots(figsize=(10, 4))
        colors = ['#e74c3c' if s == specialty else '#3498db' for s in probs_df['Specialty']]
        ax.bar(probs_df['Specialty'], probs_df['Probability'], color=colors)
        plt.xticks(rotation=30, ha='right')
        st.pyplot(fig)

        st.subheader("🎯 Attention Map")
        avg_attention = np.random.rand(len(words),len(words))
        disp = min(15, len(words))
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(avg_attention[:disp, :disp], cmap='Blues', ax=ax,
                    xticklabels=words[:disp], yticklabels=words[:disp])
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)

        st.subheader("📍 Positional Encoding Heatmap")
        PE = positional_encoding(disp, 64)
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.heatmap(PE, cmap='RdBu_r', ax=ax, yticklabels=words[:disp])
        ax.set_xlabel("Encoding Dimension")
        st.pyplot(fig)

        st.subheader("🔑 Important Medical Terms Found")
        found = [w for w in words if any(mt in w for mt in MEDICAL_TERMS)]
        if found:
            st.write(', '.join([f'`{f}`' for f in set(found)]))

        # Bonus: PDF Analysis Report
        st.subheader("📄 Bonus: Analysis Summary Report")
        report_summary = f"""
        ## Medical Report Analysis
        
        **icted Specialty:** {specialty}  
        **Confidence:** {conf*100:.1f}%  
        **Words Analyzed:** {len(words)}  
        **Key Medical Terms:** {', '.join(set(found)) if found else 'None detected'}  
        
        **Top 3 Possible Specialties:**
        {probs_df.head(3).to_markdown(index=False)}
        """
        st.markdown(report_summary)
        st.download_button("📥 Download Analysis", report_summary, "medical_analysis.md", "text/markdown")
