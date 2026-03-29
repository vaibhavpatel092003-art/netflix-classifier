import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

# Always load CSV from same folder as app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "cleaned_netflix_final (3).csv")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
import xgboost as xgb

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Netflix Content Classifier",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Netflix Content Type Classifier")
st.markdown("Predict whether a Netflix title is a **Movie** or **TV Show** using Machine Learning.")
st.markdown("---")

# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH)
    # Handle missing values
    df['rating']       = df['rating'].fillna(df['rating'].mode()[0])
    df['duration']     = df['duration'].fillna(df['duration'].mode()[0])
    df['duration_int'] = df['duration_int'].fillna(df['duration_int'].median()).astype(int)
    return df

df = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Model Settings")
selected_model = st.sidebar.selectbox(
    "Choose a model",
    ["Logistic Regression", "Random Forest", "XGBoost"]
)
test_size = st.sidebar.slider("Test set size (%)", 10, 40, 20) / 100

st.sidebar.markdown("---")
st.sidebar.header("🔮 Try a Prediction")
input_year     = st.sidebar.number_input("Release Year", 1990, 2024, 2020)
input_duration = st.sidebar.number_input("Duration (min / seasons)", 1, 300, 90)
input_rating   = st.sidebar.selectbox("Rating", ["TV-MA", "TV-14", "TV-PG", "R", "PG-13", "PG", "TV-Y7", "TV-Y", "TV-G", "NR"])
input_country  = st.sidebar.text_input("Country", "United States")
input_genre    = st.sidebar.text_input("Genre", "Action & Adventure, Thrillers")

# ── Feature Engineering ───────────────────────────────────────────────────────
@st.cache_data
def engineer_features(df):
    df = df.copy()
    df['label']      = (df['type'] == 'Movie').astype(int)
    le_rating        = LabelEncoder()
    df['rating_enc'] = le_rating.fit_transform(df['rating'])
    top_countries    = df['country'].value_counts().nlargest(20).index
    for c in top_countries:
        df[f'country_{c}'] = (df['country'] == c).astype(int)
    tfidf        = TfidfVectorizer(max_features=30, token_pattern=r'[A-Za-z &]+')
    genre_matrix = tfidf.fit_transform(df['listed_in']).toarray()
    genre_cols   = [f'genre_{i}' for i in range(genre_matrix.shape[1])]
    genre_df     = pd.DataFrame(genre_matrix, columns=genre_cols, index=df.index)
    df           = pd.concat([df, genre_df], axis=1)
    feature_cols = (['release_year', 'duration_int', 'rating_enc']
                    + [f'country_{c}' for c in top_countries]
                    + genre_cols)
    return df, feature_cols, le_rating, tfidf, top_countries

df_feat, feature_cols, le_rating, tfidf, top_countries = engineer_features(df)

X = df_feat[feature_cols].values
y = df_feat['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y)

# ── Train Model ───────────────────────────────────────────────────────────────
@st.cache_data
def train_model(model_name, X_train, y_train, X_test, y_test, X, y):
    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    else:
        model = xgb.XGBClassifier(n_estimators=200, random_state=42,
                                  eval_metric='logloss', verbosity=0)
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    cv      = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    return model, y_pred, y_proba, cv

model, y_pred, y_proba, cv_scores = train_model(
    selected_model, X_train, y_train, X_test, y_test, X, y)

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
cm  = confusion_matrix(y_test, y_pred)

# ── Tab Layout ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 EDA", "📈 Model Results", "🔲 Confusion Matrix",
    "📉 ROC Curve", "🌟 Feature Importance"
])

# ─── TAB 1: EDA ───────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Titles", f"{len(df):,}")
    col2.metric("Movies",       f"{(df['type']=='Movie').sum():,}")
    col3.metric("TV Shows",     f"{(df['type']=='TV Show').sum():,}")
    col4.metric("Columns",      df.shape[1])

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Content Type Distribution**")
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        counts = df['type'].value_counts()
        ax1.pie(counts, labels=counts.index, autopct='%1.1f%%',
                colors=['#E50914', '#564d4d'], startangle=140)
        st.pyplot(fig1)
        plt.close()

    with col_b:
        st.markdown("**Top 10 Ratings**")
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        rating_counts = df['rating'].value_counts().head(10)
        ax2.barh(rating_counts.index, rating_counts.values, color='#E50914')
        ax2.invert_yaxis()
        ax2.set_xlabel("Count")
        st.pyplot(fig2)
        plt.close()

    st.markdown("**Release Year Distribution**")
    fig3, ax3 = plt.subplots(figsize=(10, 3))
    ax3.hist(df['release_year'], bins=30, color='#E50914', edgecolor='black', alpha=0.8)
    ax3.set_xlabel("Year")
    ax3.set_ylabel("Count")
    st.pyplot(fig3)
    plt.close()

    st.markdown("**Sample Data**")
    st.dataframe(df[['type','title','rating','release_year','duration','listed_in','country']].head(10))

# ─── TAB 2: Model Results ─────────────────────────────────────────────────────
with tab2:
    st.subheader(f"Results — {selected_model}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy",       f"{acc:.4f}")
    c2.metric("AUC-ROC",        f"{auc:.4f}")
    c3.metric("CV Accuracy",    f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    st.markdown("---")
    st.markdown("**Classification Report**")
    report = classification_report(y_test, y_pred,
                                   target_names=['TV Show','Movie'],
                                   output_dict=True)
    report_df = pd.DataFrame(report).transpose().round(4)
    st.dataframe(report_df)

    st.markdown("**5-Fold Cross Validation Scores**")
    fig_cv, ax_cv = plt.subplots(figsize=(8, 3))
    ax_cv.bar([f"Fold {i+1}" for i in range(5)], cv_scores, color='#E50914')
    ax_cv.set_ylim(0.95, 1.01)
    ax_cv.axhline(cv_scores.mean(), color='black', linestyle='--',
                  label=f"Mean = {cv_scores.mean():.4f}")
    ax_cv.legend()
    ax_cv.set_ylabel("Accuracy")
    st.pyplot(fig_cv)
    plt.close()

# ─── TAB 3: Confusion Matrix ──────────────────────────────────────────────────
with tab3:
    st.subheader("Confusion Matrix")
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax_cm,
                xticklabels=['TV Show','Movie'],
                yticklabels=['TV Show','Movie'],
                annot_kws={'fontsize': 16, 'fontweight': 'bold'})
    ax_cm.set_xlabel("Predicted Label")
    ax_cm.set_ylabel("Actual Label")
    ax_cm.set_title(selected_model)
    st.pyplot(fig_cm)
    plt.close()

    st.info("""
    **How to read:**
    - Top-left  → True Negatives  (TV Show correctly predicted)
    - Top-right → False Positives (TV Show incorrectly predicted as Movie)
    - Bottom-left → False Negatives (Movie incorrectly predicted as TV Show)
    - Bottom-right → True Positives (Movie correctly predicted)
    """)

# ─── TAB 4: ROC Curve ─────────────────────────────────────────────────────────
with tab4:
    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig_roc, ax_roc = plt.subplots(figsize=(7, 5))
    ax_roc.plot(fpr, tpr, color='#E50914', linewidth=2.5,
                label=f'{selected_model}  (AUC = {auc:.4f})')
    ax_roc.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Baseline')
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve")
    ax_roc.legend()
    ax_roc.grid(alpha=0.4)
    st.pyplot(fig_roc)
    plt.close()

# ─── TAB 5: Feature Importance ────────────────────────────────────────────────
with tab5:
    st.subheader("Feature Importance")
    if selected_model == "Logistic Regression":
        st.info("Logistic Regression doesn't have feature importances. Choose Random Forest or XGBoost.")
    else:
        imp      = model.feature_importances_
        top_idx  = np.argsort(imp)[-15:]
        top_vals = imp[top_idx]
        def clean(n):
            return (n.replace('country_','').replace('genre_','genre_')
                     .replace('rating_enc','Rating')
                     .replace('release_year','Release Year')
                     .replace('duration_int','Duration'))
        top_lbls = [clean(feature_cols[i]) for i in top_idx]
        fig_fi, ax_fi = plt.subplots(figsize=(8, 6))
        ax_fi.barh(top_lbls, top_vals, color='#E50914')
        ax_fi.set_xlabel("Importance Score")
        ax_fi.set_title(f"Top 15 Features — {selected_model}")
        ax_fi.grid(axis='x', alpha=0.4)
        st.pyplot(fig_fi)
        plt.close()

# ── Sidebar Prediction ────────────────────────────────────────────────────────
st.sidebar.markdown("---")
if st.sidebar.button("🔮 Predict!", use_container_width=True):
    custom = np.zeros(len(feature_cols))
    custom[0] = input_year
    custom[1] = input_duration
    custom[2] = (le_rating.transform([input_rating])[0]
                 if input_rating in le_rating.classes_ else 0)
    country_col = f'country_{input_country}'
    if country_col in feature_cols:
        custom[feature_cols.index(country_col)] = 1
    genre_vec   = tfidf.transform([input_genre]).toarray()[0]
    genre_start = feature_cols.index('genre_0')
    custom[genre_start:genre_start + len([f for f in feature_cols if f.startswith('genre_')])] = genre_vec

    pred  = model.predict([custom])[0]
    proba = model.predict_proba([custom])[0]
    label = "🎬 Movie" if pred == 1 else "📺 TV Show"

    st.sidebar.success(f"**Prediction: {label}**")
    st.sidebar.write(f"Movie confidence: **{proba[1]:.2%}**")
    st.sidebar.write(f"TV Show confidence: **{proba[0]:.2%}**")
