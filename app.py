import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import io
import re
import textwrap

st.set_page_config(
    page_title="AI Learning Assistant 2.0",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown("""

<style>

/* =========================================================
   AI LEARNING ASSISTANT 2.0
   FULL PURPLE / LAVENDER DASHBOARD THEME
   ========================================================= */


/* =========================================================
   MAIN PAGE BACKGROUND
   ========================================================= */

.stApp {
    background: #F3F0FF !important;
    color: #29243D !important;
}


/* Main content area */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* =========================================================
   SIDEBAR
   ========================================================= */

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #38206B 0%,
        #5B3FA3 50%,
        #7656C5 100%
    ) !important;
}


/* Sidebar general text */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: white !important;
}


/* Sidebar navigation text */
[data-testid="stSidebar"] [role="radiogroup"] label {
    color: white !important;
}


/* Radio text */
[data-testid="stSidebar"] [role="radiogroup"] label p {
    color: white !important;
    font-weight: 500;
}


/* Sidebar navigation hover */
[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: rgba(255,255,255,0.12);
    border-radius: 10px;
}


/* =========================================================
   SIDEBAR FILE UPLOADER
   ========================================================= */

[data-testid="stFileUploader"] {
    background: white !important;
    border: 2px dashed #BBA7E8 !important;
    border-radius: 15px !important;
    padding: 12px !important;
}


/* File uploader inner area */
[data-testid="stFileUploader"] section {
    background: white !important;
    border-radius: 12px !important;
}


/* File uploader text */
[data-testid="stFileUploader"] span {
    color: #4B2E83 !important;
}


/* File uploader small text */
[data-testid="stFileUploader"] small {
    color: #6B6280 !important;
}


/* Browse files button */
[data-testid="stFileUploader"] button {
    background: #6C4AB6 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}


/* Browse button hover */
[data-testid="stFileUploader"] button:hover {
    background: #4B2E83 !important;
    color: white !important;
}


/* =========================================================
   SIDEBAR BUTTONS
   ========================================================= */

[data-testid="stSidebar"] .stButton button {
    background: rgba(255,255,255,0.15) !important;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.30) !important;
    border-radius: 10px !important;
}


/* =========================================================
   HERO / WELCOME BANNER
   ========================================================= */

.hero {
    background: linear-gradient(
        110deg,
        #4B2E83 0%,
        #6C4AB6 50%,
        #9B72CF 100%
    ) !important;

    color: white !important;

    padding: 28px 30px;

    border-radius: 20px;

    margin-bottom: 22px;

    box-shadow:
        0 8px 25px rgba(75, 46, 131, 0.25);
}


/* Hero heading */
.hero h1 {
    margin: 0 !important;
    color: white !important;
    font-size: 30px !important;
    font-weight: 750 !important;
}


/* Hero paragraph */
.hero p {
    margin-top: 8px !important;
    color: #F7F3FF !important;
    font-size: 15px !important;
}


/* =========================================================
   METRIC CARDS
   ========================================================= */

.metric-card {
    background: white !important;

    border: 1px solid #DDD4F2 !important;

    padding: 20px;

    border-radius: 16px;

    box-shadow:
        0 5px 18px rgba(75, 46, 131, 0.10);

    transition: 0.2s;
}


/* Metric card hover */
.metric-card:hover {
    transform: translateY(-2px);

    box-shadow:
        0 8px 22px rgba(75, 46, 131, 0.16);
}


/* Metric label */
.metric-label {
    color: #756A8A !important;
    font-size: 13px;
}


/* Metric value */
.metric-value {
    color: #4B2E83 !important;
    font-size: 27px;
    font-weight: 750;
}


/* =========================================================
   STREAMLIT DEFAULT METRIC
   ========================================================= */

div[data-testid="stMetric"] {
    background: white !important;

    border: 1px solid #DDD4F2 !important;

    border-radius: 16px !important;

    padding: 15px 18px !important;

    box-shadow:
        0 5px 18px rgba(75, 46, 131, 0.10);
}


/* Metric label */
div[data-testid="stMetric"] label {
    color: #756A8A !important;
}


/* Metric number */
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #4B2E83 !important;
}


/* Metric delta */
div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    color: #6C4AB6 !important;
}


/* =========================================================
   SECTION TITLES
   ========================================================= */

.section-title {
    font-size: 20px;

    font-weight: 750;

    color: #4B2E83 !important;

    margin-bottom: 10px;
}


/* =========================================================
   NORMAL TEXT
   ========================================================= */

p {
    color: #403A50;
}


.small-muted {
    color: #81778F !important;

    font-size: 13px;
}


/* =========================================================
   TABS
   ========================================================= */

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;

    background: #E8E0FA !important;

    padding: 6px;

    border-radius: 12px;
}


/* Individual tab */
.stTabs [data-baseweb="tab"] {
    background: white !important;

    color: #4B2E83 !important;

    border-radius: 9px;

    padding: 8px 18px;

    font-weight: 600;
}


/* Active tab */
.stTabs [aria-selected="true"] {
    background: #6C4AB6 !important;

    color: white !important;
}


/* =========================================================
   BUTTONS
   ========================================================= */

.stButton > button {
    background: linear-gradient(
        90deg,
        #5B3FA3,
        #8B63C7
    ) !important;

    color: white !important;

    border: none !important;

    border-radius
    </style>
""", unsafe_allow_html=True)

# ---------- Sample dataset (used when no CSV is uploaded) ----------
@st.cache_data
def make_demo_data(n=800):
    rng = np.random.default_rng(42)
    study_hours = np.clip(rng.normal(5.5, 2.0, n), 0.2, 12)
    completion = np.clip(rng.normal(0.76, 0.17, n), 0.05, 1.0)
    attempts = rng.integers(1, 8, n)
    interactions = rng.integers(2, 35, n)
    help_count = rng.integers(0, 10, n)
    attendance = np.clip(rng.normal(0.82, 0.13, n), 0.2, 1.0)
    # Score is derived from learning activity plus noise. Previous activity can predict it.
    score = np.clip(
        0.24 + 0.035*study_hours + 0.28*completion + 0.13*attendance
        + 0.006*interactions - 0.012*help_count + rng.normal(0, 0.10, n),
        0.15, 1.0
    )
    df = pd.DataFrame({
        "student_id": [f"STU-{i+1:04d}" for i in range(n)],
        "study_hours": study_hours.round(1),
        "completion_rate": completion.round(2),
        "quiz_attempts": attempts,
        "interactions": interactions,
        "help_count": help_count,
        "attendance_rate": attendance.round(2),
        "average_score": score.round(2),
    })
    # Target labels are based on score for demonstration, but score itself is excluded from predictors.
    df["performance_level"] = pd.cut(
        df["average_score"], bins=[0, .60, .78, 1.01],
        labels=["Needs Support", "On Track", "High Performance"], include_lowest=True
    ).astype(str)
    df["subject"] = rng.choice(["Python", "Data Structures", "Mathematics", "Machine Learning", "Cyber Security"], n)
    return df

FEATURES = ["study_hours", "completion_rate", "quiz_attempts", "interactions", "help_count", "attendance_rate"]

def train_models(df):
    work = df.copy()
    # For user CSVs, create a performance label if absent.
    if "performance_level" not in work.columns:
        score_col = "average_score" if "average_score" in work.columns else None
        if score_col:
            vals = work[score_col].astype(float)
            # Supports 0-1 or 0-100 scores.
            if vals.max() > 1.5:
                vals = vals / 100
            work["performance_level"] = pd.cut(vals, [0, .60, .78, 1.01],
                labels=["Needs Support", "On Track", "High Performance"], include_lowest=True).astype(str)
        else:
            work["performance_level"] = "On Track"
    available = [c for c in FEATURES if c in work.columns]
    for c in FEATURES:
        if c not in work.columns:
            work[c] = float(df[c].median()) if c in df.columns else 0
    X = work[FEATURES].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median()).fillna(0)
    y = work["performance_level"].astype(str).fillna("On Track")
    if len(work) < 10 or y.nunique() < 2:
        model = RandomForestClassifier(n_estimators=120, random_state=42, class_weight="balanced")
        model.fit(X, y)
        return model, X, y, None
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify
    )
    model = RandomForestClassifier(n_estimators=160, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, preds),
        "Precision (weighted)": precision_score(y_test, preds, average="weighted", zero_division=0),
        "Recall (weighted)": recall_score(y_test, preds, average="weighted", zero_division=0),
        "F1-score (weighted)": f1_score(y_test, preds, average="weighted", zero_division=0),
        "Test rows": len(y_test)
    }
    return model, X, y, metrics

def score_fraction(value):
    try:
        v = float(value)
        return v / 100 if v > 1.5 else v
    except Exception:
        return 0.0

def make_recommendations(row, predicted):
    recs = []
    if float(row["completion_rate"]) < .65:
        recs.append(("Finish one short lesson", "Your completion rate is low. Complete one small lesson before starting another.", "20 min"))
    if float(row["attendance_rate"]) < .75:
        recs.append(("Review missed concepts", "Review the topics covered in classes you missed.", "25 min"))
    if float(row["help_count"]) >= 5:
        recs.append(("Use guided practice", "Read an example, then solve a similar question without looking.", "20 min"))
    if float(row["study_hours"]) < 4:
        recs.append(("Try a focused study session", "Use two 25-minute sessions with a short break.", "50 min"))
    if predicted == "Needs Support":
        recs.append(("Take a foundation quiz", "Start with easier questions and review each explanation.", "15 min"))
    elif predicted == "On Track":
        recs.append(("Practice a mixed quiz", "Combine familiar topics with one new concept.", "20 min"))
    else:
        recs.append(("Try an advanced challenge", "Apply the topic to a small real-world problem.", "30 min"))
    recs.append(("Review yesterday's notes", "Use active recall: close your notes and explain the main idea.", "10 min"))
    return recs[:4]

# ---------- Sidebar ----------
st.sidebar.markdown("## 🎓 AI Learning Assistant 2.0")
st.sidebar.caption("Smarter learning. Brighter future.")
page = st.sidebar.radio(
    "NAVIGATION",
    ["Dashboard", "Performance Prediction", "Course Recommendations",
     "Adaptive Learning Path", "AI Tutor (RAG)", "Knowledge Map", "Quizzes & Practice", "Study Plan"],
    index=0
)
st.sidebar.markdown("---")
st.sidebar.markdown("### 📁 Student data")
uploaded_csv = st.sidebar.file_uploader("Upload student CSV (optional)", type=["csv"])
st.sidebar.caption("No file? The app runs with a built-in demo dataset.")
if uploaded_csv:
    try:
        data = pd.read_csv(uploaded_csv)
        st.sidebar.success(f"Loaded {len(data)} rows")
    except Exception as e:
        st.sidebar.error(f"Could not read CSV: {e}")
        data = make_demo_data()
else:
    data = make_demo_data()

# Ensure columns are present and numeric
for col in FEATURES:
    if col not in data.columns:
        data[col] = 0
for col in FEATURES:
    data[col] = pd.to_numeric(data[col], errors="coerce")
data[FEATURES] = data[FEATURES].fillna(data[FEATURES].median()).fillna(0)
if "student_id" not in data.columns:
    data["student_id"] = [f"STU-{i+1:04d}" for i in range(len(data))]
if "subject" not in data.columns:
    data["subject"] = "General"
if "average_score" not in data.columns:
    data["average_score"] = np.clip(
        .25 + .035*data["study_hours"] + .28*data["completion_rate"] +
        .13*data["attendance_rate"] + .006*data["interactions"] -
        .012*data["help_count"], 0, 1
    )
if data["average_score"].max() > 1.5:
    data["average_score"] = data["average_score"] / 100
data["average_score"] = data["average_score"].clip(0, 1)
if "performance_level" not in data.columns:
    data["performance_level"] = pd.cut(
        data["average_score"], [0, .60, .78, 1.01],
        labels=["Needs Support", "On Track", "High Performance"], include_lowest=True
    ).astype(str)

model, X_all, y_all, model_metrics = train_models(data)
avg_score = float(data["average_score"].mean())
avg_completion = float(data["completion_rate"].mean())
avg_attendance = float(data["attendance_rate"].mean())
selected_id = st.sidebar.selectbox("Select demo student", data["student_id"].astype(str).tolist(), index=0)
student_row = data.loc[data["student_id"].astype(str) == selected_id].iloc[0]
student_x = pd.DataFrame([{c: float(student_row[c]) for c in FEATURES}])
predicted = str(model.predict(student_x)[0])
try:
    probs = model.predict_proba(student_x)[0]
    prob = float(max(probs))
except Exception:
    prob = 0.75

# ---------- Header ----------
st.markdown(f"""
<div class="hero">
  <h1>Welcome back, {selected_id} 👋</h1>
  <p>Your personal learning dashboard — progress, predictions, and next steps in one place.</p>
</div>
""", unsafe_allow_html=True)

def metric(label, value, help_text=None):
    st.metric(label, value, help=help_text)

# ---------- Pages ----------
if page == "Dashboard":
    c1,c2,c3,c4 = st.columns(4)
    with c1: metric("Students in dataset", f"{len(data):,}")
    with c2: metric("Activities completed", f"{int(data['interactions'].sum()):,}")
    with c3: metric("Average score", f"{avg_score:.1%}")
    with c4: metric("Average completion", f"{avg_completion:.1%}")

    left, right = st.columns([1.15, 1])
    with left:
        st.markdown('<div class="section-title">📈 Learning progress by subject</div>', unsafe_allow_html=True)
        subject_df = data.groupby("subject", as_index=False).agg(
            Average_Score=("average_score","mean"),
            Completion=("completion_rate","mean")
        )
        subject_df["Average Score (%)"] = subject_df["Average_Score"] * 100
        fig = px.bar(subject_df, x="subject", y="Average Score (%)", color="subject",
                     text=subject_df["Average Score (%)"].map(lambda x: f"{x:.0f}%"))
        fig.update_layout(showlegend=False, yaxis_range=[0,100], height=330,
                          margin=dict(l=10,r=10,t=20,b=10), plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown('<div class="section-title">🧠 Your performance prediction</div>', unsafe_allow_html=True)
        st.subheader(predicted)
        st.progress(min(max(prob, 0), 1))
        st.caption(f"Model confidence: {prob:.0%} (not a guarantee)")
        st.info("Prediction uses study hours, completion, attempts, interactions, help requests, and attendance.")
        if predicted == "Needs Support":
            st.warning("Suggested focus: revise foundations and complete short practice tasks.")
        elif predicted == "On Track":
            st.success("You're progressing. Keep practicing and revisit weak topics.")
        else:
            st.success("You're doing well. Try an advanced challenge to deepen understanding.")

    a,b = st.columns([1,1])
    with a:
        st.markdown('<div class="section-title">⚠️ Students who may need support</div>', unsafe_allow_html=True)
        risk = data.copy()
        risk["Risk Score"] = (
            (1-risk["completion_rate"])*0.35 + (1-risk["attendance_rate"])*0.25
            + (risk["help_count"]/max(float(risk["help_count"].max()),1))*0.15
            + (1-risk["average_score"])*0.25
        )
        st.dataframe(risk.sort_values("Risk Score", ascending=False)[
            ["student_id","performance_level","average_score","completion_rate","attendance_rate","Risk Score"]
        ].head(6).style.format({"average_score":"{:.0%}","completion_rate":"{:.0%}","attendance_rate":"{:.0%}","Risk Score":"{:.2f}"}),
        use_container_width=True, hide_index=True)
    with b:
        st.markdown('<div class="section-title">✨ Recommended next steps</div>', unsafe_allow_html=True)
        for title, detail, duration in make_recommendations(student_row, predicted):
            with st.container(border=True):
                st.markdown(f"**{title}**  ·  `{duration}`")
                st.caption(detail)

elif page == "Performance Prediction":
    st.markdown('<div class="section-title">🔮 Predict performance from learning activity</div>', unsafe_allow_html=True)
    st.write("Change the values to see how a student's activity profile affects the model prediction.")
    c1,c2,c3 = st.columns(3)
    with c1:
        study_hours = st.slider("Study hours per week", 0.0, 20.0, float(student_row["study_hours"]), 0.5)
        completion = st.slider("Completion rate", 0, 100, int(float(student_row["completion_rate"])*100)) / 100
    with c2:
        attempts = st.slider("Quiz attempts", 1, 15, int(np.clip(student_row["quiz_attempts"],1,15)))
        interactions = st.slider("Learning interactions", 0, 50, int(np.clip(student_row["interactions"],0,50)))
    with c3:
        help_count = st.slider("Help requests", 0, 20, int(np.clip(student_row["help_count"],0,20)))
        attendance = st.slider("Attendance rate", 0, 100, int(float(student_row["attendance_rate"])*100)) / 100
    new_x = pd.DataFrame([{"study_hours":study_hours,"completion_rate":completion,
        "quiz_attempts":attempts,"interactions":interactions,"help_count":help_count,
        "attendance_rate":attendance}])[FEATURES]
    new_prediction = str(model.predict(new_x)[0])
    try:
        class_probs = model.predict_proba(new_x)[0]
        conf = float(max(class_probs))
    except Exception:
        conf = 0.0
    x,y,z = st.columns(3)
    with x: st.metric("Predicted level", new_prediction)
    with y: st.metric("Model confidence", f"{conf:.0%}")
    with z: st.metric("Average dataset score", f"{avg_score:.0%}")
    st.progress(min(max(conf,0),1))
    if model_metrics:
        st.markdown("#### Holdout evaluation")
        st.caption("Metrics below use a random train/test split. For real deployment, evaluate on later-in-time student records.")
        m1,m2,m3,m4 = st.columns(4)
        for col, key in zip([m1,m2,m3,m4],["Accuracy","Precision (weighted)","Recall (weighted)","F1-score (weighted)"]):
            with col: st.metric(key, f"{model_metrics[key]:.1%}")
    st.warning("Demo note: labels in the built-in dataset are derived from scores. For a valid future-performance model, train on earlier activity and predict a later outcome; avoid using the same score to create the target and as an input feature.")

elif page == "Course Recommendations":
    st.markdown('<div class="section-title">📚 Courses selected for your learning profile</div>', unsafe_allow_html=True)
    courses = pd.DataFrame([
        ["Python for Data Science","Python","Beginner",4.7, "Practice variables, functions, and data analysis."],
        ["Data Structures & Algorithms","Data Structures","Intermediate",4.6, "Build problem-solving skills with arrays, stacks, queues, and trees."],
        ["Machine Learning Foundations","Machine Learning","Intermediate",4.8, "Learn model training, evaluation, and feature engineering."],
        ["Cyber Security Essentials","Cyber Security","Beginner",4.7, "Understand threats, safe practices, and network basics."],
        ["Applied AI Projects","Machine Learning","Advanced",4.9, "Build and evaluate a practical AI application."],
        ["Mathematics for AI","Mathematics","Intermediate",4.5, "Strengthen probability, statistics, and linear algebra."]
    ], columns=["Course","Subject","Level","Rating","Description"])
    subject_pref = st.selectbox("Choose your interest", ["All"] + sorted(courses["Subject"].unique()))
    level_pref = st.selectbox("Preferred difficulty", ["All","Beginner","Intermediate","Advanced"])
    filtered = courses.copy()
    if subject_pref != "All": filtered = filtered[filtered["Subject"] == subject_pref]
    if level_pref != "All": filtered = filtered[filtered["Level"] == level_pref]
    for start in range(0,len(filtered),2):
        cols = st.columns(2)
        for col, (_, course) in zip(cols, filtered.iloc[start:start+2].iterrows()):
            with col:
                with st.container(border=True):
                    st.markdown(f"### 🎯 {course['Course']}")
                    st.caption(f"{course['Subject']} · {course['Level']} · ⭐ {course['Rating']}")
                    st.write(course["Description"])
                    if st.button("Add to my study plan", key="course_"+course["Course"]):
                        st.session_state.setdefault("added_courses", [])
                        if course["Course"] not in st.session_state["added_courses"]:
                            st.session_state["added_courses"].append(course["Course"])
                        st.success("Added to your study plan!")
    if st.session_state.get("added_courses"):
        st.markdown("#### Your selected courses")
        for course in st.session_state["added_courses"]:
            st.write("✅", course)

elif page == "Adaptive Learning Path":
    st.markdown('<div class="section-title">🛤️ Adaptive learning pathway</div>', unsafe_allow_html=True)
    st.write("Your next activity is selected using the profile and predicted performance.")
    steps = make_recommendations(student_row, predicted)
    path = [
        ("1. Review", "Revisit the last topic and write down three key ideas.", "Completed" if student_row["completion_rate"] > .8 else "Recommended"),
        ("2. Practice", steps[0][0], "Next"),
        ("3. Check understanding", "Complete a short 5-question quiz.", "Up next"),
        ("4. Level up", "Move to the next topic when quiz accuracy reaches 80%.", "Locked until ready")
    ]
    for title, desc, status in path:
        with st.container(border=True):
            l,r = st.columns([4,1])
            with l:
                st.markdown(f"**{title}**")
                st.write(desc)
            with r:
                st.info(status)
    st.markdown("#### Recommended focus")
    for title, detail, duration in steps:
        st.write(f"**{title}** · {duration} — {detail}")

elif page == "AI Tutor (RAG)":
    st.markdown("### 🤖 NLP AI Tutor")

user_question = st.text_input(
    "Ask any NLP question:",
    placeholder="Example: What is Tokenization?"
)

if st.button("Get Answer"):
    if user_question.strip():
        sample_qa = {
            "what is nlp": "NLP (Natural Language Processing) helps computers understand human language.",
            "what is tokenization": "Tokenization splits text into smaller units such as words or sentences.",
            "what is stemming": "Stemming reduces words to their root form by removing prefixes or suffixes.",
            "what is lemmatization": "Lemmatization converts a word into its meaningful base form.",
            "what is tf-idf": "TF-IDF measures how important a word is in a document.",
            "what is n-gram": "An N-gram is a sequence of N consecutive words.",
            "what is word2vec": "Word2Vec converts words into numerical vectors based on their context.",
            "what is sentiment analysis": "Sentiment analysis identifies whether text is positive, negative, or neutral.",
            "what is stop word": "Stop words are common words such as 'is', 'the', and 'a'.",
            "what is text preprocessing": "Text preprocessing cleans and prepares text before NLP analysis."
        }

        question = user_question.strip().lower()

        if question in sample_qa:
            st.success("Answer")
            st.write(sample_qa[question])
        else:
            st.warning(
                "This demo knows 10 sample NLP questions. "
                "Try a supported question, or connect an AI model "
                "to answer other questions."
            )
    else:
        st.warning("Please type a question first.")
    st.markdown('<div class="section-title">🤖 AI Tutor — ask questions from your own notes</div>', unsafe_allow_html=True)
    st.info("This is a local retrieval-based tutor: upload notes or a PDF, and it finds the most relevant passages. It does not call an external AI API.")
    docs = st.file_uploader("Upload notes (TXT, PDF)", type=["txt","pdf"], accept_multiple_files=True)
    chunks = []
    chunk_sources = []
    if docs:
        for doc in docs:
            text_content = ""
            if doc.name.lower().endswith(".txt"):
                text_content = doc.getvalue().decode("utf-8", errors="ignore")
            else:
                try:
                    from pypdf import PdfReader
                    reader = PdfReader(io.BytesIO(doc.getvalue()))
                    text_content = "\n".join(page.extract_text() or "" for page in reader.pages)
                except Exception:
                    st.error("Could not read this PDF. Install pypdf using: pip install pypdf")
            # Split text into overlapping-ish paragraph chunks
            paras = [p.strip() for p in re.split(r"\n\s*\n", text_content) if p.strip()]
            if not paras:
                paras = [text_content[i:i+900] for i in range(0,len(text_content),750) if text_content[i:i+900].strip()]
            for p in paras:
                if p:
                    chunks.append(p[:1800])
                    chunk_sources.append(doc.name)
    question = st.text_input("Your question", placeholder="Example: Explain supervised learning in simple words")
    if st.button("Ask AI Tutor", type="primary"):
        if not question.strip():
            st.warning("Enter a question first.")
        elif not chunks:
            st.warning("Upload a TXT or PDF note first. Without notes, this tutor has no source material to search.")
        else:
            try:
                vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
                matrix = vectorizer.fit_transform(chunks + [question])
                sims = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
                top = np.argsort(sims)[::-1][:3]
                useful = [i for i in top if sims[i] > 0.02]
                if not useful:
                    st.warning("I couldn't find a close match in these notes. Try using keywords from your document.")
                else:
                    st.markdown("### Answer from your notes")
                    st.write("I found these relevant passages. Use them as the basis for your answer:")
                    for rank, i in enumerate(useful,1):
                        with st.container(border=True):
                            st.caption(f"Source: {chunk_sources[i]} · Match score: {sims[i]:.2f}")
                            st.write(chunks[i])
                    st.caption("Note: This version retrieves relevant text; it does not generate new explanations like a hosted LLM.")
            except Exception as e:
                st.error(f"Retrieval error: {e}")
    st.markdown("#### Quick question examples")
    st.write("Try: **What is machine learning?** · **Explain neural networks** · **What are the steps in data preprocessing?**")

elif page == "Knowledge Map":
    st.markdown('<div class="section-title">🧩 Knowledge map</div>', unsafe_allow_html=True)
    st.write("A sample concept map. Mark topics based on your own understanding.")
    topics = ["Python basics","Variables & data types","Conditions & loops","Functions","Data structures","OOP","NumPy & Pandas","Machine learning","Model evaluation"]
    cols = st.columns(3)
    mastery = {}
    for i, topic in enumerate(topics):
        with cols[i%3]:
            mastery[topic] = st.select_slider(topic, options=["Needs practice","Learning","Mastered"], value="Learning", key="mastery_"+topic)
    st.markdown("#### Suggested next topic")
    weak = [k for k,v in mastery.items() if v == "Needs practice"]
    if weak:
        st.warning(f"Start with: {weak[0]}. Review the prerequisite concept before moving forward.")
    else:
        st.success("No topic marked as needing practice. Try a mixed quiz to check retention.")
    st.markdown("#### Topic status")
    map_df = pd.DataFrame({"Topic":list(mastery.keys()),"Status":list(mastery.values())})
    st.dataframe(map_df, use_container_width=True, hide_index=True)

elif page == "Quizzes & Practice":
    st.markdown('<div class="section-title">📝 Quick practice quiz</div>', unsafe_allow_html=True)
    quiz = [
        ("Which Python data type stores key-value pairs?", ["List","Dictionary","Tuple","Set"], "Dictionary",
         "A dictionary stores data as key-value pairs."),
        ("Which metric is the harmonic mean of precision and recall?", ["Accuracy","Recall","F1-score","Specificity"], "F1-score",
         "F1-score combines precision and recall using their harmonic mean."),
        ("Which learning type uses labelled examples?", ["Unsupervised learning","Supervised learning","Reinforcement only","Clustering"], "Supervised learning",
         "Supervised learning learns from examples that include the correct target."),
        ("What does train/test split help estimate?", ["Font size","Generalization to unseen data","File size","Internet speed"], "Generalization to unseen data",
         "The test set checks model performance on data not used for fitting.")
    ]
    if "quiz_answers" not in st.session_state:
        st.session_state.quiz_answers = {}
    for i,(q,opts,answer,explain) in enumerate(quiz):
        st.markdown(f"**Q{i+1}. {q}**")
        st.session_state.quiz_answers[i] = st.radio("Choose one:", opts, key=f"quiz_{i}", label_visibility="collapsed")
    if st.button("Submit quiz", type="primary"):
        score = sum(st.session_state.quiz_answers.get(i) == q[2] for i,q in enumerate(quiz))
        st.success(f"Your score: {score}/{len(quiz)} ({score/len(quiz):.0%})")
        for i,(q,opts,answer,explain) in enumerate(quiz):
            if st.session_state.quiz_answers.get(i) == answer:
                st.write(f"✅ Q{i+1}: Correct — {explain}")
            else:
                st.write(f"❌ Q{i+1}: Correct answer: **{answer}**. {explain}")

elif page == "Study Plan":
    st.markdown('<div class="section-title">📅 Your daily study plan</div>', unsafe_allow_html=True)
    st.write("Tick off tasks as you complete them. This plan is for the current session.")
    tasks = [
        ("Review notes from the last class", "15 min"),
        ("Complete one topic lesson", "25 min"),
        ("Solve 5 practice questions", "20 min"),
        ("Review mistakes and write corrections", "10 min"),
        ("Take a short recall quiz", "10 min"),
    ]
    done = 0
    for i,(task,duration) in enumerate(tasks):
        checked = st.checkbox(f"{task} — {duration}", key=f"task_{i}")
        done += int(checked)
    st.progress(done/len(tasks))
    st.metric("Tasks completed", f"{done}/{len(tasks)}")
    if done == len(tasks):
        st.balloons()
        st.success("Great work! You completed today's plan.")
    st.markdown("#### Personalized tip")
    st.info(make_recommendations(student_row, predicted)[0][1])

st.markdown("---")
st.caption("AI Learning Assistant 2.0 · Educational demo. Predictions are advisory only, and the included data is synthetic.")
