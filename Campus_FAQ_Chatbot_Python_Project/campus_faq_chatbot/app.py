
import streamlit as st
import sqlite3
import pandas as pd
import re
import math
from collections import Counter
from datetime import datetime
from itertools import combinations
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="CampusConnect | NLP Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

DB_NAME = "campus_chatbot.db"

CAMPUS_LINKS = {
    "University Website": "https://karunya.edu.in",
    "Student Portal": "https://eduserve.karunya.edu",
    "Online Fee Payment": "https://eduserve.karunya.edu/online/payfees.aspx"
}

# =========================================================
# THEME
# =========================================================

if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

def apply_theme():
    dark = st.session_state.theme == "Dark"

    if dark:
        bg = "#101123"
        sidebar = "#191a35"
        card = "#20213d"
        text = "#f4f2ff"
        muted = "#b4b1ca"
        border = "#383958"
        accent = "#b49aff"
        input_bg = "#292a49"
    else:
        bg = "#f3f4fb"
        sidebar = "#ffffff"
        card = "#ffffff"
        text = "#20213a"
        muted = "#55566c"
        border = "#dedfee"
        accent = "#7046c8"
        input_bg = "#ffffff"

    st.markdown(f"""
    <style>
    .stApp {{
        background: {bg};
        color: {text};
    }}

    [data-testid="stSidebar"] {{
        background: {sidebar};
        border-right: 1px solid {border};
    }}

    [data-testid="stSidebar"] * {{
        color: {text} !important;
    }}

    h1, h2, h3, h4, p, label, li {{
        color: {text};
    }}

    .main-title {{
        font-size: 35px;
        font-weight: 800;
        color: {accent};
        margin-bottom: 4px;
    }}

    .subtitle {{
        font-size: 14px;
        color: {muted};
        margin-bottom: 25px;
    }}

    .welcome {{
        background: linear-gradient(120deg, #49318c, #7652c7, #a16ccf);
        padding: 28px;
        border-radius: 18px;
        color: white;
        margin-bottom: 22px;
    }}

    .welcome h2, .welcome p {{
        color: white !important;
    }}

    .metric-card {{
        background: {card};
        border: 1px solid {border};
        border-radius: 15px;
        padding: 20px;
        min-height: 125px;
    }}

    .metric-value {{
        font-size: 31px;
        font-weight: 800;
        color: {accent};
    }}

    .metric-label {{
        font-size: 13px;
        color: {muted};
    }}

    .metric-detail {{
        font-size: 12px;
        color: {muted};
        margin-top: 5px;
    }}

    .section-card {{
        background: {card};
        border: 1px solid {border};
        padding: 18px;
        border-radius: 14px;
        margin-bottom: 12px;
    }}

    .small-muted {{
        color: {muted};
        font-size: 13px;
    }}

    .link-card {{
        display: block;
        background: {card};
        border: 1px solid {border};
        padding: 16px;
        border-radius: 12px;
        text-decoration: none !important;
        margin-bottom: 10px;
        color: {text} !important;
    }}

    .link-card:hover {{
        border-color: {accent};
        color: {accent} !important;
    }}

    .stTextInput input, .stTextArea textarea {{
        background-color: {input_bg} !important;
        color: {text} !important;
    }}

    [data-testid="stMetric"] {{
        background: {card};
        border: 1px solid {border};
        padding: 14px;
        border-radius: 12px;
    }}

    .stButton button {{
        background: {accent};
        color: white;
        border-radius: 9px;
        border: none;
    }}

    hr {{
        border-color: {border};
    }}
    </style>
    """, unsafe_allow_html=True)

apply_theme()

# =========================================================
# DATABASE
# =========================================================

def get_connection():
    return sqlite3.connect(DB_NAME)

def initialize_database():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS faqs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT,
            question TEXT,
            answer TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_message TEXT,
            bot_response TEXT,
            timestamp TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS unanswered_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            timestamp TEXT
        )
    """)

    # Migrate older chat history databases
    cur.execute("PRAGMA table_info(chat_history)")
    columns = [row[1] for row in cur.fetchall()]

    if "user_message" not in columns:
        cur.execute(
            "ALTER TABLE chat_history ADD COLUMN user_message TEXT DEFAULT ''"
        )

    if "bot_response" not in columns:
        cur.execute(
            "ALTER TABLE chat_history ADD COLUMN bot_response TEXT DEFAULT ''"
        )

    if "timestamp" not in columns:
        cur.execute(
            "ALTER TABLE chat_history ADD COLUMN timestamp TEXT DEFAULT ''"
        )

    cur.execute("PRAGMA table_info(chat_history)")
    columns = [row[1] for row in cur.fetchall()]

    if "message" in columns:
        cur.execute("""
            UPDATE chat_history
            SET user_message = message
            WHERE user_message IS NULL OR user_message = ''
        """)

    if "response" in columns:
        cur.execute("""
            UPDATE chat_history
            SET bot_response = response
            WHERE bot_response IS NULL OR bot_response = ''
        """)

    cur.execute("""
        UPDATE chat_history SET timestamp = ?
        WHERE timestamp IS NULL OR timestamp = ''
    """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),))

    cur.execute("SELECT COUNT(*) FROM faqs")
    if cur.fetchone()[0] == 0:

        sample_faqs = [
            ("Admissions", "How can I apply for admission?",
             "You can apply through the official university admission portal. Visit https://karunya.edu.in for admission details."),

            ("Admissions", "What documents are required for admission?",
             "Academic certificates, transfer certificate, identification proof, photographs and other documents specified by the admission office may be required."),

            ("Academics", "How can I access the student portal?",
             "You can access the student portal at https://eduserve.karunya.edu"),

            ("Academics", "Where can I check my attendance?",
             "Log in to the student portal at https://eduserve.karunya.edu to view attendance details."),

            ("Examinations", "How can I check my exam timetable?",
             "Check the student portal at https://eduserve.karunya.edu and official university announcements for examination timetables."),

            ("Examinations", "How do I get my exam results?",
             "Examination results are generally published through the student portal at https://eduserve.karunya.edu."),

            ("Fees", "How can I pay my college fees?",
             "You can pay your fees through the online payment portal: https://eduserve.karunya.edu/online/payfees.aspx"),

            ("Fees", "Can I download my fee receipt?",
             "After successful payment, log in to https://eduserve.karunya.edu/online/payfees.aspx and check the payment history or receipt section."),

            ("Hostel", "How can I apply for hostel accommodation?",
             "Contact the hostel administration or student affairs office for hostel application procedures."),

            ("Hostel", "Are hostel facilities available?",
             "Contact the university hostel office for room availability and accommodation details."),

            ("Placements", "How can I register for placements?",
             "Contact the placement and training department and follow the placement registration announcements."),

            ("Placements", "What is the placement training process?",
             "Placement training generally includes aptitude preparation, technical training, communication skills and interview preparation."),

            ("Transport", "Is college bus transportation available?",
             "Contact the transport department for available bus routes, timings and fee details."),

            ("Scholarships", "How can I apply for scholarships?",
             "Contact the scholarship section or student affairs office to learn about available scholarships and eligibility."),

            ("Facilities", "Does the campus have a library?",
             "Contact the library or visit https://karunya.edu.in for university information."),

            ("General", "What is the official university website?",
             "The official university website is https://karunya.edu.in"),

            ("General", "Where can I contact the university?",
             "Visit https://karunya.edu.in for official contact information.")
        ]

        cur.executemany("""
            INSERT INTO faqs (category, question, answer)
            VALUES (?, ?, ?)
        """, sample_faqs)

    conn.commit()
    conn.close()

initialize_database()

# =========================================================
# DATABASE FUNCTIONS
# =========================================================

def get_faqs():
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM faqs ORDER BY id DESC", conn
    )
    conn.close()
    return df

def get_history():
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT id, user_message, bot_response, timestamp
        FROM chat_history ORDER BY id DESC
    """, conn)
    conn.close()
    return df

def get_unanswered():
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT * FROM unanswered_questions ORDER BY id DESC
    """, conn)
    conn.close()
    return df

def save_chat(question, answer):
    conn = get_connection()
    conn.execute("""
        INSERT INTO chat_history
        (user_message, bot_response, timestamp)
        VALUES (?, ?, ?)
    """, (
        question,
        answer,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
    conn.close()

def save_unanswered(question):
    conn = get_connection()
    conn.execute("""
        INSERT INTO unanswered_questions (question, timestamp)
        VALUES (?, ?)
    """, (
        question,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
    conn.close()

# =========================================================
# TEXT PREPROCESSING
# =========================================================

STOPWORDS = {
    "is", "am", "are", "the", "a", "an", "of", "in", "on",
    "to", "for", "and", "or", "what", "where", "when", "how",
    "can", "i", "do", "does", "my", "me", "it", "at", "with",
    "please", "tell", "about", "could", "would"
}

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def tokenize(text):
    return re.findall(r"\b[\w']+\b", str(text).lower())

def preprocess_text(text):
    tokens = tokenize(text)
    filtered = [word for word in tokens if word not in STOPWORDS]
    return {
        "Original": text,
        "Tokens": tokens,
        "Filtered Tokens": filtered,
        "Token Count": len(tokens),
        "Filtered Count": len(filtered)
    }

# =========================================================
# NLP CHATBOT
# =========================================================

def chatbot_response(user_question):
    faqs = get_faqs()

    if faqs.empty:
        return "The knowledge base is empty.", False

    question = clean_text(user_question)

    if not question:
        return "Please enter a question.", False

    if question in ["hi", "hello", "hey", "good morning", "good evening"]:
        return "Hello! Welcome to CampusConnect. How can I help you today?", True

    if question in ["thanks", "thank you", "thankyou"]:
        return "You're welcome! Feel free to ask another campus-related question.", True

    faq_questions = faqs["question"].fillna("").tolist()
    categories = faqs["category"].fillna("").tolist()

    documents = [
        clean_text(q + " " + c)
        for q, c in zip(faq_questions, categories)
    ]

    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(documents + [question])

        scores = cosine_similarity(
            matrix[-1], matrix[:-1]
        ).flatten()

        best_index = scores.argmax()

        if scores[best_index] >= 0.12:
            return faqs.iloc[best_index]["answer"], True

    except ValueError:
        pass

    words = set(question.split())
    best_score = 0
    best_index = -1

    for i, faq_question in enumerate(faq_questions):
        faq_words = set(clean_text(faq_question).split())
        score = len(words.intersection(faq_words))

        if score > best_score:
            best_score = score
            best_index = i

    if best_index >= 0 and best_score >= 2:
        return faqs.iloc[best_index]["answer"], True

    save_unanswered(user_question)

    return (
        "I couldn't find a suitable answer in my campus knowledge base. "
        "Please try rephrasing your question or contact the concerned department.",
        False
    )

# =========================================================
# NLP LAB FUNCTIONS
# =========================================================

def levenshtein_distance(a, b):
    a, b = a.lower(), b.lower()
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]

    for i in range(len(a) + 1):
        dp[i][0] = i

    for j in range(len(b) + 1):
        dp[0][j] = j

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    return dp[-1][-1]

def ngram_counts(text, n):
    tokens = tokenize(text)
    grams = [
        tuple(tokens[i:i+n])
        for i in range(len(tokens) - n + 1)
    ]
    return Counter(grams)

def pmi_values(text):
    tokens = tokenize(text)
    pairs = list(zip(tokens, tokens[1:]))
    word_counts = Counter(tokens)
    pair_counts = Counter(pairs)
    total = max(len(tokens), 1)

    results = []

    for (w1, w2), count in pair_counts.most_common(15):
        p_xy = count / max(len(pairs), 1)
        p_x = word_counts[w1] / total
        p_y = word_counts[w2] / total
        pmi = math.log2(p_xy / (p_x * p_y))
        results.append({
            "Word Pair": f"{w1} + {w2}",
            "PMI": round(pmi, 3),
            "Frequency": count
        })

    return pd.DataFrame(results)

def rule_based_pos(text):
    result = []

    for word in tokenize(text):
        if word in STOPWORDS:
            tag = "STOPWORD"
        elif word.endswith("ing"):
            tag = "VERB"
        elif word.endswith("ed"):
            tag = "VERB"
        elif word.endswith("ly"):
            tag = "ADVERB"
        elif word.endswith(("ous", "ful", "ive", "al")):
            tag = "ADJECTIVE"
        elif word in ["is", "are", "was", "were", "be", "have", "has"]:
            tag = "VERB"
        else:
            tag = "NOUN / OTHER"

        result.append({"Token": word, "Predicted POS": tag})

    return pd.DataFrame(result)

def rule_based_ner(text):
    entities = []

    patterns = {
        "EMAIL": r"\b[\w.-]+@[\w.-]+\.\w+\b",
        "DATE": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        "TIME": r"\b\d{1,2}(:\d{2})?\s?(AM|PM|am|pm)\b",
        "URL": r"https?://\S+"
    }

    for label, pattern in patterns.items():
        for match in re.finditer(pattern, text):
            entities.append({
                "Entity": match.group(),
                "Type": label
            })

    for match in re.finditer(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b", text):
        entities.append({
            "Entity": match.group(),
            "Type": "PROPER NOUN (RULE)"
        })

    return pd.DataFrame(entities)

def aspect_analysis(text):
    aspects = {
        "hostel": ["hostel", "room", "accommodation"],
        "teaching": ["teacher", "faculty", "teaching", "class"],
        "transport": ["bus", "transport", "travel"],
        "fees": ["fee", "fees", "payment", "cost"],
        "facilities": ["library", "wifi", "facility", "lab"]
    }

    positive = [
        "good", "excellent", "great", "helpful", "clean",
        "comfortable", "easy", "nice", "best"
    ]

    negative = [
        "bad", "poor", "worst", "difficult", "dirty",
        "expensive", "slow", "problem", "uncomfortable"
    ]

    tokens = tokenize(text)
    output = []

    for aspect, keywords in aspects.items():
        if any(word in tokens for word in keywords):
            pos = sum(word in tokens for word in positive)
            neg = sum(word in tokens for word in negative)

            sentiment = (
                "Positive" if pos > neg
                else "Negative" if neg > pos
                else "Neutral"
            )

            output.append({
                "Aspect": aspect.title(),
                "Sentiment": sentiment
            })

    return pd.DataFrame(output)

def simple_srl(text):
    words = text.strip().split()

    if not words:
        return pd.DataFrame()

    verb_index = next(
        (i for i, word in enumerate(words)
         if word.lower().endswith(("ing", "ed"))
         or word.lower() in ["is", "are", "was", "were", "has", "have", "paid", "apply"]),
        None
    )

    if verb_index is None:
        return pd.DataFrame([
            {"Semantic Role": "Sentence", "Text": text}
        ])

    return pd.DataFrame([
        {"Semantic Role": "Agent / Subject", "Text": " ".join(words[:verb_index])},
        {"Semantic Role": "Predicate / Verb", "Text": words[verb_index]},
        {"Semantic Role": "Object / Context", "Text": " ".join(words[verb_index+1:])}
    ])

# =========================================================
# SYLLABUS MAPPING DATA
# =========================================================

SYLLABUS = {
    "UNIT I – NLP Foundations and Language Modelling": [
        ("Text preprocessing", "Implemented", "Lowercasing, punctuation removal, tokenization and stopword filtering."),
        ("Sentence segmentation", "Implemented", "Sentence splitting and text boundary handling."),
        ("Edit distance", "Implemented", "Levenshtein distance between two words."),
        ("Unigram and bigram models", "Implemented", "Word and two-word frequency counts."),
        ("Language modelling", "Demonstrated", "N-gram frequency-based language modelling."),
        ("Smoothing", "Conceptual", "Laplace smoothing explained; advanced probability modelling can be extended.")
    ],
    "UNIT II – POS, NER and Parsing": [
        ("Part-of-Speech tagging", "Demonstrated", "Rule-based POS tag demonstration in NLP Laboratory."),
        ("Named Entity Recognition", "Demonstrated", "Rule-based extraction of names, dates, email addresses and URLs."),
        ("Hidden Markov Models", "Conceptual", "HMM sequence-tagging principle explained."),
        ("Context-Free Grammar", "Conceptual", "CFG production rules and parsing concepts explained."),
        ("Dependency parsing", "Conceptual", "Dependency relations and syntactic structure explained."),
        ("Dynamic programming parsing", "Conceptual", "Parsing optimisation concepts explained.")
    ],
    "UNIT III – Syntax and Semantics": [
        ("Constituency parsing", "Conceptual", "Phrase structure and parse tree concepts."),
        ("Dependency relations", "Conceptual", "Subject, verb and object relationship demonstration."),
        ("Semantic representation", "Demonstrated", "Simple semantic role labelling demonstration."),
        ("WordNet", "Conceptual", "Lexical relationships such as synonymy and hypernymy explained."),
        ("Vector similarity", "Implemented", "TF-IDF cosine similarity for FAQ matching.")
    ],
    "UNIT IV – Embeddings and Information Extraction": [
        ("TF-IDF", "Implemented", "FAQ document vectorisation."),
        ("Pointwise Mutual Information", "Implemented", "Adjacent word-pair PMI demonstration."),
        ("Word embeddings", "Conceptual", "Distributional representation and embedding properties explained."),
        ("Embedding evaluation", "Conceptual", "Similarity and analogy evaluation explained."),
        ("Relation extraction", "Demonstrated", "Simple rule-based subject-predicate-object extraction."),
        ("Event extraction", "Demonstrated", "Rule-based identification of event-related words."),
        ("Temporal information", "Demonstrated", "Date and time expression extraction.")
    ],
    "UNIT V – Advanced NLP Applications": [
        ("Aspect-based analysis", "Demonstrated", "Aspect identification and simple sentiment classification."),
        ("Temporal analysis / TimeBank", "Demonstrated", "Time-expression extraction and TimeBank-style event-time example."),
        ("Semantic Role Labelling", "Demonstrated", "Simplified agent, predicate and object extraction."),
        ("Machine Translation", "Conceptual", "Translation workflow and evaluation concepts explained."),
        ("Question Answering", "Implemented", "Campus FAQ question-answer retrieval."),
        ("Information Retrieval", "Implemented", "TF-IDF-based FAQ search and ranking."),
        ("Chatbots", "Implemented", "Interactive campus FAQ chatbot."),
        ("ASR and TTS", "Future Enhancement", "Speech recognition and text-to-speech can be added.")
    ]
}

# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

st.sidebar.markdown("# 🎓 CampusConnect")
st.sidebar.caption("SMART CAMPUS INFORMATION ASSISTANT")
st.sidebar.markdown("---")

theme_choice = st.sidebar.toggle(
    "☀️ Light Theme",
    value=(st.session_state.theme == "Light")
)

new_theme = "Light" if theme_choice else "Dark"

if new_theme != st.session_state.theme:
    st.session_state.theme = new_theme
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### ✦ WORKSPACE")

page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Dashboard",
        "💬 FAQ Chatbot",
        "📚 FAQ Explorer",
        "🗂️ Knowledge Base",
        "🧪 NLP Laboratory",
        "📈 Analytics & Reports",
        "❓ Unanswered Questions",
        "🎓 NLP Syllabus Mapping",
        "ℹ️ About Project"
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.caption("NLP • Information Retrieval • SQLite")

# =========================================================
# COMMON DATA AND CHART STYLE
# =========================================================

faqs = get_faqs()
history = get_history()
unanswered = get_unanswered()

dark = st.session_state.theme == "Dark"
plot_template = "plotly_dark" if dark else "plotly_white"
plot_bg = "#20213d" if dark else "#ffffff"
font_color = "#f4f2ff" if dark else "#20213a"

def style_chart(fig, height=350):
    fig.update_layout(
        template=plot_template,
        height=height,
        paper_bgcolor=plot_bg,
        plot_bgcolor=plot_bg,
        font=dict(color=font_color),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def heading(title, subtitle):
    st.markdown(
        f'<div class="main-title">{title}</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="subtitle">{subtitle}</div>',
        unsafe_allow_html=True
    )

def metric(label, value, detail):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-detail">{detail}</div>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# DASHBOARD
# =========================================================

if page == "📊 Dashboard":

    st.markdown("""
    <div class="welcome">
        <h2>Welcome to CampusConnect 🎓</h2>
        <p>Your smart campus information assistant. Explore university
        services, ask questions and discover information through NLP.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric("📚 TOTAL FAQs", len(faqs), "Knowledge base records")

    with c2:
        metric("💬 CONVERSATIONS", len(history), "Recorded interactions")

    with c3:
        metric(
            "🗂️ CATEGORIES",
            faqs["category"].nunique() if not faqs.empty else 0,
            "Campus information areas"
        )

    with c4:
        metric("❓ UNANSWERED", len(unanswered), "Questions to improve")

    st.markdown("### 📈 Campus Activity")

    left, right = st.columns([1.5, 1])

    with left:
        st.markdown("#### Conversation Trend")

        if not history.empty:
            data = history.copy()
            data["timestamp"] = pd.to_datetime(
                data["timestamp"], errors="coerce"
            )
            data = data.dropna(subset=["timestamp"])

            if not data.empty:
                data["Date"] = data["timestamp"].dt.date
                daily = data.groupby("Date").size().reset_index(name="Chats")

                fig = px.area(
                    daily,
                    x="Date",
                    y="Chats",
                    markers=True,
                    title="Daily chatbot conversations"
                )
                st.plotly_chart(
                    style_chart(fig),
                    use_container_width=True
                )
            else:
                st.info("Conversation data will appear here.")
        else:
            st.info("Start using the chatbot to generate activity.")

    with right:
        st.markdown("#### FAQ Categories")

        if not faqs.empty:
            cat = faqs["category"].value_counts().reset_index()
            cat.columns = ["Category", "Count"]

            fig = px.pie(
                cat,
                names="Category",
                values="Count",
                hole=0.55,
                title="Knowledge base distribution"
            )
            st.plotly_chart(
                style_chart(fig),
                use_container_width=True
            )

    st.markdown("### 🔗 Quick Campus Access")

    l1, l2, l3 = st.columns(3)

    with l1:
        st.markdown(
            f'<a class="link-card" href="{CAMPUS_LINKS["University Website"]}" target="_blank">🌐 <b>University Website</b><br><span class="small-muted">Official Karunya website</span></a>',
            unsafe_allow_html=True
        )

    with l2:
        st.markdown(
            f'<a class="link-card" href="{CAMPUS_LINKS["Student Portal"]}" target="_blank">🎓 <b>Student Portal</b><br><span class="small-muted">Student academic services</span></a>',
            unsafe_allow_html=True
        )

    with l3:
        st.markdown(
            f'<a class="link-card" href="{CAMPUS_LINKS["Online Fee Payment"]}" target="_blank">💳 <b>Fee Payment</b><br><span class="small-muted">Online fee payment portal</span></a>',
            unsafe_allow_html=True
        )

    st.markdown("### 🧭 Campus Information Areas")

    q1, q2, q3, q4, q5 = st.columns(5)

    q1.info("📖 Academics")
    q2.info("📝 Examinations")
    q3.info("💼 Placements")
    q4.info("🏠 Hostel")
    q5.info("🚌 Transport")

    st.markdown("### 🕘 Recent Conversations")

    if not history.empty:
        st.dataframe(
            history.head(5)[
                ["user_message", "bot_response", "timestamp"]
            ],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No conversations recorded yet.")

# =========================================================
# FAQ CHATBOT
# =========================================================

elif page == "💬 FAQ Chatbot":

    heading(
        "Campus FAQ Chatbot",
        "Ask questions about academics, admissions, fees and campus services."
    )

    st.caption("Powered by text preprocessing, TF-IDF and cosine similarity.")

    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Hello! Welcome to CampusConnect. Ask me about academics, fees, examinations, hostel, placements or transport."
        }]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_input = st.chat_input("💬 Type your campus-related question...")

    if user_input:
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user"):
            st.write(user_input)

        response, matched = chatbot_response(user_input)

        with st.chat_message("assistant"):
            st.write(response)

            if not matched:
                st.caption("No sufficiently relevant FAQ was found.")

        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })

        save_chat(user_input, response)

    st.markdown("### 🔗 Useful Student Links")

    st.markdown(
        f"- [🌐 University Website]({CAMPUS_LINKS['University Website']})\n"
        f"- [🎓 Student Portal]({CAMPUS_LINKS['Student Portal']})\n"
        f"- [💳 Online Fee Payment]({CAMPUS_LINKS['Online Fee Payment']})"
    )

# =========================================================
# FAQ EXPLORER
# =========================================================

elif page == "📚 FAQ Explorer":

    heading(
        "FAQ Explorer",
        "Search, filter and explore the campus knowledge base."
    )

    search = st.text_input(
        "🔍 Search FAQs",
        placeholder="Search for fees, hostel, attendance..."
    )

    categories = ["All Categories"] + (
        sorted(faqs["category"].dropna().unique().tolist())
        if not faqs.empty else []
    )

    selected = st.selectbox("📂 Choose Category", categories)

    filtered = faqs.copy()

    if selected != "All Categories":
        filtered = filtered[filtered["category"] == selected]

    if search:
        filtered = filtered[
            filtered["question"].str.contains(
                search, case=False, na=False
            )
            |
            filtered["answer"].str.contains(
                search, case=False, na=False
            )
        ]

    st.write(f"**{len(filtered)} matching questions**")

    if not filtered.empty:
        for _, row in filtered.iterrows():
            with st.expander(
                f"❔ {row['question']}  |  {row['category']}"
            ):
                st.write(row["answer"])
    else:
        st.warning("No matching FAQ found.")

# =========================================================
# KNOWLEDGE BASE
# =========================================================

elif page == "🗂️ Knowledge Base":

    heading(
        "Knowledge Base Management",
        "Add, view and manage campus questions and answers."
    )

    tab1, tab2 = st.tabs(["➕ Add FAQ", "📋 Manage FAQs"])

    with tab1:
        with st.form("add_faq", clear_on_submit=True):

            category = st.selectbox(
                "Category",
                [
                    "Admissions", "Academics", "Examinations",
                    "Fees", "Hostel", "Placements", "Transport",
                    "Scholarships", "Facilities", "General"
                ]
            )

            question = st.text_input("Question")
            answer = st.text_area("Answer", height=130)

            submitted = st.form_submit_button("Save FAQ")

            if submitted:
                if question.strip() and answer.strip():
                    conn = get_connection()
                    conn.execute("""
                        INSERT INTO faqs (category, question, answer)
                        VALUES (?, ?, ?)
                    """, (category, question.strip(), answer.strip()))
                    conn.commit()
                    conn.close()

                    st.success("FAQ added successfully.")
                    st.rerun()
                else:
                    st.error("Enter both question and answer.")

    with tab2:
        faqs = get_faqs()

        st.dataframe(
            faqs,
            use_container_width=True,
            hide_index=True
        )

        if not faqs.empty:
            selected_id = st.selectbox(
                "Select FAQ ID",
                faqs["id"].tolist()
            )

            if st.button("Delete Selected FAQ"):
                conn = get_connection()
                conn.execute(
                    "DELETE FROM faqs WHERE id = ?",
                    (selected_id,)
                )
                conn.commit()
                conn.close()

                st.success("FAQ deleted.")
                st.rerun()

# =========================================================
# NLP LABORATORY
# =========================================================

elif page == "🧪 NLP Laboratory":

    heading(
        "NLP Laboratory",
        "Experiment with Natural Language Processing techniques used in this project."
    )

    experiment = st.selectbox(
        "🧪 Select NLP Experiment",
        [
            "Text Preprocessing & Tokenization",
            "Sentence Segmentation",
            "Levenshtein Edit Distance",
            "Unigram and Bigram Model",
            "TF-IDF and Cosine Similarity",
            "Pointwise Mutual Information",
            "POS Tagging",
            "Named Entity Recognition",
            "Relation and Event Extraction",
            "Aspect-Based Sentiment Analysis",
            "Semantic Role Labelling",
            "TimeBank-Style Temporal Analysis"
        ]
    )

    if experiment == "Levenshtein Edit Distance":
        st.markdown("### 🔤 Edit Distance Calculator")

        word1 = st.text_input("First word", "kitten")
        word2 = st.text_input("Second word", "sitting")

        if word1 and word2:
            distance = levenshtein_distance(word1, word2)
            st.metric("Levenshtein Distance", distance)
            st.write(
                "The distance represents the minimum number of insertions, "
                "deletions and substitutions required to transform one word into another."
            )

    else:
        default_text = (
            "Students can access the student portal to check attendance "
            "and examination results. The university provides hostel and "
            "transport facilities."
        )

        lab_text = st.text_area(
            "Enter text for NLP analysis",
            default_text,
            height=130,
            key="lab_text"
        )

        if experiment == "Text Preprocessing & Tokenization":
            st.markdown("### Text Preprocessing")

            if lab_text:
                result = preprocess_text(lab_text)

                c1, c2 = st.columns(2)
                c1.metric("Total Tokens", result["Token Count"])
                c2.metric("After Stopword Removal", result["Filtered Count"])

                st.write("**Original Text**")
                st.write(result["Original"])

                st.write("**Tokens**")
                st.write(result["Tokens"])

                st.write("**Filtered Tokens**")
                st.write(result["Filtered Tokens"])

        elif experiment == "Sentence Segmentation":
            st.markdown("### Sentence Segmentation")

            sentences = [
                s.strip()
                for s in re.split(r"(?<=[.!?])\s+", lab_text)
                if s.strip()
            ]

            st.metric("Number of Sentences", len(sentences))

            for i, sentence in enumerate(sentences, 1):
                st.write(f"**Sentence {i}:** {sentence}")

        elif experiment == "Unigram and Bigram Model":
            st.markdown("### N-gram Language Model")

            unigrams = ngram_counts(lab_text, 1)
            bigrams = ngram_counts(lab_text, 2)

            left, right = st.columns(2)

            with left:
                st.write("**Unigram Frequencies**")
                unigram_df = pd.DataFrame(
                    unigrams.most_common(20),
                    columns=["N-gram", "Frequency"]
                )
                unigram_df["N-gram"] = unigram_df["N-gram"].apply(
                    lambda x: " ".join(x)
                )
                st.dataframe(unigram_df, use_container_width=True)

            with right:
                st.write("**Bigram Frequencies**")
                bigram_df = pd.DataFrame(
                    bigrams.most_common(20),
                    columns=["N-gram", "Frequency"]
                )
                bigram_df["N-gram"] = bigram_df["N-gram"].apply(
                    lambda x: " ".join(x)
                )
                st.dataframe(bigram_df, use_container_width=True)

        elif experiment == "TF-IDF and Cosine Similarity":
            st.markdown("### Semantic Similarity Demonstration")

            query = st.text_input(
                "Enter a query",
                "How can I check my attendance?"
            )

            faq_data = get_faqs()

            if query and not faq_data.empty:
                docs = faq_data["question"].fillna("").tolist()

                vectorizer = TfidfVectorizer(ngram_range=(1, 2))
                matrix = vectorizer.fit_transform(docs + [query])

                scores = cosine_similarity(
                    matrix[-1], matrix[:-1]
                ).flatten()

                result = faq_data[["category", "question", "answer"]].copy()
                result["Similarity Score"] = scores
                result = result.sort_values(
                    "Similarity Score", ascending=False
                )

                st.dataframe(
                    result.head(10),
                    use_container_width=True,
                    hide_index=True
                )

                st.write(
                    "Cosine similarity measures how closely the query "
                    "matches the FAQ question in vector space."
                )

        elif experiment == "Pointwise Mutual Information":
            st.markdown("### PMI Word Association")

            result = pmi_values(lab_text)

            if not result.empty:
                st.dataframe(result, use_container_width=True)

                fig = px.bar(
                    result,
                    x="Word Pair",
                    y="PMI",
                    title="Pointwise Mutual Information"
                )
                st.plotly_chart(
                    style_chart(fig),
                    use_container_width=True
                )
            else:
                st.info("Enter more text to calculate PMI.")

        elif experiment == "POS Tagging":
            st.markdown("### Rule-Based POS Tagging")

            result = rule_based_pos(lab_text)
            st.dataframe(result, use_container_width=True)

            st.caption(
                "This is a lightweight rule-based demonstration, not a trained POS tagger."
            )

        elif experiment == "Named Entity Recognition":
            st.markdown("### Rule-Based Named Entity Recognition")

            result = rule_based_ner(lab_text)

            if not result.empty:
                st.dataframe(result, use_container_width=True)
            else:
                st.info("No supported entities detected.")

            st.caption(
                "This demonstration uses regular-expression and capitalization rules."
            )

        elif experiment == "Relation and Event Extraction":
            st.markdown("### Relation and Event Demonstration")

            words = lab_text.split()
            verbs = [
                w for w in words
                if w.lower().endswith(("ing", "ed"))
                or w.lower() in ["is", "are", "was", "were", "has", "have", "provides", "offers", "check"]
            ]

            st.write("**Possible event/action terms**")
            st.write(verbs if verbs else "No obvious event terms identified.")

            st.write("**Simple subject-predicate-object pattern**")
            st.dataframe(
                simple_srl(lab_text),
                use_container_width=True
            )

            st.caption(
                "Rule-based demonstration of event and relation patterns."
            )

        elif experiment == "Aspect-Based Sentiment Analysis":
            st.markdown("### Aspect-Based Sentiment Analysis")

            result = aspect_analysis(lab_text)

            if not result.empty:
                st.dataframe(result, use_container_width=True)

                fig = px.bar(
                    result,
                    x="Aspect",
                    color="Sentiment",
                    title="Aspect sentiment overview"
                )
                st.plotly_chart(
                    style_chart(fig),
                    use_container_width=True
                )
            else:
                st.info(
                    "Try including words such as hostel, teaching, fees, transport or library."
                )

            st.caption(
                "Sentiment is determined using a small rule-based vocabulary."
            )

        elif experiment == "Semantic Role Labelling":
            st.markdown("### Simplified Semantic Role Labelling")

            result = simple_srl(lab_text)
            st.dataframe(result, use_container_width=True)

            st.caption(
                "This simplified demonstration identifies a possible agent, predicate and object/context."
            )

        elif experiment == "TimeBank-Style Temporal Analysis":
            st.markdown("### Event and Time Expression Extraction")

            date_pattern = r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|today|tomorrow|yesterday|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b"
            time_pattern = r"\b\d{1,2}(?::\d{2})?\s?(?:AM|PM|am|pm)\b"

            dates = re.findall(date_pattern, lab_text, flags=re.I)
            times = re.findall(time_pattern, lab_text)

            event_words = [
                word for word in tokenize(lab_text)
                if word.endswith(("ing", "ed"))
                or word in ["exam", "meeting", "class", "payment", "registration", "submission"]
            ]

            temporal_data = []

            for event in event_words:
                temporal_data.append({
                    "Event": event,
                    "Temporal Expression": ", ".join(dates + times) or "Not identified"
                })

            if temporal_data:
                st.dataframe(
                    pd.DataFrame(temporal_data),
                    use_container_width=True
                )
            else:
                st.info("No obvious event terms identified.")

            st.caption(
                "TimeBank-style demonstration only. It does not use a trained TimeBank model."
            )

    st.markdown("---")
    st.markdown("### 📘 NLP Concepts Used")

    st.write("""
    - Text preprocessing and tokenization
    - Statistical language modelling
    - TF-IDF and vector similarity
    - Lexical and syntactic analysis
    - Information extraction
    - Semantic and temporal analysis
    - Question answering and information retrieval
    """)

# =========================================================
# ANALYTICS AND REPORTS
# =========================================================

elif page == "📈 Analytics & Reports":

    heading(
        "Analytics & Reports",
        "Explore chatbot activity and knowledge-base performance."
    )

    a, b, c = st.columns(3)

    a.metric("Total Conversations", len(history))
    b.metric("Total FAQs", len(faqs))
    c.metric("Unanswered Questions", len(unanswered))

    st.markdown("### 📊 FAQ Category Distribution")

    if not faqs.empty:
        cat = faqs["category"].value_counts().reset_index()
        cat.columns = ["Category", "Count"]

        fig = px.bar(
            cat,
            x="Category",
            y="Count",
            color="Category",
            text="Count",
            title="FAQs across campus categories"
        )
        st.plotly_chart(
            style_chart(fig),
            use_container_width=True
        )

    st.markdown("### 💬 Chat Activity")

    if not history.empty:
        data = history.copy()
        data["timestamp"] = pd.to_datetime(
            data["timestamp"], errors="coerce"
        )
        data = data.dropna(subset=["timestamp"])

        if not data.empty:
            data["Date"] = data["timestamp"].dt.date
            daily = data.groupby("Date").size().reset_index(name="Chats")

            fig = px.line(
                daily,
                x="Date",
                y="Chats",
                markers=True,
                title="Daily conversation activity"
            )
            st.plotly_chart(
                style_chart(fig),
                use_container_width=True
            )

        st.markdown("### Recent Chat Records")
        st.dataframe(history, use_container_width=True, hide_index=True)
    else:
        st.info("No chat records available yet.")

    st.markdown("### ❓ Unanswered Question Keywords")

    if not unanswered.empty:
        words = " ".join(
            unanswered["question"].fillna("").tolist()
        ).lower()

        counts = Counter(
            w for w in re.findall(r"\b[a-z]{3,}\b", words)
            if w not in STOPWORDS
        )

        if counts:
            keyword_df = pd.DataFrame(
                counts.most_common(12),
                columns=["Keyword", "Frequency"]
            )

            fig = px.bar(
                keyword_df,
                x="Frequency",
                y="Keyword",
                orientation="h",
                title="Frequently occurring unanswered terms"
            )
            st.plotly_chart(
                style_chart(fig),
                use_container_width=True
            )
    else:
        st.success("No unanswered questions recorded.")

# =========================================================
# UNANSWERED QUESTIONS
# =========================================================

elif page == "❓ Unanswered Questions":

    heading(
        "Unanswered Questions",
        "Review questions that need additional knowledge-base coverage."
    )

    unanswered = get_unanswered()

    st.metric("Total Unanswered Questions", len(unanswered))

    if not unanswered.empty:
        st.dataframe(
            unanswered,
            use_container_width=True,
            hide_index=True
        )

        csv_data = unanswered.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download Unanswered Questions",
            csv_data,
            "campus_unanswered_questions.csv",
            "text/csv"
        )
    else:
        st.success("No unanswered questions recorded yet.")

# =========================================================
# NLP SYLLABUS MAPPING
# =========================================================

elif page == "🎓 NLP Syllabus Mapping":

    heading(
        "NLP Syllabus Mapping",
        "Mapping CampusConnect features to the concepts covered in NLP Units I–V."
    )

    st.markdown("""
    This project applies Natural Language Processing concepts to a
    campus information retrieval and question-answering system.
    The following mapping identifies the practical demonstrations,
    implemented components and concepts explained theoretically.
    """)

    status_data = []

    for unit, topics in SYLLABUS.items():
        implemented = sum(
            1 for _, status, _ in topics if status == "Implemented"
        )
        demonstrated = sum(
            1 for _, status, _ in topics if status == "Demonstrated"
        )
        conceptual = sum(
            1 for _, status, _ in topics
            if status in ["Conceptual", "Future Enhancement"]
        )

        status_data.append({
            "Unit": unit.split("–")[0].strip(),
            "Implemented": implemented,
            "Demonstrated": demonstrated,
            "Conceptual / Future": conceptual
        })

    st.markdown("### 📊 Syllabus Coverage Overview")

    coverage_df = pd.DataFrame(status_data)

    fig = px.bar(
        coverage_df,
        x="Unit",
        y=["Implemented", "Demonstrated", "Conceptual / Future"],
        barmode="group",
        title="NLP topic implementation status by unit"
    )

    st.plotly_chart(
        style_chart(fig),
        use_container_width=True
    )

    st.markdown("### 📖 Detailed Unit-wise Mapping")

    for unit, topics in SYLLABUS.items():
        with st.expander(unit, expanded=False):
            mapping_df = pd.DataFrame(
                topics,
                columns=[
                    "Syllabus Topic",
                    "Implementation Status",
                    "Project Application"
                ]
            )

            st.dataframe(
                mapping_df,
                use_container_width=True,
                hide_index=True
            )

    st.markdown("### 🧠 Core NLP Techniques in CampusConnect")

    st.markdown("""
    | NLP Technique | Application in Project |
    |---|---|
    | Text Preprocessing | Cleans and tokenizes user questions |
    | TF-IDF | Converts questions into numerical vectors |
    | Cosine Similarity | Finds the closest matching FAQ |
    | Information Retrieval | Retrieves relevant campus answers |
    | POS / NER Demonstrations | Shows grammatical and entity analysis |
    | Semantic Role Demonstration | Identifies possible sentence roles |
    | Aspect Analysis | Demonstrates sentiment by campus aspect |
    | Temporal Extraction | Demonstrates date and event identification |
    | Chatbot | Provides a conversational interface |
    """)

    st.info(
        "Implementation status is intentionally distinguished: "
        "rule-based demonstrations are not presented as trained NLP models. "
        "HMM, CFG, full parsing, trained SRL, machine translation and speech "
        "processing are conceptual or future extensions in this version."
    )

# =========================================================
# ABOUT PROJECT
# =========================================================

elif page == "ℹ️ About Project":

    heading(
        "About CampusConnect",
        "An NLP-based campus FAQ chatbot and information retrieval system."
    )

    st.markdown("""
    <div class="welcome">
        <h2>CampusConnect 🎓</h2>
        <p>Making campus information easier to discover through
        Natural Language Processing.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🎯 Project Objective")

    st.write("""
    CampusConnect is designed to help students access common university
    information through a conversational interface. It uses text
    preprocessing, TF-IDF vectorisation and cosine similarity to match
    student questions with relevant entries in a campus FAQ knowledge base.
    """)

    st.markdown("### ⚙️ Technologies Used")

    t1, t2, t3, t4 = st.columns(4)

    t1.info("🐍 Python")
    t2.info("🖥️ Streamlit")
    t3.info("🧠 Scikit-learn")
    t4.info("🗄️ SQLite")

    st.markdown("### 🔗 Official Campus Links")

    for label, url in CAMPUS_LINKS.items():
        st.markdown(f"- [{label}]({url})")

    st.markdown("### 🧩 Main Modules")

    st.markdown("""
    - Dashboard and campus quick links
    - FAQ chatbot
    - FAQ search and exploration
    - Knowledge-base management
    - NLP laboratory
    - Analytics and reports
    - Unanswered question tracking
    - NLP syllabus mapping
    """)

    st.markdown("### ⚠️ Project Scope")

    st.write("""
    The chatbot uses retrieval-based question answering from the stored
    FAQ database. Its NLP laboratory contains lightweight practical
    demonstrations of several syllabus concepts. Some advanced topics
    are included as conceptual explanations rather than trained models.
    """)

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")
st.markdown(
    '<div class="small-muted" style="text-align:center;">'
    '🎓 CampusConnect | NLP-Based Campus FAQ Chatbot | Units I–V'
    '</div>',
    unsafe_allow_html=True
)
