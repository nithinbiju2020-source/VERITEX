import streamlit as st
import joblib
import time


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="VERITEX",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# ==========================================
# LOAD MODELS
# ==========================================

fake_model_path = "models/fake_news_model.pkl"
clickbait_model_path = "models/clickbait_model.pkl"

try:
    fake_model = joblib.load(fake_model_path)
    clickbait_model = joblib.load(clickbait_model_path)

except Exception as e:
    st.error("Could not load the ML models.")
    st.code(str(e))
    st.stop()


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.stApp {
    background: radial-gradient(circle at 50% -10%, #18213b 0%, #080b14 48%);
    color: white;
}

.block-container {
    max-width: 900px;
    padding-top: 3rem;
    padding-bottom: 3rem;
}


/* HEADER */

.logo {
    text-align: center;
    font-size: clamp(44px, 9vw, 72px);
    font-weight: 900;
    letter-spacing: -3px;
    color: white;
    margin: 0;
}

.tagline {
    text-align: center;
    color: #9ca3af;
    font-size: clamp(14px, 3vw, 18px);
    margin-top: 5px;
    margin-bottom: 42px;
}


/* INPUT */

.input-label {
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 10px;
}

textarea {
    background: #111827 !important;
    color: white !important;
    border: 1px solid #374151 !important;
    border-radius: 14px !important;
}

textarea::placeholder {
    color: #6b7280 !important;
}


/* BUTTON */

.stButton > button {
    width: 100%;
    height: 54px;
    border-radius: 13px;
    border: 1px solid #4b5563;
    background: linear-gradient(135deg, #151c2e, #0f1422);
    color: white;
    font-size: 15px;
    font-weight: 800;
    letter-spacing: 1px;
    transition: all 0.25s ease;
}

.stButton > button:hover {
    border-color: #8b5cf6;
    background: linear-gradient(135deg, #1c2540, #12182a);
    transform: translateY(-1px);
}


/* SCANNING */

.scan-box {
    margin: 25px 0;
    padding: 24px;
    border-radius: 16px;
    background: rgba(17, 24, 39, 0.95);
    border: 1px solid #374151;
    text-align: center;
}

.scan-title {
    font-size: 17px;
    font-weight: 800;
    letter-spacing: 1px;
}

.scan-subtitle {
    color: #9ca3af;
    font-size: 13px;
    margin-top: 7px;
}

.scanner {
    width: 80%;
    height: 4px;
    background: #1f2937;
    border-radius: 20px;
    margin: 18px auto 0;
    overflow: hidden;
}

.scanner-bar {
    height: 100%;
    width: 35%;
    background: linear-gradient(90deg, transparent, #8b5cf6, #60a5fa, transparent);
    border-radius: 20px;
    animation: scanning 1.2s infinite ease-in-out;
}

@keyframes scanning {
    0% {
        transform: translateX(-200%);
    }

    100% {
        transform: translateX(300%);
    }
}

.loading-dots::after {
    content: "";
    animation: dots 1.4s infinite;
}

@keyframes dots {
    0% { content: ""; }
    25% { content: "."; }
    50% { content: ".."; }
    75% { content: "..."; }
    100% { content: ""; }
}


/* RESULTS */

.results-heading {
    text-align: center;
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 2px;
    color: #6b7280;
    margin: 30px 0 18px;
}

.result-card {
    padding: 28px 20px;
    min-height: 190px;
    border-radius: 18px;
    text-align: center;
    background: linear-gradient(145deg, #182034, #0c111d);
    border: 1px solid #334155;
    box-shadow: 0 15px 40px rgba(0,0,0,0.22);
    box-sizing: border-box;
}

.result-title {
    font-size: 13px;
    color: #94a3b8;
    font-weight: 800;
    letter-spacing: 1.5px;
}

.result-icon {
    font-size: 28px;
    margin: 10px 0;
}

.result-value {
    font-size: clamp(22px, 5vw, 31px);
    font-weight: 900;
    color: white;
}

.confidence {
    color: #9ca3af;
    font-size: 14px;
    margin-top: 10px;
}

.confidence-track {
    height: 7px;
    background: #1f2937;
    border-radius: 20px;
    margin-top: 15px;
    overflow: hidden;
}

.confidence-fill {
    height: 100%;
    border-radius: 20px;
    background: linear-gradient(90deg, #6366f1, #8b5cf6, #60a5fa);
}


/* SUMMARY */

.summary-box {
    margin-top: 25px;
    padding: 22px;
    border-radius: 16px;
    background: #0f172a;
    border: 1px solid #263449;
}

.summary-title {
    font-size: 16px;
    font-weight: 800;
    margin-bottom: 14px;
}

.summary-item {
    color: #cbd5e1;
    font-size: 14px;
    line-height: 1.6;
    margin: 10px 0;
}


/* DISCLAIMER */

.disclaimer {
    margin-top: 25px;
    padding: 15px;
    border-radius: 12px;
    background: rgba(30,41,59,0.55);
    border: 1px solid #263449;
    color: #94a3b8;
    font-size: 12px;
    line-height: 1.6;
    text-align: center;
}


/* FOOTER */

.footer {
    text-align: center;
    color: #4b5563;
    margin-top: 45px;
    font-size: 12px;
}


/* MOBILE */

@media (max-width: 640px) {

    .block-container {
        padding-left: 16px;
        padding-right: 16px;
        padding-top: 2rem;
    }

    .tagline {
        margin-bottom: 30px;
    }

    .result-card {
        min-height: auto;
        margin-bottom: 14px;
        padding: 25px 18px;
    }

    .scan-box {
        padding: 20px 14px;
    }

    .scanner {
        width: 90%;
    }

    .summary-box {
        padding: 18px;
    }
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# HEADER
# ==========================================

st.markdown(
    '<div class="logo">VERITEX</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="tagline">AI-Powered Fake News & Clickbait Analyzer</div>',
    unsafe_allow_html=True
)


# ==========================================
# INPUT
# ==========================================

st.markdown(
    '<div class="input-label">📰 Enter a headline or news article</div>',
    unsafe_allow_html=True
)

text = st.text_area(
    "News content",
    height=180,
    placeholder="Paste a news headline or article here...",
    label_visibility="collapsed"
)


# ==========================================
# ANALYZE
# ==========================================

if st.button("🔍  ANALYZE", use_container_width=True):

    if not text.strip():

        st.warning("Please enter a headline or article first.")

    else:

        # ==================================
        # LOADING ANIMATION
        # ==================================

        loader = st.empty()

        loader.markdown(
            '<div class="scan-box">'
            '<div class="scan-title">VERITEX IS SCANNING<span class="loading-dots"></span></div>'
            '<div class="scan-subtitle">Analyzing linguistic patterns and content signals</div>'
            '<div class="scanner"><div class="scanner-bar"></div></div>'
            '</div>',
            unsafe_allow_html=True
        )

        time.sleep(1.6)

        loader.empty()


        # ==================================
        # FAKE NEWS
        # ==================================

        fake_prediction = fake_model.predict([text])[0]
        fake_probability = fake_model.predict_proba([text])[0]

        if fake_prediction == 1:
            fake_label = "LIKELY FAKE"
            fake_confidence = fake_probability[1] * 100
            fake_icon = "⚠️"
        else:
            fake_label = "LIKELY REAL"
            fake_confidence = fake_probability[0] * 100
            fake_icon = "✓"


        # ==================================
        # CLICKBAIT
        # ==================================

        clickbait_prediction = clickbait_model.predict([text])[0]
        clickbait_probability = clickbait_model.predict_proba([text])[0]

        if clickbait_prediction == 1:
            clickbait_label = "CLICKBAIT"
            clickbait_confidence = clickbait_probability[1] * 100
            clickbait_icon = "🎯"
        else:
            clickbait_label = "NON-CLICKBAIT"
            clickbait_confidence = clickbait_probability[0] * 100
            clickbait_icon = "✓"


        # ==================================
        # RESULTS HEADER
        # ==================================

        st.markdown(
            '<div class="results-heading">ANALYSIS RESULTS</div>',
            unsafe_allow_html=True
        )


        # ==================================
        # RESULT CARDS
        # ==================================

        col1, col2 = st.columns(2, gap="medium")


        # FAKE NEWS CARD

        with col1:

            fake_html = (
                '<div class="result-card">'
                '<div class="result-title">FAKE NEWS</div>'
                f'<div class="result-icon">{fake_icon}</div>'
                f'<div class="result-value">{fake_label}</div>'
                f'<div class="confidence">Confidence: {fake_confidence:.2f}%</div>'
                '<div class="confidence-track">'
                f'<div class="confidence-fill" style="width:{fake_confidence:.2f}%"></div>'
                '</div>'
                '</div>'
            )

            st.markdown(
                fake_html,
                unsafe_allow_html=True
            )


        # CLICKBAIT CARD

        with col2:

            clickbait_html = (
                '<div class="result-card">'
                '<div class="result-title">CLICKBAIT</div>'
                f'<div class="result-icon">{clickbait_icon}</div>'
                f'<div class="result-value">{clickbait_label}</div>'
                f'<div class="confidence">Confidence: {clickbait_confidence:.2f}%</div>'
                '<div class="confidence-track">'
                f'<div class="confidence-fill" style="width:{clickbait_confidence:.2f}%"></div>'
                '</div>'
                '</div>'
            )

            st.markdown(
                clickbait_html,
                unsafe_allow_html=True
            )


        # ==================================
        # SUMMARY
        # ==================================

        summary_html = (
            '<div class="summary-box">'
            '<div class="summary-title">🔎 Analysis Summary</div>'
        )

        if fake_prediction == 1:

            summary_html += (
                '<div class="summary-item">'
                f'⚠️ The model predicts this content as <b>likely fake</b> '
                f'with <b>{fake_confidence:.2f}%</b> confidence.'
                '</div>'
            )

        else:

            summary_html += (
                '<div class="summary-item">'
                f'✓ The model predicts this content as <b>likely real</b> '
                f'with <b>{fake_confidence:.2f}%</b> confidence.'
                '</div>'
            )


        if clickbait_prediction == 1:

            summary_html += (
                '<div class="summary-item">'
                f'🎯 The content shows <b>clickbait patterns</b> '
                f'with <b>{clickbait_confidence:.2f}%</b> confidence.'
                '</div>'
            )

        else:

            summary_html += (
                '<div class="summary-item">'
                f'✓ The content does not strongly match <b>clickbait patterns</b> '
                f'({clickbait_confidence:.2f}% confidence).'
                '</div>'
            )


        summary_html += '</div>'

        st.markdown(
            summary_html,
            unsafe_allow_html=True
        )


        # ==================================
        # DISCLAIMER
        # ==================================

        disclaimer_html = (
            '<div class="disclaimer">'
            'VERITEX identifies linguistic patterns learned from training data. '
            'A prediction is not proof that a news claim is factually true or false.'
            '</div>'
        )

        st.markdown(
            disclaimer_html,
            unsafe_allow_html=True
        )


# ==========================================
# FOOTER
# ==========================================

st.markdown(
    '<div class="footer">'
    'VERITEX • Machine Learning Micro Project • KTU 2024 Scheme'
    '</div>',
    unsafe_allow_html=True
)