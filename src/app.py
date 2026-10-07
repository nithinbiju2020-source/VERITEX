import streamlit as st
import joblib
import os


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="VERITEX",
    page_icon="📰",
    layout="centered"
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
    background: #080b14;
    color: white;
}

.main-title {
    text-align: center;
    font-size: 52px;
    font-weight: 800;
    margin-bottom: 0;
    color: #ffffff;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #9ca3af;
    margin-bottom: 35px;
}

.result-card {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    background: #111827;
    border: 1px solid #374151;
}

.result-title {
    font-size: 18px;
    color: #9ca3af;
}

.result-value {
    font-size: 32px;
    font-weight: 700;
    margin-top: 8px;
}

.confidence {
    font-size: 16px;
    color: #9ca3af;
}

.footer {
    text-align: center;
    color: #6b7280;
    margin-top: 40px;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# HEADER
# ==========================================

st.markdown(
    '<div class="main-title">VERITEX</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-Powered Fake News & Clickbait Analyzer</div>',
    unsafe_allow_html=True
)


# ==========================================
# INPUT
# ==========================================

st.markdown("### 📰 Enter a headline or news article")

text = st.text_area(
    "",
    height=180,
    placeholder="Paste a news headline or article here..."
)


# ==========================================
# ANALYZE BUTTON
# ==========================================

if st.button("🔍 ANALYZE", use_container_width=True):

    if not text.strip():

        st.warning("Please enter a headline or article first.")

    else:

        # ==================================
        # FAKE NEWS PREDICTION
        # ==================================

        fake_prediction = fake_model.predict([text])[0]

        fake_probability = fake_model.predict_proba([text])[0]

        # label:
        # 0 = Real
        # 1 = Fake

        if fake_prediction == 1:
            fake_label = "LIKELY FAKE"
            fake_confidence = fake_probability[1] * 100
        else:
            fake_label = "LIKELY REAL"
            fake_confidence = fake_probability[0] * 100


        # ==================================
        # CLICKBAIT PREDICTION
        # ==================================

        clickbait_prediction = clickbait_model.predict([text])[0]

        clickbait_probability = clickbait_model.predict_proba([text])[0]

        # label:
        # 0 = Non-clickbait
        # 1 = Clickbait

        if clickbait_prediction == 1:
            clickbait_label = "CLICKBAIT"
            clickbait_confidence = clickbait_probability[1] * 100
        else:
            clickbait_label = "NON-CLICKBAIT"
            clickbait_confidence = clickbait_probability[0] * 100


        # ==================================
        # RESULTS
        # ==================================

        st.markdown("---")

        col1, col2 = st.columns(2)


        # Fake News Result

        with col1:

            st.markdown(
                f"""
                <div class="result-card">

                <div class="result-title">
                FAKE NEWS ANALYSIS
                </div>

                <div class="result-value">
                {fake_label}
                </div>

                <div class="confidence">
                Confidence: {fake_confidence:.2f}%
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        # Clickbait Result

        with col2:

            st.markdown(
                f"""
                <div class="result-card">

                <div class="result-title">
                CLICKBAIT ANALYSIS
                </div>

                <div class="result-value">
                {clickbait_label}
                </div>

                <div class="confidence">
                Confidence: {clickbait_confidence:.2f}%
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        # ==================================
        # SIMPLE INTERPRETATION
        # ==================================

        st.markdown("### 🔎 Analysis Summary")

        if fake_prediction == 1:
            st.error(
                f"⚠️ The model predicts this content as likely fake "
                f"with {fake_confidence:.2f}% confidence."
            )
        else:
            st.success(
                f"✓ The model predicts this content as likely real "
                f"with {fake_confidence:.2f}% confidence."
            )


        if clickbait_prediction == 1:
            st.warning(
                f"🎯 The headline shows strong clickbait patterns "
                f"with {clickbait_confidence:.2f}% confidence."
            )
        else:
            st.info(
                f"✓ The headline does not strongly match clickbait "
                f"patterns ({clickbait_confidence:.2f}% confidence)."
            )


        # ==================================
        # DISCLAIMER
        # ==================================

        st.markdown("---")

        st.caption(
            "VERITEX uses machine-learning patterns learned from training data. "
            "It does not independently verify whether a claim is factually true."
        )


# ==========================================
# FOOTER
# ==========================================

st.markdown(
    '<div class="footer">VERITEX • Machine Learning Micro Project • KTU 2024 Scheme</div>',
    unsafe_allow_html=True
)