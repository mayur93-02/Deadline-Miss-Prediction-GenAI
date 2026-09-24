import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
from groq import Groq

# Load API key from .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ============================================================
# Model Load
# ============================================================
model = joblib.load('deadline_miss_model.pkl')

# ============================================================
# Background data for SHAP (loaded once, cached for speed)
# ============================================================
@st.cache_resource
def load_background():
    try:
        return pd.read_csv('background_sample.csv')
    except FileNotFoundError:
        return None

background_data = load_background()

st.set_page_config(page_title="Project Deadline Risk Predictor", page_icon="🚨")
st.title("🚨 Project Deadline Miss Predictor")
st.write("Project की details डालिए, deadline miss होने का risk जानिए")

st.subheader("📋 Basic Project Info")
team_exp = st.slider("Team Experience (years)", 0.0, 12.0, 3.0)
manager_exp = st.slider("Manager Experience (years)", 0.0, 12.0, 3.0)
year_end = st.number_input("Year End (e.g. 85 for 1985)", min_value=80, max_value=99, value=85)
length = st.slider("Planned Length (months)", 1, 40, 12)

st.subheader("🧩 Complexity Metrics")
transactions = st.number_input("Transactions Count", 0, 900, 200)
entities = st.number_input("Entities Count", 0, 200, 50)
points_nonadjust = st.number_input("Points Non-Adjust", 0, 1500, 300)
adjustment = st.slider("Adjustment Factor", 0, 50, 25)

st.subheader("💻 Language")
language = st.selectbox("Programming Language", ["Language 1", "Language 2", "Language 3"])

# ============================================================
# Derived / Engineered Features (must match training exactly)
# ============================================================
points_adjust = points_nonadjust * (0.65 + adjustment * 0.01)

transactions_per_length = transactions / length if length != 0 else 0
entities_per_length = entities / length if length != 0 else 0
transactions_per_entity = transactions / entities if entities != 0 else 0
adjustment_ratio = adjustment / points_nonadjust if points_nonadjust != 0 else 0

experience_gap = team_exp - manager_exp
total_experience = team_exp + manager_exp
experience_ratio = team_exp / (manager_exp + 1)  # +1 avoids divide-by-zero

complexity_score = (points_adjust + entities + transactions) / 3

# Language one-hot
language_1 = 1 if language == "Language 1" else 0
language_2 = 1 if language == "Language 2" else 0
language_3 = 1 if language == "Language 3" else 0

# Point size bucket (based on PointsAjust) -- one-hot
# bins: Small (0-100), Medium (100-250), Large (250-500), Very Large (500+)
if points_adjust <= 100:
    size_bucket = "Small"
elif points_adjust <= 250:
    size_bucket = "Medium"
elif points_adjust <= 500:
    size_bucket = "Large"
else:
    size_bucket = "Very Large"

point_size_small = 1 if size_bucket == "Small" else 0
point_size_medium = 1 if size_bucket == "Medium" else 0
point_size_large = 1 if size_bucket == "Large" else 0
point_size_very_large = 1 if size_bucket == "Very Large" else 0

# ============================================================
# Build input row -- EXACT same column order as X_train
# ============================================================
input_data = pd.DataFrame([{
    'TeamExp': team_exp,
    'ManagerExp': manager_exp,
    'YearEnd': year_end,
    'Length': length,
    'Transactions': transactions,
    'Entities': entities,
    'PointsNonAdjust': points_nonadjust,
    'Adjustment': adjustment,
    'PointsAjust': points_adjust,
    'Transactions_per_Length': transactions_per_length,
    'Entities_per_Length': entities_per_length,
    'Transactions_per_Entity': transactions_per_entity,
    'Adjustment_Ratio': adjustment_ratio,
    'Experience_Gap': experience_gap,
    'Language_1': language_1,
    'Language_2': language_2,
    'Language_3': language_3,
    'Complexity_Score': complexity_score,
    'Experience_Ratio': experience_ratio,
    'Point_size_Small': point_size_small,
    'Point_size_Medium': point_size_medium,
    'Point_size_Large': point_size_large,
    'Point_size_Very Large': point_size_very_large,
    'Total_Experience': total_experience,
}])

# ============================================================
# Predict
# ============================================================
if st.button("🔮 Predict Risk"):
    prediction = model.predict(input_data)[0]

    # Not all models support predict_proba (e.g. some SVM configs) -- handle safely
    try:
        probability = model.predict_proba(input_data)[0][1]
        prob_text = f" (Probability: {probability:.1%})"
    except AttributeError:
        prob_text = ""

    if prediction == 1:
        st.error(f"⚠️ High Risk! This project is likely to miss its deadline.{prob_text}")
    else:
        st.success(f"✅ Low Risk. This project is likely to stay on schedule.{prob_text}")

    with st.expander("🔍 See input data sent to model"):
        st.dataframe(input_data)

    # ========================================================
    # WHY? -- SHAP explanation for THIS specific prediction
    # ========================================================
    st.subheader("🧠 Kyun ye risk dikh raha hai (Top Reasons)")

    with st.spinner("Analyzing risk factors... (10-15 seconds lag sakte hain)"):
        try:
            # Tree-based models (Random Forest, XGBoost, DecisionTree) -- fast
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_data)
            if isinstance(shap_values, list):
                row_shap = shap_values[1][0]
            else:
                row_shap = shap_values[0]
        except Exception:
            # Non-tree models (KNN, SVM, Logistic Regression)
            if background_data is None:
                st.warning("⚠️ background_sample.csv nahi mili — explanation kam accurate ho sakta hai. "
                           "Notebook mein X_train.sample(50).to_csv('background_sample.csv') chalaakar "
                           "is app.py ke folder mein daaliye.")
                background = input_data  # last-resort fallback only
            else:
                background = background_data[input_data.columns]  # same column order

            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values_full = explainer.shap_values(input_data, nsamples=100, silent=True)
            if isinstance(shap_values_full, list):
                row_shap = shap_values_full[1][0]
            else:
                row_shap = shap_values_full[0, :, 1]

    # Build a readable table: feature, value, impact (+ increases risk, - decreases risk)
    contrib_df = pd.DataFrame({
        'Feature': input_data.columns,
        'Value': input_data.iloc[0].values,
        'Impact': row_shap
    })
    contrib_df['AbsImpact'] = contrib_df['Impact'].abs()
    contrib_df = contrib_df.sort_values('AbsImpact', ascending=False).head(5)

    for _, row in contrib_df.iterrows():
        direction = "🔺 बढ़ा रहा है risk" if row['Impact'] > 0 else "🔻 घटा रहा है risk"
        st.write(f"**{row['Feature']}** = {row['Value']:.2f} → {direction}")

    # ========================================================
    # HOW TO FIX -- simple rule-based suggestions based on top risk-increasing factors
    # ========================================================
    st.subheader("🛠️ Risk Kaise Kam Karein (Suggestions)")

    risk_increasing = contrib_df[contrib_df['Impact'] > 0]['Feature'].tolist()

    suggestions_map = {
        'Complexity_Score': "Project scope को छोटे modules में divide कीजिए, ya complexity कम करने के लिए requirements simplify कीजिए।",
        'PointsAjust': "Function points ज़्यादा हैं — feature scope को phase-wise release में बांटने पर विचार कीजिए।",
        'Entities': "Data model बहुत complex है — कम entities वाला simpler design सोचिए, या extra QA/design review जोड़िए।",
        'Transactions': "System में बहुत ज़्यादा operations हैं — कम-priority transactions को बाद के phase में डालिए।",
        'Length': "Planned duration लंबा है — milestones छोटे रखिए ताकि progress track करना आसान हो।",
        'Adjustment': "Technical complexity ज़्यादा है — senior/experienced developers इस project में लगाइए।",
        'Experience_Gap': "Team और Manager के experience में बड़ा gap है — mentoring ya knowledge-sharing sessions रखिए।",
        'Total_Experience': "Team का overall experience कम है — कम से कम 1 senior resource add करने पर विचार कीजिए।",
        'Transactions_per_Length': "Timeline के हिसाब से workload ज़्यादा है — या तो timeline बढ़ाइए या scope घटाइए।",
    }

    if len(risk_increasing) == 0:
        st.write("✅ कोई major risk factor नहीं मिला — project अच्छी स्थिति में है!")
    else:
        for feature in risk_increasing:
            suggestion = suggestions_map.get(feature, "Is factor ko closely monitor kijiye aur team ke saath discuss kijiye.")
            st.write(f"- **{feature}**: {suggestion}")

    # ========================================================
    # GenAI Summary -- turn the SHAP data into a readable paragraph
    # ========================================================
    st.subheader("🤖 AI-Generated Summary")

    if not GROQ_API_KEY:
        st.info("Groq API key nahi mili. `.env` file check kijiye.")
    else:
        with st.spinner("AI summary likh raha hai..."):
            try:
                client = Groq(api_key=GROQ_API_KEY)

                # Build a clear prompt describing this specific project's risk
                top_factors_text = "\n".join(
                    [f"- {row['Feature']} = {row['Value']:.2f} ({'risk badhata hai' if row['Impact'] > 0 else 'risk ghataata hai'})"
                     for _, row in contrib_df.iterrows()]
                )

                risk_label = "High Risk" if prediction == 1 else "Low Risk"

                prompt = f"""Tum ek experienced Project Management consultant ho. Neeche diye gaye
data ke aadhar par, ek chhota (4-5 lines), professional Hindi-English mix (Hinglish) mein
paragraph likho jo project manager ko samjhaye:
1. Project ka overall risk status
2. Sabse bade 2-3 risk factors kya hain aur kyun
3. Ek practical sujhav (recommendation)

Project Status: {risk_label}
Top Contributing Factors:
{top_factors_text}

Sirf paragraph likho, koi heading ya bullet points nahi chahiye."""

                response = client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[{"role": "user", "content": prompt}]
                )

                ai_summary = response.choices[0].message.content
                st.write(ai_summary)

            except Exception as e:
                if "429" in str(e) or "rate_limited" in str(e).lower():
                    st.warning("⏳ AI summary abhi generate nahi ho payi — rate limit lag gayi hai. Thodi der (30-60 seconds) ruk kar dobara try kijiye.")
                else:
                    st.warning(f"AI summary generate nahi ho paayi: {e}")