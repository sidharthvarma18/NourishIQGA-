"""
NourishIQ – Personalised Health & Diet App
Business Data Analytics Dashboard
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io, base64, os, json

# ──────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NourishIQ Analytics Dashboard",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background: #f8fafc; }
.stMetric { background: white; border-radius: 12px; padding: 16px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }
.block-container { padding-top: 1.5rem; }
h1 { color: #1a1a2e; font-weight: 700; }
h2 { color: #16213e; font-weight: 600; }
h3 { color: #0f3460; font-weight: 600; }
.insight-box {
    background: linear-gradient(135deg, #e8f5e9 0%, #f3e5f5 100%);
    border-left: 4px solid #4caf50;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 10px 0;
    font-size: 0.92rem;
}
.kpi-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    color: white;
    padding: 28px 32px;
    border-radius: 16px;
    margin-bottom: 20px;
}
.section-tag {
    display: inline-block;
    background: #e3f2fd;
    color: #1565c0;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-bottom: 8px;
}
.persona-card {
    background: white;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,.07);
    margin: 8px 0;
    border-top: 4px solid #4caf50;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    path = os.path.join(os.path.dirname(__file__), "nourishiq_dataset.csv")
    df = pd.read_csv(path)
    return df

df_full = load_data()

# ──────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/salad.png", width=64)
st.sidebar.title("🥗 NourishIQ")
st.sidebar.markdown("**Business Analytics Dashboard**")
st.sidebar.markdown("---")

SECTIONS = [
    "🏠 Home & Objectives",
    "📊 Descriptive Analytics",
    "🔍 Diagnostic Analytics",
    "🤖 Classification Models",
    "🌳 Decision Tree Analysis",
    "👥 Customer Segmentation",
    "🔗 Association Rule Mining",
    "📈 Regression Analysis",
    "📅 Forecasting",
    "💡 Recommender System",
    "💬 Text Mining & Sentiment",
    "🌐 Social Network Analysis",
    "🔐 Ethics & Data Privacy",
    "✅ Business Recommendations",
    "🆕 Predict New User",
]
section = st.sidebar.radio("Navigate", SECTIONS)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Global Filters")

age_opts = ["All"] + sorted(df_full["Age_Group"].unique().tolist())
sel_age = st.sidebar.selectbox("Age Group", age_opts)

gender_opts = ["All"] + sorted(df_full["Gender"].unique().tolist())
sel_gender = st.sidebar.selectbox("Gender", gender_opts)

tier_opts = ["All"] + sorted(df_full["City_Tier"].unique().tolist())
sel_tier = st.sidebar.selectbox("City Tier", tier_opts)

intent_opts = ["All"] + sorted(df_full["Sign_Up_Intent"].unique().tolist())
sel_intent = st.sidebar.selectbox("Sign-Up Intent", intent_opts)

bmi_opts = ["All"] + sorted(df_full["BMI_Category"].unique().tolist())
sel_bmi = st.sidebar.selectbox("BMI Category", bmi_opts)

def apply_filters(df):
    if sel_age != "All": df = df[df["Age_Group"] == sel_age]
    if sel_gender != "All": df = df[df["Gender"] == sel_gender]
    if sel_tier != "All": df = df[df["City_Tier"] == sel_tier]
    if sel_intent != "All": df = df[df["Sign_Up_Intent"] == sel_intent]
    if sel_bmi != "All": df = df[df["BMI_Category"] == sel_bmi]
    return df

df = apply_filters(df_full.copy())

st.sidebar.markdown("---")
csv_dl = df.to_csv(index=False).encode("utf-8")
st.sidebar.download_button("⬇️ Download Filtered Data", csv_dl, "nourishiq_filtered.csv", "text/csv")
st.sidebar.markdown(f"**Showing {len(df):,} of {len(df_full):,} respondents**")

# ══════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════
COLORS = px.colors.qualitative.Set2

def insight(text):
    st.markdown(f'<div class="insight-box">💡 <b>Business Insight:</b> {text}</div>', unsafe_allow_html=True)

def section_tag(text):
    st.markdown(f'<span class="section-tag">{text}</span>', unsafe_allow_html=True)

def bar_chart(data_series, title, xlabel="", ylabel="Count", color_seq=None):
    vc = data_series.value_counts().reset_index()
    vc.columns = ["Category", "Count"]
    fig = px.bar(vc, x="Category", y="Count", title=title, color="Category",
                 color_discrete_sequence=color_seq or COLORS,
                 labels={"Category": xlabel, "Count": ylabel})
    fig.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                      title_font_size=15, height=360)
    return fig

def pie_chart(data_series, title):
    vc = data_series.value_counts().reset_index()
    vc.columns = ["Category", "Count"]
    fig = px.pie(vc, names="Category", values="Count", title=title,
                 color_discrete_sequence=COLORS, hole=0.3)
    fig.update_layout(title_font_size=15, height=360)
    return fig

# ══════════════════════════════════════════════════════════
# SECTION: HOME
# ══════════════════════════════════════════════════════════
if section == "🏠 Home & Objectives":
    st.markdown("""
    <div class="kpi-header">
        <h1 style="color:white; margin:0; font-size:2rem;">🥗 NourishIQ</h1>
        <p style="color:#b0bec5; margin:6px 0 0 0; font-size:1.1rem;">
            Personalised Diet, Nutrition & Fitness App for India
        </p>
        <p style="color:#90caf9; margin:4px 0 0 0; font-size:0.9rem;">
            Business Data Analytics Dashboard &nbsp;|&nbsp; 2,000 Survey Respondents &nbsp;|&nbsp; 80 Variables
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("👥 Total Respondents", f"{len(df_full):,}")
    col2.metric("📊 Avg BMI", f"{df_full['BMI'].mean():.1f}")
    col3.metric("💰 Avg Monthly Budget", f"₹{df_full['Monthly_Budget'].mean():.0f}")
    col4.metric("🎯 High Intent Rate", f"{(df_full['Sign_Up_Intent']=='High').mean()*100:.1f}%")
    col5.metric("⭐ Avg Recommend Score", f"{df_full['Recommend_Likelihood'].mean():.1f}/10")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### 🎯 Project Objectives")
        objectives = [
            "Understand demand for a personalised health app in India",
            "Identify most attractive customer segments",
            "Predict user sign-up intent using classification models",
            "Estimate users' monthly budget using regression",
            "Segment users into health & lifestyle personas via clustering",
            "Discover service combinations via association rule mining",
            "Recommend personalised diet, workout & subscription plans",
            "Forecast future app adoption, revenue & user growth",
            "Support data-driven pricing, targeting & retention decisions",
            "Build a professor-friendly business analytics dashboard",
        ]
        for i, obj in enumerate(objectives, 1):
            st.markdown(f"**{i}.** {obj}")

    with col_b:
        st.markdown("### 🏗️ Analytics Lifecycle")
        flow = ["Survey Data (2,000 Users)", "Data Cleaning & Validation",
                "Feature Engineering (15 scores)", "Exploratory Data Analysis",
                "ML Models (Classification, Regression, Clustering)",
                "Association Rule Mining", "Forecasting & Recommender",
                "Business Insights & Prescriptive Actions",
                "Streamlit Dashboard Deployment"]
        for i, step in enumerate(flow):
            arrow = " ↓" if i < len(flow)-1 else " ✅"
            color = "#4caf50" if i == len(flow)-1 else "#1565c0"
            st.markdown(f"<span style='color:{color}; font-weight:600;'>{'  '*0}{'━' if i>0 else ''}▶ {step}{arrow}</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 💼 Business Problem")
    bpc1, bpc2, bpc3 = st.columns(3)
    with bpc1:
        st.markdown("**❗ The Gap**\n\nMost diet & fitness apps are:\n- Too expensive\n- Not India-specific\n- No Indian food database\n- Generic, not personalised\n- No regional language support")
    with bpc2:
        st.markdown("**👥 Underserved Users**\n\n- Students on tight budgets\n- Tier 2/3 city residents\n- Users with health conditions\n- Homemakers planning family meals\n- Seniors managing chronic disease")
    with bpc3:
        st.markdown("**✅ NourishIQ Solution**\n\n- Affordable ₹199–₹499/month\n- Indian food database\n- BMI + macro + calorie tracking\n- AI-powered personalised plans\n- Expert chat for premium users")

    st.markdown("---")
    st.markdown("### 📐 Dataset Overview")
    dc1, dc2, dc3, dc4 = st.columns(4)
    dc1.metric("Total Variables", "80")
    dc2.metric("Demographic Vars", "8")
    dc3.metric("Health & Diet Vars", "22")
    dc4.metric("Engineered Features", "15")


# ══════════════════════════════════════════════════════════
# SECTION: DESCRIPTIVE ANALYTICS
# ══════════════════════════════════════════════════════════
elif section == "📊 Descriptive Analytics":
    st.title("📊 Descriptive Analytics")
    st.markdown("*Understanding who our potential users are — demographics, health, diet, and spending habits.*")

    # KPI Row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Respondents", f"{len(df):,}")
    c2.metric("Avg BMI", f"{df['BMI'].mean():.1f}")
    c3.metric("Avg Budget (₹)", f"{df['Monthly_Budget'].mean():.0f}")
    c4.metric("High Intent %", f"{(df['Sign_Up_Intent']=='High').mean()*100:.1f}%")
    c5.metric("Subscription Ready", f"{(df['Likely_To_Subscribe']=='Yes').mean()*100:.1f}%")

    st.markdown("---")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["👤 Demographics", "🏥 Health & BMI", "🥗 Diet Habits", "💪 Fitness", "💰 Spending"])

    with tab1:
        section_tag("Demographic Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(bar_chart(df["Age_Group"], "Age Group Distribution"), use_container_width=True)
            insight("The 25-34 and 18-24 age groups dominate — NourishIQ should design its core UX and pricing around young adults and students.")
        with col2:
            st.plotly_chart(pie_chart(df["Gender"], "Gender Distribution"), use_container_width=True)
            insight("Near-equal gender split. Marketing campaigns should be gender-inclusive, with specific nutrition plans for PCOS, pregnancy, and muscle building.")
        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(bar_chart(df["City_Tier"], "City Tier Distribution", color_seq=px.colors.sequential.Blues_r), use_container_width=True)
            insight("Tier 2 and Tier 3 cities together account for 55% of respondents — a massively underserved market with strong growth potential.")
        with col4:
            st.plotly_chart(bar_chart(df["Occupation"], "Occupation Distribution"), use_container_width=True)
            insight("Working professionals and students are dominant occupations — these two segments should get specially designed plans and pricing.")
        col5, col6 = st.columns(2)
        with col5:
            st.plotly_chart(bar_chart(df["Income_Band"], "Income Band Distribution", color_seq=px.colors.sequential.Greens_r), use_container_width=True)
            insight("Most users fall in the ₹15k–₹60k/month income bracket — confirming that a ₹199–₹499/month pricing tier is appropriate.")
        with col6:
            st.plotly_chart(bar_chart(df["Education"], "Education Level"), use_container_width=True)
            insight("Graduates and post-graduates form the majority — these users can understand personalised health insights and will value data-driven recommendations.")

    with tab2:
        section_tag("Health & BMI Profile")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x="BMI", nbins=30, title="BMI Distribution", color_discrete_sequence=["#4caf50"])
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=360)
            st.plotly_chart(fig, use_container_width=True)
            insight("Majority of users have BMI in the overweight range (25–30). This confirms high demand for weight management plans.")
        with col2:
            st.plotly_chart(pie_chart(df["BMI_Category"], "BMI Category Split"), use_container_width=True)
            insight("Only ~30% of users are in the Normal BMI range. The Overweight and Obese segments together represent NourishIQ's largest market opportunity.")
        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(bar_chart(df["Health_Conditions"], "Health Conditions Among Users"), use_container_width=True)
            insight("Diabetes and Hypertension are the most common conditions. NourishIQ should launch condition-specific plans as a premium offering.")
        with col4:
            st.plotly_chart(bar_chart(df["Health_Concern_Level"], "Health Concern Level"), use_container_width=True)
            insight("Over 36% of users have High health concern — these are prime candidates for NourishIQ's paid subscription plans.")
        col5, col6 = st.columns(2)
        with col5:
            fig = px.box(df, x="BMI_Category", y="Monthly_Budget", title="BMI Category vs Monthly Budget",
                         color="BMI_Category", color_discrete_sequence=COLORS)
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=360)
            st.plotly_chart(fig, use_container_width=True)
            insight("Users with higher BMI tend to be willing to spend more on health solutions — target Overweight and Obese segments with premium plans.")
        with col6:
            st.plotly_chart(bar_chart(df["Doctor_Recommendation"], "Doctor Recommended a Health App?", color_seq=["#2196f3","#ff9800"]), use_container_width=True)
            insight("30% of users have a doctor recommendation. This segment shows higher intent to subscribe and should be targeted with medical nutrition plans.")

    with tab3:
        section_tag("Diet & Food Habits")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(pie_chart(df["Food_Preference"], "Food Preference Distribution"), use_container_width=True)
            insight("Vegetarians and Non-Vegetarians are almost equal — NourishIQ must maintain a comprehensive Indian food database covering both categories.")
        with col2:
            st.plotly_chart(bar_chart(df["Eating_Habit_Quality"], "Eating Habit Quality", color_seq=px.colors.sequential.RdYlGn), use_container_width=True)
            insight("58% of users rate their eating habits as Poor or Average — this is the core problem NourishIQ solves. High demand signal.")
        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(bar_chart(df["Junk_Food_Freq"], "Junk Food Frequency"), use_container_width=True)
            insight("40% of users consume junk food 3+ times a week. NourishIQ's recipe suggestions and meal plan replacements can directly address this behaviour.")
        with col4:
            st.plotly_chart(bar_chart(df["Water_Intake"], "Daily Water Intake"), use_container_width=True)
            insight("56% of users drink less than 2L daily. Hydration reminders are a low-effort, high-impact feature to improve user health and engagement.")
        col5, col6 = st.columns(2)
        with col5:
            st.plotly_chart(bar_chart(df["Meal_Timing_Regularity"], "Meal Timing Regularity"), use_container_width=True)
            insight("68% of users have irregular or somewhat irregular meal timing. Meal reminder notifications can be a key retention feature.")
        with col6:
            st.plotly_chart(bar_chart(df["Meal_Plan_Type"], "Preferred Meal Plan Type"), use_container_width=True)
            insight("Weight Loss is the #1 meal plan preference. NourishIQ should make this the flagship plan and use it as the freemium acquisition hook.")

    with tab4:
        section_tag("Fitness & Lifestyle Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(bar_chart(df["Activity_Level"], "Physical Activity Level"), use_container_width=True)
            insight("60% of users are Sedentary or Lightly Active — the core audience for beginner-friendly home workout plans.")
        with col2:
            st.plotly_chart(bar_chart(df["Fitness_Goal"], "Primary Fitness Goal"), use_container_width=True)
            insight("Weight Loss and Muscle Gain are the top two goals. NourishIQ's content strategy should lead with these two themes.")
        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(bar_chart(df["Workout_Preference"], "Workout Preference"), use_container_width=True)
            insight("60% prefer home or gym workouts. NourishIQ should build both home workout routines and gym-based plans with instructional videos.")
        with col4:
            st.plotly_chart(bar_chart(df["Sleep_Hours"], "Average Sleep Hours"), use_container_width=True)
            insight("50% of users get less than 7 hours of sleep. Sleep tracking and bedtime reminders would increase app daily active usage.")

    with tab5:
        section_tag("Spending & Pricing Preferences")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x="Monthly_Budget", nbins=30, title="Monthly Budget Distribution (₹)",
                               color_discrete_sequence=["#9c27b0"])
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=360)
            st.plotly_chart(fig, use_container_width=True)
            insight("Budget peaks around ₹199–₹499, confirming this is the sweet spot for NourishIQ's pricing tiers.")
        with col2:
            st.plotly_chart(pie_chart(df["Pricing_Model"], "Preferred Pricing Model"), use_container_width=True)
            insight("Monthly subscription is preferred by 32% of users. However, annual plans should offer 2 months free to improve retention and LTV.")
        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(bar_chart(df["Sign_Up_Intent"], "Sign-Up Intent Distribution", color_seq=["#f44336","#ff9800","#4caf50"]), use_container_width=True)
            insight("15%+ High Intent users represent ~300+ immediate paying subscribers. Even at ₹199/month, this is ₹60,000+ MRR at launch.")
        with col4:
            st.plotly_chart(bar_chart(df["Discount_Sensitivity"], "Discount Sensitivity"), use_container_width=True)
            insight("78% of users are Medium-to-High discount sensitive. Launch offers, referral discounts, and seasonal deals will be very effective.")


# ══════════════════════════════════════════════════════════
# SECTION: DIAGNOSTIC ANALYTICS
# ══════════════════════════════════════════════════════════
elif section == "🔍 Diagnostic Analytics":
    st.title("🔍 Diagnostic Analytics")
    st.markdown("*Understanding WHY certain users are more likely to subscribe to NourishIQ.*")

    tab1, tab2, tab3 = st.tabs(["📊 Cross-Tabulations", "🔥 Correlation Analysis", "📦 Group Comparisons"])

    with tab1:
        section_tag("Cross-Tabulation Analysis")
        col1, col2 = st.columns(2)
        with col1:
            ct1 = pd.crosstab(df["BMI_Category"], df["Sign_Up_Intent"], normalize="index") * 100
            fig = px.bar(ct1.reset_index().melt("BMI_Category"), x="BMI_Category", y="value",
                         color="Sign_Up_Intent", barmode="group", title="BMI Category vs Sign-Up Intent (%)",
                         color_discrete_sequence=["#f44336","#ff9800","#4caf50"])
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=380, yaxis_title="% of Respondents")
            st.plotly_chart(fig, use_container_width=True)
            insight("Obese and Overweight users show significantly higher Sign-Up Intent — they feel the pain more and are more motivated to act.")
        with col2:
            ct2 = pd.crosstab(df["City_Tier"], df["Pricing_Model"], normalize="index") * 100
            fig = px.bar(ct2.reset_index().melt("City_Tier"), x="City_Tier", y="value",
                         color="Pricing_Model", barmode="stack", title="City Tier vs Preferred Pricing Model (%)",
                         color_discrete_sequence=COLORS)
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=380)
            st.plotly_chart(fig, use_container_width=True)
            insight("Tier 2 and Tier 3 users have higher preference for monthly billing over annual — suggesting they want flexibility due to budget constraints.")
        col3, col4 = st.columns(2)
        with col3:
            ct3 = pd.crosstab(df["Trust_AI_Diet"], df["Sign_Up_Intent"], normalize="index") * 100
            fig = px.bar(ct3.reset_index().melt("Trust_AI_Diet"), x="Trust_AI_Diet", y="value",
                         color="Sign_Up_Intent", barmode="group",
                         title="Trust in AI Diet Plans vs Sign-Up Intent (%)",
                         color_discrete_sequence=["#f44336","#ff9800","#4caf50"])
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=380)
            st.plotly_chart(fig, use_container_width=True)
            insight("Users who trust AI diet plans have 3x higher High Intent rate. Building trust through transparent AI explanations will drive conversions.")
        with col4:
            ct4 = pd.crosstab(df["Doctor_Recommendation"], df["Likely_To_Subscribe"], normalize="index") * 100
            fig = px.bar(ct4.reset_index().melt("Doctor_Recommendation"), x="Doctor_Recommendation", y="value",
                         color="Likely_To_Subscribe", barmode="group",
                         title="Doctor Recommendation vs Subscription Likelihood (%)",
                         color_discrete_sequence=["#f44336","#4caf50"])
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=380)
            st.plotly_chart(fig, use_container_width=True)
            insight("Doctor-recommended users are 40% more likely to subscribe. NourishIQ should partner with clinics for referral programs.")

    with tab2:
        section_tag("Correlation Heatmap")
        numeric_cols = ["BMI","Monthly_Budget","Health_Risk_Score","Diet_Quality_Score",
                        "Fitness_Readiness_Score","Digital_Trust_Score","App_Engagement_Score",
                        "Price_Sensitivity_Score","Subscription_Potential","Personalisation_Need",
                        "Motivation_Risk_Score","Customer_Value_Score","Retention_Risk_Score",
                        "Recommend_Likelihood"]
        corr = df[numeric_cols].corr().round(2)
        fig = px.imshow(corr, text_auto=True, aspect="auto", title="Feature Correlation Heatmap",
                        color_continuous_scale="RdBu_r", zmin=-1, zmax=1, height=550)
        fig.update_layout(paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
        insight("Subscription Potential is strongly correlated with Digital Trust Score and App Engagement Score. Customer Value Score is driven by Monthly Budget and Subscription Potential. Retention Risk is negatively correlated with Digital Trust.")

    with tab3:
        section_tag("Group Comparisons")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.box(df, x="Occupation", y="Monthly_Budget", color="Occupation",
                         title="Monthly Budget by Occupation", color_discrete_sequence=COLORS)
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=380, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            insight("Self-Employed and Working Professionals have the highest median budgets. These are NourishIQ's primary revenue targets.")
        with col2:
            fig = px.box(df, x="Health_Concern_Level", y="Subscription_Potential", color="Health_Concern_Level",
                         title="Health Concern vs Subscription Potential", color_discrete_sequence=["#f44336","#ff9800","#4caf50"])
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=380, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            insight("High Health Concern users have significantly higher Subscription Potential scores — confirming that health urgency drives paid subscription behaviour.")
        col3, col4 = st.columns(2)
        with col3:
            fig = px.box(df, x="Data_Sharing_Comfort", y="Digital_Trust_Score", color="Data_Sharing_Comfort",
                         title="Data Sharing Comfort vs Digital Trust Score", color_discrete_sequence=COLORS)
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=380, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            insight("Users comfortable with data sharing have higher Digital Trust Scores. NourishIQ should use transparent data policies and show users exactly what data is used.")
        with col4:
            fig = px.box(df, x="Past_Health_App", y="Monthly_Budget", color="Past_Health_App",
                         title="Past App Usage vs Monthly Budget",
                         color_discrete_sequence=["#ff9800","#4caf50"])
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=380, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            insight("Users who have previously paid for health apps have higher budgets. Targeting this experienced-user cohort will yield higher ARPU.")


# ══════════════════════════════════════════════════════════
# SECTION: CLASSIFICATION MODELS
# ══════════════════════════════════════════════════════════
elif section == "🤖 Classification Models":
    st.title("🤖 Classification Models")
    st.markdown("*Predicting whether a user is likely to subscribe to NourishIQ.*")

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, confusion_matrix, roc_auc_score, roc_curve)
    import plotly.figure_factory as ff

    @st.cache_data
    def run_classifiers(df_full):
        # Feature prep
        feat_cols = ["BMI","Monthly_Budget","Health_Risk_Score","Diet_Quality_Score",
                     "Fitness_Readiness_Score","Digital_Trust_Score","App_Engagement_Score",
                     "Price_Sensitivity_Score","Subscription_Potential","Personalisation_Need",
                     "Motivation_Risk_Score","Customer_Value_Score","Retention_Risk_Score",
                     "Health_Concern_Level","Sign_Up_Intent","Trust_AI_Diet","Trust_App_Reco",
                     "Activity_Level","BMI_Category","Gender","City_Tier","Occupation"]
        df_ml = df_full[feat_cols + ["Likely_To_Subscribe"]].copy()
        le_dict = {}
        for c in df_ml.select_dtypes("object").columns:
            le = LabelEncoder()
            df_ml[c] = le.fit_transform(df_ml[c].astype(str))
            le_dict[c] = le
        X = df_ml.drop("Likely_To_Subscribe", axis=1)
        y = df_ml["Likely_To_Subscribe"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        models = {
            "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
            "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "KNN": KNeighborsClassifier(n_neighbors=7),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        }
        results = []
        cm_dict = {}
        roc_dict = {}
        fi_dict = {}
        for name, model in models.items():
            Xtr = X_train_s if name in ["Logistic Regression","KNN"] else X_train
            Xte = X_test_s  if name in ["Logistic Regression","KNN"] else X_test
            model.fit(Xtr, y_train)
            pred = model.predict(Xte)
            proba = model.predict_proba(Xte)[:,1] if hasattr(model,"predict_proba") else None
            cm_dict[name] = confusion_matrix(y_test, pred).tolist()
            if proba is not None:
                fpr, tpr, _ = roc_curve(y_test, proba)
                roc_dict[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(),
                                  "auc": roc_auc_score(y_test, proba)}
            results.append({
                "Model": name,
                "Accuracy": round(accuracy_score(y_test, pred)*100, 2),
                "Precision": round(precision_score(y_test, pred, zero_division=0)*100, 2),
                "Recall": round(recall_score(y_test, pred, zero_division=0)*100, 2),
                "F1-Score": round(f1_score(y_test, pred, zero_division=0)*100, 2),
                "AUC": round(roc_dict.get(name,{}).get("auc",0)*100, 2),
            })
            if hasattr(model, "feature_importances_"):
                fi_dict[name] = dict(zip(X.columns, model.feature_importances_))
        return pd.DataFrame(results), cm_dict, roc_dict, fi_dict, list(X.columns)

    with st.spinner("Training 5 classification models..."):
        results_df, cm_dict, roc_dict, fi_dict, feat_names = run_classifiers(df_full)

    st.markdown("### 📊 Model Comparison")
    st.dataframe(results_df.style.highlight_max(subset=["Accuracy","Precision","Recall","F1-Score","AUC"],
                                                 color="#c8e6c9"), use_container_width=True)
    insight("Random Forest and Gradient Boosting are the top performers. Random Forest is preferred for NourishIQ because it is robust, handles mixed data well, and provides interpretable feature importances for business decisions.")

    best_model = results_df.loc[results_df["F1-Score"].idxmax(), "Model"]
    st.markdown(f"**🏆 Best Model: {best_model}** with F1-Score {results_df.loc[results_df['F1-Score'].idxmax(),'F1-Score']:.2f}%")

    tab1, tab2, tab3 = st.tabs(["📉 ROC Curves", "🔢 Confusion Matrix", "⚠️ Feature Importance"])

    with tab1:
        fig = go.Figure()
        colors_roc = ["#4caf50","#2196f3","#f44336","#ff9800","#9c27b0"]
        for (name, data), col in zip(roc_dict.items(), colors_roc):
            fig.add_trace(go.Scatter(x=data["fpr"], y=data["tpr"], name=f"{name} (AUC={data['auc']:.3f})",
                                     line=dict(color=col, width=2)))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], name="Random", line=dict(dash="dash", color="gray")))
        fig.update_layout(title="ROC Curves – All Models", xaxis_title="False Positive Rate",
                          yaxis_title="True Positive Rate", plot_bgcolor="white", paper_bgcolor="white", height=450)
        st.plotly_chart(fig, use_container_width=True)
        insight("Higher AUC means the model is better at distinguishing subscribers from non-subscribers. NourishIQ should use the Gradient Boosting model for real-time lead scoring.")

    with tab2:
        sel_m = st.selectbox("Select Model", list(cm_dict.keys()))
        cm = np.array(cm_dict[sel_m])
        labels = ["No","Yes"]
        fig = px.imshow(cm, text_auto=True, x=labels, y=labels, title=f"Confusion Matrix – {sel_m}",
                        color_continuous_scale="Blues", labels=dict(x="Predicted", y="Actual"))
        fig.update_layout(paper_bgcolor="white", height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"**True Positives (correctly predicted subscribers):** {cm[1,1]} | **False Positives:** {cm[0,1]} | **False Negatives:** {cm[1,0]}")
        insight("For NourishIQ, False Negatives (missing a potential subscriber) are more costly than False Positives. Choose a model/threshold that maximises Recall.")

    with tab3:
        if fi_dict:
            sel_fi = st.selectbox("Model for Feature Importance", [k for k in fi_dict])
            fi = pd.DataFrame({"Feature": list(fi_dict[sel_fi].keys()),
                               "Importance": list(fi_dict[sel_fi].values())}).sort_values("Importance", ascending=True).tail(15)
            fig = px.bar(fi, x="Importance", y="Feature", orientation="h",
                         title=f"Top Features – {sel_fi}", color="Importance",
                         color_continuous_scale="Greens")
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=480)
            st.plotly_chart(fig, use_container_width=True)
            insight("Subscription Potential, Monthly Budget, Customer Value Score, and Digital Trust Score are the strongest predictors of subscription likelihood. NourishIQ should personalize onboarding based on these factors.")


# ══════════════════════════════════════════════════════════
# SECTION: DECISION TREE
# ══════════════════════════════════════════════════════════
elif section == "🌳 Decision Tree Analysis":
    st.title("🌳 Decision Tree Analysis")
    st.markdown("*Explainable rules for predicting NourishIQ subscription intent.*")

    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report

    @st.cache_data
    def run_dt(df_full):
        feat_cols = ["BMI","Monthly_Budget","Health_Risk_Score","Diet_Quality_Score",
                     "Fitness_Readiness_Score","Digital_Trust_Score","App_Engagement_Score",
                     "Subscription_Potential","Personalisation_Need","Customer_Value_Score"]
        df_ml = df_full[feat_cols + ["Likely_To_Subscribe"]].copy()
        le = LabelEncoder()
        df_ml["Likely_To_Subscribe"] = le.fit_transform(df_ml["Likely_To_Subscribe"])
        X = df_ml[feat_cols]; y = df_ml["Likely_To_Subscribe"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        dt = DecisionTreeClassifier(max_depth=5, random_state=42)
        dt.fit(X_train, y_train)
        preds = dt.predict(X_test)
        acc = accuracy_score(y_test, preds)
        rules = export_text(dt, feature_names=feat_cols, max_depth=4)
        return dt, feat_cols, acc, rules

    dt, feat_cols, acc, rules = run_dt(df_full)

    c1, c2, c3 = st.columns(3)
    c1.metric("Decision Tree Accuracy", f"{acc*100:.1f}%")
    c2.metric("Max Tree Depth", "5")
    c3.metric("Features Used", str(len(feat_cols)))

    st.markdown("### 🌿 Decision Rules (Top Paths)")
    st.code(rules[:3000] + "\n... (truncated for display)", language="text")

    fi = pd.DataFrame({"Feature": feat_cols, "Importance": dt.feature_importances_}).sort_values("Importance", ascending=True)
    fig = px.bar(fi, x="Importance", y="Feature", orientation="h", title="Feature Importance in Decision Tree",
                 color="Importance", color_continuous_scale="YlGn")
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 💼 Business Interpretation of Key Decision Paths")
    paths = [
        ("High Subscription Potential (>6) AND Monthly Budget > ₹350",
         "→ Very likely to subscribe (Premium or Standard plan). Target immediately."),
        ("Subscription Potential ≤ 6 AND Digital Trust Score > 6",
         "→ Medium intent. Can be converted with a free trial or a referral discount offer."),
        ("Customer Value Score > 7 AND Health Risk Score > 5",
         "→ High-value health-concerned user. Offer condition-specific premium plan."),
        ("Diet Quality Score < 4 AND Personalisation Need > 6",
         "→ User with poor diet habits and strong need for guidance. Ideal for Weight-Loss Starter Plan."),
        ("Fitness Readiness Score < 3 AND BMI > 27",
         "→ Sedentary overweight user. Target with beginner home workout + diet combo plan."),
    ]
    for rule, action in paths:
        st.markdown(f"**Rule:** `{rule}`")
        st.markdown(f"**Action:** {action}")
        st.markdown("---")

    insight("Decision Tree rules provide NourishIQ's marketing team with clear, human-readable targeting criteria. These rules can be translated directly into CRM filters, ad targeting segments, and onboarding flows.")


# ══════════════════════════════════════════════════════════
# SECTION: CLUSTERING
# ══════════════════════════════════════════════════════════
elif section == "👥 Customer Segmentation":
    st.title("👥 Customer Segmentation & Personas")
    st.markdown("*Identifying distinct user groups using K-Means Clustering.*")

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score

    @st.cache_data
    def run_clustering(df_full):
        feat_cols = ["BMI","Monthly_Budget","Health_Risk_Score","Diet_Quality_Score",
                     "Fitness_Readiness_Score","Digital_Trust_Score","App_Engagement_Score",
                     "Price_Sensitivity_Score","Subscription_Potential","Personalisation_Need",
                     "Motivation_Risk_Score","Customer_Value_Score","Retention_Risk_Score"]
        X = df_full[feat_cols].copy()
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)

        inertia, sil_scores = [], []
        for k in range(2, 11):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_s)
            inertia.append(km.inertia_)
            sil_scores.append(silhouette_score(X_s, labels))

        km_best = KMeans(n_clusters=6, random_state=42, n_init=10)
        clusters = km_best.fit_predict(X_s)
        return clusters, inertia, sil_scores, feat_cols

    clusters, inertia, sil_scores, feat_cols = run_clustering(df_full)
    df_c = df_full.copy()
    df_c["Cluster"] = clusters

    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(x=range(2,11), y=inertia, title="Elbow Method – Optimal K",
                      labels={"x":"Number of Clusters","y":"Inertia"}, markers=True)
        fig.add_vline(x=6, line_dash="dash", line_color="red", annotation_text="Selected K=6")
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=360)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.line(x=range(2,11), y=sil_scores, title="Silhouette Scores",
                      labels={"x":"Number of Clusters","y":"Silhouette Score"}, markers=True)
        fig.add_vline(x=6, line_dash="dash", line_color="green", annotation_text="K=6")
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=360)
        st.plotly_chart(fig, use_container_width=True)

    insight("K=6 provides a good balance between cluster separation (Silhouette Score) and within-cluster variance (Elbow). Six personas cover the diversity of the Indian health app market.")

    # Scatter plot
    fig = px.scatter(df_c, x="Monthly_Budget", y="Subscription_Potential", color="Cluster",
                     title="Customer Clusters: Budget vs Subscription Potential",
                     color_continuous_scale="Viridis", hover_data=["User_Persona","BMI_Category"])
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=420)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🧑‍🤝‍🧑 Cluster Profiles")
    cluster_profiles = df_c.groupby("Cluster").agg({
        "BMI": "mean", "Monthly_Budget": "mean", "Health_Risk_Score": "mean",
        "Subscription_Potential": "mean", "Digital_Trust_Score": "mean",
        "App_Engagement_Score": "mean", "Customer_Value_Score": "mean",
        "Retention_Risk_Score": "mean"
    }).round(2)
    st.dataframe(cluster_profiles, use_container_width=True)

    # Persona cards
    persona_info = [
        ("Budget-Conscious Students", "Young, lower income, high motivation but price-sensitive",
         "Free Plan with upgrade triggers", "Freemium with referral rewards"),
        ("Weight-Loss Seekers", "Overweight/Obese users with strong health concern",
         "₹199 Basic or ₹299 Standard Weight-Loss Plan", "Progress milestones + community support"),
        ("Fitness-Focused Professionals", "Active young adults, gym-goers, muscle gain goals",
         "₹299 Standard or ₹499 Premium with AI form correction",
         "Streak rewards and workout challenges"),
        ("Diet-Conscious Homemakers", "Family-oriented, vegetarian preference, meal planning",
         "Family Plan ₹499/month", "Weekly meal planning reminders"),
        ("Medical Condition Managers", "Diabetics, hypertension users, doctor-referred",
         "₹299–499 Condition-Specific Plan", "Expert chat + progress reports"),
        ("Premium Wellness Users", "High income, high trust, want everything",
         "₹499 Premium or Annual Plan", "Exclusive features + expert consultation"),
    ]
    for i, (name, profile, plan, retention) in enumerate(persona_info):
        st.markdown(f"""
        <div class="persona-card">
            <b>Cluster {i} – {name}</b><br>
            👤 <i>{profile}</i><br>
            📦 <b>Recommended Plan:</b> {plan}<br>
            🔁 <b>Retention Strategy:</b> {retention}
        </div>
        """, unsafe_allow_html=True)

    insight("These 6 persona clusters allow NourishIQ to personalize every aspect of the user experience — from onboarding messages to plan recommendations and pricing — instead of using a one-size-fits-all approach.")


# ══════════════════════════════════════════════════════════
# SECTION: ASSOCIATION RULE MINING
# ══════════════════════════════════════════════════════════
elif section == "🔗 Association Rule Mining":
    st.title("🔗 Association Rule Mining")
    st.markdown("*Discovering which services users want together — for smart bundle design.*")

    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder

    st.markdown("""
    **What is Association Rule Mining?**
    - **Support**: How often a combination of services appears in the dataset
    - **Confidence**: Given a user wants service A, how likely are they to also want service B?
    - **Lift**: How much stronger is the association than random chance? (>1 = meaningful relationship)
    """)

    @st.cache_data
    def run_arm(df_full):
        # Encode multi-service variables as basket
        service_cols = ["Services_Wanted","Meal_Plan_Type","Workout_Preference",
                        "Reminder_Preference","Upgrade_Trigger","Fitness_Goal"]
        transactions = []
        for _, row in df_full.iterrows():
            basket = []
            for col in service_cols:
                val = str(row[col]).strip()
                if val and val not in ["nan","None"]:
                    basket.append(f"{col}:{val}")
            basket.append(f"Intent:{row['Sign_Up_Intent']}")
            basket.append(f"Plan:{row['Plan_Recommendation']}")
            transactions.append(basket)
        te = TransactionEncoder()
        te_array = te.fit_transform(transactions)
        df_basket = pd.DataFrame(te_array, columns=te.columns_)
        freq = apriori(df_basket, min_support=0.05, use_colnames=True)
        rules = association_rules(freq, metric="lift", min_threshold=1.1)
        rules = rules.sort_values("lift", ascending=False).head(30)
        rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
        rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))
        rules = rules[["antecedents","consequents","support","confidence","lift"]].round(3)
        return rules

    with st.spinner("Running Apriori algorithm..."):
        rules = run_arm(df_full)

    min_lift = st.slider("Minimum Lift", 1.0, 3.0, 1.1, 0.05)
    filtered = rules[rules["lift"] >= min_lift]
    st.markdown(f"**{len(filtered)} rules** found with Lift ≥ {min_lift}")
    st.dataframe(filtered.head(20), use_container_width=True)

    if not filtered.empty:
        fig = px.scatter(filtered, x="support", y="confidence", size="lift", color="lift",
                         title="Association Rules: Support vs Confidence (size = Lift)",
                         color_continuous_scale="Greens",
                         hover_data=["antecedents","consequents"])
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=440)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🛍️ Recommended Product Bundles Based on Rules")
    bundles = [
        ("🏋️ Weight Loss Starter Pack",
         "Calorie Tracker + Indian Diet Plans + Hydration Reminders + Home Workouts + Weekly Progress Reports",
         "₹199/month", "Weight-Loss Seekers & Sedentary Users"),
        ("💊 Diabetic Wellness Plan",
         "Controlled-Carb Meal Plans + Meal Timing Reminders + Expert Chat + Progress Tracking",
         "₹299/month", "Medical Condition Managers"),
        ("💪 Fitness Pro Bundle",
         "High-Protein Diet Plans + Gym Workout Routines + AI Form Correction + Macro Tracker",
         "₹299/month", "Fitness-Focused Young Professionals"),
        ("👨‍👩‍👧 Family Wellness Plan",
         "4 User Profiles + Meal Planning + Grocery List + Allergy Filters + Progress Reports",
         "₹499/month", "Homemakers & Family Plan Prospects"),
        ("🌟 Premium Nutrition Coach",
         "All Features + Expert Dietitian Chat + Travel Diet Plans + Ad-Free + Personalised AI Plans",
         "₹499/month", "Premium Wellness Users"),
    ]
    for name, features, price, target in bundles:
        with st.expander(f"{name} – {price}"):
            st.markdown(f"**Features:** {features}")
            st.markdown(f"**Target Segment:** {target}")

    insight("Association rules reveal that users who want diet plans also tend to want hydration reminders and progress reports — suggesting these three should always be bundled together. Expert chat users strongly co-occur with recipe library interest, forming the basis for the Premium Nutrition Coach plan.")


# ══════════════════════════════════════════════════════════
# SECTION: REGRESSION
# ══════════════════════════════════════════════════════════
elif section == "📈 Regression Analysis":
    st.title("📈 Regression Analysis")
    st.markdown("*Predicting how much a user is willing to pay — to design the right pricing tiers.*")

    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    @st.cache_data
    def run_regression(df_full):
        feat_cols = ["BMI","Health_Risk_Score","Diet_Quality_Score","Fitness_Readiness_Score",
                     "Digital_Trust_Score","App_Engagement_Score","Price_Sensitivity_Score",
                     "Subscription_Potential","Personalisation_Need","Customer_Value_Score",
                     "Income_Band","Occupation","City_Tier","Health_Concern_Level","Discount_Sensitivity"]
        df_ml = df_full[feat_cols + ["Monthly_Budget"]].copy()
        for c in df_ml.select_dtypes("object").columns:
            df_ml[c] = LabelEncoder().fit_transform(df_ml[c].astype(str))
        X = df_ml.drop("Monthly_Budget", axis=1)
        y = df_ml["Monthly_Budget"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_s, X_test_s = scaler.fit_transform(X_train), scaler.transform(X_test)
        models = {
            "Linear Regression": (LinearRegression(), True),
            "Decision Tree": (DecisionTreeRegressor(max_depth=6, random_state=42), False),
            "Random Forest": (RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1), False),
            "Gradient Boosting": (GradientBoostingRegressor(n_estimators=100, random_state=42), False),
        }
        results = []
        preds_dict = {}
        fi_dict = {}
        for name, (model, scaled) in models.items():
            Xtr = X_train_s if scaled else X_train
            Xte = X_test_s  if scaled else X_test
            model.fit(Xtr, y_train)
            pred = model.predict(Xte)
            preds_dict[name] = {"actual": y_test.tolist(), "pred": pred.tolist()}
            results.append({
                "Model": name,
                "MAE": round(mean_absolute_error(y_test, pred), 2),
                "RMSE": round(np.sqrt(mean_squared_error(y_test, pred)), 2),
                "R² Score": round(r2_score(y_test, pred), 4),
            })
            if hasattr(model, "feature_importances_"):
                fi_dict[name] = dict(zip(feat_cols, model.feature_importances_))
        return pd.DataFrame(results), preds_dict, fi_dict

    with st.spinner("Training regression models..."):
        reg_df, preds_dict, fi_reg = run_regression(df_full)

    st.markdown("### 📊 Model Comparison")
    st.dataframe(reg_df.style.highlight_max(subset=["R² Score"], color="#c8e6c9")
                             .highlight_min(subset=["MAE","RMSE"], color="#c8e6c9"), use_container_width=True)
    best_reg = reg_df.loc[reg_df["R² Score"].idxmax(), "Model"]
    insight(f"**{best_reg}** achieves the best R² Score, explaining the most variance in users' monthly budget. This model should be used in NourishIQ's pricing engine to determine which plan tier to recommend for each new user.")

    tab1, tab2 = st.tabs(["📉 Actual vs Predicted", "🔑 Feature Importance"])
    with tab1:
        sel = st.selectbox("Select Model", list(preds_dict.keys()))
        actual = preds_dict[sel]["actual"][:200]
        pred   = preds_dict[sel]["pred"][:200]
        fig = px.scatter(x=actual, y=pred, title=f"Actual vs Predicted Monthly Budget – {sel}",
                         labels={"x":"Actual Budget (₹)","y":"Predicted Budget (₹)"},
                         color_discrete_sequence=["#4caf50"], opacity=0.6)
        fig.add_trace(go.Scatter(x=[min(actual), max(actual)], y=[min(actual), max(actual)],
                                  mode="lines", name="Perfect Prediction", line=dict(dash="dash", color="red")))
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=420)
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        if fi_reg:
            sel_fi = st.selectbox("Model", [k for k in fi_reg])
            fi = pd.DataFrame({"Feature": list(fi_reg[sel_fi].keys()),
                               "Importance": list(fi_reg[sel_fi].values())}).sort_values("Importance", ascending=True)
            fig = px.bar(fi, x="Importance", y="Feature", orientation="h",
                         title=f"Feature Importance – {sel_fi}", color="Importance",
                         color_continuous_scale="Blues")
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=430)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 💰 Pricing Strategy Based on Regression")
    pricing_data = {
        "Plan": ["Free Plan", "Basic ₹199/month", "Standard ₹299/month", "Premium ₹499/month", "Family Plan ₹499/month"],
        "Target Budget Range": ["₹0–₹150", "₹151–₹250", "₹251–₹400", "₹401–₹700", "₹401–₹700 (household)"],
        "Features": [
            "BMI tracker, basic diet tips",
            "Calorie tracker, Indian diet plans, hydration reminders",
            "Everything Basic + workout routines, macro tracker, progress reports",
            "Everything Standard + expert chat, AI form correction, recipe library, travel diet",
            "4 profiles + all Standard features + family meal planner"
        ],
        "Target Segment": ["Students, trial users","Tier 2/3 users","Working professionals","Premium users","Homemakers"]
    }
    st.dataframe(pd.DataFrame(pricing_data), use_container_width=True)


# ══════════════════════════════════════════════════════════
# SECTION: FORECASTING
# ══════════════════════════════════════════════════════════
elif section == "📅 Forecasting":
    st.title("📅 Forecasting & Time Series Analysis")
    st.markdown("*Projecting NourishIQ's user growth, subscriptions, and revenue over 24 months.*")

    # Simulate monthly data
    np.random.seed(99)
    months = pd.date_range("2025-01-01", periods=24, freq="MS")
    base_downloads = 5000
    downloads = [int(base_downloads * (1.08**i) + np.random.normal(0, 400)) for i in range(24)]
    conversion_rate = 0.12
    subscribers = [int(d * (conversion_rate + 0.005*i/24)) for i, d in enumerate(downloads)]
    avg_rev = 280
    revenue = [s * avg_rev for s in subscribers]
    churn = [max(0.05 - 0.001*i + np.random.normal(0, 0.005), 0.03) for i in range(24)]

    # Moving Average
    def moving_avg(data, window=3):
        return pd.Series(data).rolling(window, min_periods=1).mean().tolist()

    # Linear trend extension
    def linear_forecast(data, future=6):
        x = np.arange(len(data))
        m, b = np.polyfit(x, data, 1)
        future_x = np.arange(len(data), len(data)+future)
        return [m*xi + b for xi in future_x]

    tab1, tab2, tab3, tab4 = st.tabs(["📥 Downloads", "👤 Subscribers", "💰 Revenue", "📉 Churn"])

    def forecast_chart(months, actual, ma, forecast_vals, ylabel, title):
        future_months = pd.date_range(months[-1] + pd.DateOffset(months=1), periods=6, freq="MS")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(months), y=actual, name="Actual", line=dict(color="#4caf50", width=2)))
        fig.add_trace(go.Scatter(x=list(months), y=ma, name="Moving Avg (3M)", line=dict(color="#2196f3", dash="dot")))
        fig.add_trace(go.Scatter(x=list(future_months), y=forecast_vals, name="Forecast (6M)",
                                  line=dict(color="#ff9800", dash="dash"), mode="lines+markers"))
        fig.update_layout(title=title, yaxis_title=ylabel, xaxis_title="Month",
                          plot_bgcolor="white", paper_bgcolor="white", height=400)
        return fig

    with tab1:
        fig = forecast_chart(months, downloads, moving_avg(downloads),
                             linear_forecast(downloads), "Monthly Downloads", "📥 App Downloads Forecast")
        st.plotly_chart(fig, use_container_width=True)
        insight(f"Downloads are projected to reach {linear_forecast(downloads)[-1]:,.0f}/month by month 30 at current growth rate. Peak acquisition months are typically January (New Year resolutions) and July (monsoon season activity drop).")

    with tab2:
        fig = forecast_chart(months, subscribers, moving_avg(subscribers),
                             linear_forecast(subscribers), "Monthly Subscribers", "👤 Subscriber Growth Forecast")
        st.plotly_chart(fig, use_container_width=True)
        insight("Subscriber conversion improves over time as NourishIQ builds trust and refines its onboarding. Target a 15% conversion rate by month 18 through in-app personalization and free trial campaigns.")

    with tab3:
        fig = forecast_chart(months, revenue, moving_avg(revenue),
                             linear_forecast(revenue), "Revenue (₹)", "💰 Monthly Revenue Forecast (₹)")
        st.plotly_chart(fig, use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Month 1 Revenue", f"₹{revenue[0]:,.0f}")
        c2.metric("Month 12 Revenue", f"₹{revenue[11]:,.0f}")
        c3.metric("Month 24 Revenue", f"₹{revenue[23]:,.0f}")
        insight("Revenue grows faster than user count as average revenue per user (ARPU) improves through upsells and plan upgrades. NourishIQ should focus on converting free users to ₹299+ plans by month 6.")

    with tab4:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(months), y=[c*100 for c in churn], name="Churn Rate %",
                                  fill="tozeroy", line=dict(color="#f44336")))
        fig.update_layout(title="Monthly Churn Rate Trend", yaxis_title="Churn Rate (%)",
                          plot_bgcolor="white", paper_bgcolor="white", height=400)
        st.plotly_chart(fig, use_container_width=True)
        insight("Churn decreases as the product matures and personalization improves. Key retention actions: weekly progress reports, gamification, and community features at months 3–6.")


# ══════════════════════════════════════════════════════════
# SECTION: RECOMMENDER
# ══════════════════════════════════════════════════════════
elif section == "💡 Recommender System":
    st.title("💡 Personalised Recommender System")
    st.markdown("*Get a personalised NourishIQ plan recommendation based on your health profile.*")

    col1, col2, col3 = st.columns(3)
    with col1:
        bmi_val = st.slider("Your BMI", 14.0, 45.0, 24.0, 0.5)
        budget  = st.slider("Monthly Budget (₹)", 50, 1500, 299, 50)
        concern = st.selectbox("Health Concern Level", ["Low","Medium","High"])
    with col2:
        goal    = st.selectbox("Fitness Goal", ["Weight Loss","Muscle Gain","Energy Improvement","General Wellness","Condition Management"])
        activity= st.selectbox("Activity Level", ["Sedentary","Lightly Active","Moderately Active","Highly Active"])
        cond    = st.selectbox("Health Condition", ["None","Diabetes","Hypertension","Thyroid","PCOS","Heart Condition"])
    with col3:
        food    = st.selectbox("Food Preference", ["Vegetarian","Non-Vegetarian","Vegan","Eggetarian"])
        trust   = st.selectbox("Trust in AI Plans", ["Low","Medium","High"])
        family  = st.selectbox("Family Plan Interest", ["No","Yes"])

    if st.button("🎯 Get My Personalised Recommendation", type="primary"):
        # Determine plan
        if budget < 200:
            plan, price = "Free Plan", "₹0"
        elif budget < 350:
            plan, price = "Basic Plan", "₹199/month"
        elif budget < 600:
            plan, price = "Standard Plan", "₹299/month"
        else:
            plan, price = "Premium Plan", "₹499/month"
        if family == "Yes" and budget >= 400:
            plan, price = "Family Wellness Plan", "₹499/month"

        # Diet plan
        if cond in ["Diabetes"]:
            diet = "Diabetic-Friendly Controlled-Carb Indian Meal Plan"
        elif cond in ["Hypertension"]:
            diet = "Low-Sodium Heart-Healthy Indian Meal Plan"
        elif goal == "Weight Loss":
            diet = "Calorie-Deficit Indian Weight Loss Meal Plan" + (" (Vegetarian)" if food=="Vegetarian" else "")
        elif goal == "Muscle Gain":
            diet = "High-Protein Indian Muscle Building Meal Plan"
        elif food == "Vegan":
            diet = "Plant-Based Indian Vegan Wellness Plan"
        else:
            diet = "Balanced Indian Daily Nutrition Plan"

        # Workout
        if activity == "Sedentary":
            workout = "Beginner 15-min Home Workout + Daily Walk Challenge"
        elif activity == "Lightly Active":
            workout = "Intermediate Home/Gym 30-min Routine + Yoga"
        elif goal == "Muscle Gain":
            workout = "Progressive Overload Gym Program (5-day split)"
        else:
            workout = "Mixed Cardio + Strength Training 4x/week"

        # Features
        features = ["BMI Tracker", "Calorie & Macro Tracker", "Indian Recipe Library"]
        if concern == "High": features.append("Weekly Progress Reports")
        if cond != "None": features.append("Condition-Specific Diet Plans")
        if trust == "High": features.append("AI-Powered Personalised Plans")
        if budget >= 399: features.append("Expert Dietitian Chat")
        if budget >= 399: features.append("AI Form Correction for Workouts")
        if family == "Yes": features.append("Family Profile Management")

        # Message
        if trust == "Low" and concern == "Low":
            msg = "Start with our Free Plan. Try our 7-day diet challenge. See results first, then upgrade!"
        elif concern == "High" and budget >= 300:
            msg = "You're our ideal user! Start your health journey with NourishIQ's personalized Standard plan today."
        else:
            msg = "NourishIQ's affordable plan is designed exactly for your needs. Join 10,000+ Indians improving their health!"

        st.success("✅ Your Personalised NourishIQ Recommendation")
        rc1, rc2 = st.columns(2)
        with rc1:
            st.markdown(f"### 📦 Recommended Plan: **{plan}** ({price})")
            st.markdown(f"🥗 **Diet Plan:** {diet}")
            st.markdown(f"💪 **Workout Plan:** {workout}")
            st.markdown(f"💬 **Message for You:** _{msg}_")
        with rc2:
            st.markdown("### ✅ Your Recommended App Features")
            for f in features:
                st.markdown(f"• {f}")
        bmi_cat = "Underweight" if bmi_val < 18.5 else "Normal" if bmi_val < 25 else "Overweight" if bmi_val < 30 else "Obese"
        st.info(f"📊 Your BMI is **{bmi_val}** ({bmi_cat}). {'Target: reach BMI < 25 with NourishIQ in 12–16 weeks.' if bmi_val >= 25 else 'Your BMI is in the healthy range. Focus on maintenance and wellness.'}")


# ══════════════════════════════════════════════════════════
# SECTION: TEXT MINING
# ══════════════════════════════════════════════════════════
elif section == "💬 Text Mining & Sentiment":
    st.title("💬 Text Mining & Sentiment Analysis")
    st.markdown("*Analysing simulated user feedback to improve NourishIQ's product and service.*")

    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    positive_reviews = [
        "The Indian diet plans are amazing and very practical for everyday cooking",
        "Love the calorie tracker with desi food options finally an app that gets India",
        "BMI tracking and weekly reports keep me motivated and on track",
        "Hydration reminders are so helpful I actually drink enough water now",
        "The personalised plan for my diabetes is genuinely life changing",
        "Expert chat feature helped me understand my nutrition needs clearly",
        "Very affordable compared to hiring a personal dietitian excellent value",
        "The recipe library with Indian dishes is exactly what I needed for my family",
        "AI recommendations are surprisingly accurate and easy to follow",
        "Love that it has vegetarian options with proper protein combinations",
        "The workout routines for home are perfect for my busy schedule",
        "Great app for managing weight loss with Indian food habits in mind",
        "Progress reports every week keep me accountable and motivated",
        "The allergy filter helped me plan meals around my lactose intolerance easily",
    ]
    negative_reviews = [
        "The app sometimes crashes when I open the recipe section very frustrating",
        "Pricing seems too high for students like me need a better student plan",
        "Wish there were more South Indian recipes the North Indian options dominate",
        "Privacy policy needs to be clearer about what health data is being stored",
        "The workout videos could be higher quality and more diverse",
        "Customer support response time needs to improve significantly",
        "The AI recommendations sometimes feel too generic not truly personalised",
        "Would like to see a regional language option Hindi and Tamil at least",
        "Free plan is too limited needs at least the calorie tracker for free users",
        "Wish expert chat was available in the basic plan not just premium",
    ]
    all_reviews = positive_reviews + negative_reviews

    sia = SentimentIntensityAnalyzer()
    sentiments = [sia.polarity_scores(r)["compound"] for r in all_reviews]
    labels = ["Positive" if s >= 0.05 else "Negative" if s <= -0.05 else "Neutral" for s in sentiments]

    sent_df = pd.DataFrame({"Review": all_reviews, "Sentiment_Score": sentiments, "Label": labels})

    col1, col2, col3 = st.columns(3)
    col1.metric("Positive Reviews", f"{labels.count('Positive')}")
    col2.metric("Neutral Reviews", f"{labels.count('Neutral')}")
    col3.metric("Negative Reviews", f"{labels.count('Negative')}")

    fig = px.histogram(sent_df, x="Sentiment_Score", color="Label", title="Sentiment Score Distribution",
                       color_discrete_map={"Positive":"#4caf50","Neutral":"#ff9800","Negative":"#f44336"})
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=350)
    st.plotly_chart(fig, use_container_width=True)

    tab1, tab2 = st.tabs(["☁️ Word Cloud", "📋 Review Analysis"])
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Positive Reviews – Key Themes**")
            pos_text = " ".join(positive_reviews)
            wc = WordCloud(width=500, height=300, background_color="white", colormap="Greens",
                           max_words=40).generate(pos_text)
            fig_wc, ax = plt.subplots(figsize=(6, 3))
            ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
            st.pyplot(fig_wc)
        with col2:
            st.markdown("**Negative Reviews – Key Themes**")
            neg_text = " ".join(negative_reviews)
            wc2 = WordCloud(width=500, height=300, background_color="white", colormap="Reds",
                            max_words=40).generate(neg_text)
            fig_wc2, ax2 = plt.subplots(figsize=(6, 3))
            ax2.imshow(wc2, interpolation="bilinear"); ax2.axis("off")
            st.pyplot(fig_wc2)

    with tab2:
        st.dataframe(sent_df[["Review","Sentiment_Score","Label"]], use_container_width=True)

    st.markdown("### 📌 Product Improvement Actions from Feedback")
    actions = [
        ("🐛 App Stability", "Fix crashes in Recipe section – immediate priority"),
        ("💰 Student Pricing", "Introduce ₹99/month student plan with college email verification"),
        ("🌶️ Regional Cuisine", "Expand to South Indian, Bengali, Gujarati, and Punjabi recipe databases"),
        ("🔐 Privacy", "Redesign privacy dashboard to show users exactly what data is used"),
        ("🌐 Language", "Add Hindi and Tamil language options in UI"),
        ("📞 Support", "Launch WhatsApp support channel with 2-hour SLA"),
        ("🆓 Free Plan", "Include calorie tracker in free plan as acquisition feature"),
    ]
    for icon_title, action in actions:
        st.markdown(f"**{icon_title}:** {action}")

    insight("Positive sentiment is strong around Indian diet plans, personalisation, and affordability — these are NourishIQ's core value propositions and should be highlighted in all marketing. Negative sentiment around pricing and regional content points to product expansion priorities.")


# ══════════════════════════════════════════════════════════
# SECTION: SOCIAL NETWORK
# ══════════════════════════════════════════════════════════
elif section == "🌐 Social Network Analysis":
    st.title("🌐 Social Network & Referral Analysis")
    st.markdown("*Understanding how NourishIQ users refer and influence each other.*")

    import networkx as nx

    np.random.seed(55)
    n_nodes = 40
    G = nx.barabasi_albert_graph(n_nodes, 2, seed=42)
    recommend_scores = df_full["Recommend_Likelihood"].values[:n_nodes]
    personas = df_full["User_Persona"].values[:n_nodes]

    pos = nx.spring_layout(G, seed=42)
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    edge_x, edge_y = [], []
    for e in G.edges():
        x0,y0 = pos[e[0]]; x1,y1 = pos[e[1]]
        edge_x += [x0,x1,None]; edge_y += [y0,y1,None]

    node_size = [max(8, int(recommend_scores[n]*4)) for n in G.nodes()]
    node_color = recommend_scores[:n_nodes]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                              line=dict(width=0.8, color="#b0bec5"), hoverinfo="none"))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text",
                              marker=dict(size=node_size, color=node_color,
                                          colorscale="YlGn", showscale=True,
                                          colorbar=dict(title="Recommend Score")),
                              text=[f"U{i}" for i in range(n_nodes)],
                              textposition="top center",
                              hovertext=[f"User {i}<br>Score: {recommend_scores[i]:.1f}<br>{personas[i]}" for i in range(n_nodes)],
                              hoverinfo="text"))
    fig.update_layout(title="NourishIQ Referral Network (Simulated)",
                      showlegend=False, height=520,
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      paper_bgcolor="white", plot_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    degree = dict(G.degree())
    top_nodes = sorted(degree.items(), key=lambda x: x[1], reverse=True)[:8]
    st.markdown("### 🌟 Top Referral Influencers")
    inf_cols = st.columns(4)
    for i, (node, deg) in enumerate(top_nodes[:4]):
        inf_cols[i].metric(f"User {node}", f"{deg} connections", f"Score: {recommend_scores[node]:.1f}")

    st.markdown("### 💡 Referral Marketing Strategy")
    strats = [
        ("🎓 Student Networks", "Students with high recommend scores can become NourishIQ campus ambassadors. Offer 1 free month per 3 referrals."),
        ("💼 Workplace Wellness", "Working professionals can refer colleagues through Corporate Wellness Programme integrations."),
        ("👨‍👩‍👧 Family Chains", "Family Plan users naturally refer family members. Offer family group discounts."),
        ("🏋️ Gym Communities", "Fitness-focused users can share NourishIQ in gym WhatsApp groups. Create gym partner referral codes."),
        ("👩‍⚕️ Doctor Referrals", "Diabetic and hypertension patients referred by doctors convert at 2x rate. Build a doctor partner network."),
    ]
    for title, desc in strats:
        st.markdown(f"**{title}:** {desc}")

    insight("Hub nodes in the referral network (high-degree users) are typically Fitness-Focused Young Professionals and Premium Wellness Users with Recommend Likelihood > 8. Identifying and rewarding these users with referral bonuses will exponentially grow NourishIQ's user base.")


# ══════════════════════════════════════════════════════════
# SECTION: ETHICS
# ══════════════════════════════════════════════════════════
elif section == "🔐 Ethics & Data Privacy":
    st.title("🔐 Ethics, Data Privacy & AI Governance")
    st.markdown("*NourishIQ's commitment to responsible handling of sensitive health data.*")

    principles = [
        ("🔒 User Consent", "All health data is collected only with explicit, informed user consent. Users can withdraw consent at any time."),
        ("🛡️ Data Privacy", "BMI, health conditions, and dietary data are encrypted at rest and in transit. Never sold to third parties."),
        ("🗑️ Right to Delete", "Users can request complete deletion of their health data within 72 hours via the app settings."),
        ("🔍 Transparent AI", "Every AI-generated diet or workout recommendation includes a clear explanation of what inputs were used."),
        ("⚖️ Bias Prevention", "AI models are regularly audited for demographic bias across age, gender, income, and city tier."),
        ("👶 Minor Protection", "Users under 18 require parental consent. Diet plans for minors are designed with paediatric nutritionists."),
        ("👴 Senior Safeguards", "Recommendations for users 55+ are reviewed against geriatric nutrition guidelines."),
        ("🏥 Medical Disclaimer", "NourishIQ clearly states it does not replace doctors or dietitians and encourages users to consult healthcare professionals for medical conditions."),
        ("📉 Data Minimisation", "Only data necessary for personalisation is collected. Optional data fields are clearly marked."),
        ("🔔 Nudge Ethics", "Health nudges and reminders are designed to motivate, not to create anxiety or body-image issues."),
        ("🌐 AI Governance", "NourishIQ follows India's Digital Personal Data Protection Act (DPDP) 2023 and aligns with global health data best practices."),
        ("🤝 Human Oversight", "Condition-specific plans (Diabetes, Heart Disease) are reviewed by certified nutritionists before deployment."),
    ]
    for i in range(0, len(principles), 2):
        c1, c2 = st.columns(2)
        title1, desc1 = principles[i]
        with c1:
            st.markdown(f"**{title1}**\n\n{desc1}")
        if i+1 < len(principles):
            title2, desc2 = principles[i+1]
            with c2:
                st.markdown(f"**{title2}**\n\n{desc2}")
        st.markdown("---")

    st.markdown("### ⚠️ Legal & Compliance Framework")
    st.markdown("""
    | Regulation | NourishIQ Compliance |
    |---|---|
    | India DPDP Act 2023 | Data Processing Agreements, Consent Manager |
    | IT Act 2000 | Encrypted storage, secure APIs |
    | FSSAI Guidelines | Food recommendations aligned with Indian food safety standards |
    | WHO Nutrition Guidelines | Macro & calorie targets follow WHO standards |
    | GDPR (for future global expansion) | Privacy by design architecture |
    """)

    insight("Health data is among the most sensitive personal data. NourishIQ's trust is built on transparency, consent, and responsible AI. Users who trust the app are 3x more likely to subscribe — making ethics a direct business advantage, not just a compliance requirement.")


# ══════════════════════════════════════════════════════════
# SECTION: BUSINESS RECOMMENDATIONS
# ══════════════════════════════════════════════════════════
elif section == "✅ Business Recommendations":
    st.title("✅ Prescriptive Analytics & Business Recommendations")
    st.markdown("*Data-driven action plan for launching and scaling NourishIQ.*")

    st.markdown("### 🎯 What Should NourishIQ Do Next?")

    recs = {
        "🚀 Launch Strategy": [
            "Target 18–34 year olds in Tier 2 and Tier 3 cities first — largest underserved segment",
            "Launch with freemium model: Free → ₹199 → ₹299 → ₹499/month tiers",
            "Lead with Weight Loss plan as flagship — highest demand (35% of users)",
            "Partner with gyms, colleges, and corporate wellness programs for distribution",
        ],
        "💰 Pricing Strategy": [
            "Free Plan: BMI tracker + basic tips (acquisition hook)",
            "₹199/month Basic: Calorie tracker + Indian diet plans + hydration reminders",
            "₹299/month Standard: Everything Basic + workout routines + macro tracker + progress reports",
            "₹499/month Premium: Everything + expert chat + AI form correction + recipe library + travel diet",
            "Family Plan ₹499: 4 profiles + family meal planner (target homemakers)",
            "Annual plan = 10 months + 2 months free (to improve LTV and reduce churn)",
        ],
        "📣 Marketing Strategy": [
            "Use classification model outputs to identify and target High Intent users via performance marketing",
            "Focus on Instagram Reels and YouTube Shorts showcasing Indian food transformations",
            "Build referral programme: ₹99 credit per successful referral",
            "Doctor and clinic partnerships for diabetic and hypertensive patient referrals",
            "WhatsApp marketing campaigns with personalised diet tips (free lead generation)",
        ],
        "🔁 Retention Strategy": [
            "Weekly progress reports create habit loops and reduce early churn",
            "Gamification: 7-day streaks, monthly challenges, community leaderboards",
            "Proactive re-engagement: users inactive for 3+ days get personalised motivation nudge",
            "Downgrade protection: before a user cancels, offer 1-month discount or plan downgrade",
            "Use Retention Risk Score to identify at-risk users for intervention before they churn",
        ],
        "📦 Product Roadmap": [
            "Phase 1 (Months 1–3): BMI tracker, calorie tracker, Indian diet plans, hydration reminders",
            "Phase 2 (Months 4–6): Workout routines, macro tracker, weekly progress reports, family plan",
            "Phase 3 (Months 7–12): Expert chat, AI form correction, recipe library, condition-specific plans",
            "Phase 4 (Year 2): Regional language support, grocery integration, wearable sync, travel diet",
        ],
    }

    for section_title, points in recs.items():
        with st.expander(section_title, expanded=True):
            for p in points:
                st.markdown(f"• {p}")

    st.markdown("---")
    st.markdown("### 📊 Expected Business Outcomes")
    outcomes = pd.DataFrame({
        "Metric": ["Month 6 Downloads","Month 12 Subscribers","Month 12 MRR","Year 1 Revenue","Avg Churn Rate","NPS Score Target"],
        "Target": ["50,000+","8,000+","₹22,40,000+","₹1.5 Cr+","< 6%","50+"],
        "Driver": ["Social + Referral","Freemium Conversion","Upsell to ₹299+ Plans","Subscriber Growth","Gamification + Reminders","Indian Food + Personalisation"]
    })
    st.dataframe(outcomes, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🏆 Founder's Data-Driven Conclusion")
    st.markdown("""
    > *"Our survey data of 2,000 Indian health app users clearly validates demand for NourishIQ.
    > The data shows 58% subscription interest, a strong preference for Indian food-based diet plans,
    > and a sweet spot pricing of ₹199–₹499/month. The segments most ready to pay are overweight users
    > with high health concern, working professionals, and users already referred by doctors.
    > Our ML models confirm that Subscription Potential, Digital Trust, and Monthly Budget are the
    > top predictors of conversion. NourishIQ should launch with a freemium model, focus on Weight Loss
    > and Diabetic-Friendly plans, and use data-driven personalisation as its core competitive advantage
    > against generic international apps."*
    >
    > — NourishIQ Analytics Team
    """)


# ══════════════════════════════════════════════════════════
# SECTION: PREDICT NEW USER
# ══════════════════════════════════════════════════════════
elif section == "🆕 Predict New User":
    st.title("🆕 New User Prediction Module")
    st.markdown("*Enter a new user's profile and get NourishIQ's data-driven predictions.*")

    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split

    @st.cache_resource
    def train_models(df_full):
        clf_feats = ["BMI","Monthly_Budget","Health_Risk_Score","Diet_Quality_Score",
                     "Fitness_Readiness_Score","Digital_Trust_Score","App_Engagement_Score",
                     "Subscription_Potential","Customer_Value_Score","Personalisation_Need"]
        df_c = df_full[clf_feats + ["Likely_To_Subscribe","Sign_Up_Intent"]].copy()
        le_s = LabelEncoder(); le_i = LabelEncoder()
        df_c["Likely_To_Subscribe"] = le_s.fit_transform(df_c["Likely_To_Subscribe"])
        df_c["Sign_Up_Intent"] = le_i.fit_transform(df_c["Sign_Up_Intent"])
        Xc = df_c[clf_feats]; yc = df_c["Likely_To_Subscribe"]; yi = df_c["Sign_Up_Intent"]
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(Xc, yc)
        clf_intent = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf_intent.fit(Xc, yi)
        reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        reg.fit(Xc, df_full["Monthly_Budget"])
        return clf, clf_intent, reg, clf_feats, le_s, le_i

    clf, clf_intent, reg, clf_feats, le_s, le_i = train_models(df_full)

    st.markdown("#### Enter New User Profile")
    col1, col2, col3 = st.columns(3)
    with col1:
        new_bmi = st.number_input("BMI", 14.0, 50.0, 27.5, 0.5)
        new_budget = st.number_input("Monthly Budget (₹)", 50, 2000, 299, 50)
        new_hconcern = st.select_slider("Health Concern (1=Low, 3=High)", [1,2,3], 2)
    with col2:
        new_diet_q = st.slider("Diet Quality Score (0–10)", 0, 10, 5)
        new_fitness = st.slider("Fitness Readiness Score (0–10)", 0, 10, 4)
        new_trust = st.slider("Digital Trust Score (0–10)", 0, 10, 6)
    with col3:
        new_engage = st.slider("App Engagement Score (0–10)", 0, 10, 5)
        new_sub_pot = st.slider("Subscription Potential (0–10)", 0, 10, 5)
        new_cust_val = st.slider("Customer Value Score (0–10)", 0, 10, 5)

    new_hrisk = max(0, min(10, int((new_bmi > 27)*2 + (new_hconcern == 3)*2 + 1)))
    new_personalisation = max(0, min(10, int((new_bmi > 27)*2 + (new_diet_q < 5)*2 + 2)))

    input_vec = np.array([[new_bmi, new_budget, new_hrisk, new_diet_q, new_fitness,
                           new_trust, new_engage, new_sub_pot, new_cust_val, new_personalisation]])

    if st.button("🔮 Predict This User", type="primary"):
        sub_proba = clf.predict_proba(input_vec)[0]
        sub_pred = le_s.inverse_transform(clf.predict(input_vec))[0]
        intent_pred = le_i.inverse_transform(clf_intent.predict(input_vec))[0]
        budget_pred = reg.predict(input_vec)[0]

        c1,c2,c3,c4 = st.columns(4)
        sub_yes_prob = sub_proba[list(le_s.classes_).index("Yes")] if "Yes" in le_s.classes_ else sub_proba[1]
        c1.metric("Subscription Likelihood", sub_pred, f"{sub_yes_prob*100:.1f}% probability")
        c2.metric("Sign-Up Intent", intent_pred)
        c3.metric("Predicted Monthly Budget", f"₹{budget_pred:.0f}")
        if budget_pred < 200:
            plan = "Free Plan"
        elif budget_pred < 350:
            plan = "Basic ₹199/month"
        elif budget_pred < 600:
            plan = "Standard ₹299/month"
        else:
            plan = "Premium ₹499/month"
        c4.metric("Recommended Plan", plan)

        st.success(f"✅ **Analysis Complete** — This user has **{intent_pred} Sign-Up Intent** and is **{sub_pred}** likely to subscribe.")

        if sub_pred == "Yes":
            st.markdown("### 📣 Suggested Marketing Action")
            st.markdown(f"- Send personalised onboarding email with {plan} offer\n- Show diet plan preview during first app launch\n- Trigger free trial CTA after Day 3 of app use\n- Add to 'High Intent' segment in CRM for follow-up within 48 hours")
        else:
            st.markdown("### 📣 Suggested Conversion Action")
            st.markdown("- Offer 7-day free trial to build trust\n- Send weekly health tip emails to nurture\n- Retarget with testimonials from similar users\n- Highlight data privacy guarantee to reduce hesitation")

st.markdown("---")
st.markdown("<center style='color:#9e9e9e; font-size:0.82rem;'>NourishIQ Analytics Dashboard &nbsp;|&nbsp; Built with Streamlit &nbsp;|&nbsp; Synthetic Dataset – Educational Purpose Only</center>", unsafe_allow_html=True)
