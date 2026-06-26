import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Churn Predictor | Charity David", page_icon="📡", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Dancing+Script:wght@700&family=Inter:wght@300;400;600;700&display=swap');

/* Global */
[data-testid="stAppViewContainer"] { background-color: #f0f4f8; }
[data-testid="stHeader"] { display: none !important; }
.block-container { padding-top: 2.5rem; padding-left: 2rem; padding-right: 2rem; }

/* Sidebar */
[data-testid="stSidebar"] { background-color: #1e3a8a; border-right: none; }
[data-testid="stSidebar"] * { color: #ffffff; }
[data-testid="stSidebar"] button {
    background-color: rgba(255,255,255,0.08) !important;
    color: #c7d8f8 !important;
    border: none !important;
    border-radius: 8px !important;
    box-shadow: none !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    text-align: left !important;
}
[data-testid="stSidebar"] button:hover {
    background-color: rgba(255,255,255,0.18) !important;
    color: #ffffff !important;
}
[data-testid="stSidebar"] button[kind="primary"] {
    background-color: #ffffff !important;
    color: #1e3a8a !important;
    font-weight: 700 !important;
}

/* Typography */
.dash-title { font-family:'Inter',sans-serif; font-size:1.9rem; font-weight:700; color:#1e3a8a; line-height:1.1; }
.dash-subtitle { font-family:'Dancing Script',cursive; font-size:1.35rem; color:#3b82f6; margin-bottom:4px; }
.dash-meta { font-family:'Inter',sans-serif; font-size:0.78rem; color:#6b7280; margin-bottom:1.2rem; }

/* Sidebar branding */
.sb-brand { font-family:'Inter',sans-serif; font-size:0.65rem; font-weight:700; color:#93c5fd; letter-spacing:0.12em; text-transform:uppercase; }
.sb-title { font-family:'Inter',sans-serif; font-size:1.5rem; font-weight:900; color:#ffffff; line-height:1; }
.sb-sub { font-family:'Inter',sans-serif; font-size:0.62rem; color:#93c5fd; letter-spacing:0.05em; }
.sb-divider { border-top:1px solid rgba(255,255,255,0.15); margin:0.8rem 0; }
.sb-footer { font-family:'Inter',sans-serif; font-size:0.7rem; color:#93c5fd; padding:0.5rem 0; }

.metric-card {
    background:#ffffff; border:1px solid #e5e7eb;
    border-radius:12px; padding:0.9rem 0.8rem;
    display:flex; align-items:center; gap:0.6rem;
    margin-bottom:1rem; box-shadow:0 1px 3px rgba(0,0,0,0.06);
}
.metric-icon { font-size:1.3rem; width:38px; height:38px; border-radius:8px; display:flex; align-items:center; justify-content:center; flex-shrink:0; }
.metric-icon-blue { background:#eff6ff; }
.metric-icon-red { background:#fef2f2; }
.metric-icon-green { background:#f0fdf4; }
.metric-icon-gold { background:#fffbeb; }
.metric-val { font-family:'Inter',sans-serif; font-size:1.4rem; font-weight:700; color:#111827; line-height:1; }
.metric-label { font-family:'Inter',sans-serif; font-size:0.65rem; color:#6b7280; text-transform:uppercase; letter-spacing:0.04em; margin-top:3px; }
.metric-sub { font-family:'Inter',sans-serif; font-size:0.68rem; color:#9ca3af; margin-top:2px; }

/* Section headers */
.section-header { font-family:'Inter',sans-serif; font-size:0.82rem; font-weight:700; color:#374151; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.8rem; display:flex; align-items:center; gap:6px; }

/* Insight cards */
.insight-card { background:#ffffff; border:1px solid #e5e7eb; border-radius:10px; padding:0.9rem 1.1rem; margin-bottom:0.5rem; box-shadow:0 1px 2px rgba(0,0,0,0.04); }
.insight-title { font-family:'Inter',sans-serif; font-size:0.75rem; font-weight:700; color:#1e3a8a; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:4px; }
.insight-stat { font-family:'Inter',sans-serif; font-size:0.95rem; font-weight:700; color:#ef4444; }
.insight-text { font-family:'Inter',sans-serif; font-size:0.82rem; color:#4b5563; }

/* Prediction result */
.pred-high { background:#fff5f5; border:2px solid #fca5a5; border-radius:14px; padding:1.5rem; text-align:center; }
.pred-low { background:#f0fdf4; border:2px solid #86efac; border-radius:14px; padding:1.5rem; text-align:center; }
.pred-prob { font-family:'Inter',sans-serif; font-size:3.2rem; font-weight:900; line-height:1; }
.pred-label { font-family:'Inter',sans-serif; font-size:1rem; font-weight:700; margin-top:8px; }
.pred-sub { font-family:'Inter',sans-serif; font-size:0.78rem; color:#6b7280; margin-top:6px; }

/* Top Reasons */
.reasons-box { background:#ffffff; border:1px solid #e5e7eb; border-radius:10px; padding:1rem 1.2rem; margin-top:1rem; box-shadow:0 1px 2px rgba(0,0,0,0.04); }
.reasons-title { font-family:'Inter',sans-serif; font-size:0.78rem; font-weight:700; color:#374151; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.6rem; }
.reason-item { font-family:'Inter',sans-serif; font-size:0.82rem; color:#374151; padding:4px 0; display:flex; align-items:center; gap:8px; border-bottom:1px solid #f3f4f6; }
.reason-item:last-child { border-bottom:none; }
.reason-dot-red { width:8px; height:8px; background:#ef4444; border-radius:50%; flex-shrink:0; }
.reason-dot-green { width:8px; height:8px; background:#22c55e; border-radius:50%; flex-shrink:0; }

/* Recommendation cards */
.rec-card { background:#ffffff; border:1px solid #e5e7eb; border-left:4px solid #1e3a8a; border-radius:0 10px 10px 0; padding:0.9rem 1.1rem; margin-bottom:0.6rem; box-shadow:0 1px 2px rgba(0,0,0,0.04); }
.rec-title { font-family:'Inter',sans-serif; font-size:0.85rem; font-weight:700; color:#1e3a8a; margin-bottom:4px; }
.rec-text { font-family:'Inter',sans-serif; font-size:0.81rem; color:#4b5563; }

/* Page title bar */
.page-title { font-family:'Inter',sans-serif; font-size:1rem; font-weight:700; color:#111827; margin-bottom:0.2rem; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background:#ffffff; border-radius:8px; border:1px solid #e5e7eb; padding:4px; gap:4px; }
.stTabs [data-baseweb="tab"] { background:transparent; border-radius:6px; color:#6b7280; font-size:0.82rem; }
.stTabs [aria-selected="true"] { background:#1e3a8a !important; color:#ffffff !important; }

/* Inputs */
[data-testid="stSelectbox"] label { color:#374151 !important; font-family:'Inter',sans-serif !important; font-size:0.8rem !important; font-weight:500 !important; }
[data-testid="stSlider"] label { color:#374151 !important; font-family:'Inter',sans-serif !important; font-size:0.8rem !important; font-weight:500 !important; }

/* Primary button */
.stButton > button[kind="primary"] {
    background-color: #1e3a8a !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
}
.stButton > button[kind="primary"]:hover { background-color: #1e40af !important; }

/* Dataframe */
[data-testid="stDataFrame"] { border-radius:8px; border:1px solid #e5e7eb; overflow:hidden; }

#MainMenu {visibility:hidden;} footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ── LOAD DATA & MODEL ─────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("C:/Users/pc/churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    df['Churn_Binary'] = df['Churn'].map({'Yes': 1, 'No': 0})
    df['SeniorCitizen'] = df['SeniorCitizen'].map({1: 'Yes', 0: 'No'})
    return df

@st.cache_resource
def load_model():
    with open("C:/Users/pc/churn-prediction/churn_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("C:/Users/pc/churn-prediction/model_features.pkl", "rb") as f:
        features = pickle.load(f)
    return model, features

df = load_data()
model, feature_cols = load_model()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1.2rem 1rem 0.8rem 1rem;'>
        <div class='sb-brand'>Customer Analytics</div>
        <div class='sb-title'>Churn</div>
        <div class='sb-sub'>IBM Telco Dataset • Random Forest</div>
    </div>
    <div class='sb-divider'></div>
    """, unsafe_allow_html=True)

    pages = {"Overview":"🏠","EDA":"📊","Predict":"🔮","Feature Importance":"📈","Recommendations":"💡"}
    if "page" not in st.session_state:
        st.session_state.page = "Overview"
    for pg, icon in pages.items():
        if st.button(f"{icon}  {pg}", key=f"nav_{pg}", use_container_width=True,
                     type="primary" if st.session_state.page == pg else "secondary"):
            st.session_state.page = pg
            st.rerun()

    st.markdown("<div class='sb-divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sb-footer'>Model: Random Forest<br>Accuracy: 79% • AUC: 0.83</div>", unsafe_allow_html=True)

page = st.session_state.page

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='dash-title'>Customer Churn Prediction Dashboard</div>
<div class='dash-subtitle'>by Charity David</div>
<div class='dash-meta'>IBM Telco Dataset • Random Forest Model • 7,043 Customers • 26.5% Churn Rate</div>
""", unsafe_allow_html=True)

def light_fig(w=12, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.tick_params(colors="#6b7280")
    ax.xaxis.label.set_color("#6b7280")
    ax.yaxis.label.set_color("#6b7280")
    for spine in ax.spines.values(): spine.set_edgecolor("#e5e7eb")
    ax.title.set_color("#111827")
    return fig, ax

def light_figs(rows, cols, w=14, h=5):
    fig, axes = plt.subplots(rows, cols, figsize=(w, h))
    fig.patch.set_facecolor("#ffffff")
    for ax in (axes.flatten() if hasattr(axes, 'flatten') else [axes]):
        ax.set_facecolor("#ffffff")
        ax.tick_params(colors="#6b7280")
        ax.xaxis.label.set_color("#6b7280")
        ax.yaxis.label.set_color("#6b7280")
        for spine in ax.spines.values(): spine.set_edgecolor("#e5e7eb")
        ax.title.set_color("#111827")
    return fig, axes

BLUE = "#1e3a8a"
LIGHT_BLUE = "#3b82f6"
RED = "#ef4444"
GOLD = "#f59e0b"
GREEN = "#22c55e"
GRAY = "#e5e7eb"

# ── OVERVIEW ──────────────────────────────────────────────────────────────────
if page == "Overview":
    churn_rate = round(df['Churn_Binary'].mean() * 100, 1)
    churned = int(df['Churn_Binary'].sum())
    retained = len(df) - churned
    avg_monthly = round(df[df['Churn']=='Yes']['MonthlyCharges'].mean(), 2)
    revenue_risk = round(churned * avg_monthly / 1000, 0)

    revenue_risk = int(round(churned * avg_monthly / 1000))

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""<div class='metric-card'><div class='metric-icon metric-icon-blue'>👥</div>
            <div><div class='metric-val'>7,043</div><div class='metric-label'>Customers</div>
            <div class='metric-sub'>Total dataset</div></div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-card'><div class='metric-icon metric-icon-red'>⚠️</div>
            <div><div class='metric-val'>{churn_rate}%</div><div class='metric-label'>Churn Rate</div>
            <div class='metric-sub'>{churned:,} lost</div></div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='metric-card'><div class='metric-icon metric-icon-green'>✅</div>
            <div><div class='metric-val'>{retained:,}</div><div class='metric-label'>Retained</div>
            <div class='metric-sub'>73.5% loyalty</div></div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class='metric-card'><div class='metric-icon metric-icon-gold'>💰</div>
            <div><div class='metric-val'>${avg_monthly}</div><div class='metric-label'>Avg Charges</div>
            <div class='metric-sub'>Churners/month</div></div></div>""", unsafe_allow_html=True)
    with c5:
        st.markdown(f"""<div class='metric-card'><div class='metric-icon metric-icon-green'>📉</div>
            <div><div class='metric-val'>${revenue_risk}K</div><div class='metric-label'>Revenue Risk</div>
            <div class='metric-sub'>Monthly loss</div></div></div>""", unsafe_allow_html=True)

    # Executive Summary
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    retain_5pct = int(round(churned * 0.05))
    revenue_saved = int(round(retain_5pct * avg_monthly / 1000))
    st.markdown(f"""
    <div style='background:#eff6ff;border:1px solid #bfdbfe;border-radius:12px;padding:1.1rem 1.4rem;margin-bottom:1rem;'>
        <div style='font-family:Inter,sans-serif;font-size:0.82rem;font-weight:700;color:#1e3a8a;margin-bottom:6px;'>📋 EXECUTIVE SUMMARY</div>
        <div style='font-family:Inter,sans-serif;font-size:0.85rem;color:#1e40af;line-height:1.6;'>
        Our model predicts that <strong>{churn_rate}%</strong> of customers are at risk of churning, representing a potential monthly revenue loss of <strong>${revenue_risk}K</strong>.
        Month-to-month contracts and fiber optic internet users are the highest-risk segments, churning at <strong>42.7%</strong> and <strong>41.9%</strong> respectively.
        Reducing churn by just 5% could retain approximately <strong>~{retain_5pct} additional customers</strong>, saving an estimated <strong>${revenue_saved}K/month</strong> in recurring revenue.
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown("<div class='section-header'>🔍 KEY RISK FACTORS</div>", unsafe_allow_html=True)
        insights = [
            ("CONTRACT TYPE", "Month-to-month customers churn at 42.7% — 15x higher than two-year contracts (2.8%)", "42.7%"),
            ("INTERNET SERVICE", "Fiber optic customers have the highest churn rate at 41.9% despite paying more", "41.9%"),
            ("PAYMENT METHOD", "Electronic check users churn at 45.3% — the riskiest payment method by far", "45.3%"),
            ("TENURE", "Average churner stays only 18 months vs 37.6 months for retained customers", "18 mo"),
            ("SENIOR CITIZENS", "Senior customers churn at 41.7% — nearly double the non-senior rate of 23.6%", "41.7%"),
        ]
        for title, text, stat in insights:
            st.markdown(f"""<div class='insight-card'>
                <div class='insight-title'>{title} — <span class='insight-stat'>{stat}</span></div>
                <div class='insight-text'>{text}</div>
            </div>""", unsafe_allow_html=True)

    with col_r:
        st.markdown("<div class='section-header'>📊 CHURN BY CONTRACT</div>", unsafe_allow_html=True)
        fig, ax = light_fig(6, 4)
        contracts = df.groupby('Contract')['Churn_Binary'].mean() * 100
        colors = [RED, GOLD, GREEN]
        bars = ax.bar(contracts.index, contracts.values, color=colors, edgecolor='none', width=0.5)
        for bar, val in zip(bars, contracts.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                    f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold', color='#111827')
        ax.set_ylabel('Churn Rate (%)', color='#6b7280')
        ax.set_title('Churn Rate by Contract Type', fontweight='bold', color='#111827')
        ax.set_ylim(0, 55)
        plt.setp(ax.get_xticklabels(), color='#6b7280', fontsize=9)
        plt.setp(ax.get_yticklabels(), color='#6b7280')
        fig.patch.set_facecolor('#f8fafc')
        ax.set_facecolor('#f8fafc')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

# ── EDA ───────────────────────────────────────────────────────────────────────
elif page == "EDA":
    st.markdown("<div class='section-header'>📊 EXPLORATORY DATA ANALYSIS</div>", unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["Churn Overview", "Tenure & Charges", "Services", "Demographics"])

    with tab1:
        fig, axes = light_figs(1, 2, 13, 5)
        axes[0].bar(['No Churn','Churned'], df['Churn'].value_counts().values,
                    color=[BLUE, RED], edgecolor='none', width=0.5)
        axes[0].set_title('Churn Count', fontweight='bold')
        axes[0].set_ylabel('Customers')
        for i, v in enumerate(df['Churn'].value_counts().values):
            axes[0].text(i, v+50, str(v), ha='center', fontweight='bold', color='#111827')
        plt.setp(axes[0].get_xticklabels(), color='#6b7280')
        plt.setp(axes[0].get_yticklabels(), color='#6b7280')

        cr = df.groupby('Contract')['Churn_Binary'].mean() * 100
        axes[1].bar(cr.index, cr.values, color=[RED, GOLD, GREEN], edgecolor='none', width=0.5)
        axes[1].set_title('Churn Rate by Contract', fontweight='bold')
        axes[1].set_ylabel('Churn Rate (%)')
        axes[1].set_ylim(0, 55)
        for i, (idx, val) in enumerate(cr.items()):
            axes[1].text(i, val+0.5, f'{val:.1f}%', ha='center', fontweight='bold', color='#111827', fontsize=9)
        plt.setp(axes[1].get_xticklabels(), color='#6b7280', fontsize=9)
        plt.setp(axes[1].get_yticklabels(), color='#6b7280')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab2:
        # Row 1: Customers by Tenure (area) + Tenure vs Churn Rate (scatter)
        fig, axes = light_figs(1, 2, 13, 5)

        # Area chart — customers by tenure
        tenure_counts = df.groupby('tenure').size()
        axes[0].fill_between(tenure_counts.index, tenure_counts.values, alpha=0.4, color='#818cf8')
        axes[0].plot(tenure_counts.index, tenure_counts.values, color='#6366f1', linewidth=2)
        axes[0].set_title('Customers by Tenure', fontweight='bold')
        axes[0].set_xlabel('Tenure (Months)')
        axes[0].set_ylabel('Customers')
        plt.setp(axes[0].get_xticklabels(), color='#6b7280')
        plt.setp(axes[0].get_yticklabels(), color='#6b7280')

        # Scatter — tenure vs churn rate by tenure bucket
        tenure_churn = df.groupby('tenure')['Churn_Binary'].mean() * 100
        axes[1].scatter(tenure_churn.index, tenure_churn.values,
                        color='#818cf8', alpha=0.6, s=25, edgecolors='none')
        axes[1].set_title('Customer Tenure vs Churn Rate', fontweight='bold')
        axes[1].set_xlabel('Tenure (Months)')
        axes[1].set_ylabel('Churn Rate (%)')
        plt.setp(axes[1].get_xticklabels(), color='#6b7280')
        plt.setp(axes[1].get_yticklabels(), color='#6b7280')
        plt.tight_layout(); st.pyplot(fig); plt.close()

        # Monthly charges distribution
        fig, axes = light_figs(1, 2, 13, 5)
        axes[0].hist(df[df['Churn']=='No']['MonthlyCharges'], bins=30, alpha=0.7, color=BLUE, label='No Churn')
        axes[0].hist(df[df['Churn']=='Yes']['MonthlyCharges'], bins=30, alpha=0.7, color=RED, label='Churned')
        axes[0].set_title('Monthly Charges Distribution', fontweight='bold')
        axes[0].set_xlabel('Monthly Charges ($)')
        axes[0].set_ylabel('Customers')
        axes[0].legend()
        plt.setp(axes[0].get_xticklabels(), color='#6b7280')
        plt.setp(axes[0].get_yticklabels(), color='#6b7280')

        avg_by_churn = df.groupby('Churn')['MonthlyCharges'].mean()
        bars = axes[1].bar(avg_by_churn.index, avg_by_churn.values,
                           color=[BLUE, RED], edgecolor='none', width=0.4)
        axes[1].set_title('Avg Monthly Charges by Churn', fontweight='bold')
        axes[1].set_ylabel('Avg Monthly Charges ($)')
        axes[1].set_ylim(0, 90)
        for bar, val in zip(bars, avg_by_churn.values):
            axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                         f'${val:.2f}', ha='center', fontweight='bold', color='#111827')
        plt.setp(axes[1].get_xticklabels(), color='#6b7280')
        plt.setp(axes[1].get_yticklabels(), color='#6b7280')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab3:
        # Cross-tab heatmap: Contract x Internet Service
        st.markdown("<div style='font-family:Inter,sans-serif;font-size:0.85rem;font-weight:600;color:#374151;margin-bottom:0.5rem'>Churn Rate by Contract & Internet Service</div>", unsafe_allow_html=True)
        cross = df.groupby(['Contract','InternetService'])['Churn_Binary'].mean() * 100
        cross_df = cross.unstack()
        fig, ax = light_fig(10, 4)
        sns.heatmap(cross_df, annot=True, fmt='.1f', cmap='RdYlGn_r',
                    ax=ax, linewidths=0.5, linecolor='#e5e7eb',
                    annot_kws={'size': 11, 'weight': 'bold'},
                    cbar_kws={'shrink': 0.8})
        ax.set_title('Churn Rate (%) by Contract Type & Internet Service', fontweight='bold')
        ax.set_xlabel('Internet Service', color='#6b7280')
        ax.set_ylabel('Contract Type', color='#6b7280')
        plt.setp(ax.get_xticklabels(), color='#374151', fontsize=10)
        plt.setp(ax.get_yticklabels(), color='#374151', fontsize=10, rotation=0)
        fig.patch.set_facecolor('#f8fafc')
        ax.set_facecolor('#f8fafc')
        plt.tight_layout(); st.pyplot(fig); plt.close()

        # Payment method churn
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        fig, axes = light_figs(1, 2, 13, 5)
        pm_churn = df.groupby('PaymentMethod')['Churn_Binary'].mean() * 100
        colors_pm = [RED if v > 30 else GOLD if v > 20 else GREEN for v in pm_churn.values]
        bars = axes[0].bar(range(len(pm_churn)), pm_churn.values, color=colors_pm, edgecolor='none', width=0.5)
        axes[0].set_title('Churn by Payment Method', fontweight='bold')
        axes[0].set_ylabel('Churn Rate (%)')
        axes[0].set_xticks(range(len(pm_churn)))
        axes[0].set_xticklabels(pm_churn.index, rotation=15, ha='right', fontsize=8, color='#6b7280')
        axes[0].set_ylim(0, 55)
        for bar, val in zip(bars, pm_churn.values):
            axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                         f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold', color='#111827')
        plt.setp(axes[0].get_yticklabels(), color='#6b7280')

        internet_churn = df.groupby('InternetService')['Churn_Binary'].mean() * 100
        bars2 = axes[1].bar(internet_churn.index, internet_churn.values,
                            color=[LIGHT_BLUE, RED, GREEN], edgecolor='none', width=0.5)
        axes[1].set_title('Churn by Internet Service', fontweight='bold')
        axes[1].set_ylabel('Churn Rate (%)')
        axes[1].set_ylim(0, 55)
        for bar, val in zip(bars2, internet_churn.values):
            axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                         f'{val:.1f}%', ha='center', fontweight='bold', color='#111827')
        plt.setp(axes[1].get_xticklabels(), color='#6b7280')
        plt.setp(axes[1].get_yticklabels(), color='#6b7280')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab4:
        fig, axes = light_figs(1, 2, 12, 5)
        for ax, col, title in zip(axes, ['SeniorCitizen','gender'], ['Senior Citizen','Gender']):
            cr = df.groupby(col)['Churn_Binary'].mean() * 100
            bars = ax.bar(cr.index, cr.values, color=[BLUE, LIGHT_BLUE], edgecolor='none', width=0.4)
            ax.set_title(f'Churn Rate by {title}', fontweight='bold')
            ax.set_ylabel('Churn Rate (%)')
            ax.set_ylim(0, cr.max()+15)
            for bar, val in zip(bars, cr.values):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                        f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold', color='#111827')
            plt.setp(ax.get_xticklabels(), color='#6b7280')
            plt.setp(ax.get_yticklabels(), color='#6b7280')
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ── PREDICT ───────────────────────────────────────────────────────────────────
elif page == "Predict":
    st.markdown("<div class='section-header'>🔮 PREDICT CUSTOMER CHURN</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-family:Inter,sans-serif;font-size:0.85rem;color:#6b7280;margin-bottom:1rem'>Fill in the customer details below to get a churn probability prediction.</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div style='font-family:Inter,sans-serif;font-size:0.78rem;color:#1e3a8a;font-weight:700;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.06em'>Demographics</div>", unsafe_allow_html=True)
        gender = st.selectbox("Gender", ["Male","Female"])
        senior = st.selectbox("Senior Citizen", ["No","Yes"])
        partner = st.selectbox("Partner", ["Yes","No"])
        dependents = st.selectbox("Dependents", ["No","Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)

    with col2:
        st.markdown("<div style='font-family:Inter,sans-serif;font-size:0.78rem;color:#1e3a8a;font-weight:700;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.06em'>Services</div>", unsafe_allow_html=True)
        phone = st.selectbox("Phone Service", ["Yes","No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No","Yes","No phone service"])
        internet = st.selectbox("Internet Service", ["DSL","Fiber optic","No"])
        online_sec = st.selectbox("Online Security", ["No","Yes","No internet service"])
        online_bk = st.selectbox("Online Backup", ["Yes","No","No internet service"])
        device = st.selectbox("Device Protection", ["No","Yes","No internet service"])
        tech = st.selectbox("Tech Support", ["No","Yes","No internet service"])
        tv = st.selectbox("Streaming TV", ["No","Yes","No internet service"])
        movies = st.selectbox("Streaming Movies", ["No","Yes","No internet service"])

    with col3:
        st.markdown("<div style='font-family:Inter,sans-serif;font-size:0.78rem;color:#1e3a8a;font-weight:700;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.06em'>Account</div>", unsafe_allow_html=True)
        contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes","No"])
        payment = st.selectbox("Payment Method", ["Electronic check","Mailed check",
                                                   "Bank transfer (automatic)","Credit card (automatic)"])
        monthly = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
        total = st.slider("Total Charges ($)", 0.0, 9000.0, float(monthly * tenure), step=10.0)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    if st.button("🔮  Predict Churn Probability", type="primary", use_container_width=True):
        input_data = {
            'gender': 1 if gender=='Male' else 0,
            'SeniorCitizen': 1 if senior=='Yes' else 0,
            'Partner': 1 if partner=='Yes' else 0,
            'Dependents': 1 if dependents=='Yes' else 0,
            'tenure': tenure,
            'PhoneService': 1 if phone=='Yes' else 0,
            'PaperlessBilling': 1 if paperless=='Yes' else 0,
            'MonthlyCharges': monthly,
            'TotalCharges': total,
            'MultipleLines_No phone service': 1 if multiple_lines=='No phone service' else 0,
            'MultipleLines_Yes': 1 if multiple_lines=='Yes' else 0,
            'InternetService_Fiber optic': 1 if internet=='Fiber optic' else 0,
            'InternetService_No': 1 if internet=='No' else 0,
            'OnlineSecurity_No internet service': 1 if online_sec=='No internet service' else 0,
            'OnlineSecurity_Yes': 1 if online_sec=='Yes' else 0,
            'OnlineBackup_No internet service': 1 if online_bk=='No internet service' else 0,
            'OnlineBackup_Yes': 1 if online_bk=='Yes' else 0,
            'DeviceProtection_No internet service': 1 if device=='No internet service' else 0,
            'DeviceProtection_Yes': 1 if device=='Yes' else 0,
            'TechSupport_No internet service': 1 if tech=='No internet service' else 0,
            'TechSupport_Yes': 1 if tech=='Yes' else 0,
            'StreamingTV_No internet service': 1 if tv=='No internet service' else 0,
            'StreamingTV_Yes': 1 if tv=='Yes' else 0,
            'StreamingMovies_No internet service': 1 if movies=='No internet service' else 0,
            'StreamingMovies_Yes': 1 if movies=='Yes' else 0,
            'Contract_One year': 1 if contract=='One year' else 0,
            'Contract_Two year': 1 if contract=='Two year' else 0,
            'PaymentMethod_Credit card (automatic)': 1 if payment=='Credit card (automatic)' else 0,
            'PaymentMethod_Electronic check': 1 if payment=='Electronic check' else 0,
            'PaymentMethod_Mailed check': 1 if payment=='Mailed check' else 0,
        }

        input_df = pd.DataFrame([input_data])[feature_cols]
        prob = model.predict_proba(input_df)[0][1]
        pct = round(prob * 100, 1)

        # Build top reasons
        risk_factors = []
        safe_factors = []
        if contract == "Month-to-month": risk_factors.append("Month-to-month contract (42.7% churn rate)")
        if contract == "Two year": safe_factors.append("Two-year contract (only 2.8% churn rate)")
        if internet == "Fiber optic": risk_factors.append("Fiber optic internet service (41.9% churn rate)")
        if payment == "Electronic check": risk_factors.append("Electronic check payment (45.3% churn rate)")
        if online_sec == "No": risk_factors.append("No online security (41.8% churn rate)")
        if tech == "No": risk_factors.append("No tech support (41.6% churn rate)")
        if tenure < 12: risk_factors.append(f"Short tenure ({tenure} months) — early churn window")
        if senior == "Yes": risk_factors.append("Senior citizen (41.7% churn rate)")
        if monthly > 70: risk_factors.append(f"High monthly charges (${monthly:.0f})")
        if online_sec == "Yes": safe_factors.append("Has online security (reduces churn risk)")
        if tech == "Yes": safe_factors.append("Has tech support (reduces churn risk)")
        if tenure >= 24: safe_factors.append(f"Long tenure ({tenure} months) — loyal customer")
        if payment in ["Credit card (automatic)","Bank transfer (automatic)"]:
            safe_factors.append("Automatic payment method (lower churn risk)")

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        r1, r2, r3 = st.columns([1, 2, 1])
        with r2:
            if prob >= 0.5:
                st.markdown(f"""<div class='pred-high'>
                    <div class='pred-prob' style='color:#ef4444'>{pct}%</div>
                    <div class='pred-label' style='color:#ef4444'>⚠️ HIGH CHURN RISK</div>
                    <div class='pred-sub'>This customer is likely to churn. Immediate action recommended.</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class='pred-low'>
                    <div class='pred-prob' style='color:#16a34a'>{pct}%</div>
                    <div class='pred-label' style='color:#16a34a'>✅ LOW CHURN RISK</div>
                    <div class='pred-sub'>This customer is likely to stay. Continue standard engagement.</div>
                </div>""", unsafe_allow_html=True)

            # Probability bar
            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 0.8))
            fig.patch.set_facecolor("#ffffff")
            ax.set_facecolor("#ffffff")
            bar_color = "#ef4444" if prob >= 0.5 else "#22c55e"
            ax.barh([''], [pct], color=bar_color, height=0.5, edgecolor='none')
            ax.barh([''], [100-pct], left=[pct], color='#e5e7eb', height=0.5, edgecolor='none')
            ax.set_xlim(0, 100)
            ax.set_xlabel('Churn Probability (%)', color='#6b7280', fontsize=8)
            ax.axvline(50, color='#f59e0b', linestyle='--', alpha=0.8, linewidth=1.5)
            ax.text(50, 0.42, '50%', ha='center', color='#f59e0b', fontsize=7)
            for spine in ax.spines.values(): spine.set_edgecolor('#e5e7eb')
            ax.tick_params(colors='#6b7280')
            plt.tight_layout()
            st.pyplot(fig); plt.close()

            # Top reasons
            reasons_html = ""
            for r in risk_factors[:5]:
                reasons_html += f"<div class='reason-item'><div class='reason-dot-red'></div>{r}</div>"
            for r in safe_factors[:3]:
                reasons_html += f"<div class='reason-item'><div class='reason-dot-green'></div>{r}</div>"

            if reasons_html:
                st.markdown(f"""<div class='reasons-box'>
                    <div class='reasons-title'>🔍 Top Reasons for This Prediction</div>
                    {reasons_html}
                </div>""", unsafe_allow_html=True)

# ── FEATURE IMPORTANCE ────────────────────────────────────────────────────────
elif page == "Feature Importance":
    st.markdown("<div class='section-header'>📈 WHAT DRIVES CHURN?</div>", unsafe_allow_html=True)

    importances = pd.Series(model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=False).head(15)

    fig, ax = light_fig(11, 7)
    colors = [RED if i < 3 else LIGHT_BLUE if i < 7 else '#93c5fd' for i in range(len(importances))]
    bars = ax.barh(importances.index[::-1], importances.values[::-1],
                   color=colors[::-1], edgecolor='none', height=0.65)
    ax.set_title('Top 15 Feature Importances — Random Forest', fontweight='bold', fontsize=13)
    ax.set_xlabel('Importance Score')
    for bar, val in zip(bars, importances.values[::-1]):
        ax.text(bar.get_width()+0.001, bar.get_y()+bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9, fontweight='bold', color='#374151')
    plt.setp(ax.get_yticklabels(), color='#374151', fontsize=9)
    plt.setp(ax.get_xticklabels(), color='#6b7280')
    fig.patch.set_facecolor('#f8fafc')
    ax.set_facecolor('#f8fafc')
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class='insight-card'>
            <div class='insight-title'>🔴 Top Drivers</div>
            <div class='insight-text'>TotalCharges, Tenure, and MonthlyCharges dominate — financial commitment and loyalty are the strongest signals of whether a customer stays.</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class='insight-card'>
            <div class='insight-title'>🔵 Contract Impact</div>
            <div class='insight-text'>Two-year contract is the 4th most important feature — locking customers into longer contracts significantly reduces churn probability.</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class='insight-card'>
            <div class='insight-title'>⚪ Gender Is Irrelevant</div>
            <div class='insight-text'>Gender appears in the top 15 but with very low importance (0.026), confirming that churn is driven by service experience, not demographics.</div>
        </div>""", unsafe_allow_html=True)

# ── RECOMMENDATIONS ───────────────────────────────────────────────────────────
elif page == "Recommendations":
    st.markdown("<div class='section-header'>💡 BUSINESS RECOMMENDATIONS</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-family:Inter,sans-serif;font-size:0.85rem;color:#6b7280;margin-bottom:1rem'>Data-driven actions to reduce churn, based on model findings.</div>", unsafe_allow_html=True)

    recs = [
        ("🔒 Push Long-Term Contracts", "Month-to-month customers churn at 42.7% vs 2.8% for two-year contracts. Offer discounts, loyalty rewards, or bundled services to convert month-to-month customers to annual contracts. Even moving to one-year cuts churn to 11.3%."),
        ("🌐 Investigate Fiber Optic Satisfaction", "Fiber optic customers pay the most but churn the most (41.9%). This suggests a value perception problem — speed, reliability, or customer service. Conduct satisfaction surveys and address service quality issues."),
        ("🔐 Bundle Security & Support Services", "Customers without Online Security churn at 41.8% and without Tech Support at 41.6%. Bundling these services — or offering them free for the first year — can meaningfully reduce churn."),
        ("💳 Incentivise Auto-Pay Adoption", "Electronic check users churn at 45.3% vs 15.2% for credit card auto-pay. Offer a monthly discount for switching to automatic payment methods."),
        ("👴 Create Senior Citizen Retention Programme", "Senior customers churn at 41.7%. Dedicated support channels, simplified billing, and senior-specific packages can improve retention in this high-risk segment."),
        ("⏱ Focus on Early Tenure Customers", "Most churn happens in the first 1-2 months. A structured onboarding programme — proactive check-ins, usage guides, and early support — can significantly improve early retention."),
        ("🎯 Use the Prediction Model Proactively", "Deploy this model to score all existing customers monthly. Flag anyone above 50% churn probability for intervention by the retention team before they cancel."),
    ]

    for title, text in recs:
        st.markdown(f"""<div class='rec-card'>
            <div class='rec-title'>{title}</div>
            <div class='rec-text'>{text}</div>
        </div>""", unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='border-top:1px solid #e5e7eb;padding-top:0.8rem;font-family:Inter,sans-serif;
font-size:0.72rem;color:#9ca3af;text-align:center;'>
Data: IBM Telco Customer Churn Dataset &nbsp;|&nbsp;
<span style='font-family:Dancing Script,cursive;font-size:0.9rem;color:#3b82f6'>Charity David</span>
&nbsp;|&nbsp; Customer Churn Prediction Dashboard
</div>""", unsafe_allow_html=True)
