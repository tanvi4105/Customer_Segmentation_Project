import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import pandas as pd
import plotly.express as px
import joblib
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- LOAD DATA ----------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

df = pd.read_csv(OUTPUT_DIR / "segmented_customers.csv")

# Load trained models
kmeans = joblib.load(MODEL_DIR / "clustering_model.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")
pca = joblib.load(MODEL_DIR / "pca.pkl")
feature_columns = joblib.load(MODEL_DIR / "feature_columns.pkl")

# Segment names
segment_names = {
    0: "High Value Loyal",
    1: "Frequent Buyer",
    2: "Occasional Customer",
    3: "At Risk"
}

# Business recommendations
recommendations = {
    "High Value Loyal": "Offer VIP rewards and premium recommendations.",
    "Frequent Buyer": "Cross-sell complementary products.",
    "Occasional Customer": "Send loyalty offers and reminders.",
    "At Risk": "Launch re-engagement campaigns."
}


# ---------------- CSS ----------------

st.markdown("""
<style>

/* ===== Remove Streamlit top white header ===== */
header[data-testid="stHeader"] {
    background: transparent;
}


/* Main content spacing */
.block-container{
    padding-top:1rem !important;
    padding-bottom:1rem !important;
    padding-left:2.2rem !important;
    padding-right:2.2rem !important;
}

/* Remove Streamlit toolbar */


div[data-testid="stDecoration"]{
    display:none;
}

/* ===== Main Background ===== */
.stApp{
    background:linear-gradient(135deg,#071329,#111827);
}

/* ===== Sidebar ===== */
section[data-testid="stSidebar"] {
    background: #071329;
    border-right: 1px solid rgba(255,255,255,.08);
    padding-top: 0.5rem;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

/* ===== Text ===== */
h1,h2,h3,h4,h5,h6,p,label,span,div{
    color:white !important;
}

/* ===== Metric Cards ===== */
div[data-testid="stMetric"]{
    background:#1b2942;
    border-radius:18px;
    padding:18px;
    border:1px solid rgba(255,255,255,.08);
    box-shadow:0 8px 20px rgba(0,0,0,.25);
}

div[data-testid="stMetricValue"]{
    color:white !important;
    font-size:34px !important;
    font-weight:bold;
}

div[data-testid="stMetricLabel"]{
    color:#cbd5e1 !important;
}

/* ===== Buttons ===== */
.stButton button{
    background:#0ea5e9;
    color:white;
    border:none;
    border-radius:10px;
    font-weight:bold;
}

.stButton button:hover{
    background:#0284c7;
}

/* ===== Inputs ===== */
.stNumberInput input,
.stTextInput input{
    background:#1b2942 !important;
    color:white !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.markdown(
    "<h1 style='margin-top:0;color:white;font-size:32px;'>🤖 AI Customer Analytics</h1>",
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Predict Segment",
        "Cluster Visualization",
        "Segment Comparison",
        "Clustering Evaluation",
        "Business Insights"
    ]
)


# ==========================================================
# DASHBOARD
# ==========================================================

if page=="Dashboard":

    st.markdown("""
<div style="margin-top:18px;margin-bottom:22px;display:flex;align-items:center;gap:18px;">
    <span style="font-size:56px;">📊</span>
    <h1 style="font-size:52px;font-weight:800;color:white;margin:0;">
        AI Customer Segmentation Dashboard
    </h1>
</div>
""", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)

    c1.metric("👥 Customers", f"{len(df):,}")
    c2.metric("🎯 Segments", f"{df['Segment_Name'].nunique()}")
    c3.metric("💰 Avg Spend", f"₹{df['Monetary'].mean():,.0f}")
    c4.metric("🛒 Avg Orders", f"{df['Frequency'].mean():.1f}")
    st.divider()

    left,right=st.columns(2)

    with left:
        # ---------------- Customer Segment Distribution ----------------
        seg = df["Segment_Name"].value_counts().reset_index()
        seg.columns = ["Segment", "Customers"]

        fig = px.pie(
            seg,
            names="Segment",
            values="Customers",
            hole=0.60,
            color_discrete_sequence=[
                "#7C3AED",  # Purple
                "#10B981",  # Green
                "#2563EB",  # Blue
                "#F59E0B"   # Orange
            ]
        )

        fig.update_traces(
            textinfo="none",
            hovertemplate="<b>%{label}</b><br>Customers: %{value}<br>%{percent}<extra></extra>"
        )

        fig.update_layout(
            title=dict(
                text="Customer Segment Distribution",
                x=0,
                font=dict(size=22, color="white")
            ),
            paper_bgcolor="#071329",
            plot_bgcolor="#071329",
            font=dict(color="white"),
            legend=dict(
                title="",
                font=dict(color="white"),
                bgcolor="rgba(0,0,0,0)",
                y=0.98,
                x=1.02
            ),
            width=500,          # Same size as bar chart
            height=420,         # Same height as bar chart
            margin=dict(t=60, b=20, l=20, r=120)
        )

        st.plotly_chart(fig, use_container_width=False)
    with right:

        spend=df.groupby("Segment_Name")["Monetary"].mean().reset_index()

        fig=px.bar(
            spend,
            x="Segment_Name",
            y="Monetary",
            color="Segment_Name",
            text_auto=".2s",
            title="Average Spend by Segment"
        )

        fig.update_layout(
                        paper_bgcolor="#0f172a",
                        plot_bgcolor="#0f172a",
                        font=dict(color="white", size=14),
                        title_font=dict(color="white", size=22),
                        legend=dict(font=dict(color="white")))

        st.plotly_chart(fig,use_container_width=True)

    st.divider()

    # left,right=st.columns(2)
    left, right = st.columns([1, 1])

    with left:

        top=df.nlargest(10,"Monetary")[["CustomerID","Monetary"]]

        fig=px.bar(
            top,
            x="CustomerID",
            y="Monetary",
            color="Monetary",
            text_auto=".2s",
            title="Top 10 Customers"
        )

        fig.update_layout(
                paper_bgcolor="#0f172a",
                plot_bgcolor="#0f172a",
                font=dict(color="white", size=14),
                title_font=dict(color="white", size=22),
                legend=dict(font=dict(color="white"))
        )

        st.plotly_chart(fig,use_container_width=True)
        

    with right:

        fig=px.histogram(
            df,
            x="Monetary",
            nbins=30,
            color_discrete_sequence=["#00E5FF"],
            title="Customer Spending Distribution"
        )

        fig.update_layout(
                                paper_bgcolor="#0f172a",
                                plot_bgcolor="#0f172a",
                                font=dict(color="white", size=14),
                                title_font=dict(color="white", size=22),
                                legend=dict(font=dict(color="white")))
        st.plotly_chart(fig,use_container_width=True)
       

# ==========================================================
# PREDICT SEGMENT
# ==========================================================

elif page == "Predict Segment":

    st.title("🎯 Predict Customer Segment")
    st.write("Enter customer information to identify the customer segment.")

    col1, col2 = st.columns(2)

    with col1:
        recency = st.number_input(
            "Recency",
            min_value=0.0,
            value=30.0
        )

        frequency = st.number_input(
            "Frequency",
            min_value=0.0,
            value=5.0
        )

        monetary = st.number_input(
            "Monetary",
            min_value=0.0,
            value=500.0
        )

    with col2:
        avg_order = st.number_input(
            "Average Order Value",
            min_value=0.0,
            value=100.0
        )

        quantity = st.number_input(
            "Total Quantity",
            min_value=0.0,
            value=20.0
        )

        country = st.number_input(
            "Country Code",
            min_value=0.0,
            value=1.0
        )

    # Small Predict Button
    st.markdown("""
    <style>
    div.stButton > button {
        width: 160px;
        height: 42px;
        font-size: 16px;
    }
    </style>
    """, unsafe_allow_html=True)

    if st.button("Predict Segment"):

    # Create input dataframe
        input_df = pd.DataFrame([{
            "Recency": recency,
            "Frequency": frequency,
            "Monetary": monetary,
            "AvgOrderValue": avg_order,
            "TotalQuantity": quantity,
            "Country": country
        }])

        try:
            # Scale features
            X = scaler.transform(input_df[feature_columns])

            # Apply PCA
            X_pca = pca.transform(X)

            # Predict cluster
            cluster = int(kmeans.predict(X_pca)[0])

            # Distance from centroid
            distance = float(np.min(kmeans.transform(X_pca)))

            # Segment name
            segment = segment_names.get(cluster, "Unknown")

            st.success("Prediction completed successfully!")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Cluster", cluster)

            with col2:
                st.metric("Segment", segment)

            with col3:
                st.metric("Distance", round(distance, 2))

            st.subheader("Business Recommendation")
            st.info(recommendations.get(segment, "No recommendation available."))

        except Exception as e:
            st.error(f"Prediction failed: {e}")
    

# ==========================================================
# PCA VISUALIZATION
# ==========================================================

elif page=="Cluster Visualization":

    st.markdown("""
<div style="margin-top:18px;margin-bottom:22px;display:flex;align-items:center;gap:18px;">
    <span style="font-size:56px;">🌌</span>
    <h1 style="font-size:52px;font-weight:800;color:white;margin:0;">
        Customer Cluster Visualization
    </h1>
</div>
""", unsafe_allow_html=True)

    fig=px.scatter(
        df,
        x="PC1",
        y="PC2",
        color="Segment_Name",
        size="Frequency",
        hover_data=["CustomerID","Monetary"],
        opacity=.8
    )

    fig.update_layout(
                            paper_bgcolor="#0f172a",
                            plot_bgcolor="#0f172a",
                            font=dict(color="white", size=14),
                            title_font=dict(color="white", size=22),
                            legend=dict(font=dict(color="white")))

    st.plotly_chart(fig,use_container_width=True)
 
# elif page == "Predict Segment":

#     st.title("🎯 Predict Customer Segment")
#     st.write("Enter customer information to identify the customer segment.")

#     st.markdown("""
#     <style>
#     .predict-card {
#         background: #1b2942;
#         padding: 25px;
#         border-radius: 18px;
#         border: 1px solid rgba(255,255,255,0.08);
#         margin-bottom: 20px;
#     }

#     .result-card {
#         background: #16243b;
#         padding: 25px;
#         border-radius: 18px;
#         border: 1px solid rgba(255,255,255,0.08);
#         margin-top: 20px;
#     }

#     div.stButton > button {
#         width: 180px;
#         height: 45px;
#         font-size: 16px;
#         font-weight: bold;
#     }
#     </style>
#     """, unsafe_allow_html=True)

#     st.markdown(
#         '<div class="predict-card">',
#         unsafe_allow_html=True
#     )

#     col1, col2 = st.columns(2)

#     with col1:

#         recency = st.number_input(
#             "Recency",
#             min_value=0.0,
#             value=30.0,
#             step=1.0,
#             help="Number of days since the customer's last purchase."
#         )

#         frequency = st.number_input(
#             "Frequency",
#             min_value=0.0,
#             value=5.0,
#             step=1.0,
#             help="Number of purchases/orders made by the customer."
#         )

#         monetary = st.number_input(
#             "Monetary",
#             min_value=0.0,
#             value=500.0,
#             step=50.0,
#             help="Total amount spent by the customer."
#         )

#     with col2:

#         avg_order = st.number_input(
#             "Average Order Value",
#             min_value=0.0,
#             value=100.0,
#             step=10.0,
#             help="Average amount spent per order."
#         )

#         quantity = st.number_input(
#             "Total Quantity",
#             min_value=0.0,
#             value=20.0,
#             step=1.0,
#             help="Total number of items purchased."
#         )

#         country = st.number_input(
#             "Country Code",
#             min_value=0.0,
#             value=0.0,
#             step=1.0,
#             help="Numerical country encoding used during model training."
#         )

#     st.markdown('</div>', unsafe_allow_html=True)

#     # Prediction button
#     if st.button("🔍 Predict Segment"):

#         payload = {
#             "Recency": recency,
#             "Frequency": frequency,
#             "Monetary": monetary,
#             "AvgOrderValue": avg_order,
#             "TotalQuantity": quantity,
#             "Country": country
#         }

#         try:

#             response = requests.post(
#                 f"{API_URL}/segment",
#                 json=payload,
#                 timeout=10
#             )

#             if response.status_code == 200:

#                 result = response.json()

#                 st.markdown(
#                     '<div class="result-card">',
#                     unsafe_allow_html=True
#                 )

#                 st.subheader("📌 Prediction Result")

#                 result_col1, result_col2, result_col3 = st.columns(3)

#                 with result_col1:
#                     st.metric(
#                         "Cluster",
#                         result.get("cluster", "N/A")
#                     )

#                 with result_col2:
#                     st.metric(
#                         "Customer Segment",
#                         result.get("segment_name", "Unknown")
#                     )

#                 with result_col3:
#                     st.metric(
#                         "Distance from Centroid",
#                         result.get("distance_from_centroid", "N/A")
#                     )

#                 st.markdown("</div>", unsafe_allow_html=True)

#                 st.markdown("### 💡 Business Recommendation")

#                 st.info(
#                     result.get(
#                         "recommendation",
#                         "No recommendation available."
#                     )
#                 )

#             else:

#                 st.error(
#                     f"Prediction failed. Backend returned status code "
#                     f"{response.status_code}"
#                 )

#                 try:
#                     st.json(response.json())
#                 except:
#                     st.write(response.text)

#         except requests.exceptions.ConnectionError:

#             st.error(
#                 "❌ Could not connect to backend. "
#                 "Please make sure FastAPI is running on http://127.0.0.1:8000"
#             )

#         except requests.exceptions.Timeout:

#             st.error(
#                 "⏱️ Backend request timed out. "
#                 "Please check whether the FastAPI server is running correctly."
#             )

#         except Exception as e:

#             st.error(f"❌ Prediction error: {str(e)}")


# ==========================================================
# SEGMENT COMPARISON
# ==========================================================

elif page == "Segment Comparison":

    st.markdown("""
    <div style="margin-top:18px;margin-bottom:22px;display:flex;align-items:center;gap:18px;">
        <span style="font-size:56px;">⚖️</span>
        <h1 style="font-size:52px;font-weight:800;color:white;margin:0;">
            Segment Comparison
        </h1>
    </div>
    """, unsafe_allow_html=True)

    # Select ALL segments by default
    selected = st.multiselect(
        "Choose Segments",
        options=df["Segment_Name"].unique(),
        default=df["Segment_Name"].unique()
    )

    compare = df[df["Segment_Name"].isin(selected)]

    summary = compare.groupby("Segment_Name")[
    ["Monetary","Frequency","Recency","AvgOrderValue","TotalQuantity"]
].mean().round(2)
    st.dataframe(summary, use_container_width=True)

    # Radar Chart
    colors = {
        "Occasional Customer": "#7C3AED",
        "Frequent Buyer": "#10B981",
        "High Value Loyal": "#2563EB",
        "At Risk": "#F59E0B"
    }

    fig = go.Figure()

    for seg in summary.index:
        fig.add_trace(go.Scatterpolar(
            r=summary.loc[seg].values,
            theta=summary.columns,
            fill="toself",
            name=seg,
            line=dict(color=colors.get(seg, "#FFFFFF"), width=3)
        ))

    fig.update_layout(
        title=dict(
            text="Customer Segment Radar Comparison",
            x=0,
            font=dict(size=22, color="white")
        ),
        paper_bgcolor="#071329",
        plot_bgcolor="#071329",
        font=dict(color="white"),
        polar=dict(
            bgcolor="#071329",
            radialaxis=dict(
                visible=True,
                gridcolor="rgba(255,255,255,0.2)",
                linecolor="rgba(255,255,255,0.3)"
            ),
            angularaxis=dict(
                gridcolor="rgba(255,255,255,0.2)"
            )
        ),
        legend=dict(
            font=dict(color="white"),
            bgcolor="rgba(0,0,0,0)",
            orientation="h",
            y=1.12,
            x=0
        ),
        margin=dict(t=80, b=30, l=30, r=30)
    )

    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
elif page == "Clustering Evaluation":

    st.title("📊 Clustering Evaluation")
    st.write("Comparison of different clustering algorithms")

    # Load evaluation results
    evaluation_path = OUTPUT_DIR / "clustering_evaluation.csv"

    if evaluation_path.exists():

        evaluation = pd.read_csv(evaluation_path)

        # -----------------------------
        # Algorithm Comparison Table
        # -----------------------------
        st.subheader("Algorithm Comparison")

        st.dataframe(
            evaluation,
            use_container_width=True,
            hide_index=True
        )

        # -----------------------------
        # Silhouette Score
        # -----------------------------
        st.subheader("Silhouette Score")

        fig1 = px.bar(
            evaluation,
            x="Algorithm",
            y="Silhouette Score",
            title="Silhouette Score Comparison",
            text_auto=True
        )

        fig1.update_layout(
            xaxis_title="Algorithm",
            yaxis_title="Silhouette Score"
        )

        st.plotly_chart(fig1, use_container_width=True)

        # -----------------------------
        # Davies-Bouldin Score
        # -----------------------------
        st.subheader("Davies-Bouldin Index")

        fig2 = px.bar(
            evaluation,
            x="Algorithm",
            y="Davies-Bouldin",
            title="Davies-Bouldin Index Comparison",
            text_auto=True
        )

        fig2.update_layout(
            xaxis_title="Algorithm",
            yaxis_title="Davies-Bouldin Index"
        )

        st.plotly_chart(fig2, use_container_width=True)

        # -----------------------------
        # Calinski-Harabasz Score
        # -----------------------------
        st.subheader("Calinski-Harabasz Score")

        fig3 = px.bar(
            evaluation,
            x="Algorithm",
            y="Calinski-Harabasz",
            title="Calinski-Harabasz Score Comparison",
            text_auto=True
        )

        fig3.update_layout(
            xaxis_title="Algorithm",
            yaxis_title="Calinski-Harabasz Score"
        )

        st.plotly_chart(fig3, use_container_width=True)

        # -----------------------------
        # Best Algorithm
        # -----------------------------
        valid = evaluation.dropna(subset=["Silhouette Score"])

        if not valid.empty:

            best_algorithm = valid.loc[
                valid["Silhouette Score"].idxmax(),
                "Algorithm"
            ]

            best_score = valid["Silhouette Score"].max()

            st.success(
                f"Best performing algorithm based on Silhouette Score: "
                f"**{best_algorithm}** ({best_score:.3f})"
            )

    else:

        st.error(
            "clustering_evaluation.csv not found in the outputs folder."
        )
# ==========================================================
# BUSINESS INSIGHTS
# ==========================================================

else:

    st.markdown("""
<div style="margin-top:18px;margin-bottom:22px;display:flex;align-items:center;gap:18px;">
    <span style="font-size:56px;">💡</span>
    <h1 style="font-size:52px;font-weight:800;color:white;margin:0;">
        Business Insights
    </h1>
</div>
""", unsafe_allow_html=True)
    
    colors={
        "High Value Loyal":"#10b981",
        "Frequent Buyer":"#3b82f6",
        "Occasional Customer":"#f59e0b",
        "At Risk":"#ef4444"
    }

    tips={
        "High Value Loyal":"Offer VIP rewards.",
        "Frequent Buyer":"Cross-sell products.",
        "Occasional Customer":"Give loyalty discounts.",
        "At Risk":"Launch win-back campaigns."
    }

    for seg in df["Segment_Name"].unique():

        sub=df[df["Segment_Name"]==seg]

        st.markdown(f"""
        <div style="background:#1e293b;
        border-left:8px solid {colors.get(seg)};
        padding:18px;
        border-radius:15px;
        margin-bottom:18px;">

        <h3>{seg}</h3>

        👥 Customers: {len(sub)}<br>
        💰 Average Spend: ₹{sub['Monetary'].mean():.0f}<br>
        🛒 Average Orders: {sub['Frequency'].mean():.1f}<br>
        📅 Average Recency: {sub['Recency'].mean():.0f}<br><br>

        <b>Strategy:</b> {tips.get(seg)}

        </div>
        """,unsafe_allow_html=True)

