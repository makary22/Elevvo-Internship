# import streamlit as st 
# import pickle 
# import numpy as np


# kmeans_model = pickle.load(open("kmeans_model.pkl", "rb"))
# dbscan_model = pickle.load(open("dbscan_model.pkl", "rb"))

# st.title("🤖 Prediction & Clustering App")


# model_choice = st.selectbox(
#     "Select Model:",
#     ["KMeans (Clustering)", "DBSCAN (Clustering)"]
# )

# annual_income = st.number_input("Enter Annual Income ($):", min_value=15.0, max_value=137.0, step=1.0)
# spending_score = st.number_input("Enter Spending Score (1-100):", min_value=1.0, max_value=100.0, step=1.0)

# input_features = np.array([[annual_income, spending_score]])


# if model_choice == "KMeans (Clustering)":
#     if st.button("Predict Cluster (KMeans)"):
#         cluster = kmeans_model.predict(input_features)[0]+1
#         st.success(f"Customer belongs to Cluster: {cluster}")


# elif model_choice == "DBSCAN (Clustering)":
#     if st.button("Predict Cluster (DBSCAN)"):
#         cluster = dbscan_model.fit_predict(input_features)[0]
#         st.success(f"Customer belongs to Cluster: {cluster}")






import streamlit as st
import pickle
import numpy as np

# ================== Load Models ==================
kmeans_model = pickle.load(open("kmeans_model.pkl", "rb"))
dbscan_model = pickle.load(open("dbscan_model.pkl", "rb"))

# ================== Page Config ==================
st.set_page_config(
    page_title="Clustering App",
    page_icon="🤖",
    layout="wide"
)

# ================== Sidebar Navigation ==================
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🔮 Prediction", "ℹ️ Model Info", "📊 Visualizations"]
)

# ================== PREDICTION PAGE ==================
if page == "🔮 Prediction":
    st.title("🤖 Customer Segmentation")

    # Split page into 2 columns
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📝 Customer Information")
        annual_income = st.slider("💵 Annual Income (k$)", 15, 137, 50)
        spending_score = st.slider("🛒 Spending Score (1-100)", 1, 100, 50)

        model_choice = st.selectbox(
            "Select Model:",
            ["KMeans (Clustering)", "DBSCAN (Clustering)"]
        )

        input_features = np.array([[annual_income, spending_score]])

        predict_btn = st.button("🚀 Predict Cluster")

    with col2:
        st.header("📊 Prediction Results")
        if predict_btn:
            if model_choice == "KMeans (Clustering)":
                cluster = kmeans_model.predict(input_features)[0] + 1
                st.success(f"✅ Customer belongs to **Cluster {cluster}**")

                # Simple recommendation
                if cluster == 1:
                    st.warning("🟡 Medium Income & Spending – Balanced customers.")
                elif cluster == 2:
                    st.info("🔵 High Income – Potential premium customers.")
                elif cluster == 3:
                    st.error("🔴 Low Spending – Need engagement strategies.")
                else:
                    st.success("🟢 High Spending – Loyal customers.")

            elif model_choice == "DBSCAN (Clustering)":
                cluster = dbscan_model.fit_predict(input_features)[0]
                if cluster == -1:
                    st.error("⚠️ Customer is considered **Noise (-1)** by DBSCAN")
                else:
                    st.success(f"✅ Customer belongs to **Cluster {cluster}**")

# ================== MODEL INFO PAGE ==================
elif page == "ℹ️ Model Info":
    st.title("📘 Model Information")
    st.info("This app uses **KMeans** and **DBSCAN** clustering algorithms "
            "to segment customers based on **Annual Income** and **Spending Score**.")

    st.subheader("📌 KMeans")
    st.write("- Groups data into fixed number of clusters.")
    st.write("- Useful for well-separated spherical clusters.")

    st.subheader("📌 DBSCAN")
    st.write("- Detects clusters of varying shapes & sizes.")
    st.write("- Can identify noise/outliers (-1).")

# ================== VISUALIZATION PAGE ==================
elif page == "📊 Visualizations":
    st.title("📊 Data Visualizations")
    st.info("You can add scatter plots here to visualize clusters "
            "and compare customer positions.")

    # Example placeholder
    st.write("👉 Visualization will be added here later.")

