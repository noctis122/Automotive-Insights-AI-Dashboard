# app.py
import streamlit as st
from analyzer import CarReviewAnalyzer

st.title("🚗 Automotive Insights Dashboard")
uploaded_file = st.file_uploader("Upload reviews CSV")

if uploaded_file:
    analyzer = CarReviewAnalyzer()
    df = pd.read_csv(uploaded_file)
    with st.spinner("Analyzing reviews..."):
        results = analyzer.analyze_reviews(df['Review'].tolist())
    
    st.subheader("Sentiment Distribution")
    sentiments = [r['label'] for r in results['sentiments']]
    st.bar_chart(pd.Series(sentiments).value_counts())
    
    st.subheader("Toxicity Analysis")
    toxic_scores = [t['scores']['toxic'] for t in results['toxicity_scores']]
    st.line_chart(toxic_scores)
    
    st.download_button("Download Report", analyzer.generate_report(results))