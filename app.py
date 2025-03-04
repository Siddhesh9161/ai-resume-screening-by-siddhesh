import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import time  # For progress updates

# Apply Background Image
page_bg_img = '''
<style>
.stApp {
    background: url("https://static.vecteezy.com/system/resources/previews/011/132/871/original/elegant-background-concept-that-will-make-your-graphic-design-or-web-design-look-professional-vector.jpg") no-repeat center center fixed;
    background-size: cover;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip() if text else "No readable text found."

# Function to rank resumes
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
    
    return cosine_similarities

# Streamlit UI
st.title("📄 AI Resume Screening & Ranking System By Siddhesh")
st.write("Upload resumes and compare them to the job description to rank candidates!")

job_description = st.text_area("📝 Enter the Job Description")
uploaded_files = st.file_uploader("📂 Upload PDF Resumes", type=["pdf"], accept_multiple_files=True)

if uploaded_files and job_description:
    with st.spinner("🔄 Processing resumes..."):
        time.sleep(1)  # Simulating loading time

    resumes = []
    progress_bar = st.progress(0)

    for i, file in enumerate(uploaded_files):
        text = extract_text_from_pdf(file)
        resumes.append(text)
        progress_bar.progress((i + 1) / len(uploaded_files))

    scores = rank_resumes(job_description, resumes)
    ranked_resumes = sorted(zip(uploaded_files, scores, resumes), key=lambda x: x[1], reverse=True)

    st.subheader("🏆 Ranked Resumes")
    
    results = []
    for i, (file, score, extracted_text) in enumerate(ranked_resumes, start=1):
        st.write(f"**{i}. {file.name}** - Score: {score:.2f}")
        with st.expander("📖 View Extracted Text Snippet"):
            st.write(extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text)
        results.append([file.name, score])

    # Convert to DataFrame for downloading
    df_results = pd.DataFrame(results, columns=["Resume Name", "Score"])
    
    # Provide CSV download option
    csv = df_results.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Rankings", data=csv, file_name="resume_rankings.csv", mime="text/csv")
