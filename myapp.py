import asyncio
import aiohttp
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from bs4 import BeautifulSoup
import streamlit as st

# SetUp


# Centered logo from GitHub
st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <img src='https://raw.githubusercontent.com/YannisBalasis/search_engine/main/logo-2025-final.png' width='200'>
    </div>
""", unsafe_allow_html=True)

# Fancy header
st.markdown("""
    <div style='
        background: linear-gradient(90deg, #004d99 0%, #007acc 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.3);
        text-align: center;
        margin-bottom: 20px;
    '>
        <h1 style='color: white; font-size: 2.5em;'>Patras Medicine iGEM 2025 - Search Engine</h1>
    </div>
""", unsafe_allow_html=True)

# Sleek info box
st.markdown("""
<div style='
    background-color: #e6f2ff;
    padding: 15px;
    border-left: 5px solid #004d99;
    border-radius: 10px;
    margin-bottom: 20px;
    font-size: 1.1em;
'>
    🔬 Αυτό είναι το εργαλείο αναζήτησης της φοιτητικής ομάδας <strong>Patras Medicine iGEM 2025</strong>.<br><br>
    ➔ Δώσε λέξεις-κλειδιά στο πεδίο παρακάτω.<br>
    ➔ Τα αποτελέσματα εμφανίζονται ταξινομημένα κατά σχετικότητα.
</div>
""", unsafe_allow_html=True)

query = st.text_input("🔍 Αναζήτησε άρθρα:", placeholder="π.χ. Molecular mechanisms of obesity-related diabetes")


# Load model once
@st.cache_resource
def load_model():
    return SentenceTransformer('pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb')

model = load_model()

# Functions
async def fetch_pubmed(session, query):
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&retmode=json&term={query}&retmax=5"
    async with session.get(url) as resp:
        data = await resp.json()
        ids = data['esearchresult'].get('idlist', [])
        articles = []
        for pmid in ids:
            summary_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={pmid}&retmode=json"
            async with session.get(summary_url) as sum_resp:
                sum_data = await sum_resp.json()
                if 'result' in sum_data and pmid in sum_data['result']:
                    result = sum_data['result'][pmid]
                    title = result.get('title', '')
                else:
                    continue
            fetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmid}&retmode=xml"
            async with session.get(fetch_url) as fetch_resp:
                xml_text = await fetch_resp.text()
                abstract = ""
                try:
                    if '<Abstract>' in xml_text:
                        raw_abstract = xml_text.split('<Abstract>')[1].split('</Abstract>')[0]
                        soup = BeautifulSoup(raw_abstract, 'lxml')
                        abstract = soup.get_text().strip()
                except:
                    pass
            articles.append({'title': title, 'abstract': abstract, 'source': 'PubMed'})
        return articles

async def fetch_europe_pmc(session, query):
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={query}&format=json&pageSize=5"
    async with session.get(url) as resp:
        data = await resp.json()
        articles = []
        for hit in data.get('resultList', {}).get('result', []):
            articles.append({'title': hit.get('title', ''), 'abstract': hit.get('abstractText', ''), 'source': 'EuropePMC'})
        return articles

async def fetch_arxiv(session, query):
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=5"
    async with session.get(url) as resp:
        text = await resp.text()
        entries = text.split('<entry>')
        articles = []
        for entry in entries[1:]:
            try:
                title = entry.split('<title>')[1].split('</title>')[0].strip()
                summary = entry.split('<summary>')[1].split('</summary>')[0].strip()
                articles.append({'title': title, 'abstract': summary, 'source': 'arXiv'})
            except IndexError:
                continue
        return articles

async def fetch_pmc(session, query):
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pmc&retmode=json&term={query}&retmax=5"
    async with session.get(url) as resp:
        data = await resp.json()
        ids = data['esearchresult'].get('idlist', [])
        articles = []
        for pmcid in ids:
            summary_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pmc&id={pmcid}&retmode=json"
            async with session.get(summary_url) as sum_resp:
                sum_data = await sum_resp.json()
                if 'result' in sum_data and pmcid in sum_data['result']:
                    result = sum_data['result'][pmcid]
                    articles.append({'title': result.get('title', ''), 'abstract': result.get('elocationid', ''), 'source': 'PubMed Central'})
        return articles

async def search_all_sources(query):
    async with aiohttp.ClientSession() as session:
        pubmed_task = asyncio.create_task(fetch_pubmed(session, query))
        europe_pmc_task = asyncio.create_task(fetch_europe_pmc(session, query))
        arxiv_task = asyncio.create_task(fetch_arxiv(session, query))
        pmc_task = asyncio.create_task(fetch_pmc(session, query))
        results = await asyncio.gather(pubmed_task, europe_pmc_task, arxiv_task, pmc_task)
        all_articles = [item for sublist in results for item in sublist]
        return all_articles

def rank_articles_semantic(articles, query):
    df = pd.DataFrame(articles)
    df['content'] = df['title'] + " " + df['abstract']
    article_embeddings = model.encode(df['content'].tolist(), convert_to_tensor=True)
    query_embedding = model.encode(query, convert_to_tensor=True)
    similarities = util.cos_sim(query_embedding, article_embeddings).cpu().numpy().flatten()
    df['similarity'] = similarities
    return df.sort_values(by='similarity', ascending=False)

# Main Logic
if query:
    with st.spinner("Searching and ranking..."):
        articles = asyncio.run(search_all_sources(query))
        if articles:
            ranked = rank_articles_semantic(articles, query)
            for idx, row in ranked.iterrows():
                st.subheader(f"{row['title']} ({row['source']})")
                st.write(row['abstract'])
                st.caption(f"Similarity: {row['similarity']:.2f}")
                st.markdown("---")
        else:
            st.warning("No articles found.")
