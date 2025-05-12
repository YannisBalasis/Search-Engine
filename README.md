# Patras Medicine iGEM 2025 - Literature Search Engine

 An intelligent and user-friendly literature search engine developed to support the student research team **Patras Medicine iGEM 2025**.

 Created by **Yannis Balasis**, as part of the **Dry Lab** of Patras Medicine iGEM 2025.

---

##  Purpose

This tool was built to help our team members efficiently **search for scientific articles** and **literature** relevant to our iGEM project. It supports search from PubMed, PubMed Central, EuropePMC, and arXiv, and uses **semantic similarity** via BioBERT to rank results by relevance.

---

##  Features

-  Searches across **four major databases**: PubMed, PMC, EuropePMC, and arXiv  
-  Uses **BioBERT** for semantic similarity ranking  
-  Fast performance via **asynchronous data fetching**  
-  Simple and interactive **Streamlit UI**  
-  Multilingual-friendly (Greek UI available)

---

## Requirements
Dependencies listed in requirements.txt:

nginx
Copy
Edit
aiohttp
pandas
sentence-transformers
beautifulsoup4
lxml
streamlit

## Interface Preview
<img src="https://raw.githubusercontent.com/YannisBalasis/search_engine/main/logo-2025-final.png" width="150" align="center">

## Credits
Developed by: Yannis Balasis, Dry Lab member of Patras Medicine iGEM 2025
Semantic Model: BioBERT via Sentence Transformers
Data Sources: PubMed, arXiv, EuropePMC, PMC



##  Installation



```bash
git clone https://github.com/YannisBalasis/literature-search-engine.git
cd literature-search-engine

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt


streamlit run app.py

