import streamlit as st
import json
import os
import re
import pandas as pd
from pypdf import PdfReader
from AIconomy import macro_analysis
import uuid
from pathlib import Path
import requests
import numpy as np
import pickle
import time

session_history = "istoric_chat.json"
# utilizator > CrewAI process > Web researcher > data calculator > critic / raport <-> Gemini
VECTOR_CACHE_DIR = "vector_cache"
os.makedirs(VECTOR_CACHE_DIR, exist_ok=True)

def get_api_key():
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return None


def chunk_text(text, chunk_size=1500, overlap=300):
    """segmntare"""
    chunks_finale = []
    pas = chunk_size - overlap

    for i in range(0, len(text), pas):
        bucata = text[i : i + chunk_size]
        chunks_finale.append(bucata)
    return chunks_finale


def preproc_emb(nume_files, rawtext, api_key):
    """vectorszeaza documentul - respecta rate-limits + salvare pe disc (.pkl)"""
    cache_path = os.path.join(VECTOR_CACHE_DIR, f"{nume_files}.pkl")

    #l am procesat deja in trecut -> incarcam de pe disc!
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    #prima oara -> procesez textul
    chunks = chunk_text(rawtext)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents?key={api_key}"

    all_data = []
    batch_size = 50  #pot stabilitate

    progrbar = st.progress(0, text=f"Processing RAG for {nume_files}... Please wait")

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        payload = [{"model": "models/gemini-embedding-001", "content": {"parts": [{"text": c}]}} for c in batch]

        resp = requests.post(url, headers={'Content-Type': 'application/json'}, json={"requests": payload})
        if resp.status_code == 200:
            vectors = [item['values'] for item in resp.json().get('embeddings', [])]
            for c, v in zip(batch, vectors):
                all_data.append({"chunk": c, "vector": v})
        else:
            st.error(f"ERROR API Embeddings at pag {i}: {resp.text}")
            break

        #updating the UI
        progres = min((i + batch_size) / len(chunks), 1.0)
        progrbar.progress(progres, text=f"vectorszation underway: {int(progres*100)}% completed...")
        time.sleep(1.5)

    progrbar.empty()

    #salvez vectorsi generati pe disc
    if all_data:
        with open(cache_path, "wb") as f:
            pickle.dump(all_data, f)
    return all_data



def extrage_text_rag(docs, query, api_key, top_k=4):
    """RAG - Gaseste top 4 cele mai relevante paragrafe pentru intrebare"""
    all_chunks = []

    #citire + vectorszare (instantaneu daca doc a mai fost citit)
    for files in docs:
        ext = files.suffix.lower()
        rawtext = ""

        try:
            if ext == ".pdf":
                reader = PdfReader(str(files))
                rawtext = "\n".join((pagina.extract_text() or "") for pagina in reader.pages)
            elif ext == ".csv":
                rawtext = pd.read_csv(files, sep=';').to_string()

            if rawtext:
                date_vect = preproc_emb(files.name, rawtext, api_key)
                for item in date_vect:
                    item["sursa"] = files.name
                all_chunks.extend(date_vect)
        except Exception as e:
            st.sidebar.error(f"ERROR WHILE PROCESSING {files.name}: {e}")

    if not all_chunks:
        return ""

    # vectorszare intrebare user
    url_query = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={api_key}"
    resp_q = requests.post(url_query, headers={'Content-Type': 'application/json'},
                           json={"model": "models/gemini-embedding-001", "content": {"parts": [{"text": query}]}})

    if resp_q.status_code != 200:
        return ""
    query_vector = resp_q.json()['embedding']['values']

    # similaritate cosinus (cu dom cautat)
    for item in all_chunks:
        v1, v2 = np.array(query_vector), np.array(item["vector"])
        item["scor"] = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    #sortare descr dupa scor si extragere
    all_chunks.sort(key=lambda x: x["scor"], reverse=True)
    top_chunks = all_chunks[:top_k]

    return "\n\n".join([f"--- extracted from: {c['sursa']} (Importance: {c['scor']:.2f}) ---\n{c['chunk']}" for c in top_chunks])



def gaseste_docs_relev(prompt: str, folder="docs"):
    """
    Cauta automat docsle din baza locala "docs/" relevante pentru intrebarea utilizatorului
    Pentru moment folosim o filtrare simpla dupa cuvinte-cheie. NU facem si NU aplicam inca semantic search.
    """
    folder_path = Path(folder)
    if not folder_path.exists():
        return []

    prompt_lower = prompt.lower()
    categorii = {
        "pib": [
            "pib",
            "gdp",
            "produs intern brut",
            "crestere economica",
            "crestere_economica",
            "creștere economică",
            "growth"
        ],

        "inflatie": [
            "inflatie",
            "inflație",
            "ipc",
            "preturi",
            "prețuri",
            "inflation"
        ],

        "fiscal": [
            "deficit",
            "buget",
            "executia",
            "execuția",
            "datorie",
            "fiscal"
        ],

        "financiar": [
            "stabilitate",
            "stabilitate_financiara",
            "stabilitate financiara",
            "stabilitate financiară",
            "banci",
            "bănci",
            "dobanda",
            "dobândă",
            "financial stability",
            "deficit"
        ]
    }

    cat_detect = []
    for categorie, cuvinte in categorii.items():
        if any(cuvant in prompt_lower for cuvant in cuvinte):
            cat_detect.append(categorie)

    # daca nu recunoastem categoria - nu pp ca orice document este relevant!!
    if not cat_detect:
        return []
    filese = []

    for files in folder_path.iterdir():
        if not files.is_file():
            continue

        nume = files.name.lower()
        scor = 0

        for categorie in cat_detect:
            if categorie == "pib":
                if "pib" in nume or "gdp" in nume:
                    scor += 3

            elif categorie == "inflatie":
                if "inflatia" in nume or "inflation" in nume:
                    scor += 3

            elif categorie == "fiscal":
                if ("executia" in nume or "buget" in nume or "fiscal" in nume):
                    scor += 3

            elif categorie == "financiar":
                if ("stabilitate" in nume or "financiar" in nume):
                    scor += 3

        if scor > 0:
            filese.append((scor, files))

    #cele mai relevante date - primele
    filese.sort(key=lambda x: x[0], reverse=True)
    return [files for scor, files in filese]



#salvarea istoricului / incarcarea
def incarca_istoric():
    if os.path.exists(session_history):
        with open(session_history, "r", encoding="utf-8") as f:
            date = json.load(f)
            # baza_date trebuie sa fie mereu un dict {chat_id: {...}}.
            # Daca filesul e corupt- tratat ca istoric gol!
            if isinstance(date, dict):
                return date
    return {}


def salveaza_istoric(mesaje):
    with open(session_history, "w", encoding="utf-8") as f:
        json.dump(mesaje, f, ensure_ascii=False, indent=4)


def extrage_grafic(text):
    """
    Extrage datele JSON pentru grafic; return (date_grafic, text_no_json).
    """
    start_idx = text.find("```")
    if start_idx == -1:
        return None, text

    end_idx = text.find("```", start_idx + 3)
    if end_idx == -1:
        return None, text

    #interiorul blocului extras + elimin "json"
    bloc_brut = text[start_idx + 3 : end_idx].strip()
    if bloc_brut.lower().startswith("json"):
        bloc_brut = bloc_brut[4:].strip()

    if '"grafic"' not in bloc_brut:
        return None, text

    #acoladele adaugte daca lipsesc
    if not bloc_brut.startswith("{"):
        bloc_brut = "{" + bloc_brut
    if not bloc_brut.endswith("}"):
        bloc_brut = bloc_brut + "}"

    #idem, sect json taiata
    text_no_json = (text[:start_idx] + text[end_idx + 3:]).strip()

    try:
        return json.loads(bloc_brut), text_no_json
    except Exception:
        return None, text


st.set_page_config(page_title="AI Economic Assistant", page_icon="📈", layout="wide")

#verif cheia api cu un mesaj clar in loc de o eroare
api_key = get_api_key()
if not api_key:
    st.error(
        " GEMINI_API_KEY is missing!.\n\n"
        "Add this as a path variable or into a file: "
        "`.streamlit/secrets.toml` (done manually, "
        " with:\n\n"
        '```toml\nGEMINI_API_KEY = "YOUR_KEY"\n```\n\n'
        "Then, please RESTRAT the app!"
    )
    st.stop()
os.environ["GEMINI_API_KEY"] = api_key


# conf stari
if "baza_date" not in st.session_state:
    st.session_state.baza_date = incarca_istoric()
if "ecran_curent" not in st.session_state:
    st.session_state.ecran_curent = "landing_page"  # mereu se incepe cu ecranul principal
if "chat_activ_id" not in st.session_state:
    st.session_state.chat_activ_id = None

# STAGE 1 - LANDING PAGE
if st.session_state.ecran_curent == "landing_page":
    st.markdown("<h1 style='text-align: center;'>🤖 AIconomy #Command Center</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: gray;'>Hybrid macroeconomic analysis system</h4>", unsafe_allow_html=True)
    st.write("---")
    col_logo1, col_logo2, col_logo3 = st.columns([2, 1, 2])  # Centrare
    with col_logo2:
        if os.path.exists("logo.png"):
            st.image("logo.png", use_container_width=True)

    # Butonul central pentru o sesiune noua
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("➕ New chat (start a new analysis)", use_container_width=True, type="primary"):
            nou_id = str(uuid.uuid4())[:8]
            st.session_state.baza_date[nou_id] = {
                "titlu": "New Analysis session",
                "mesaje": []
            }
            salveaza_istoric(st.session_state.baza_date)

            st.session_state.chat_activ_id = nou_id
            st.session_state.ecran_curent = "dashboard"
            st.rerun()

    st.write("<br><br>", unsafe_allow_html=True)
    st.subheader(f"📂Previous sessions: ({len(st.session_state.baza_date)})")

    all_sessions = list(st.session_state.baza_date.keys())
    new_sessions = all_sessions[-16:]

    if len(new_sessions) > 0:
        cols_grid = st.columns(8)

        for i, ses_id in enumerate(reversed(new_sessions)):
            col_curenta = cols_grid[i % 8]
            with col_curenta:
                with st.container(border=True):
                    titlu = st.session_state.baza_date[ses_id].get("titlu", "Analysis")
                    titlu_scurt = titlu[:30] + "..." if len(titlu) > 30 else titlu
                    st.markdown("<h1 style='text-align: center; margin-bottom: 0rem; margin-top: 0rem;'>📊</h1>", unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align: center; font-size: 14px; font-weight: 500; height: 40px;'>{titlu_scurt}</p>", unsafe_allow_html=True)
                    sub_col_chat, sub_col_del = st.columns([3, 1])


                    with sub_col_chat:
                        if st.button("Open", key=f"btn_{ses_id}", use_container_width=True):
                            st.session_state.chat_activ_id = ses_id
                            st.session_state.ecran_curent = "dashboard"
                            st.rerun()

                    with sub_col_del:
                            if st.button("🗑️", key=f"del_{ses_id}", help="Delete this session"):
                                del st.session_state.baza_date[ses_id]
                                salveaza_istoric(st.session_state.baza_date)
                                st.rerun()

    else:
        st.info("Currently, there are no conversations saved at this moment! Press the upper button to start a new one.")


# caz 2 - folosit - intrare in dashboard
elif st.session_state.ecran_curent == "dashboard":

    chat_id = st.session_state.chat_activ_id
    this_session = st.session_state.baza_date[chat_id]
    st.title(this_session["titlu"])

    with st.sidebar:
        if st.button("⬅️ Back to main menu", use_container_width=True):
            st.session_state.ecran_curent = "landing_page"
            st.rerun()

        st.divider()
        st.header("📂 Local Romanian Database")
        docs_path = Path("docs")

        if docs_path.exists():
            filese_docs = [
                f
                for f in docs_path.iterdir()
                if f.is_file()
                and f.suffix.lower() in [".pdf", ".csv"]
            ]
            st.write(f"📚 {len(filese_docs)} documents available locally")
        else:
            filese_docs = []
            st.warning("The folder <<docs/>> does NOT exist")

        st.divider()
        st.subheader("➕ Add document")

        filese_incarcate = st.file_uploader(
            "Upload documents (.pdf / .csv)",
            type=["pdf", "csv"],
            accept_multiple_files=True
        )

        if filese_incarcate:
            docs_path.mkdir(exist_ok=True)
            for files in filese_incarcate:
                cale_files = docs_path / files.name
                with open(cale_files, "wb") as f:
                    f.write(files.getbuffer())
            st.success(f"✅ {len(filese_incarcate)} document(s) added to local database")

    # CHAT AREA
    for msg in this_session["mesaje"]:
        with st.chat_message(msg["role"]):
#trecem si mesajele din istoric prin functia de decupare a graficului
                date_grafic, text_no_json = extrage_grafic(msg["content"])

                if date_grafic and "grafic" in date_grafic:
                    st.markdown(text_no_json)

                    df = pd.DataFrame(list(date_grafic["grafic"].items()), columns=["Period", "Value"])
                    df.set_index("Period", inplace=True)
                    st.divider()
                    st.subheader("📊 Graphic visualization (History):")

                    if date_grafic.get("tip_grafic") == "bar_chart":
                        st.bar_chart(df)
                    else:
                        st.line_chart(df)
                else:
                    st.markdown(msg["content"])

    if prompt := st.chat_input("What do you want to know? Ask me anything regarding the economy activity or predictions"):
        this_session["mesaje"].append({"role": "user", "content": prompt})

        if len(this_session["mesaje"]) == 1:
            this_session["titlu"] = (f"Analysis: {prompt[:30]}...")

        salveaza_istoric(st.session_state.baza_date)

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Processing..."):

                #sliding window pentru conversatie (pana la 16 mesaje)
                istoric_lim = this_session["mesaje"][-16:]

                context_recent = ""
                for msg in istoric_lim:
                    context_recent += f"{msg['role'].upper()}: {msg['content']}\n"

                #identif docsle relevante si extragem text brut din ele
                docs_relev = gaseste_docs_relev(prompt)
                docs_local = False

                if docs_relev:
                    st.sidebar.write("🔎 Documents selected for this query:")
                    for doc in docs_relev:
                        st.sidebar.write(f"- {doc.name}")

                    context_docs = extrage_text_rag(docs_relev, prompt, api_key)

                    if context_docs:
                        context_recent += f"\n\n[LOCAL DOCUMENTS CONTEXT]\n{context_docs}"
                        docs_local = True

                #apelam CrewAI
                rezultat_final = macro_analysis(prompt, context_recent, docs_local)
                text_complet = str(rezultat_final)

                # parsare JSON + grafic
                date_grafic, text_no_json = extrage_grafic(text_complet)

                if date_grafic and "grafic" in date_grafic:
                    st.markdown(text_no_json)

                    df = pd.DataFrame(list(date_grafic["grafic"].items()), columns=["Period", "Value"])
                    df.set_index("Period", inplace=True)
                    st.divider()
                    st.subheader("📊 Graphic visualization:")

                    if date_grafic.get("tip_grafic") == "bar_chart":
                        st.bar_chart(df)
                    else:
                        st.line_chart(df)
                else:
                    st.markdown(text_complet)

        err_msj = ("ERROR" in text_complet) or ("MISSING INFORMATION" in text_complet) or ("crashed" in text_complet)
        if err_msj:
            this_session["mesaje"].pop()
            salveaza_istoric(st.session_state.baza_date)

        else:
            this_session["mesaje"].append({"role": "assistant", "content": text_complet})
            salveaza_istoric(st.session_state.baza_date)
