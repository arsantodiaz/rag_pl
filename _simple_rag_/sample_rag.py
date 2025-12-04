import streamlit as st
import os
import glob
import time
import streamlit_authenticator as stauth
import chromadb
import requests
import json

# --- IMPORTS MODERN (LCEL) ---
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma

# --- CONFIG ---
st.set_page_config(page_title="RAG PrecentiaLife (Server Mode)", page_icon="⚡", layout="wide")

# --- KONFIGURASI KONEKSI ---
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = os.getenv("CHROMA_PORT", 8000)
COLLECTION_NAME = "precentialife_collection"
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# --- PATH LOKAL ---
WORKSPACE_ROOT = "D:/Workspace_PL"
DIR_DOCS = os.path.join(WORKSPACE_ROOT, "PL_UserGuide")
DIR_CODE = os.path.join(WORKSPACE_ROOT, "precentialife")
DIR_DDL = os.path.join(WORKSPACE_ROOT, "precentialife_ddl")


# --- SOLUSI FINAL: KELAS EMBEDDING KUSTOM (DIPERBAIKI) ---
class DirectOllamaEmbeddings(Embeddings):
    def __init__(self, model: str, base_url: str, timeout: int = 3600):
        self.model = model
        # Endpoint /api/embed lebih stabil untuk embedding
        self.base_url = base_url.rstrip('/') + "/api/embed"
        self.timeout = timeout

    def _embed(self, text: str) -> list[float]:
        try:
            # PERBAIKAN 1: Menggunakan key "input" alih-alih "prompt"
            payload = {
                "model": self.model,
                "input": text
            }

            res = requests.post(
                self.base_url,
                json=payload,
                timeout=self.timeout
            )
            res.raise_for_status()
            response_json = res.json()

            # PERBAIKAN 2: Cek 'embeddings' (jamak) dulu, lalu fallback ke 'embedding'
            if "embeddings" in response_json:
                # Endpoint /api/embed mengembalikan list of lists, kita ambil yang pertama
                return response_json["embeddings"][0]
            elif "embedding" in response_json:
                return response_json["embedding"]
            else:
                raise KeyError(
                    f"Kunci 'embedding' tidak ditemukan. Keys received: {list(response_json.keys())}")

        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Gagal terhubung ke Ollama di {self.base_url}: {e}")
        except json.JSONDecodeError:
            raise ValueError(f"Gagal mendekode JSON dari respons Ollama. Respons mentah: {res.text}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


# --- DEFINISI FUNGSI ---

def get_vectorstore(client, embeddings):
    return Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )


def initialize_knowledge_base(status_container, embeddings, client):
    all_documents = []
    status_log = []

    def process_files(directory, file_patterns, category):
        found_files = False
        if os.path.exists(directory):
            for pattern, loader_cls in file_patterns.items():
                search_path = os.path.join(directory, "**", pattern)
                for file_path in glob.glob(search_path, recursive=True):
                    found_files = True
                    try:
                        status_container.write(f"📄 Membaca: {os.path.basename(file_path)}")
                        loader = loader_cls(file_path)
                        docs = loader.load()
                        for d in docs:
                            d.metadata['category'] = category
                        all_documents.extend(docs)
                    except Exception as e:
                        status_container.write(f"❌ Gagal membaca {os.path.basename(file_path)}: {e}")
            if not found_files:
                status_log.append(f"ℹ️ Tidak ada file yang cocok ditemukan di {directory}")
        else:
            status_log.append(f"⚠️ Folder tidak ditemukan: {directory}")

    # Memproses file sesuai path yang didefinisikan
    process_files(DIR_DOCS, {"*.pdf": PyPDFLoader, "*.docx": Docx2txtLoader, "*.txt": TextLoader}, "USER_GUIDE")
    process_files(DIR_DDL, {"*.sql": TextLoader}, "DATABASE_SCHEMA")
    process_files(DIR_CODE, {"*.java": TextLoader, "*.xml": TextLoader, "*.properties": TextLoader}, "SOURCE_CODE")

    if not all_documents:
        status_log.append("❌ Gagal: Tidak ada dokumen yang berhasil dimuat.")
        return None, status_log

    status_container.write("📚 Memecah dokumen menjadi chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    doc_chunks = text_splitter.split_documents(all_documents)
    status_log.append(f"✅ Total Chunks: {len(doc_chunks)}")

    status_container.write("🧠 Mengirim data ke ChromaDB Server... Ini mungkin butuh waktu lama.")

    vectordb = Chroma.from_documents(
        documents=doc_chunks,
        embedding=embeddings,
        client=client,
        collection_name=COLLECTION_NAME,
    )

    return vectordb, status_log


def get_lcel_chain(vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 7})
    llm = ChatOllama(model="llama3", temperature=0, base_url=OLLAMA_URL)
    template = """Anda adalah Senior Architect. Jawab pertanyaan berdasarkan Context berikut:
    CONTEXT: {context}
    PERTANYAAN: {question}
    Jelaskan alurnya (Business -> Logic -> Data). Jika tidak ada di context, bilang tidak tahu."""
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    return rag_chain, retriever


# --- AUTH ---
config_data = {
    'credentials': {
        'usernames': {
            'admin': {'name': 'Admin', 'password': '$2b$12$41rI12HT4lM8S511cIgvJe26O0znj3EHVj/k.AjR4qIjHY6KLxjS.'},
            'user': {'name': 'Staff', 'password': '$2b$12$41rI12HT4lM8S511cIgvJe26O0znj3EHVj/k.AjR4qIjHY6KLxjS.'}
        }
    },
    'cookie': {'expiry_days': 1, 'key': 'secret', 'name': 'rag_cookie'},
    'preauthorized': {'emails': []}
}
authenticator = stauth.Authenticate(config_data['credentials'], config_data['cookie']['name'],
                                    config_data['cookie']['key'], config_data['cookie']['expiry_days'])
authenticator.login()

# --- LOGIKA UTAMA APLIKASI ---
if st.session_state["authentication_status"]:
    if "chroma_client" not in st.session_state:
        st.session_state["chroma_client"] = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

    with st.sidebar:
        st.write(f"Login: {st.session_state['name']}")
        authenticator.logout('Logout', 'main')
        if st.session_state.get('username') == 'admin':
            if st.button("🚀 Re-Scan"):
                with st.status("Memulai proses Re-Scan...", expanded=True) as status:
                    status.write("Menghapus koleksi lama dari server ChromaDB...")
                    try:
                        st.session_state["chroma_client"].delete_collection(name=COLLECTION_NAME)
                        status.write("Koleksi lama berhasil dihapus.")
                    except Exception as e:
                        status.warning(f"Tidak dapat menghapus koleksi (mungkin belum ada): {e}")

                    st.cache_resource.clear()

                    status.write("Memulai indexing untuk database baru...")
                    current_embeddings = DirectOllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_URL)
                    db, logs = initialize_knowledge_base(status, current_embeddings, st.session_state["chroma_client"])

                    for log_entry in logs:
                        status.write(log_entry)

                    if db:
                        st.session_state['vectordb'] = db
                        status.update(label="Re-Scan Selesai!", state="complete", expanded=False)
                    else:
                        status.update(label="Re-Scan Gagal!", state="error")

    if 'vectordb' not in st.session_state:
        try:
            current_embeddings = DirectOllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_URL)
            st.session_state['vectordb'] = get_vectorstore(st.session_state["chroma_client"], current_embeddings)
        except Exception as e:
            st.sidebar.warning(f"Belum ada database. Harap lakukan Re-Scan. Error: {e}")

    st.title("🧬 RAG PrecentiaLife (Server Mode)")
    if prompt := st.chat_input("Tanya PrecentiaLife..."):
        st.chat_message("user").write(prompt)
        if 'vectordb' in st.session_state:
            with st.chat_message("assistant"):
                with st.spinner("Menganalisa..."):
                    try:
                        chain, retriever = get_lcel_chain(st.session_state['vectordb'])
                        response = chain.invoke(prompt)
                        st.write(response)
                        with st.expander("Lihat Referensi Dokumen"):
                            docs = retriever.invoke(prompt)
                            for d in docs:
                                st.caption(
                                    f"[{d.metadata.get('category')}] {os.path.basename(d.metadata.get('source', ''))}")
                    except Exception as e:
                        st.error(f"Terjadi error saat memproses pertanyaan: {e}")
        else:
            st.error("Database belum siap. Harap lakukan 'Re-Scan' melalui sidebar.")

elif st.session_state["authentication_status"] is False:
    st.error('Login Gagal')
elif st.session_state["authentication_status"] is None:
    st.warning('Silakan Login')