import os
import tempfile
import streamlit as st
import pandas as pd
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from typing import List, Tuple
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate as CoreChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
st.set_page_config(page_title="AI Legal Toolkit", layout="wide", initial_sidebar_state="expanded")

try:
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)
except FileNotFoundError:
    st.error("config.yaml not found. Please create the config file.")
    st.stop()

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
)


if 'authentication_status' not in st.session_state or st.session_state['authentication_status'] is None:
    
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        try:
            authenticator.login()
        except Exception as e:
            st.error(e)
            
    with tab2:
        try:
            result = authenticator.register_user(location='main')
            
            if result:
                email, username, name = result
                st.success('User registered successfully')
                with open('config.yaml', 'w') as file:
                    yaml.dump(config, file, default_flow_style=False)
        except Exception as e:
            st.error(f"Registration Error: {e}")

if st.session_state.get("authentication_status"):

    st.markdown(
        """
        <style>
        :root{
          --bg:#071019;
          --panel:#0c1220;
          --muted:#9aa4b2;
          --text:#e6eef7;
          --accent:#f6b042;
        }
        .stApp { background: linear-gradient(180deg,var(--bg), #041018); color:var(--text); }
        .header {
          display:flex; align-items:center; gap:16px; margin-bottom:18px;
          background: linear-gradient(90deg, rgba(12,18,28,0.65), rgba(8,12,18,0.6));
          padding:18px; border-left:4px solid var(--accent); border-radius:8px;
          box-shadow: 0 10px 30px rgba(2,6,23,0.6);
        }
        .brand { font-size:20px; font-weight:700; }
        .tagline { color:var(--muted); margin-top:4px; font-size:13px; }
        .panel {
          background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
          border-radius: 8px; padding: 16px; box-shadow: 0 8px 26px rgba(5,8,12,0.45);
          border: 1px solid rgba(255,255,255,0.03);
        }
        .panel h3 { margin:0; color:var(--text); font-size:16px; }
        .panel p { margin-top:6px; color:var(--muted); font-size:13px; }
        .muted { color:var(--muted); font-size:13px; }
        .accent { color:var(--accent); font-weight:700; }
        .stTextInput, .stFileUploader, .stTextArea { margin-bottom: 8px; }
        .stCodeBlock pre { background: rgba(255,255,255,0.02) !important; color:var(--muted) !important; border-radius:6px; padding:10px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


    with st.sidebar:
        st.write(f"Welcome, **{st.session_state.get('name','User')}**")
        authenticator.logout('Logout', 'main')
        st.markdown("---")
        
        st.title("AI Legal Toolkit")
        st.write("Courtroom simulator • Argument generator • General legal QA")
        module = st.radio("Module", ["Home", "Courtroom Simulation", "Argument Generator", "General QA"], index=0)
        st.markdown("---")
        st.subheader("Configuration")
        
        CONSTITUTION_PATH = os.getenv("CONSTITUTION_CSV", r"C:\Users\Karth\Desktop\hackathon_project\Constitution Of India.csv")
        BNS_PATH = os.getenv("BNS_CSV", r"C:\Users\Karth\Desktop\hackathon_project\bns_sections.csv")
        
        st.write("Model: **OPENAI (GPT-4o)**")
        st.caption("Ensure GPT-4o is running locally `).")
        st.markdown("---")
        st.markdown("Helpful: Use small PDFs for iteration. CSVs must match expected schema (Article/Description and Section/Description).")


    st.markdown(
        """
        <div class="header">
          <div style="display:flex;flex-direction:column;">
            <div class="brand">⚖️ AI Legal Toolkit</div>
            <div class="tagline">Courtroom simulation • Retrieval-augmented argument generation • Law research (Constitution & BNS)</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    @st.cache_data
    def load_law_csvs(constitution_path: str, bns_path: str) -> Tuple[str, str, List[Document]]:
        constitution_text = ""
        bns_text = ""
        docs: List[Document] = []
        try:
            constitution_df = pd.read_csv(constitution_path, encoding="utf-8")
        except Exception:
            try:
                constitution_df = pd.read_csv(constitution_path, encoding="latin1")
            except FileNotFoundError:
                st.error(f"File not found: {constitution_path}")
                return "", "", []

        try:
            bns_df = pd.read_csv(bns_path, encoding="utf-8")
        except Exception:
            try:
                bns_df = pd.read_csv(bns_path, encoding="latin1")
            except FileNotFoundError:
                st.error(f"File not found: {bns_path}")
                return "", "", []

        def df_to_text(df: pd.DataFrame, key_col_a=None, key_col_b=None, prefix=""):
            lines = []
            if key_col_a and key_col_b and key_col_a in df.columns and key_col_b in df.columns:
                for _, row in df.iterrows():
                    lines.append(f"{prefix} {row[key_col_a]}: {row[key_col_b]}")
            else:
                for _, row in df.iterrows():
                    lines.append(" | ".join([f"{col}: {row[col]}" for col in df.columns]))
            return "\n".join(lines)

        constitution_text = df_to_text(constitution_df, key_col_a="Article", key_col_b="Description", prefix="Article")
        bns_text = df_to_text(bns_df, key_col_a="Section", key_col_b="Description", prefix="Section")

        for _, row in constitution_df.iterrows():
            text = " | ".join([f"{col}: {row[col]}" for col in constitution_df.columns])
            docs.append(Document(page_content=text, metadata={"source": os.path.basename(constitution_path)}))

        for _, row in bns_df.iterrows():
            text = " | ".join([f"{col}: {row[col]}" for col in bns_df.columns])
            docs.append(Document(page_content=text, metadata={"source": os.path.basename(bns_path)}))

        return constitution_text, bns_text, docs

    constitution_text, bns_text, law_docs = load_law_csvs(CONSTITUTION_PATH, BNS_PATH)


    def load_fir_text(uploaded_file) -> str:
        if not uploaded_file:
            return ""
        if uploaded_file.name.lower().endswith(".pdf"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            text = " ".join([page.page_content for page in pages])
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        else:
            text = uploaded_file.read().decode("utf-8", errors="ignore")
        return text.strip()


    @st.cache_resource
    def get_llm(model_name: str = "gpt-4o", temperature: float = 0.7):
        
        return ChatOpenAI(model=model_name, temperature=temperature,api_key = api_key)

    @st.cache_resource
    def get_embeddings():
        return OpenAIEmbeddings(model="text-embedding-ada-002",api_key = api_key)
    


    SIM_PROMPT_TEMPLATE = CoreChatPromptTemplate.from_template("""
   You are an expert **Indian Courtroom Stenographer & High-Fidelity Legal Hearing Simulator**.  
Your responsibility is to generate a **FULL, VERBATIM, REALISTIC TRANSCRIPT** of a Metropolitan Magistrate or Sessions Court hearing in India.

The output must feel like an **actual court record**, with no missing dialogue, no implied actions, no summaries, no gaps.

# ------------------------------------------------------------
# ABSOLUTE RULES — NON-NEGOTIABLE
# ------------------------------------------------------------
1. **NO SUMMARIES WHATSOEVER**  
   You must write *every spoken line*, exactly as if typed in a courtroom transcript.  
   Never say “arguments were heard”, “discussion took place”, etc.  
   Every character must speak their own lines.

2. **NO PLACEHOLDER TEXT**  
   Forbidden: [Accused], [Date], [Witness], [Location].  
   You MUST INVENT realistic and specific details such as:
   - Names (e.g., Rajesh Malhotra, Insp. Kavita Deshmukh)  
   - Dates and times (e.g., “Hearing on 14 February 2025 at 11:32 AM”)  
   - Court Number, Bench, Police Station, Case ID  
   These must remain consistent throughout the transcript.

3. **NO ABRUPT ENDINGS**  
   The hearing must conclude with the **Judge dictating a full Daily Order (Roznamcha)** containing:
   - Appearance of parties  
   - Submissions  
   - Findings  
   - Final direction (“Bail granted/rejected”, “Police custody granted”, “Matter posted to…”)

4. **DIALOGUE MUST BE LONG & DETAILED**  
   The transcript should resemble a **20–30 minute hearing**, containing:
   - Long arguments  
   - Multiple objections  
   - Judge interrogations  
   - Evidence debates  
   - Procedural clarifications  

5. **LEGAL ACCURACY**  
   When citing BNS, CrPC-equivalent procedures, bail standards, seizure protocols, etc., maintain realism.

# ------------------------------------------------------------
# MANDATORY STRUCTURE (EVERY SECTION REQUIRED)
# ------------------------------------------------------------

### 1. COURT CLERK OPENING
Include:
- Court name (e.g., “Court of Metropolitan Magistrate-05, Saket Courts, New Delhi”)  
- Item number  
- Case title  
- Case number  
- Presence of parties  

### 2. PUBLIC PROSECUTOR — CHARGE READING
A long, detailed reading of FIR allegations referencing **specific BNS Sections**, such as:
- §101 (Theft)  
- §111 (Robbery)  
- §115 (Causing Hurt)  
- §64 (Forgery)  
- §35 (Arrest Procedure)  
Or whichever are relevant from {context}.

### 3. IMMEDIATE DEFENSE OBJECTION
Defense Counsel must interrupt with a formal objection.  
Accepted grounds include:
- Illegal arrest under **§35 BNS**  
- Invalid seizure memo  
- FIR contradictions  
- Lack of Section 79 search compliance  
- Failure to serve notice  

The objection must escalate into a detailed argument exchange.

### 4. EXTENDED EVIDENTIARY ARGUMENTS
A long back-and-forth sequence covering:
- **CDR analysis**  
- **CCTV descriptions**  
- **Witness inconsistencies**  
- **Medical report issues**  
- **Chain of custody flaws**  
- **Forensic gaps**

Prosecutor defends evidence.  
Defense attacks credibility.  
Both must cite specifics, timestamps, item numbers, documents.

### 5. JUDGE’S CROSS-EXAMINATION / INTERROGATION
The Judge must interrupt repeatedly with pointed questions. Examples:
- “Show me the medical report. Where is the MLC annexed?”  
- “Why did the IO not issue notice under §35(3)?”  
- “Counsel, this CCTV timestamp mismatch — explain.”  
- “Prosecutor, where is the seizure memo signed by an independent witness?”

Judge may ask the Investigating Officer to stand and answer questions.

### 6. FINAL DAILY ORDER (ROZNAMCHA)
The Judge must dictate the **entire** order, including:
- Court name  
- Case ID  
- Proceedings summary  
- Submissions considered  
- Findings  
- Direction (Bail/PC/JC/Notice to IO/etc.)  
- Next date of hearing  
- Time of dictation and signature note  

Must be 12–20 lines minimum.

# ------------------------------------------------------------
# ADDITIONAL STYLE REQUIREMENTS
# ------------------------------------------------------------
- Use **timestamp markers** occasionally: (11:43 AM), (12:02 PM)  
- Use courtroom tone: “My Lord”, “Your Honour”, “Learned Counsel”  
- Maintain formal Indian legal English  
- Use stenographer cues sparingly: (Counsel rises), (Accused produced), etc.  
- No modern slang, no narrative writing. Pure transcript.

# ------------------------------------------------------------
# CONTEXT (FIR, LAWS, FACTS)
# ------------------------------------------------------------
{context}

# ------------------------------------------------------------
# HEARING FOCUS / QUESTION
# ------------------------------------------------------------
{question}

# ------------------------------------------------------------
# PAST HISTORY (IF ANY)
# ------------------------------------------------------------
{history}

# ------------------------------------------------------------
# NOW GENERATE THE COMPLETE, VERBATIM COURTROOM TRANSCRIPT
# ------------------------------------------------------------
    """)

    def create_context_for_simulation(fir_text: str, constitution_text: str, bns_text: str) -> str:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        fir_chunks = splitter.split_text(f"FIR DOCUMENT:\n{fir_text}") if fir_text else []
        laws_context = (
            f"\nINDIAN CONSTITUTION EXTRACTS:\n{constitution_text[:8000]}"
            f"\n\nBNS SECTIONS:\n{bns_text[:8000]}"
        )
        return "\n".join(fir_chunks) + "\n" + laws_context

    def run_simulation(fir_text: str, user_question: str, history: ChatMessageHistory) -> str:
        if not user_question or not user_question.strip():
            user_question = "Extensive arguments on Bail Application and validity of arrest"
            
        agent = get_llm(model_name="gpt-4o", temperature=0.7)
        context = create_context_for_simulation(fir_text, constitution_text, bns_text)

        try:
            formatted_prompt = SIM_PROMPT_TEMPLATE.format(
                context=context,
                history=(getattr(history, 'messages', history)) if history else [],
                question=user_question
            )
            if hasattr(formatted_prompt, 'to_string'):
                prompt_text = formatted_prompt.to_string()
            elif hasattr(formatted_prompt, '__str__'):
                prompt_text = str(formatted_prompt)
            else:
                prompt_text = repr(formatted_prompt)
        except Exception:
            prompt_text = f"CONTEXT:\n{context}\n\nQUESTION:\n{user_question}\n\nHISTORY:\n{getattr(history, 'messages', history) if history else ''}"

        resp = agent.invoke(prompt_text)
        try:
            content = resp.content if hasattr(resp, "content") else (resp[0].text if isinstance(resp, (list, tuple)) else str(resp))
        except Exception:
            content = str(resp)
        return content

    @st.cache_resource
    def build_vectordb_for_pdf_and_law(uploaded_file):
        embeddings = get_embeddings()
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = os.path.join(tmpdir, uploaded_file.name)
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            pdf_docs = PyPDFLoader(pdf_path).load()
        all_docs = pdf_docs + law_docs
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(all_docs)
        
        if not chunks:
            st.error("No documents to process.")
            return None
            
        vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings)
        return vectordb

    @st.cache_resource
    def build_vectordb_for_law_only():
        embeddings = get_embeddings()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(law_docs)
        
        if not chunks:
            return None
            
        vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings)
        return vectordb

    

    ARGUMENT_PROMPT = CoreChatPromptTemplate.from_messages([
        ("system", """You are an experienced senior legal researcher and argument writer.
    Your task is to analyze the given case and produce a comprehensive legal analysis and courtroom-style document.

    You must:
    - Refer to the provided context (from the uploaded case PDF, Constitution of India, and BNS laws).
    - Expand every point thoroughly, writing at least **three detailed paragraphs** for each heading.
    - Where there are subpoints or bullet points, expand each of them into **three coherent paragraphs**.
    - Use appropriate legal terminology, structured reasoning, and references to relevant sections, doctrines, or constitutional principles.
    - Write in a formal, persuasive tone, suitable for submission to court or legal counsel.

    Follow this detailed structure:

    1. **Client-Side Arguments:** - Present the key arguments supporting the client’s case, each with at least **three paragraphs**.  
       - Include supporting references from BNS sections, constitutional articles, and principles of justice.

    2.  **Opponent’s Possible Arguments:** - Anticipate **at least three** counterpoints that the opposing counsel may present.  
       - Expand each with **three paragraphs** exploring their rationale and potential strength.

    3. **Counter-Arguments:** - For each opponent argument, provide a detailed rebuttal with **at least three paragraphs**.  
       - Cite legal reasoning, statutory interpretation, and case precedents when appropriate.

    4. **Suggestions & Strategy:** - Provide **at least three paragraphs** suggesting strategic actions, courtroom approaches, or pre-trial considerations.
       - Include references to legal precedents, judicial behavior, and procedural recommendations.

    Be analytical, detailed, and structured. Avoid repetition.  
    The content must read like a professional legal memorandum prepared for court submission.

    Context:
    {context}
    """),
        ("user", "Analyze the uploaded case and generate detailed, expanded arguments, counterarguments, and strategy in full paragraphs under each section(mention the laws and sections and articles if required).")
    ])

    def generate_arguments_rag(vectordb) -> str:
        retriever = vectordb.as_retriever(search_kwargs={"k": 5})
        llm = get_llm(model_name="gpt-4o", temperature=0.3)
        qa_chain = create_stuff_documents_chain(llm, ARGUMENT_PROMPT)
        rag_chain = create_retrieval_chain(retriever, qa_chain)
        out = rag_chain.invoke({"input": "Generate arguments based on the case"})
        try:
            return out.get("answer") if isinstance(out, dict) else str(out)
        except Exception:
            return str(out)


    GENERAL_Q_PROMPT = CoreChatPromptTemplate.from_messages([
        ("system", """You are a legal research assistant specialized in the Constitution of India and BNS-like statutes.
    When answering a user's question, follow this structured output EXACTLY:
    1) long (3 - 4 paras)
    2) Legal Basis (list relevant Articles / Sections with short parenthetical explanation)
    3) Explanation (2-3 short paragraphs with legal reasoning)
    4) Practical Implication (1 paragraph)
    5) Sources (list retrieved doc names or identifiers)

    Always ground responses ONLY in the supplied context. If the context doesn't directly state a rule, say "Based on the supplied materials..." and avoid inventing case law.
    Context:
    {context}
    """),
        ("user", "{input}")
    ])

    @st.cache_resource
    def get_general_qa_chain():
        llm = get_llm(model_name="gpt-4o", temperature=0.2)
        qa_chain = create_stuff_documents_chain(llm, GENERAL_Q_PROMPT)
        
        vdb = build_vectordb_for_law_only()
        if vdb:
            retriever = vdb.as_retriever(search_kwargs={"k": 5})
            rag_chain = create_retrieval_chain(retriever, qa_chain)
            return rag_chain
        else:
            return None

    with st.container():
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            st.markdown(
                """
                <div class="panel">
                  <h3>Courtroom Simulation</h3>
                  <p class="muted">Upload an FIR to simulate a full courtroom dialogue (Judge / Defense / Opposition).</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                """
                <div class="panel">
                  <h3>Argument Generator (RAG)</h3>
                  <p class="muted">Upload a case PDF to generate detailed legal arguments referencing the law corpus.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c3:
            if st.button("Open selected module"):
                pass

    st.write("")


    if module == "Courtroom Simulation":
        st.header("⚖️ Courtroom Simulation")
        st.markdown("""
        <div class="panel">
          <strong>Instructions:</strong>
          <ol>
            <li>Upload FIR (PDF/TXT).</li>
            <li>Click <strong>Simulate Court Trial</strong>. (The AI will generate the entire conversation automatically).</li>
          </ol>
        </div>
        """, unsafe_allow_html=True)

        with st.form("sim_form"):
            col1, col2 = st.columns([2, 1])
            with col1:
                uploaded_fir = st.file_uploader("Upload FIR (PDF or TXT)", type=["pdf", "txt"])
                case_question = st.text_input("Simulation Focus (Optional)", placeholder="Leave blank for automatic Bail/Charge hearing simulation")
            with col2:
                st.write("")
                st.write("")
                simulate_btn = st.form_submit_button("Simulate Court Trial")

        if simulate_btn:
            if not uploaded_fir:
                st.warning("Please upload an FIR first.")
            else:
                with st.spinner("Generating realistic courtroom dialogue (this may take a moment)..."):
                    fir_text = load_fir_text(uploaded_fir)
                    if "history" not in st.session_state:
                        st.session_state.history = ChatMessageHistory()
                    try:
                        sim_text = run_simulation(fir_text, case_question, st.session_state.history)
                        st.success("Simulation complete.")
                        st.markdown("### Courtroom Proceedings")
                        st.write(sim_text)
                        try:
                            st.session_state.history.add_user_message("Start Simulation")
                            st.session_state.history.add_ai_message(sim_text)
                        except Exception:
                            pass
                    except Exception as e:
                        st.error(f"Simulation error: {e}")

        with st.expander("Law excerpts (Constitution / BNS)"):
            st.markdown("**Constitution (excerpt)**")
            st.code(constitution_text[:1800])
            st.markdown("**BNS (excerpt)**")
            st.code(bns_text[:1800])


    elif module == "Argument Generator":
        st.header("🧠 Argument Generator (RAG)")
        st.markdown("""
        <div class="panel">
          <strong>Instructions:</strong> Upload a case PDF. The system builds a retrieval index over the PDF + law corpus and generates an extended legal memorandum.
        </div>
        """, unsafe_allow_html=True)

        with st.form("arg_form"):
            uploaded_case = st.file_uploader("Upload case document (PDF)", type=["pdf"])
            gen_args_btn = st.form_submit_button("Generate Arguments")

        if gen_args_btn:
            if not uploaded_case:
                st.warning("Please upload a PDF.")
            else:
                with st.spinner("Building index and generating arguments with Llama 3..."):
                    try:
                        vectordb = build_vectordb_for_pdf_and_law(uploaded_case)
                        if vectordb:
                            ans = generate_arguments_rag(vectordb)
                            st.success("Arguments generated.")
                            st.markdown(ans)
                        else:
                            st.error("Failed to build vector database.")
                    except Exception as e:
                        st.error(f"Error: {e}")

        with st.expander("Law references (short)"):
            st.markdown("**Constitution (excerpt)**")
            st.code(constitution_text[:1200])
            st.markdown("**BNS (excerpt)**")
            st.code(bns_text[:1200])

    elif module == "General QA":
        st.header("💬 General Legal QA — Constitution & BNS/IPC")
        st.markdown("""
        <div class="panel">
          Ask general legal questions. Answers are structured: Short Answer (bullets), Legal Basis (Articles/Sections), Explanation, Practical Implication, and Sources.
        </div>
        """, unsafe_allow_html=True)

        if "qa_history" not in st.session_state:
            st.session_state.qa_history = []

        q_col, btn_col = st.columns([5, 1])
        with q_col:
            user_q = st.text_input("Ask a legal question (e.g., 'What are advantages of the Indian Constitution?')", key="general_q")
        with btn_col:
            ask_btn = st.button("Ask")

        if ask_btn:
            if not user_q or not user_q.strip():
                st.warning("Please type a question.")
            else:
                with st.spinner("Searching law corpus and composing structured answer..."):
                    try:
                        rag_chain = get_general_qa_chain()
                        if rag_chain:
                            response = rag_chain.invoke({"input": user_q})
                            if isinstance(response, dict) and response.get("answer"):
                                ans = response.get("answer")
                            else:
                                ans = str(response)
                            st.session_state.qa_history.append({"q": user_q, "a": ans})
                            st.markdown("### Answer")
                            st.markdown(ans)
                            with st.expander("Show law excerpts used (best-effort)"):
                                st.markdown("Displayed excerpts are the top documents from the law corpus used for retrieval.")
                                try:
                                    vdb_law = build_vectordb_for_law_only()
                                    if vdb_law:
                                        retriever = vdb_law.as_retriever(search_kwargs={"k": 5})
                                        # Use get_relevant_documents for compatibility
                                        docs = retriever.get_relevant_documents(user_q)
                                        for i, d in enumerate(docs[:5], 1):
                                            src = d.metadata.get("source", "law_corpus")
                                            snippet = (d.page_content[:600] + "...") if len(d.page_content) > 600 else d.page_content
                                            st.markdown(f"**{i}. Source:** `{src}`")
                                            st.code(snippet)
                                except Exception as e:
                                    st.info(f"Source excerpts not available: {e}")
                        else:
                            st.error("Could not initialize QA chain (data missing?).")
                    except Exception as e:
                        st.error(f"Error answering QA: {e}")

        if st.session_state.qa_history:
            st.markdown("---")
            st.markdown("#### Recent QA")
            for item in reversed(st.session_state.qa_history[-6:]):
                st.markdown(f"**Q:** {item['q']}")
                st.markdown(f"**A:** {item['a'][:800]}{'...' if len(item['a'])>800 else ''}")
                st.write("")


    st.markdown("---")
    st.caption("⚠️ Disclaimer: This tool provides simulations and research assistance. It is not a substitute for professional legal advice.")
elif st.session_state.get("authentication_status") is False:
    st.error('Username/password is incorrect')
