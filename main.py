import os
import json
import re
import pandas as pd
import plotly.express as px
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.ollama import Ollama
import tempfile
import uuid
from pathlib import Path
from io import BytesIO

# Configurações globais
PROGRESS_CSV = "progress.csv"
PDF_DIR = "pdf_storage"
QUESTIONS_DIR = "question_bank"
STYLES = """
<style>
    .header { font-size: 2.5em; color: #2E86C1; margin-bottom: 20px; }
    .metric-box { padding: 20px; border-radius: 10px; margin: 10px; background-color: #EBF5FB; }
    .question-card { padding: 15px; margin: 10px 0; border-left: 5px solid #2E86C1; background-color: black; }
    .correct { border-color: #28B463 !important; }
    .incorrect { border-color: #E74C3C !important; }
    .explanation { padding: 10px; margin-top: 10px; background-color: black ; border-radius: 5px; }
    .pdf-list { max-height: 300px; overflow-y: auto; }
</style>
"""

# Funções auxiliares
def setup_directories():
    """Cria os diretórios necessários"""
    Path(PDF_DIR).mkdir(exist_ok=True)
    Path(QUESTIONS_DIR).mkdir(exist_ok=True)

def get_pdf_files():
    """Retorna lista de PDFs armazenados"""
    return [f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')]

def get_question_file(pdf_name):
    """Retorna o caminho do arquivo de questões para um PDF"""
    return Path(QUESTIONS_DIR) / f"{Path(pdf_name).stem}_questions.csv"

# Funções de dados
def load_questions(pdf_name=None):
    """Carrega questões de arquivos CSV"""
    if pdf_name:
        q_file = get_question_file(pdf_name)
        if q_file.exists():
            df = pd.read_csv(q_file)
            df['options'] = df['options'].apply(json.loads)
            return df
        return pd.DataFrame()
    
    all_questions = []
    for f in Path(QUESTIONS_DIR).glob('*.csv'):
        df = pd.read_csv(f)
        df['options'] = df['options'].apply(json.loads)
        all_questions.append(df)
    return pd.concat(all_questions) if all_questions else pd.DataFrame()

def save_questions(questions, pdf_name):
    """Salva questões no arquivo correspondente ao PDF"""
    q_file = get_question_file(pdf_name)
    df = pd.DataFrame(questions)
    df['options'] = df['options'].apply(json.dumps)
    df.to_csv(q_file, index=False)

def load_progress():
    """Carrega o progresso do usuário"""
    if Path(PROGRESS_CSV).exists():
        return pd.read_csv(PROGRESS_CSV)
    return pd.DataFrame()

def save_progress(progress):
    """Salva o progresso do usuário"""
    progress.to_csv(PROGRESS_CSV, index=False)

# Processamento de PDF
def process_pdf(file):
    """Processa o PDF e extrai texto"""
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(file.getvalue())
        loader = PDFPlumberLoader(tmp_file.name)
        docs = loader.load()
        text_splitter = SemanticChunker(HuggingFaceEmbeddings())
        return text_splitter.split_documents(docs)

# Geração de questões
def generate_questions(text, num_questions=5):
    """Gera questões a partir do texto usando Ollama"""
    prompt = f"""
    Gere exatamente {num_questions} questões técnicas em formato JSON seguindo ESTE MODELO:

    {{
        "questions": [
            {{
                "question": "Pergunta",
                "options": ["Opção 1", "Opção 2", "Opção 3", "Opção 4"],
                "correct_option": 0,
                "difficulty": "easy|medium|hard",
                "topic": "Tópico",
                "explanation": "Explicação detalhada"
            }}
        ]
    }}

    REGRAS:
    1. Use somente aspas duplas
    2. Numere as opções começando em 0
    3. Dificuldade: easy, medium ou hard
    4. Formato válido SEM erros

    Texto de referência: {text}
    """

    try:
        response = Ollama(model="llama3:8b")(prompt)
        json_str = re.sub(r'[\s\S]*?({[\s\S]*})[\s\S]*', r'\1', response)
        json_str = re.sub(r"(?<!\\)'", '"', json_str)
        data = json.loads(json_str)
        
        questions = []
        for q in data.get('questions', []):
            q['options'] = q.get('options', [])
            q['correct_option'] = int(q.get('correct_option', 0))
            q['difficulty'] = q.get('difficulty', 'medium').lower()
            q['topic'] = q.get('topic', 'Geral')
            q['explanation'] = q.get('explanation', '')
            questions.append(q)
        
        return questions

    except Exception as e:
        st.error(f"Erro na geração: {str(e)}")
        return []

# Interface
def render_study_page():
    """Página de estudo interativo"""
    st.markdown("<div class='header'>📚 Modo de Estudo</div>", unsafe_allow_html=True)
    
    pdf_files = get_pdf_files()
    if not pdf_files:
        st.warning("Nenhum PDF disponível. Carregue um PDF primeiro!")
        return
    
    selected_pdf = st.selectbox("Selecione o conteúdo:", pdf_files)
    questions = load_questions(selected_pdf)
    
    if questions.empty:
        st.warning("Nenhuma questão disponível para este conteúdo.")
        return
    
    # Sistema de dificuldade adaptativo
    progress = load_progress()
    if not progress.empty:
        user_performance = progress.groupby('difficulty')['is_correct'].mean()
        difficulty_levels = user_performance.sort_values().index.tolist()
        selected_difficulty = st.selectbox("Nível de dificuldade recomendado:", difficulty_levels)
    else:
        selected_difficulty = st.selectbox("Nível de dificuldade:", ["easy", "medium", "hard"])
    
    filtered = questions[questions['difficulty'] == selected_difficulty]
    
    if filtered.empty:
        st.warning("Nenhuma questão encontrada com este nível de dificuldade")
        return
    
    for _, row in filtered.iterrows():
        with st.container():
            st.markdown(f"""
            <div class='question-card' id='q{row['id']}'>
                <b>Questão #{row['id']}</b> ({row['difficulty'].capitalize()})<br>
                {row['question']}
            </div>
            """, unsafe_allow_html=True)
            
            options = row['options']
            selected = st.radio(
                "Selecione sua resposta:",
                options,
                key=f"opt_{row['id']}"
            )
            
            if st.button("Submeter resposta", key=f"btn_{row['id']}"):
                correct_idx = row['correct_option']
                is_correct = (options.index(selected) == correct_idx)
                
                new_progress = pd.DataFrame([{
                    'timestamp': pd.Timestamp.now(),
                    'question_id': row['id'],
                    'pdf_source': selected_pdf,
                    'difficulty': row['difficulty'],
                    'is_correct': is_correct
                }])
                
                if Path(PROGRESS_CSV).exists():
                    new_progress.to_csv(PROGRESS_CSV, mode='a', header=False, index=False)
                else:
                    new_progress.to_csv(PROGRESS_CSV, index=False)
                
                if is_correct:
                    st.success("✅ Resposta Correta!")
                else:
                    st.error(f"❌ Resposta Incorreta. A correta é: {options[correct_idx]}")
                
                st.markdown(f"<div class='explanation'>{row['explanation']}</div>", unsafe_allow_html=True)

def render_dashboard():
    """Página de dashboard analítico"""
    st.markdown("<div class='header'>📊 Dashboard de Progresso</div>", unsafe_allow_html=True)
    
    progress = load_progress()
    if progress.empty:
        st.info("Nenhum dado de progresso disponível")
        return
    
    # Métricas principais
    col1, col2, col3 = st.columns(3)
    with col1:
        total = len(progress)
        st.metric("Total de Respostas", total)
    with col2:
        accuracy = progress['is_correct'].mean() * 100
        st.metric("Taxa de Acerto", f"{accuracy:.1f}%")
    with col3:
        recent = progress.tail(10)['is_correct'].mean() * 100
        st.metric("Acertos Recentes", f"{recent:.1f}%")
    
    # Gráfico de desempenho por dificuldade
    st.subheader("Desempenho por Dificuldade")
    if not progress.empty:
        fig = px.bar(
            progress.groupby('difficulty', as_index=False)
                .agg(Acertos=('is_correct', 'sum'), Total=('is_correct', 'count')),
            x='difficulty',
            y=['Acertos', 'Total'],
            barmode='group',
            labels={'difficulty': 'Dificuldade', 'value': 'Contagem'}
        )
        st.plotly_chart(fig)
    
    # Distribuição de conteúdos
    st.subheader("Desempenho por Conteúdo")
    if not progress.empty:
        fig = px.pie(
            progress, 
            names='pdf_source',
            hole=0.3,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig)

def main():
    setup_directories()
    st.set_page_config(page_title="Study Assistant Pro", page_icon="🧠", layout="wide")
    st.markdown(STYLES, unsafe_allow_html=True)
    
    st.sidebar.title("Navegação")
    page = st.sidebar.radio("", ["Gerar Questões", "Modo de Estudo", "Dashboard", "Biblioteca PDF"])
    
    if page == "Gerar Questões":
        st.markdown("<div class='header'>🎯 Gerar Novas Questões</div>", unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Carregue seu material didático (PDF)", type="pdf")
        if uploaded_file:
            unique_id = str(uuid.uuid4())[:8]
            original_name = Path(uploaded_file.name).stem
            pdf_name = f"{original_name}_{unique_id}.pdf"
            pdf_path = Path(PDF_DIR) / pdf_name
            
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            if st.button("Gerar Questões"):
                with st.spinner("Processando PDF e gerando questões..."):
                    docs = process_pdf(uploaded_file)
                    text = " ".join([d.page_content for d in docs])
                    generated = generate_questions(text)
                    
                    if generated:
                        existing = load_questions(pdf_name)
                        new_ids = range(len(existing)+1, len(existing)+len(generated)+1)
                        
                        for i, q in enumerate(generated):
                            q['id'] = new_ids[i]
                            q['pdf_source'] = pdf_name
                        
                        save_questions(generated, pdf_name)
                        st.success(f"✅ {len(generated)} questões geradas com sucesso para {pdf_name}!")
    
    elif page == "Modo de Estudo":
        render_study_page()
    
    elif page == "Dashboard":
        render_dashboard()
    
    elif page == "Biblioteca PDF":
        st.markdown("<div class='header'>📚 Biblioteca de Conteúdos</div>", unsafe_allow_html=True)
        pdf_files = get_pdf_files()
        
        if pdf_files:
            st.markdown("<div class='pdf-list'>", unsafe_allow_html=True)
            for pdf in pdf_files:
                col1, col2 = st.columns([3,1])
                with col1:
                    st.markdown(f"📄 **{pdf}**")
                with col2:
                    if st.button("Gerar mais questões", key=f"btn_{pdf}"):
                        with st.spinner("Gerando novas questões..."):
                            # Abre o PDF armazenado
                            pdf_path = Path(PDF_DIR) / pdf
                            with open(pdf_path, "rb") as f:
                                file_bytes = f.read()
                            # Cria um objeto similar ao do st.file_uploader
                            pdf_file = BytesIO(file_bytes)
                            
                            # Processa o PDF e extrai o texto
                            docs = process_pdf(pdf_file)
                            text = " ".join([d.page_content for d in docs])
                            
                            # Gera novas questões usando a mesma função
                            novas_questoes = generate_questions(text)
                            
                            if novas_questoes:
                                # Carrega as questões já existentes para este PDF (se houver)
                                existing_df = load_questions(pdf)
                                existing = existing_df.to_dict(orient="records") if not existing_df.empty else []
                                
                                # Atribui novos IDs às questões geradas
                                new_ids = range(len(existing) + 1, len(existing) + len(novas_questoes) + 1)
                                for i, q in enumerate(novas_questoes):
                                    q['id'] = new_ids[i]
                                    q['pdf_source'] = pdf
                                
                                # Combina as questões antigas com as novas
                                todas_questoes = existing + novas_questoes
                                
                                # Salva as questões combinadas no arquivo CSV correspondente
                                save_questions(todas_questoes, pdf)
                                st.success(f"Novas questões geradas para {pdf}")
                            else:
                                st.error("Falha ao gerar novas questões.")
                                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("Nenhum PDF armazenado ainda")

if __name__ == "__main__":
    main()