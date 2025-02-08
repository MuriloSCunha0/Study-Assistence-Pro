# 📚 Plataforma Inteligente de Geração de Questões com IA 🤖

## 📌 Sobre o Projeto
Este projeto utiliza **Inteligência Artificial e Processamento de Linguagem Natural (NLP)** para gerar automaticamente **questões de múltipla escolha** a partir de **arquivos PDF**. A plataforma adapta as questões ao nível do usuário, proporcionando um **aprendizado personalizado e dinâmico**.

---

## 🎯 Objetivo
Criar uma ferramenta que **automatiza a geração de questões educacionais** com base em materiais de estudo enviados pelos usuários. O sistema também ajusta a dificuldade das questões conforme o desempenho do usuário, otimizando o aprendizado.

---

## 🔍 Funcionalidades

✅ **Upload de PDFs** - O usuário pode enviar documentos para processamento.
✅ **Processamento Inteligente** - Segmentação do texto com técnicas de **Semantic Chunking**.
✅ **Geração de Questões com IA** - Utilização de **LLM (Llama3:8b) e embeddings do Hugging Face** para criar perguntas de múltipla escolha.
✅ **Modo de Estudo Adaptativo** - Ajusta a dificuldade com base no desempenho do usuário.
✅ **Dashboard Interativo** - Gráficos e estatísticas sobre o progresso do usuário (usando **Plotly e Streamlit**).
✅ **Histórico e Feedback** - Armazena o histórico de respostas para análise de desempenho.

---

## 🛠 Tecnologias Utilizadas

🔹 **Python** - Linguagem principal do projeto  
🔹 **Streamlit** - Interface interativa e intuitiva  
🔹 **Llama3:8b** - Modelo LLM para geração de questões  
🔹 **LangChain** - Manipulação avançada de textos  
🔹 **Hugging Face Embeddings** - Conversão de trechos em vetores semânticos  
🔹 **Pandas** - Processamento e análise de dados  
🔹 **Plotly** - Visualização de estatísticas e progresso  

---

## 🚀 Como Executar o Projeto

### 1️⃣ Clone este repositório:
```bash
git clone https://github.com/seu-usuario/projeto-questoes-ia.git
cd projeto-questoes-ia
```

### 2️⃣ Crie um ambiente virtual e ative-o:
```bash
python -m venv venv  # Criar ambiente virtual
source venv/bin/activate  # Ativar no Linux/macOS
venv\Scripts\activate  # Ativar no Windows
```

### 3️⃣ Instale as dependências:
```bash
pip install -r requirements.txt
```

### 4️⃣ Execute a aplicação:
```bash
streamlit run app.py
```

---

## 🏆 Benefícios e Aplicações
🎓 **Educação** - Alunos e professores podem gerar questões personalizadas.  
📖 **Preparação para Concursos** - Simulados automáticos para estudo.  
🏢 **Treinamento Corporativo** - Empresas podem criar avaliações personalizadas.  
📈 **Aprendizado Adaptativo** - Questões ajustadas conforme o progresso do usuário.  

---

## 📄 Licença
Este projeto é de código aberto sob a licença **MIT**.

---
