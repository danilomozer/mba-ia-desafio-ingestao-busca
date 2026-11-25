# Ingestão e Busca Semântica com LangChain e Postgres

Desafio do MBA em Engenharia de Software com IA (Full Cycle).

O projeto lê um PDF, divide o texto em chunks, gera embeddings e salva tudo num PostgreSQL com pgVector. Depois, um chat no terminal responde perguntas usando **apenas** o conteúdo do PDF.

```
Faça sua pergunta (digite 'sair' para encerrar):

PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: O faturamento foi de 10 milhões de reais.

---

PERGUNTA: Quantos clientes temos em 2024?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```

## Como funciona

**Ingestão (`src/ingest.py`)**
1. Carrega o PDF com `PyPDFLoader`.
2. Divide o texto com `RecursiveCharacterTextSplitter` em chunks de 1000 caracteres com overlap de 150.
3. Gera o embedding de cada chunk e grava no Postgres com `PGVector`.

A collection é recriada a cada ingestão, então rodar o script de novo não duplica os chunks.

**Busca (`src/search.py`)**
1. Vetoriza a pergunta.
2. Busca os 10 chunks mais relevantes com `similarity_search_with_score(pergunta, k=10)`.
3. Monta o prompt do desafio com os chunks concatenados no `CONTEXTO`.
4. Chama a LLM e devolve a resposta.

**Chat (`src/chat.py`)**: loop no terminal que recebe a pergunta, chama a busca e mostra a resposta. Digite `sair` (ou Ctrl+C) para encerrar.

## Provedores e modelos

Funciona com **OpenAI** ou **Gemini**, escolhido pelo `.env`:

| Provedor | Embeddings | LLM |
|---|---|---|
| OpenAI | `text-embedding-3-small` | `gpt-6-luna` |
| Gemini | `gemini-embedding-001` | `gemini-3.5-flash-lite` |

São os modelos mais leves e baratos de cada provedor disponíveis em novembro de 2026, conforme a documentação oficial. Se algum for descontinuado, troque o nome no `.env`. Não precisa mudar o código.

> **Trocou o modelo de embeddings?** Cada modelo gera vetores de uma dimensão diferente. Rode `python src/ingest.py` de novo para recriar a collection com o novo modelo.

## Pré-requisitos

- Python 3.10 ou superior
- Docker e Docker Compose
- API key da OpenAI ou do Google (Gemini)

## Como executar

### 1. Clonar o repositório

```bash
git clone <url-deste-repositorio>
cd mba-ia-desafio-ingestao-busca
```

### 2. Criar o ambiente virtual e instalar as dependências

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configurar as variáveis de ambiente

```bash
cp .env.example .env
```

Abra o `.env` e preencha **uma** das API keys (`OPENAI_API_KEY` ou `GOOGLE_API_KEY`). Os outros valores já vêm prontos para o docker-compose do projeto.

| Variável | Descrição |
|---|---|
| `PROVIDER` | `openai` ou `gemini`. Se ficar vazia, usa o provedor que tiver API key. |
| `OPENAI_API_KEY` / `GOOGLE_API_KEY` | API key do provedor escolhido. |
| `OPENAI_EMBEDDING_MODEL` / `GOOGLE_EMBEDDING_MODEL` | Modelo de embeddings. |
| `OPENAI_LLM_MODEL` / `GOOGLE_LLM_MODEL` | Modelo que gera a resposta. |
| `DATABASE_URL` | Conexão com o Postgres (`postgresql+psycopg://postgres:postgres@localhost:5432/rag`). |
| `PG_VECTOR_COLLECTION_NAME` | Nome da collection de vetores. |
| `PDF_PATH` | Caminho do PDF, relativo à raiz do projeto ou absoluto (`document.pdf`). |

### 4. Subir o banco de dados

```bash
docker compose up -d
```

Sobe o PostgreSQL com pgVector e cria a extensão `vector` no banco `rag`.

### 5. Executar a ingestão do PDF

```bash
python src/ingest.py
```

### 6. Rodar o chat

```bash
python src/chat.py
```

## Estrutura

```
├── docker-compose.yml    # Postgres + pgVector
├── requirements.txt      # Dependências
├── .env.example          # Template das variáveis de ambiente
├── src/
│   ├── ingest.py         # Ingestão do PDF
│   ├── search.py         # Busca semântica, prompt e LLM
│   ├── chat.py           # CLI de chat
├── document.pdf          # PDF para ingestão
└── README.md
```

## Problemas comuns

- **`connection refused` na porta 5432**: o banco não está de pé. Rode `docker compose up -d` e espere alguns segundos.
- **Erro de dimensão do vetor**: o modelo de embeddings mudou desde a última ingestão. Rode `python src/ingest.py` de novo. Se ainda falhar, apague o volume com `docker compose down -v`, suba o banco de novo e refaça a ingestão.
- **Erro 404 ou "model not found"**: o modelo foi descontinuado. Consulte a documentação do provedor e atualize o nome no `.env`.
