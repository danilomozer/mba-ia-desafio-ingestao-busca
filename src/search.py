import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_postgres import PGVector

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOP_K = 10

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


class ConfigError(Exception):
    """Erro de configuração (variável de ambiente ausente ou inválida)."""


def _require_env(name):
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigError(f"Variável de ambiente {name} não definida. Confira o arquivo .env.")
    return value


def get_provider():
    """Retorna 'openai' ou 'gemini'.

    Usa PROVIDER se estiver definida; senão escolhe pelo que tiver API key.
    """
    provider = os.getenv("PROVIDER", "").strip().lower()
    if provider in ("openai", "gemini"):
        return provider
    if provider:
        raise ConfigError(f"PROVIDER inválido: '{provider}'. Use 'openai' ou 'gemini'.")
    if os.getenv("OPENAI_API_KEY", "").strip():
        return "openai"
    if os.getenv("GOOGLE_API_KEY", "").strip():
        return "gemini"
    raise ConfigError("Defina OPENAI_API_KEY ou GOOGLE_API_KEY no arquivo .env.")


def get_embeddings():
    provider = get_provider()
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        _require_env("OPENAI_API_KEY")
        return OpenAIEmbeddings(model=_require_env("OPENAI_EMBEDDING_MODEL"))

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    _require_env("GOOGLE_API_KEY")
    return GoogleGenerativeAIEmbeddings(model=_require_env("GOOGLE_EMBEDDING_MODEL"))


def get_llm():
    provider = get_provider()
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=_require_env("OPENAI_LLM_MODEL"))

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=_require_env("GOOGLE_LLM_MODEL"))


def get_vector_store(embeddings, pre_delete_collection=False):
    return PGVector(
        embeddings=embeddings,
        collection_name=_require_env("PG_VECTOR_COLLECTION_NAME"),
        connection=_require_env("DATABASE_URL"),
        use_jsonb=True,
        pre_delete_collection=pre_delete_collection,
    )


def search_context(store, question, k=TOP_K):
    """Vetoriza a pergunta, busca os k chunks mais relevantes e concatena o texto."""
    results = store.similarity_search_with_score(question, k=k)
    return "\n\n".join(doc.page_content.strip() for doc, _score in results)


def search_prompt(question=None):
    """Monta a chain de busca + LLM.

    Sem argumento, retorna a chain (recebe a pergunta e devolve a resposta).
    Com uma pergunta, já executa a chain e devolve a resposta.
    Retorna None se a configuração estiver incompleta.
    """
    try:
        store = get_vector_store(get_embeddings())
        llm = get_llm()
    except ConfigError as exc:
        print(f"Erro de configuração: {exc}")
        return None

    prompt = PromptTemplate(
        input_variables=["contexto", "pergunta"],
        template=PROMPT_TEMPLATE,
    )

    chain = (
        {
            "contexto": RunnableLambda(lambda q: search_context(store, q)),
            "pergunta": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    if question is not None:
        return chain.invoke(question)
    return chain
