
"""
TP : Chatbot basé sur un RAG (Retrieval-Augmented Generation)
Code complété jusqu'à la question 15.
"""

import os
import time
import requests
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# --- Configuration ---

PROVIDER = os.getenv("LLM_PROVIDER") or (
    "openai" if os.getenv("OPENAI_API_KEY") else "mistral"
)
PROVIDER = PROVIDER.strip().lower()

DEFAULT_MODELS = {
    "openai": ("gpt-4.1-mini", "text-embedding-3-small"),
    "mistral": ("mistral-small-latest", "mistral-embed"),
}

if PROVIDER not in DEFAULT_MODELS:
    raise ValueError(
        f"LLM_PROVIDER inconnu : '{PROVIDER}' "
        "(attendu : openai ou mistral)"
    )

CHAT_MODEL = os.getenv(
    "CHAT_MODEL", DEFAULT_MODELS[PROVIDER][0]
)
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", DEFAULT_MODELS[PROVIDER][1]
)
PERSIST_DIR = os.getenv(
    "CHROMA_DIR", f"chroma_db_{PROVIDER}"
)

PAGES = [
    "Intelligence_artificielle_générative",
    "Transformeur_génératif_préentraîné",
    "Google_Gemini",
    "Grand_modèle_de_langage",
    "ChatGPT",
    "LLaMA",
    "Réseaux_antagonistes_génératifs",
    "Apprentissage_auto-supervisé",
    "Apprentissage_par_renforcement",
    "DALL-E",
    "Midjourney",
    "Stable_Diffusion",
]


# --- Modèles fournis ---

def get_embeddings():
    """Renvoie le modèle d'embeddings du fournisseur choisi."""

    if PROVIDER == "mistral":
        from langchain_mistralai import MistralAIEmbeddings
        from tokenizers import (
            Regex, Tokenizer, models, pre_tokenizers
        )

        tokenizer = Tokenizer(
            models.WordLevel(
                {"[UNK]": 0}, unk_token="[UNK]"
            )
        )
        tokenizer.pre_tokenizer = pre_tokenizers.Split(
            Regex(r"[\s\S]"), behavior="isolated"
        )

        return MistralAIEmbeddings(
            model=EMBEDDING_MODEL,
            tokenizer=tokenizer,
            max_retries=10,
            wait_time=5,
        )

    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def get_llm():
    """Renvoie le modèle de chat du fournisseur choisi."""

    if PROVIDER == "mistral":
        from langchain_mistralai import ChatMistralAI
        return ChatMistralAI(
            model=CHAT_MODEL,
            temperature=0,
            max_retries=5,
        )

    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=CHAT_MODEL)


# --- Question 5 : récupérer une page Wikipédia ---

def get_wikipedia_page(title: str):
    """Récupère le texte brut d'une page Wikipédia."""

    url = "https://fr.wikipedia.org/w/api.php"

    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
        "redirects": 1,
    }

    headers = {"User-Agent": "RAG_project/0.0.1"}

    response = requests.get(
        url,
        params=params,
        headers=headers,
        timeout=30,
    )

    response.raise_for_status()
    data = response.json()

    page = next(iter(data["query"]["pages"].values()))
    return page["extract"] if "extract" in page else None


# --- Question 6 : charger les documents ---

def load_documents():
    """Télécharge les pages et conserve leurs métadonnées."""

    documents = []

    for titre in PAGES:
        try:
            texte = get_wikipedia_page(titre)

            if not texte or not texte.strip():
                print(f"Page vide ignorée : {titre}")
                continue

            document = Document(
                page_content=texte,
                metadata={
                    "title": titre.replace("_", " "),
                    "source": (
                        f"https://fr.wikipedia.org/wiki/{titre}"
                    ),
                },
            )

            documents.append(document)
            print(f"Page chargée : {titre}")

        except requests.RequestException as erreur:
            print(f"Erreur pour {titre} : {erreur}")

            if (
                isinstance(erreur, requests.HTTPError)
                and erreur.response is not None
                and erreur.response.status_code == 429
            ):
                print(
                    "Wikipédia limite les requêtes. "
                    "Arrêt temporaire."
                )
                break

        finally:
            time.sleep(3)

    return documents


# --- Question 7 : découper les documents ---

def split_into_chunks(documents):
    """Crée des chunks de 1000 caractères avec overlap de 200."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    return splitter.split_documents(documents)


# --- Questions 11 et 12 : base vectorielle ChromaDB ---

def get_retriever():
    """
    Construit ou recharge la base ChromaDB.
    Les embeddings existants ne sont pas recalculés.
    """

    vectorstore = Chroma(
        collection_name="wikipedia_rag",
        embedding_function=get_embeddings(),
        persist_directory=PERSIST_DIR,
    )

    nombre_chunks = vectorstore._collection.count()

    if nombre_chunks > 0:
        print(
            f"Base Chroma existante : "
            f"{nombre_chunks} chunks déjà indexés."
        )
        print(
            "Réutilisation de la base : "
            "aucun embedding de document recalculé."
        )

    else:
        print("Base Chroma vide : création de l'index...")

        documents = load_documents()

        if not documents:
            raise RuntimeError(
                "Aucune page Wikipédia téléchargée."
            )

        chunks = split_into_chunks(documents)

        print(
            f"Documents chargés : "
            f"{len(documents)} / {len(PAGES)}"
        )
        print(f"Chunks créés : {len(chunks)}")

        vectorstore.add_documents(chunks)

        print(
            "Chunks indexés dans Chroma :",
            vectorstore._collection.count()
        )

    return vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )


# --- Question 13 : préparer le contexte documentaire ---

def format_docs(docs):
    """Regroupe les chunks et les titres de leurs pages."""

    passages = []

    for doc in docs:
        titre = doc.metadata.get(
            "title", "Source inconnue"
        )
        contenu = doc.page_content

        passages.append(
            f"Titre de la page : {titre}\n{contenu}"
        )

    separateur = "\n\n" + "-" * 50 + "\n\n"

    return separateur.join(passages)


# --- Question 14 : construire le prompt ---

def get_prompt():
    """Crée le prompt du chatbot RAG."""

    return ChatPromptTemplate.from_messages([
        (
            "system",
            "Tu es un assistant spécialisé dans "
            "l'intelligence artificielle générative. "
            "Réponds toujours en français en t'appuyant "
            "uniquement sur les informations présentes "
            "dans le contexte fourni. "
            "N'invente pas d'informations. "
            "Si le contexte ne permet pas de répondre "
            "à la question, indique clairement : "
            "'Je ne sais pas à partir des documents fournis.'\n\n"
            "Contexte documentaire :\n{context}"
        ),
        (
            "human",
            "{question}"
        ),
    ])


# --- Question 15 : générer une réponse avec ses sources ---

def rag_answer(query, retriever=None, stream=True):
    """
    Répond à une question à partir des documents Wikipédia.

    Étapes :
    1. Récupérer les chunks pertinents.
    2. Construire la chaîne LangChain.
    3. Générer la réponse en streaming.
    4. Afficher les sources sans doublons.
    5. Retourner la réponse et les documents utilisés.
    """

    # Réutiliser le retriever déjà chargé si disponible
    if retriever is None:
        retriever = get_retriever()

    # Étape A : récupérer les documents pertinents
    docs = retriever.invoke(query)

    # Préparer le contexte documentaire
    contexte = format_docs(docs)

    # Étape B : construire la chaîne LangChain
    prompt = get_prompt()
    llm = get_llm()

    chain = prompt | llm | StrOutputParser()

    donnees = {
        "context": contexte,
        "question": query,
    }

    # Étape C : générer et afficher la réponse
    if stream:
        morceaux = []

        print("Réponse :")

        for token in chain.stream(donnees):
            print(token, end="", flush=True)
            morceaux.append(token)

        reponse = "".join(morceaux)
        print()

    else:
        # Mode sans affichage pour la future interface Streamlit
        reponse = chain.invoke(donnees)

    # Étape D : récupérer les sources sans doublons
    sources = list(dict.fromkeys(
        doc.metadata.get("source", "")
        for doc in docs
        if doc.metadata.get("source")
    ))

    if stream:
        print("\nSources :")

        if sources:
            for url in sources:
                print(f"- {url}")
        else:
            print("Aucune source retrouvée.")

    # Étape E : retourner réponse et documents
    return reponse, docs


# --- Programme principal ---

if __name__ == "__main__":

    print(
        f"Fournisseur : {PROVIDER} "
        f"(chat : {CHAT_MODEL}, "
        f"embeddings : {EMBEDDING_MODEL})"
    )

    print("Building retriever...")
    retriever = get_retriever()

    try:
        while True:
            print("-" * 50)
            print("Posez une question :")

            question = input("> ")
            print()

            rag_answer(question, retriever=retriever)
            print("\n")

    except (KeyboardInterrupt, EOFError):
        print("\nExiting...")
