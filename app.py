
"""
TP RAG — Question 25
Gestion des secrets Streamlit et interface chatbot RAG.
"""

import os
import streamlit as st

# --- Question 25 : gestion des secrets ---

# Les secrets Streamlit doivent être récupérés AVANT
# l'importation de main.py.
# En local, le fichier .env reste utilisé.

try:
    for nom in (
        "OPENAI_API_KEY",
        "MISTRAL_API_KEY",
        "LLM_PROVIDER",
        "CHAT_MODEL",
        "EMBEDDING_MODEL",
    ):
        if nom in st.secrets and st.secrets[nom]:
            os.environ[nom] = str(st.secrets[nom])

except FileNotFoundError:
    # Aucun fichier secrets.toml en local :
    # main.py utilisera le fichier .env.
    pass


# Importer main.py après la configuration
from main import get_retriever, rag_answer


# --- Configuration de l'interface ---

st.set_page_config(
    page_title="Chatbot RAG — IA générative",
    page_icon="💬",
    layout="centered",
)

st.title("Chatbot RAG — IA générative")

st.caption("Explorez l’intelligence artificielle générative avec notre chatbot RAG !")


# --- Chargement du retriever ---

@st.cache_resource
def charger_retriever():
    """Charger le retriever une seule fois."""
    return get_retriever()


with st.spinner("Connexion à la base documentaire..."):
    retriever = charger_retriever()


# --- Historique de conversation ---

if "messages" not in st.session_state:
    st.session_state.messages = []


# --- Affichage des sources ---

def afficher_sources(sources):
    """Afficher les sources Wikipédia."""

    if not sources:
        return

    st.markdown("**Sources :**")

    for source in sources:
        st.markdown(
            f"- [{source['title']}]({source['url']})"
        )


def preparer_sources(docs):
    """Récupérer les sources sans doublons."""

    sources = []
    urls_vues = set()

    for doc in docs:
        url = doc.metadata.get("source", "")

        if url and url not in urls_vues:
            urls_vues.add(url)

            sources.append({
                "title": doc.metadata.get(
                    "title", "Wikipédia"
                ),
                "url": url,
            })

    return sources


# --- Affichage des passages récupérés ---

def preparer_passages(docs):
    """Conserver les chunks et leurs métadonnées."""

    return [
        {
            "title": doc.metadata.get(
                "title", "Source inconnue"
            ),
            "url": doc.metadata.get("source", ""),
            "content": doc.page_content,
        }
        for doc in docs
    ]


def afficher_passages(passages):
    """Afficher les chunks dans une zone repliable."""

    if not passages:
        return

    with st.expander(
        f"Passages récupérés ({len(passages)})",
        expanded=False
    ):

        for i, passage in enumerate(passages, start=1):

            st.markdown(
                f"**Passage {i} — {passage['title']}**"
            )

            if passage["url"]:
                st.markdown(
                    f"[Consulter la source]({passage['url']})"
                )

            st.write(passage["content"])

            if i < len(passages):
                st.divider()


# --- Réafficher les anciens messages ---

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant":

            afficher_sources(
                message.get("sources", [])
            )

            afficher_passages(
                message.get("passages", [])
            )


# --- Zone de saisie ---

question = st.chat_input(
    "Posez votre question sur l'IA générative..."
)


# --- Génération des réponses ---

if question:

    # Enregistrer la question utilisateur
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner(
            "Recherche et génération de la réponse..."
        ):

            try:
                reponse, docs = rag_answer(
                    question,
                    retriever=retriever,
                    stream=False,
                )

            except Exception as erreur:

                st.error(
                    f"Impossible de générer une réponse : {erreur}"
                )

            else:

                sources = preparer_sources(docs)
                passages = preparer_passages(docs)

                # Afficher la réponse
                st.markdown(reponse)

                # Afficher les sources
                afficher_sources(sources)

                # Afficher les passages
                afficher_passages(passages)

                # Conserver l'historique
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": reponse,
                    "sources": sources,
                    "passages": passages,
                })
