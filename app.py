
"""
TP RAG - Question 20
Interface Streamlit avec affichage des passages récupérés.
"""

import streamlit as st
from main import get_retriever, rag_answer

# --- Configuration de l'interface ---

st.set_page_config(
    page_title="Chatbot RAG - IA générative",
    page_icon="💬",
    layout="centered",
)

st.title("Chatbot RAG - IA générative")

st.caption(
    "Posez vos questions sur l'intelligence artificielle générative."
)


# --- Charger le retriever une seule fois ---

@st.cache_resource
def charger_retriever():
    return get_retriever()


with st.spinner("Connexion à la base documentaire..."):
    retriever = charger_retriever()


# --- Historique des conversations ---

if "messages" not in st.session_state:
    st.session_state.messages = []


# --- Affichage des sources ---

def afficher_sources(sources):
    """Affiche les sources Wikipédia sans doublons."""

    if not sources:
        return

    st.markdown("**Sources :**")

    for source in sources:
        st.markdown(
            f"- [{source['title']}]({source['url']})"
        )


def preparer_sources(docs):
    """Récupère les sources uniques des documents."""

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


# --- Question 20 : récupérer les passages ---

def preparer_passages(docs):
    """Conserve les textes et métadonnées des chunks."""

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
    """Affiche les chunks dans une zone repliable."""

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


# --- Réafficher l'historique ---

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


# --- Génération des nouvelles réponses ---

if question:

    # Enregistrer la question
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Afficher la réponse de l'IA
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

                # Préparer les sources et passages
                sources = preparer_sources(docs)
                passages = preparer_passages(docs)

                # Afficher la réponse
                st.markdown(reponse)

                # Afficher les sources
                afficher_sources(sources)

                # Afficher les chunks dans un expander
                afficher_passages(passages)

                # Enregistrer toute la réponse
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": reponse,
                    "sources": sources,
                    "passages": passages,
                })
