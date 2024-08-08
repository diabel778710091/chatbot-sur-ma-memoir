import os
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import spacy
import nltk
from nltk.tokenize import sent_tokenize

# Téléchargement des ressources NLTK
nltk.download('punkt')

# Chargement du modèle spaCy pour le français
nlp = spacy.load('fr_core_news_sm')

# Chargement du modèle SentenceTransformer
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

def load_and_preprocess_text(file_path):
    if not os.path.exists(file_path):
        st.error(f"Le fichier {file_path} n'existe pas.")
        return []
    try:
        with open(file_path, 'r', encoding='latin-1') as f:
            data = f.read().replace('\n', ' ')
        return sent_tokenize(data)
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier : {e}")
        return []

def preprocess(phrase):
    doc = nlp(phrase)
    mots = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(mots)

def preprocess_corpus(phrases):
    return [preprocess(phrase) for phrase in phrases]

def initialize_vectorizer(corpus):
    try:
        return model.encode(corpus)
    except Exception as e:
        st.error(f"Erreur lors de la vectorisation : {e}")
        return None

def get_most_relevant_sentence(question, vectors, phrases):
    try:
        processed_question = preprocess(question)
        query_vec = model.encode([processed_question])
        similarities = util.pytorch_cos_sim(query_vec, vectors).flatten()
        index = similarities.argmax()
        return phrases[index]
    except Exception as e:
        st.error(f"Erreur lors du calcul de la similarité : {e}")
        return "Désolé, je n'ai pas pu trouver une réponse pertinente."

def main():
    st.title("Chatbot sur ma mémoire de L3")
    st.write("Bonjour ! Posez-moi des questions sur le sujet.")

    file_path = 'Amélioration_des_services_bancaires.txt'
    phrases = load_and_preprocess_text(file_path)

    if phrases:
        corpus = preprocess_corpus(phrases)
        vectors = initialize_vectorizer(corpus)

        if vectors is not None:
            question = st.text_input("Vous :", placeholder="Entrez votre question ici...")

            if st.button("Soumettre", key="submit_button"):
                if question:
                    response = get_most_relevant_sentence(question, vectors, phrases)
                    st.write("El Hadji Diabel : " + response)
                else:
                    st.warning("Veuillez entrer une question avant de soumettre.")
        else:
            st.warning("Erreur dans la vectorisation des phrases.")
    else:
        st.warning("Aucune phrase n'a été chargée.")

if __name__ == "__main__":
    main()
