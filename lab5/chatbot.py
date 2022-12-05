# Write a chatbot that reacts on specific dialog [[Question, Answer], ...]
# You can follow:
# 1. Create TFIDF vectors for each question (try without stop words first)
# 2. Write REPL that reads question, convert it to TFIDF vector, compute cos similarity, finds best match, and returns answer to the best-matched question.

from nltk.tokenize import casual_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

conversation = [["Hello", "Hi"], ["How are you?", "I am fine"], [
    "What is your name?", "My name is Chatbot"], ["Bye", "Bye"]]
questions = [q for q, a in conversation]
answers = [a for q, a in conversation]

vectorizer = TfidfVectorizer(tokenizer=casual_tokenize, stop_words='english')
tfidf = vectorizer.fit_transform(questions)

while True:
    question = input("You: ")
    if question == "Bye":
        print("Chatbot: Bye")
        break
    question_tfidf = vectorizer.transform([question])
    similarity = cosine_similarity(question_tfidf, tfidf)
    best_match_index = similarity.argmax()
    answer = answers[best_match_index]
    print("Chatbot:", answer)

# q: how to run python script from command line
# a: python <script_name>.py
