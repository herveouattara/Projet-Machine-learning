import pandas as pd

# Read the CSV file
sms_raw = pd.read_csv("sms_spam.csv")

# View the first 5 rows
# Compter le nombre de fois que "ham" apparaît dans la première colonne
count_ham = sms_raw.iloc[:, 0].value_counts().get("ham", 0)

# Compter le nombre de fois que "spam" apparaît dans la première colonne
count_spam = sms_raw.iloc[:, 0].value_counts().get("spam", 0)

# Afficher le résultat
print("Nombre de fois que 'ham' apparaît dans la deuxième colonne:", count_ham)
print("Nombre de fois que 'spam' apparaît dans la deuxième colonne:", count_spam)

X = sms_raw.iloc[:, 1]  # SMS, supposant que les SMS sont dans la deuxième colonne
Y = sms_raw.iloc[:, 0]  # Labels (spam/ham), supposant qu'ils sont dans la première colonne

import nltk
nltk.download("punkt")

from nltk.tokenize import sent_tokenize, word_tokenize


token_words = X.apply(word_tokenize)

print(token_words)

from nltk . stem import PorterStemmer
from nltk . stem import LancasterStemmer
# create an object of class PorterStemmer
porter = PorterStemmer ()
lancaster = LancasterStemmer ()
# A list of words to be stemmed
word_list = ["Friend ","friendship","friends","friendships","stabil","destabilize","misunderstanding","railroad","moonlight","football"]
print("{0:20}{1:20}{2:20}".format("Word","Porter Stemmer","lancaster Stemmer"))
for word in word_list :
 print("{0:20}{1:20}{2:20}".format(word,porter.stem(word),lancaster.stem(word)))

tokenizer = nltk.RegexpTokenizer("\w+")
new_words = X.apply(tokenizer.tokenize)
print(new_words)

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download('stopwords')

porter = PorterStemmer()

def stemSentence(sentence):
    tokenizer = nltk.RegexpTokenizer(r"[a-zA-Z]+")
    token_words = tokenizer.tokenize(sentence)
    stop_words = set(stopwords.words('english'))
    stem_sentence = []

    for word in token_words:
        if word not in stop_words:
            stem_sentence.append(porter.stem(word))
            stem_sentence.append(" ")

    return "".join(stem_sentence).strip()


X=X.apply(stemSentence)

print(X)

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(min_df=5)
 
count_matrix= count_vect.fit_transform(X)
count_array= count_matrix.toarray()
print(count_array)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, stratify=Y, random_state=42)

print("Distribution dans l'ensemble d'entraînement:")
print(Y_train.value_counts(normalize=True))
print("\nDistribution dans l'ensemble de test:")
print(Y_test.value_counts(normalize=True))



import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Séparer les spams et les messages normaux basés sur X
spam_texts = ' '.join(X[Y == 'spam'])
ham_texts = ' '.join(X[Y == 'ham'])

# Initialiser une figure avec 1 ligne et 2 colonnes
plt.figure(figsize=(20,8))

# Premier subplot pour les spams
plt.subplot(1, 2, 1)
spam_wordcloud = WordCloud(max_words=50,width=600, height=400, background_color='white').generate(spam_texts)
plt.imshow(spam_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Nuage de mots pour les spams')

# Deuxième subplot pour les messages normaux
plt.subplot(1, 2, 2)
ham_wordcloud = WordCloud(max_words=50,width=600, height=400, background_color='white').generate(ham_texts)
plt.imshow(ham_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Nuage de mots pour les messages normaux')

# Afficher la figure
plt.tight_layout()
plt.show()
