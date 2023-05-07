import re
import spacy
from unidecode import unidecode
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer 


class TextProcessing:

    def __init__(self, 
                 language="italian", 
                 is_cleaned=True, 
                 is_lemmatized=False,
                 is_stopwords_removed=True,
                 is_tokenized=True):

        self.language = language
        self.is_cleaned = is_cleaned
        self.is_lemmatized = is_lemmatized
        self.is_stopwords_removed = is_stopwords_removed
        self.is_tokenized = is_tokenized

        if self.language == "italian":
            self.model = spacy.load("it_core_news_sm")
        else:
            self.model = spacy.load("en_core_web_sm")
        
        self.stopwords = self.model.Defaults.stop_words


    def __call__(self, text):

        if self.is_lemmatized:
            text = self._lemmatize_text(text)

        if self.is_cleaned:
            text = self._clean_text(text)

        if self.is_stopwords_removed:
            text = self._remove_stopwords(text)

        return text

    def _clean_text(self, text):
        """Clean text by removing special characters and multiple spaces."""
        text = unidecode(text)  # translate to ASCII text
        text = re.sub(r'[^\w\s]', r' ', text)  # remove special symbols (?!...)
        text = " ".join(text.split())
        return text

    def _lemmatize_text(self, text):
        """Lemmatize text using a pre-trained lemmatization model."""
        doc = self.model(text)
        lem_arr = [token.lemma_ for token in doc]
        
        return " ".join(lem_arr)

    def _remove_stopwords(self, text):
        """Remove stopwords from text."""
        words = [word for word in text.split() if word.lower() not in self.stopwords]
        return " ".join(words)
    
    def tokenize(self, text):
        
        doc = self.model(text)
        return [token.text for token in doc]



    
    
    

class Vectorization:



    @staticmethod
    def tfidf_vectorize(train_set, test_set, tokenizer, lowercase=True):
 
        vectorizer = TfidfVectorizer(tokenizer=tokenizer, lowercase=lowercase)
        vectorizer.fit(train_set)
        tokens_train_vec =  vectorizer.transform(list(train_set))
        tokens_test_vec = vectorizer.transform(list(test_set))
        features_names = vectorizer.get_feature_names_out()
        return tokens_train_vec, tokens_test_vec, features_names