# import dependencies
import string
from nltk.corpus import gutenberg
from gensim.models.phrases import Phraser, Phrases
from gensim.models import Word2Vec

# load data
gberg_sents = gutenberg.sents()

# lowering case and removing punctuation
lower_sents = []
for s in gberg_sents:
    lower_sents.append([w.lower() for w in s if w.lower()
                        not in list(string.punctuation)])

# calculating bigrams from corpus
lower_bigram = Phraser(Phrases(lower_sents,
                               min_count=32,
                               threshold=64))

# including bigrams in corpus
clean_sents = []
for s in lower_sents:
    clean_sents.append(lower_bigram[s])

# calculating word embeddings
model = Word2Vec(sentences=clean_sents,
                 size=64, sg=1, window=10,
                 iter=5, min_count=10, workers=6)

# save word embeddings
model.save('data/clean_gutenberg_model.w2v')
