import multiprocessing as mp
import numpy as np

ARG_SPECIAL_TAG = 'ARGSPECIAL'
PRED_SPECIAL_TAG = 'PREDSPECIAL'

def make_embeded_form(word):
    if word:
        # return word[1].encode('utf8')
        return u"{}_{}".format(word[1], word[0])
    else:
        return word

class EmbedderMap:
    def __init__(self, embeddings, X):
        self.X_ = X
        self.embeddings_ = embeddings

    def __call__(self, i):
        result = np.zeros((len(self.X_[0]),
                           self.embeddings_.vector_size))

        for j in range(len(self.X_[0])):
            word = self.X_[i][j]
            tag = word[0] if word else str()

            if tag == ARG_SPECIAL_TAG or tag == ARG_SPECIAL_TAG:
                result[j, :] = np.ones(self.embeddings_.vector_size)
            elif word and word in self.embeddings_:
                result[j, :] = self.embeddings_[word]

        return result


def embed(X, embeddings):
    pool = mp.Pool(4)
    result = pool.map(EmbedderMap(embeddings, X), X.index, 1000)
    pool.close()
    return np.asarray(result)


class EmbedderSingleMap:
    def __init__(self, embeddings, X):
        self.X_ = X
        self.embeddings_ = embeddings

    def __call__(self, i):
        #word = make_embeded_form(self.X_[i])
        word = self.X_[i]
        if word in self.embeddings_:
            return self.embeddings_[word]
        else:
            return np.zeros((self.embeddings_.vector_size,))


def embed_single(embeddings, X):
    pool = mp.Pool(4)
    result = pool.map(EmbedderSingleMap(embeddings, X), X.index, 1000)
    pool.close()

    return np.asarray(result)

