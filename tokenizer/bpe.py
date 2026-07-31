from collections import Counter
import time

start = time.perf_counter()

with open("shakespeare.txt", "r") as f:
    file = f.read()

# ---------------------------

def buildVocab(text: str):
    return sorted(set(text))

def buildCorpus(text: str):
    splitted_text = text.split()

    corpus = []
    for word in splitted_text:
        corpus.append(list(word))

    return corpus


def countPairs(corpus: list):
    pairs = []
    for word in corpus:
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pairs.append(pair)

    pairCounter = Counter(pairs)
    return pairCounter

def findBestPair(pair_counter: dict):
    maxPair = max(pair_counter, key=pair_counter.get)
    return maxPair

def mergePair(corpus, max_pair):
    new_corpus = []
    for word in corpus:
        new_word = []
        i = 0
        while i < len(word) - 1:
            if (word[i], word[i + 1]) == max_pair:
                new_word.append(word[i] + word[i + 1])
                i = i + 2
            else:
                new_word.append(word[i])
                i = i + 1

        if i == len(word) - 1:
            new_word.append(word[-1])

        new_corpus.append(new_word)

    return new_corpus

def updateVocab(vocab, max_pair):

    new_token = max_pair[0] + max_pair[1]
    vocab.append(new_token)
    return vocab



def train_bpe(text, vocab_size):
    vocab = buildVocab(text)
    corpus = buildCorpus(text)
    merges = []

    while len(vocab) < vocab_size:
        pairCounter = countPairs(corpus)

        if not pairCounter:
            break

        maxPair = findBestPair(pairCounter)

        merges.append(maxPair)

        corpus = mergePair(corpus, maxPair)
        vocab = updateVocab(vocab, maxPair)

    return vocab, merges, corpus

def encode(text: str, merges):
    splitted_text = buildCorpus(text)

    for pair in merges:
        splitted_text = mergePair(splitted_text, pair)

    return splitted_text

def decode(splitted_text):
    text = []
    for word in splitted_text:
        new_word = "".join(word)
        text.append(new_word)

    return " ".join(text)


print(train_bpe(file, 1000))
print(time.perf_counter() - start) # 2.4312329160020454


