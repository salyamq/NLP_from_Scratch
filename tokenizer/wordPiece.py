from collections import Counter
import time


with open("shakespeare.txt", "r") as f:
    file = f.read()

# ---------------------------

def buildVocab(corpus):
    vocab = set()
    for word in corpus:
        vocab.update(word)
    return sorted(vocab)

def buildCorpus(text: str):
    splitted_text = text.split()

    corpus = []
    for word in splitted_text:
        corpus.append(list(word))

    return corpus

def countTokens(corpus):
    counter = Counter()

    for word in corpus:
        counter.update(word)

    return counter



def countPairs(corpus: list):
    pairs = []
    for word in corpus:
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pairs.append(pair)

    pairCounter = Counter(pairs)
    return pairCounter

def findBestPair(pair_counter: dict, token_counter):
    best_pair = None
    best_score = -1

    for pair, frequency in pair_counter.items():
        score = frequency / ( token_counter[pair[0]] * token_counter[pair[1]])

        if score > best_score:
            best_score = score
            best_pair = pair

    return best_pair


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



def train_wordpiece(text, vocab_size):
    corpus = buildCorpus(text)
    vocab = buildVocab(corpus)

    while len(vocab) < vocab_size:
        pairCounter = countPairs(corpus)
        tokenCounter = countTokens(corpus)

        if not pairCounter:
            break

        bestPair = findBestPair(pairCounter, tokenCounter)

        corpus = mergePair(corpus, bestPair)
        vocab = updateVocab(vocab, bestPair)

    return vocab, corpus

# ------------------
model = train_wordpiece(file, 1000)
def encode(text: str, vocab):
    words = text.split()
    tokens = []

    for word in words:
        word_tokens = []
        start = 0

        while start < len(word):
            end = len(word)
            found = False

            while end > start:
                piece = word[start:end]
                if piece in vocab:
                    word_tokens.append(piece)
                    start = end
                    found = True
                    break
                else:
                    end = end - 1

            if not found:
                word_tokens.append("<unk>")
                break

        tokens.append(word_tokens)

    return tokens


def decode(tokens):
    words = []

    for word in tokens:
        words.append("".join(word))

    return " ".join(words)



