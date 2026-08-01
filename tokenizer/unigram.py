from collections import Counter


# у нас есть корпус, нам надо найти словарь токенов V
# т.е V* = argmax_V P(Corpus | V)
# или точнее V* = argmax_V log P(Corpus | V) - λ * |V|
# Иными словами, мы ищем оптимальный набор токенов (или размер словаря),
# при котором наш корпус данных объясняется (или сжимается) наилучшим образом.

corpus = ['lowest', 'lower', 'low', 'newest', 'new']
def get_vocab(corpus):

    vocab = set()
    for word in corpus:
        for i in range(len(word)):
            for j in range(i + 1, len(word) + 1):
                vocab.add(word[i:j])

    return vocab

def get_initial_probabilities(corpus):
    counter = Counter()

    for word in corpus:
        for i in range(len(word)):
            for j in range(i + 1, len(word) + 1):
                counter[word[i:j]] = counter[word[i:j]] + 1

    total = sum(counter.values())

    probs = {}
    for token, freq in counter.items():
        probs[token] = freq / total


    return probs

probs = get_initial_probabilities(corpus)
def segment(word, probs):

    if not word:
        return []

    every_first_tokens = [word[:i] for i in range(1, len(word) + 1)]
    candidates = []
    best_prob = 0
    best_cand = None
    for token in every_first_tokens:

        if token not in probs:
            continue

        rest =  word[len(token):]
        best_pair = segment(rest, probs)

        if best_pair is None:
            continue

        candidates.append([token] + best_pair)

    for candidate in candidates:
        product = 1
        candidate_valid = True
        for subword in candidate:
            if subword not in probs:
                candidate_valid = False
                break
            product = product * probs[subword]

        if not candidate_valid:
            continue

        if best_prob < product:
            best_prob = product
            best_cand = candidate

    return best_cand








print(segment("lower", probs))










