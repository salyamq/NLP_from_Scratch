from collections import Counter
import math


# у нас есть корпус, нам надо найти словарь токенов V
# т.е V* = argmax_V P(Corpus | V)
# или точнее V* = argmax_V log P(Corpus | V) - λ * |V|
# Иными словами, мы ищем оптимальный набор токенов (или размер словаря),
# при котором наш корпус данных объясняется (или сжимается) наилучшим образом.

with open("shakespeare.txt", "r") as f:
    file = f.read()  # читаем весь файл в одну строку / read the whole file into one string


def get_vocab(corpus: list[str]) -> set[str]:
    # строит начальный словарь из вообще всех возможных подстрок слов корпуса
    # builds the initial vocabulary from every possible substring of the corpus words

    vocab = set()  # множество для уникальных подстрок / set for unique substrings
    for word in corpus:  # проходим по каждому слову корпуса / iterate over each word in the corpus
        for i in range(len(word)):  # i - начало подстроки / i is the start of the substring
            for j in range(i + 1, len(word) + 1):  # j - конец подстроки (не включительно) / j is the end of the substring (exclusive)
                vocab.add(word[i:j])  # добавляем все подстроки слова, начиная с i / add every substring of the word starting at i

    return vocab

# EM (Expectation-Maximization)
def get_initial_probabilities(corpus: list[str]) -> dict[str, float]:
    # считает начальные вероятности всех подстрок как их частоту в корпусе
    # computes initial probabilities of all substrings as their frequency in the corpus
    counter = Counter()  # счётчик частот подстрок / counter for substring frequencies

    for word in corpus:  # проходим по каждому слову корпуса / iterate over each word in the corpus
        for i in range(len(word)):  # i - начало подстроки / i is the start of the substring
            for j in range(i + 1, len(word) + 1):  # j - конец подстроки (не включительно) / j is the end of the substring (exclusive)
                counter[word[i:j]] = counter[word[i:j]] + 1  # увеличиваем счётчик для этой подстроки / bump the counter for this substring

    total = sum(counter.values())  # общее число всех подстрок / total count of all substrings

    probs = {}  # словарь вероятностей токен -> вероятность / dictionary of token -> probability
    for token, freq in counter.items():  # проходим по всем подстрокам и их частотам / iterate over all substrings and their frequencies
        probs[token] = freq / total  # вероятность = частота делённая на общее число / probability = frequency divided by the total count


    return probs


def segment(word: str, probs: dict[str, float]) -> list[str] | None:
    # рекурсивно находит самое вероятное разбиение слова на токены из словаря
    # recursively finds the most probable segmentation of a word into vocabulary tokens

    if not word:  # базовый случай рекурсии: пустое слово / base case of the recursion: empty word
        return []  # пустое разбиение для пустой строки / empty segmentation for an empty string

    every_first_tokens = [word[:i] for i in range(1, len(word) + 1)]  # все возможные первые куски слова, от одного символа до всего слова / all possible first pieces of the word, from one character up to the whole word
    candidates = []  # список кандидатов-разбиений / list of candidate segmentations
    best_prob = 0  # лучшая найденная вероятность среди кандидатов / best probability found among the candidates
    best_cand = None  # лучшее найденное разбиение / best segmentation found so far
    for token in every_first_tokens:  # перебираем каждый возможный первый кусок / try every possible first piece

        if token not in probs:  # если такого токена нет в словаре, пропускаем / if this token isn't in the vocabulary, skip it
            continue

        rest =  word[len(token):]  # оставшаяся часть слова после первого куска / the remaining part of the word after the first piece
        best_pair = segment(rest, probs)  # рекурсивно разбиваем остаток слова / recursively segment the rest of the word

        if best_pair is None:  # если остаток не удалось разбить, этот вариант не подходит / if the rest couldn't be segmented, this option doesn't work
            continue

        candidates.append([token] + best_pair)  # собираем кандидат: первый кусок плюс разбиение остатка / build the candidate: first piece plus the segmentation of the rest

    for candidate in candidates:  # проходим по всем собранным кандидатам-разбиениям / iterate over all collected candidate segmentations
        product = 1  # произведение вероятностей токенов кандидата / product of the candidate's token probabilities
        candidate_valid = True  # флаг, что все токены кандидата есть в словаре / flag that all of the candidate's tokens are in the vocabulary
        for subword in candidate:  # проходим по каждому токену кандидата / iterate over each token of the candidate
            if subword not in probs:  # если токена нет в словаре / if the token isn't in the vocabulary
                candidate_valid = False  # кандидат недействителен / candidate is invalid
                break
            product = product * probs[subword]  # перемножаем вероятности токенов / multiply the token probabilities together

        if not candidate_valid:  # пропускаем недействительных кандидатов / skip invalid candidates
            continue

        if best_prob < product:  # если это разбиение вероятнее найденного ранее / if this segmentation is more probable than the one found so far
            best_prob = product  # обновляем лучшую вероятность / update the best probability
            best_cand = candidate  # обновляем лучшее разбиение / update the best segmentation

    return best_cand

def calculate_likelihood(corpus: list[str], probs: dict[str, float]) -> float:
    # считает суммарное логарифмическое правдоподобие корпуса при данных вероятностях токенов
    # computes the total log-likelihood of the corpus given the current token probabilities
    score = 0  # накопленный логарифм правдоподобия / accumulated log-likelihood

    for word in corpus:  # проходим по каждому слову корпуса / iterate over each word in the corpus
        segmented = segment(word, probs)  # находим лучшее разбиение слова / find the best segmentation of the word
        if segmented is None:  # если слово вообще нельзя разбить текущим словарём / if the word can't be segmented with the current vocabulary at all
            return float("-inf")  # правдоподобие считается минус бесконечностью / the likelihood is treated as negative infinity

        prob = 1  # вероятность этого конкретного разбиения слова / probability of this particular word segmentation
        for token in segmented:  # проходим по токенам разбиения / iterate over the tokens of the segmentation
            prob *= probs[token]  # перемножаем вероятности токенов / multiply the token probabilities together

        score += math.log(prob)  # добавляем логарифм вероятности слова к общему счёту / add the log-probability of the word to the total score

    return score



def prune_vocab(corpus: list[str], probs: dict[str, float], vocab_size: int) -> dict[str, float]:
    # удаляет токены, потеря которых меньше всего снижает правдоподобие корпуса
    # removes tokens whose removal hurts the corpus likelihood the least
    losses = {}  # потеря правдоподобия для каждого токена при его удалении / likelihood loss for each token if it were removed
    original_score = calculate_likelihood(corpus, probs)  # правдоподобие с полным текущим словарём / likelihood with the full current vocabulary

    for token in probs:  # проходим по каждому токену словаря / iterate over every token in the vocabulary

        temp_probs = probs.copy()  # копия словаря вероятностей, чтобы не портить оригинал / a copy of the probability dict so the original isn't modified
        del temp_probs[token]  # временно убираем проверяемый токен / temporarily remove the token being tested

        new_score = calculate_likelihood(corpus, temp_probs)  # правдоподобие без этого токена / likelihood without this token
        loss = original_score - new_score  # насколько упало правдоподобие после удаления / how much the likelihood dropped after removal
        losses[token] = loss  # сохраняем потерю для этого токена / store the loss for this token

    sorted_losses = sorted(losses.items(), key=lambda x: x[1])  # сортируем токены по возрастанию потери / sort tokens by ascending loss
    remove_count = len(probs) - vocab_size  # сколько токенов нужно удалить, чтобы дойти до целевого размера / how many tokens need to be removed to reach the target size
    tokens_to_remove = sorted_losses[:remove_count]  # берём токены с наименьшей потерей - их не жалко удалить / take the tokens with the smallest loss, they're safest to remove


    for token, loss in tokens_to_remove:  # проходим по токенам, отобранным на удаление / iterate over the tokens selected for removal
        del probs[token]  # удаляем токен из словаря вероятностей / remove the token from the probability dictionary


    return probs


def train_unigram(corpus: list[str], vocab_size: int) -> dict[str, float]:
    # обучает unigram-модель, постепенно урезая словарь до нужного размера
    # trains the unigram model by gradually shrinking the vocabulary to the target size
    probs = get_initial_probabilities(corpus)  # стартовые вероятности из всех возможных подстрок / starting probabilities from every possible substring


    while len(probs) > vocab_size:  # пока словарь больше целевого размера / while the vocabulary is bigger than the target size
        segmented_words = []  # разбиения всех слов текущим словарём (шаг E) / segmentations of all words with the current vocabulary (E-step)
        for word in corpus:  # проходим по каждому слову корпуса / iterate over each word in the corpus
            segmented_word = segment(word, probs)  # находим лучшее разбиение слова / find the best segmentation of the word
            segmented_words.append(segmented_word)  # сохраняем разбиение слова / store the word's segmentation

        counts = Counter(item for sublist in segmented_words for item in sublist)  # считаем частоту каждого токена во всех разбиениях / count the frequency of every token across all segmentations
        total_tokens = sum(counts.values())  # общее число токенов во всех разбиениях / total number of tokens across all segmentations

        new_probs = {}  # обновлённые вероятности токенов (шаг M) / updated token probabilities (M-step)
        for token in counts:  # проходим по каждому встретившемуся токену / iterate over every token that occurred
            p_token = counts[token] / total_tokens  # новая вероятность токена по его частоте / new token probability based on its frequency
            new_probs[token] = p_token  # сохраняем обновлённую вероятность / store the updated probability

        probs = prune_vocab(  # урезаем словарь, убирая наименее полезные токены / prune the vocabulary by removing the least useful tokens
            corpus,
            new_probs,
            vocab_size
        )

    return probs




corpus = ['lowest', 'lower', 'low', 'newest', 'new']
print(train_unigram(file.split(), 58))