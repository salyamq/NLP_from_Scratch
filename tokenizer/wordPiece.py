from collections import Counter
import time


with open("shakespeare.txt", "r") as f:
    file = f.read()  # читаем весь файл в одну строку / read the whole file into one string

# ---------------------------

def buildVocab(corpus: list[list[str]]) -> list[str]:
    # строит начальный словарь из уникальных символов корпуса
    # builds the initial vocabulary from unique characters in the corpus
    vocab = set()  # множество для уникальных символов / set for unique characters
    for word in corpus:  # проходим по каждому слову корпуса / iterate over each word in the corpus
        vocab.update(word)  # добавляем символы слова в множество / add the word's characters to the set
    return sorted(vocab)  # возвращаем отсортированный список символов / return a sorted list of characters

def buildCorpus(text: str) -> list[list[str]]:
    # разбивает текст на слова и каждое слово превращает в список символов
    # splits text into words and turns each word into a list of characters
    splitted_text = text.split()  # разбиваем текст по пробелам на слова / split text into words by whitespace

    corpus = []  # сюда будем складывать слова-списки символов / will hold words as lists of characters
    for word in splitted_text:  # проходим по каждому слову / iterate over each word
        corpus.append(list(word))  # превращаем слово в список отдельных символов / turn the word into a list of individual characters

    return corpus

def countTokens(corpus: list[list[str]]) -> Counter:
    # считает частоту каждого отдельного токена во всём корпусе
    # counts the frequency of each individual token across the whole corpus
    counter = Counter()  # счётчик токенов / counter for tokens

    for word in corpus:  # проходим по каждому слову корпуса / iterate over each word in the corpus
        counter.update(word)  # увеличиваем счётчик для каждого токена слова / bump the counter for each token in the word

    return counter



def countPairs(corpus: list) -> Counter:
    # считает частоту всех соседних пар токенов в корпусе
    # counts the frequency of all adjacent token pairs in the corpus
    pairs = []  # список всех найденных пар токенов / list of all found token pairs
    for word in corpus:  # проходим по каждому слову корпуса / iterate over each word in the corpus
        for i in range(len(word) - 1):  # идём по индексам токенов кроме последнего / iterate over indices except the last one
            pair = (word[i], word[i + 1])  # берём текущий и следующий токен как пару / take the current and next token as a pair
            pairs.append(pair)  # добавляем пару в общий список / add the pair to the overall list

    pairCounter = Counter(pairs)  # считаем сколько раз встретилась каждая пара / count how many times each pair occurs
    return pairCounter

def findBestPair(pair_counter: dict, token_counter: Counter) -> tuple[str, str] | None:
    # находит пару с наивысшим wordpiece-скором, а не просто самую частую
    # finds the pair with the highest wordpiece score, not just the most frequent one
    best_pair = None  # лучшая найденная пара / best pair found so far
    best_score = -1  # лучший найденный скор / best score found so far

    for pair, frequency in pair_counter.items():  # проходим по всем парам и их частотам / iterate over all pairs and their frequencies
        score = frequency / ( token_counter[pair[0]] * token_counter[pair[1]])  # скор нормализует частоту пары на частоты её частей / score normalizes pair frequency by the frequencies of its parts

        if score > best_score:  # если текущий скор лучше найденного / if the current score is better than the best found so far
            best_score = score  # обновляем лучший скор / update the best score
            best_pair = pair  # обновляем лучшую пару / update the best pair

    return best_pair


def mergePair(corpus: list[list[str]], max_pair: tuple[str, str]) -> list[list[str]]:
    # объединяет все вхождения заданной пары в один токен по всему корпусу
    # merges all occurrences of the given pair into a single token across the corpus
    new_corpus = []  # новый корпус после слияния пары / new corpus after merging the pair
    for word in corpus:  # проходим по каждому слову старого корпуса / iterate over each word in the old corpus
        new_word = []  # новое слово, собранное после слияния / new word assembled after merging
        i = 0  # индекс текущего токена в слове / index of the current token in the word
        while i < len(word) - 1:  # пока не дошли до предпоследнего токена / while we haven't reached the second-to-last token
            if (word[i], word[i + 1]) == max_pair:  # если текущая пара совпадает с искомой / if the current pair matches the target pair
                new_word.append(word[i] + word[i + 1])  # склеиваем пару в один токен / merge the pair into a single token
                i = i + 2  # пропускаем оба слитых токена / skip both merged tokens
            else:
                new_word.append(word[i])  # токен остаётся отдельным / keep the token separate
                i = i + 1  # переходим к следующему токену / move to the next token

        if i == len(word) - 1:  # если остался последний токен без пары / if the last token was left unpaired
            new_word.append(word[-1])  # добавляем его как есть / append it as is

        new_corpus.append(new_word)  # добавляем собранное слово в новый корпус / add the assembled word to the new corpus

    return new_corpus

def updateVocab(vocab: list[str], max_pair: tuple[str, str]) -> list[str]:
    # добавляет новый объединённый токен в словарь
    # adds the new merged token to the vocabulary

    new_token = max_pair[0] + max_pair[1]  # склеиваем пару в новый токен / concatenate the pair into a new token
    vocab.append(new_token)  # добавляем токен в словарь / add the token to the vocabulary
    return vocab



def train_wordpiece(text: str, vocab_size: int) -> tuple[list[str], list[list[str]]]:
    # обучает wordpiece-токенизатор пока словарь не достигнет нужного размера
    # trains the wordpiece tokenizer until the vocabulary reaches the target size
    corpus = buildCorpus(text)  # начальный корпус, слова как списки символов / initial corpus, words as lists of characters
    vocab = buildVocab(corpus)  # начальный словарь из отдельных символов / initial vocabulary of individual characters

    while len(vocab) < vocab_size:  # пока словарь меньше целевого размера / while the vocabulary is smaller than the target size
        pairCounter = countPairs(corpus)  # считаем частоты пар в текущем корпусе / count pair frequencies in the current corpus
        tokenCounter = countTokens(corpus)  # считаем частоты отдельных токенов / count frequencies of individual tokens

        if not pairCounter:  # если пар больше не осталось / if there are no more pairs left
            break  # прекращаем обучение / stop training

        bestPair = findBestPair(pairCounter, tokenCounter)  # выбираем пару с лучшим wordpiece-скором / pick the pair with the best wordpiece score

        corpus = mergePair(corpus, bestPair)  # применяем слияние ко всему корпусу / apply the merge to the whole corpus
        vocab = updateVocab(vocab, bestPair)  # добавляем новый токен в словарь / add the new token to the vocabulary

    return vocab, corpus

# ------------------
model = train_wordpiece(file, 1000)  # обучаем модель на всём файле / train the model on the whole file
def encode(text: str, vocab: list[str]) -> list[list[str]]:
    # кодирует текст жадно разбивая каждое слово на самые длинные подстроки из словаря
    # encodes text by greedily splitting each word into the longest substrings found in the vocabulary
    words = text.split()  # разбиваем текст на слова / split text into words
    tokens = []  # сюда складываем токены каждого слова / will hold the tokens for each word

    for word in words:  # проходим по каждому слову / iterate over each word
        word_tokens = []  # токены текущего слова / tokens of the current word
        start = 0  # начало текущего непокрытого куска слова / start of the current uncovered chunk of the word

        while start < len(word):  # пока слово не разобрано полностью / while the word hasn't been fully split
            end = len(word)  # изначально пробуем весь остаток слова / initially try the whole remaining word
            found = False  # флаг, что подходящий кусок найден / flag that a matching piece was found

            while end > start:  # уменьшаем кусок пока не найдём совпадение / shrink the piece until a match is found
                piece = word[start:end]  # текущий проверяемый кусок / the current candidate piece
                if piece in vocab:  # если кусок есть в словаре / if the piece exists in the vocabulary
                    word_tokens.append(piece)  # добавляем его как токен / add it as a token
                    start = end  # сдвигаем начало за пределы найденного куска / move start past the found piece
                    found = True  # отмечаем, что нашли совпадение / mark that a match was found
                    break  # выходим из внутреннего цикла поиска / exit the inner search loop
                else:
                    end = end - 1  # уменьшаем кусок на один символ и пробуем снова / shrink the piece by one character and try again

            if not found:  # если ни один кусок не подошёл / if no piece matched at all
                word_tokens.append("<unk>")  # помечаем слово как неизвестное / mark the word as unknown
                break  # прекращаем разбор этого слова / stop processing this word

        tokens.append(word_tokens)  # добавляем токены слова в общий список / add the word's tokens to the overall list

    return tokens


def decode(tokens: list[list[str]]) -> str:
    # декодирует список токенов обратно в текст
    # decodes the list of tokens back into text
    words = []  # сюда складываем восстановленные слова / will hold the reconstructed words

    for word in tokens:  # проходим по каждому закодированному слову / iterate over each encoded word
        words.append("".join(word))  # склеиваем токены слова обратно в строку / join the word's tokens back into a string

    return " ".join(words)  # соединяем слова пробелами обратно в текст / join words with spaces back into text