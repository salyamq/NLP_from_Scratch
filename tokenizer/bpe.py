from collections import Counter
import time

start = time.perf_counter()  # засекаем время старта / mark the start time

with open("shakespeare.txt", "r") as f:
    file = f.read()  # читаем весь файл в одну строку / read the whole file into one string

# ---------------------------

def buildVocab(text: str) -> list[str]:
    # возвращает отсортированный список уникальных символов текста
    # returns a sorted list of unique characters in the text
    return sorted(set(text))  # set убирает дубликаты, sorted упорядочивает / set removes duplicates, sorted orders them

def buildCorpus(text: str) -> list[list[str]]:
    # разбивает текст на слова и каждое слово превращает в список символов
    # splits text into words and turns each word into a list of characters
    splitted_text = text.split()  # разбиваем текст по пробелам на слова / split text into words by whitespace

    corpus = []  # сюда будем складывать слова-списки символов / will hold words as lists of characters
    for word in splitted_text:  # проходим по каждому слову / iterate over each word
        corpus.append(list(word))  # превращаем слово в список отдельных символов / turn the word into a list of individual characters

    return corpus


def countPairs(corpus: list[list[str]]) -> Counter:
    # считает частоту всех соседних пар символов в корпусе
    # counts the frequency of all adjacent character pairs in the corpus
    pairs = []  # список всех найденных пар символов / list of all found character pairs
    for word in corpus:  # проходим по каждому слову корпуса / iterate over each word in the corpus
        for i in range(len(word) - 1):  # идём по индексам символов кроме последнего / iterate over indices except the last one
            pair = (word[i], word[i + 1])  # берём соседний символ и текущий как пару / take the current and next character as a pair
            pairs.append(pair)  # добавляем пару в общий список / add the pair to the overall list

    pairCounter = Counter(pairs)  # считаем сколько раз встретилась каждая пара / count how many times each pair occurs
    return pairCounter

def findBestPair(pair_counter: dict) -> tuple[str, str]:
    # находит самую частую пару символов
    # finds the most frequent character pair
    maxPair = max(pair_counter, key=pair_counter.get)  # ищем ключ с максимальным значением счётчика / find the key with the highest counter value
    return maxPair

def mergePair(corpus: list[list[str]], max_pair: tuple[str, str]) -> list[list[str]]:
    # объединяет все вхождения заданной пары в один токен по всему корпусу
    # merges all occurrences of the given pair into a single token across the corpus
    new_corpus = []  # новый корпус после слияния пары / new corpus after merging the pair
    for word in corpus:  # проходим по каждому слову старого корпуса / iterate over each word in the old corpus
        new_word = []  # новое слово, собранное после слияния / new word assembled after merging
        i = 0  # индекс текущего символа в слове / index of the current character in the word
        while i < len(word) - 1:  # пока не дошли до предпоследнего символа / while we haven't reached the second-to-last character
            if (word[i], word[i + 1]) == max_pair:  # если текущая пара совпадает с искомой / if the current pair matches the target pair
                new_word.append(word[i] + word[i + 1])  # склеиваем пару в один токен / merge the pair into a single token
                i = i + 2  # пропускаем оба слитых символа / skip both merged characters
            else:
                new_word.append(word[i])  # символ остаётся отдельным / keep the character separate
                i = i + 1  # переходим к следующему символу / move to the next character

        if i == len(word) - 1:  # если остался последний символ без пары / if the last character was left unpaired
            new_word.append(word[-1])  # добавляем его как есть / append it as is

        new_corpus.append(new_word)  # добавляем собранное слово в новый корпус / add the assembled word to the new corpus

    return new_corpus

def updateVocab(vocab: list[str], max_pair: tuple[str, str]) -> list[str]:
    # добавляет новый объединённый токен в словарь
    # adds the new merged token to the vocabulary

    new_token = max_pair[0] + max_pair[1]  # склеиваем пару в новый токен / concatenate the pair into a new token
    vocab.append(new_token)  # добавляем токен в словарь / add the token to the vocabulary
    return vocab



def train_bpe(text: str, vocab_size: int) -> tuple[list[str], list[tuple[str, str]], list[list[str]]]:
    # обучает bpe-токенизатор пока словарь не достигнет нужного размера
    # trains the bpe tokenizer until the vocabulary reaches the target size
    vocab = buildVocab(text)  # начальный словарь из отдельных символов / initial vocabulary of individual characters
    corpus = buildCorpus(text)  # начальный корпус, слова как списки символов / initial corpus, words as lists of characters
    merges = []  # список выполненных слияний в порядке их применения / list of merges performed, in order of application

    while len(vocab) < vocab_size:  # пока словарь меньше целевого размера / while the vocabulary is smaller than the target size
        pairCounter = countPairs(corpus)  # считаем частоты пар в текущем корпусе / count pair frequencies in the current corpus

        if not pairCounter:  # если пар больше не осталось / if there are no more pairs left
            break  # прекращаем обучение / stop training

        maxPair = findBestPair(pairCounter)  # выбираем самую частую пару / pick the most frequent pair

        merges.append(maxPair)  # запоминаем это слияние / remember this merge

        corpus = mergePair(corpus, maxPair)  # применяем слияние ко всему корпусу / apply the merge to the whole corpus
        vocab = updateVocab(vocab, maxPair)  # добавляем новый токен в словарь / add the new token to the vocabulary

    return vocab, merges, corpus

def encode(text: str, merges: list[tuple[str, str]]) -> list[list[str]]:
    # кодирует текст применяя обученные слияния по порядку
    # encodes text by applying the trained merges in order
    splitted_text = buildCorpus(text)  # разбиваем текст на слова из символов / split text into words of characters

    for pair in merges:  # применяем каждое слияние по очереди в том же порядке, что при обучении / apply each merge in turn, in the same order as during training
        splitted_text = mergePair(splitted_text, pair)  # объединяем найденную пару во всём тексте / merge the given pair throughout the text

    return splitted_text

def decode(splitted_text: list[list[str]]) -> str:
    # декодирует список токенов обратно в текст
    # decodes the list of tokens back into text
    text = []  # сюда складываем восстановленные слова / will hold the reconstructed words
    for word in splitted_text:  # проходим по каждому закодированному слову / iterate over each encoded word
        new_word = "".join(word)  # склеиваем токены слова обратно в строку / join the word's tokens back into a string
        text.append(new_word)  # добавляем восстановленное слово / add the reconstructed word

    return " ".join(text)  # соединяем слова пробелами обратно в текст / join words with spaces back into text


print(train_bpe(file, 1000))
print(time.perf_counter() - start) # 2.4312329160020454