# ===== RU =====
import numpy as np

# после токенизации мы получаем предложение
sentence = ["I", "like", "cats"]

# Теперь предложение превращается просто в числа: Я - I, like - 2, cats - 3
sentence_num = [1, 2, 3]

# 1, 2 или 3 это просто индекс, то есть просто id, но это надо превратить в инф-ию
# например, embedding_dim = 4, тогда
embedding_matrix = [
    ("I", 1, [0.01, 0.53, -0.12, 0.41]),
    ("like", 2, [0.84, -0.35, 0.17, -0.90]),
    ("cats", 3, [0.29, -0.55, 0.66, -0.18])
]

# например, есть id = 2, тогда модель получает
embedding = embedding_matrix[1][2] # [0.84, -0.35, 0.17, -0.90]

# эти числа в начале обучения случайны, модель сама их обучает
# вначале делают случайную инициализацию (Xavier, Kaiming..., нормальное распределение)

# на практике embedding_matrix это матрица размера (vocab_size, embedding_dim)

# embedding dimension мы вправре выбирать сами, будь 4, будь 4096, но разница в том
# сколько смысла мы можем заложить в эмбеддинг, т.е чем больше embedding_dim то больше смысла в векторе
# но его дольше обучать, а чем меньше embedding_dim, то меньше смысла, легче и быстрее обучать


# ===== EN =====
import numpy as np

# after tokenization we get a sentence
sentence = ["I", "like", "cats"]

# Now the sentence is turned into just numbers: I - 1, like - 2, cats - 3
sentence_num = [1, 2, 3]

# 1, 2, or 3 is just an index, i.e. just an id, but it needs to be turned into information
# for example, embedding_dim = 4, then
embedding_matrix = [
    ("I", 1, np.array([0.01, 0.53, -0.12, 0.41])),
    ("like", 2, [0.84, -0.35, 0.17, -0.90]),
    ("cats", 3, [0.29, -0.55, 0.66, -0.18])
]

# for example, if id = 2, then the model gets
embedding = embedding_matrix[1][2] # [0.84, -0.35, 0.17, -0.90]

# these numbers are random at the start of training, the model learns them itself
# at the start they do random initialization (Xavier, Kaiming..., normal distribution)

# in practice embedding_matrix is a matrix of size (vocab_size, embedding_dim)

# we're free to choose the embedding dimension ourselves, be it 4, be it 4096, but the difference is in
# how much meaning we can pack into the embedding, i.e. the bigger embedding_dim, the more meaning in the vector
# but it takes longer to train, and the smaller embedding_dim, the less meaning, but easier and faster to train