import numpy as np

# задаем гиперпараметры
# setting the hyperparameters
embedding_dim = 10  # d_model
max_seq_len = 128

# создаем пустую матрицу (max_seq_len, embedding_dim), чтобы заполнить ее
# create empty matrix (max_seq_len, embedding_dim), in order to fill this
pe = np.zeros((max_seq_len, embedding_dim))

# проходимся по каждой позиции токена (от 0 до 127)
# iterate through each token value (from 0 to 127)
for pos in range(max_seq_len):

    # проходимся по половине измерений эмбеддинга
    # один i заполняет сразу две координаты:
    # 2*i     -> sin, 2*i + 1 -> cos

    # iterate through half of the embedding dimensions
    # one i fill two coordinates
    # 2*i     -> sin, 2*i + 1 -> cos
    for i in range(embedding_dim // 2):
        # знаменатель из формулы:
        # the denominator from the formula:
        div_term = 10000 ** (2 * i / embedding_dim)

        # четные индексы эмбеддинга
        # even indexes of embedding
        pe[pos, 2 * i] = np.sin(pos / div_term)

        # нечетные, odd
        pe[pos, 2 * i + 1] = np.cos(pos / div_term)

print(pe[0])
print(pe.shape)

# ---------- vectorized

denominator = 10000 ** (2 * np.arange(embedding_dim // 2) / embedding_dim)
pe[:, 0::2] = np.sin(np.arange(max_seq_len).reshape(-1, 1)/ denominator)
pe[:, 1::2] = np.cos(np.arange(max_seq_len).reshape(-1, 1) / denominator)

print("-" * 20)
print(pe[0])
print(pe.shape)