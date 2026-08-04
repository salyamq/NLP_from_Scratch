import numpy as np

# задаем гиперпараметры
# setting the hyperparameters
vocab_size = 1000
embedding_dim = 10
max_seq_len = 128

# генерируем матрицы со средним 0 и отклонением 1
# generating matrixes using mean = 0, std = 1
token_embedding_matrix = np.random.randn(vocab_size, embedding_dim)
positional_embedding_matrix = np.random.randn(max_seq_len, embedding_dim)

# id слов / words id
sentence_ids = [5, 2, 9, 1]

# достаем векторы только для слов из нашего предложения
# extract vectors only for the words in our sentence
token_embeddings = token_embedding_matrix[sentence_ids]
print(token_embeddings.shape)

positions = np.arange(len(sentence_ids))

# достаем позиционные векторы для позиций
# retrieve the position vectors for the positions
positional_embeddings = positional_embedding_matrix[positions]
print(positional_embeddings.shape)

# складываем векторы слов и позиций вместе
# combine the vectors of words and positions
final_embeddings = token_embeddings + positional_embeddings
print(final_embeddings.shape)
