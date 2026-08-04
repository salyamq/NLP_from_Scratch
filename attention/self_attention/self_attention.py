import numpy as np

tokens = 4 # создаем кол-во токенов в "предложении"
d_model = 8 # размерность эмбеддинга

# создаем матрицу эмбеддингов
embeddings = np.random.standard_normal((tokens, d_model))

# создаем обучаемые весовые матрицы для механизма внимания
W_Q = np.random.standard_normal((d_model, d_model))
W_K = np.random.standard_normal((d_model, d_model))
W_V = np.random.standard_normal((d_model, d_model))

#  вычисляем проекции через умножение эмбеддингов на весовые матрицы
Q = embeddings @ W_Q
K = embeddings @ W_K
V = embeddings @ W_V # 4, 8

print(embeddings.shape, W_Q.shape, Q.shape)

# вычисляем QK^T
attention_scores = Q @ K.T # 4, 8 x 8, 4 = 4, 4
print(attention_scores.shape) # (4, 4)
# масштабируем attention_scores и вычисляем вероятности
scaled_attention_score = attention_scores / np.sqrt(d_model)
scaled_attention_score_softmax = (np.exp(scaled_attention_score) /
                                  np.sum(np.exp(scaled_attention_score),
                                         axis = 1,
                                         keepdims=True))

attention = scaled_attention_score_softmax @ V # 4, 4 x 4, 8 = 4, 8
print(attention.shape)




