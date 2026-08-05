import numpy as np

tokens = 5 # создаем кол-во токенов в "предложении"
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

print(embeddings, "\n", embeddings.shape)
print("x" * 100)

# создаем позиции для токенов и пар
pe = np.arange(tokens)
pair_indx = np.arange(d_model // 2)


def rope(matrix):
    frequencies = 1 / 10_000 ** ((2 * pair_indx) / d_model)
    rotation_angle = pe.reshape(-1, 1) * frequencies.reshape(1, -1)  # 5, 4

    cos_matrix = np.cos(rotation_angle)
    sin_matrix = np.sin(rotation_angle)

    matrix_a = matrix[:, ::2]
    matrix_b = matrix[:, 1::2]

    matrix_a_rot = matrix_a * cos_matrix - matrix_b * sin_matrix
    matrix_b_rot = matrix_a * sin_matrix + matrix_b * cos_matrix

    new_matrix = np.empty((tokens, d_model))
    new_matrix[:, ::2] = matrix_a_rot
    new_matrix[:, 1::2] = matrix_b_rot

    return new_matrix


attention_scores = rope(Q) @ rope(K).T
scaled_scores = attention_scores / np.sqrt(d_model)
scaled_attention_score_softmax = (np.exp(scaled_scores) /
                                  np.sum(np.exp(scaled_scores),
                                         axis = 1,
                                         keepdims=True))

attention = scaled_attention_score_softmax @ V
print(attention, "\n", attention.shape)


