import numpy as np

tokens = 5 # создаем кол-во токенов в "предложении"
d_model = 8 # размерность эмбеддинга

# создаем матрицу эмбеддингов
embeddings = np.random.standard_normal((tokens, d_model))

# создаем позиции для токенов и пар
pe = np.arange(tokens)
pair_indx = np.arange(d_model // 2)

# создаем массив частот
frequencies = 1 / 10_000 ** ((2 * pair_indx) / d_model)

# создаем angle of rotation
# то есть у нас есть частоты 1, 0.1, 0.01, 0.001
# есть позиции токенов: 0, 1, 2, 3, 4
# вычисляем угол поворота φ для каждой пары координат
# каждого токена: φ = позиция_токена × частота_пары
rotation_angle = pe.reshape(-1, 1) * frequencies.reshape(1, -1) # 5, 4
print(rotation_angle, rotation_angle.shape)
print("-" * 50)

# вычисляем матрицы cos, sin
cos_matrix = np.cos(rotation_angle)
sin_matrix = np.sin(rotation_angle)


# разбиваем эмбеддинг на две матрицы
A = embeddings[:, ::2]
B = embeddings[:, 1::2]
print(A.shape)

# вращение пар
A_rot = A * cos_matrix - B * sin_matrix
B_rot = A * sin_matrix + B * cos_matrix
print(A_rot.shape)

# собираем эмбеддинг
new_embedding = np.empty((tokens, d_model))
new_embedding[:, ::2] = A_rot
new_embedding[:, 1::2] = B_rot
print(new_embedding.shape, embeddings.shape)



