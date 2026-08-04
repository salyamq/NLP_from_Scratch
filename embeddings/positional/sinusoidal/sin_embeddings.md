Вместо того чтобы заводить обучаемую матрицу (max_seq_len, embedding_dim) 
со случайной инициализацией, которую учит градиентный спуск - берут фиксированную, 
заранее посчитанную по формуле матрицу (через sin и cos разных частот). 
Она вообще не является параметром модели, не обучается, просто константа, 
которая складывается с token embedding. (это из статьи attention is all you need)

То есть, мы генерируем positional_embeddings - математически. Формула такая:
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model)) # допустим d_model = 8 (это размерность эмбеддинга)
Для позиции pos (0, 1, 2, ...) и индекса размерности i (0, 1, ..., embedding_dim/2 - 1)

Здесь мы четные позиции внутри вектора заполняются синусом, а нечетные косинусом.
Делитель здесь, т.е 10000^(2i/d_model) это некий изменитель частоты, например для
позиции i = 0, 10000^0 = 1, значит sin(pos / 1) = sin(pos)
позиции i = 2, 10000^0.5 = 100, значит sin(pos / 100)

Если бы везде была одна частота - sin(pos),
то из-за периодичности синуса позиция pos = 0 и позиция pos = 2π ≈ 6.28 дали бы 
почти одинаковое значение (sin(0)=0, sin(6.28)≈0). 
Модель бы их путала - не могла отличить позицию 0 от позиции 6 или 7 в предложении.

--------------------
Instead of creating a trainable matrix of size `(max_seq_len, embedding_dim)` with random initialization optimized by gradient descent, a fixed matrix pre-calculated using a formula (via sine and cosine of different frequencies) is used instead. It is not a model parameter at all, does not train, and is simply a constant added to the token embedding (this is from the "Attention Is All You Need" paper).

That is, we generate positional_embeddings mathematically. The formula is:
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model)) # suppose d_model = 8 (which is the embedding dimension)
For position pos (0, 1, 2, ...) and dimension index i (0, 1, ..., embedding_dim/2 - 1)

Here, even positions inside the vector are filled with sine, and odd positions with cosine.
The denominator here, i.e., 10000^(2i/d_model), acts as a frequency modifier. For example, for:
position i = 0, 10000^0 = 1, meaning sin(pos / 1) = sin(pos)
position i = 2, 10000^0.5 = 100, meaning sin(pos / 100)

If there were a single frequency everywhere — sin(pos) — then due to the periodicity of the sine function, position pos = 0 and position pos = 2π ≈ 6.28 would yield almost identical values (sin(0)=0, sin(6.28)≈0).
The model would confuse them and fail to distinguish position 0 from position 6 or 7 in a sentence.



