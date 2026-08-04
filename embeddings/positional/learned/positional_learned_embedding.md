Для модели "Мужчина бежит за девушкой" и "Девушка бежит за мужчиной"
одно и то же, почему? Все потому что токен "мужчина" в обоих предложениях получит 
один и тот же вектор, "девушка" - тоже одинаковый, "бежит" - тоже одинаковый. 
Модель на уровне эмбеддингов получит просто набор векторов - как мешок, 
без понятия о том, в каком порядке они шли. В самом трансформере (attention) 
операция сама по себе перестановочно-инвариантна (permutation invariant), то есть 
все токены как бы "мешок со словами"

Самый просто подход решить эту проблему - просто завести 
еще одну эмбеддинг матрицу - positional embedding matrix размера
(max_seq_len, embedding_dim), max_seq_len это контекстное окно, т.е это
максимальное кол-во токенов сколько модель может удержать у себя в голове
То есть, теперь у нас есть:
    token_embedding[token_id] — "что это за слово"
    positional_embedding[position] — "где оно стоит"

Тогда, финальный вектор с ее позицией это:
    final_embedding = token_embedding[token_id] + positional_embedding[position]

-----------------
For the model, "A man is running after a girl" and "A girl is running after a man" 
are the same thing. Why? Because the token "man" in both sentences gets 
the exact same vector, "girl" is also the same, and "running" is the same too. 
At the embedding level, the model simply receives a set of vectors — like 
a bag — with no idea of the order in which they came. In the transformer itself, 
the attention operation is permutation-invariant, 
meaning all tokens act as a "bag of words."

The simplest approach to solve this problem is to just introduce another 
embedding matrix — a positional embedding matrix of size (max_seq_len, embedding_dim). 
Here, max_seq_len is the context window, meaning the 
maximum number of tokens the model can hold in its head at once.

That is, we now have:
    token_embedding[token_id] — "what word it is"
    positional_embedding[position] — "where it is located"

Then, the final vector with its position is:
    final_embedding = token_embedding[token_id] + positional_embedding[position]