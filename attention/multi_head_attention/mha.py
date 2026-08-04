import numpy as np

tokens = 4 # создаем кол-во токенов в "предложении"
d_model = 12 # размерность эмбеддинга
heads = 4 # кол-во голов

embeddings = np.random.standard_normal((tokens, d_model))
d_k = d_model // heads # размерность для 1 головы.

class AttentionClass:
    def __init__(self, embeddings, d_k, d_model):
        self.embeddings = embeddings
        self.d_k = d_k

        self.W_Q = np.random.standard_normal((d_model, d_k))
        self.W_K = np.random.standard_normal((d_model, d_k))
        self.W_V = np.random.standard_normal((d_model, d_k))

        self.Q = self.embeddings @ self.W_Q
        self.K = self.embeddings @ self.W_K
        self.V = self.embeddings @ self.W_V  # 4, 8


    def calculate_attention(self):
        attention_scores = self.Q @ self.K.T
        scaled_attention_score = attention_scores / np.sqrt(self.d_k)

        scaled_attention_score_softmax = (np.exp(scaled_attention_score) /
                                          np.sum(np.exp(scaled_attention_score),
                                                 axis=1,
                                                 keepdims=True))

        return scaled_attention_score_softmax @ self.V # 4, 4 x 4, 8 = 4, 8


heads_list = []
for i in range(heads):
    head = AttentionClass(embeddings, d_k, d_model)
    heads_list.append(head)

outs_list = []
for head in heads_list:
    out = head.calculate_attention()
    outs_list.append(out)

# конкатинируем
multi_head_output = np.concatenate(outs_list, axis=-1)
print(multi_head_output.shape) # 4, 12

# наша обучаемая матрица
W_O = np.random.standard_normal((d_model, d_model))
print(W_O.shape) # 12, 12

# наш финальный attention
final_attention = multi_head_output @ W_O
# как мы видим, мы вернулись к размерности 4, 12
print(f"attention: {final_attention.shape}, embed_matrix: {embeddings.shape}")

