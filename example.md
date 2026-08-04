llm-from-scratch/
│
├── tokenizer/
│   ├── bpe.py                 
│   └── simple_tokenizer.py    
│
├── embeddings/
│   ├── token_embedding.py
│   └── positional/
│       ├── sinusoidal.py     
│       ├── learned.py
│       ├── rope.py            
│       ├── longRope.py
|       └── nope.py
│
├── attention/
│   ├── self_attention.py      
│   ├── causal_masking.py
│   ├── multi_head.py
│   ├── multi_query.py         
│   ├── grouped_query.py       
│   ├── flash_attention.py     
│   └── sliding_window.py      
│
├── normalization/
│   ├── layernorm.py
│   ├── rmsnorm.py              
│   └── pre_vs_post_norm.py     
│
├── feedforward/
│   ├── mlp_gelu.py              
│   └── glu_variants/
│       ├── swiglu.py            
│       └── geglu.py
│
├── block/
│   └── transformer_block.py     
│
├── model/
│   ├── gpt_style.py              
│   ├── weight_tying.py
│   └── moe/                      
│       ├── router.py
│       └── expert_layer.py
│
├── training/
│   ├── loss.py                   
│   ├── optimizer.py               
│   ├── lr_schedule.py             
│   └── grad_clipping.py
│
├── inference/
│   ├── kv_cache.py
│   ├── sampling.py                
│   └── beam_search.py
│
└── finetuning/                   
    ├── lora.py
    └── dpo.py