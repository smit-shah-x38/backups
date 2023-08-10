import torch
from sentence_transformers import SentenceTransformer, util
embedder = SentenceTransformer('xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
corpus = ['boy', 'girl', 'india', 'fruit']
# corpus1=['प्रररसिद्ध','शुरू','पास','नदी','मध्य','मूल','चार','देशों','रंग']
# queries1=['पहली']
corpus_embed = embedder.encode(corpus, convert_to_tensor=True)
queries = ['orange']
top_k = 4
for query in queries:
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embed)[0]
    cos_scores = cos_scores.cpu()
    top_results = torch.topk(cos_scores, k=top_k)
    print("Query:", query)
    print("\nTop most similar sentences in corpus:")

    for score, idx in zip(top_results[0], top_results[1]):
        print(corpus[idx], "(Score: %.4f)" % (score))
