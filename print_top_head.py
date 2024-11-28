## load head score file, llama-2-7b-80k for example
import json
import numpy as np
with open('./head_score/Llama-3.2-1B-Instruct.json') as file:
    head_list = json.loads(file.readline())
## use the average retrieval score and ranking
head_score_list = [([int(ll) for ll in l[0].split("-")],np.mean(l[1])) for l in head_list.items()]
head_score_list = sorted(head_score_list, key=lambda x: x[1], reverse=True) 
top_retrieval_heads = [[l[0],  round(np.mean(l[1]), 2)] for l in head_score_list][:10]
print(top_retrieval_heads)
'''
Head:[16, 19],   Retrieval Score: 0.94      Head:[11, 15],   Retrieval Score: 0.92      
Head:[8, 26],    Retrieval Score: 0.8       Head:[6, 9],     Retrieval Score: 0.62        
Head:[7, 12],    Retrieval Score: 0.61      Head:[17, 22],   Retrieval Score: 0.56
Head:[11, 2],    Retrieval Score: 0.46      Head:[6, 16],    Retrieval Score: 0.44
Head:[19, 15],   Retrieval Score: 0.42      Head:[21, 30],   Retrieval Score: 0.4
'''
