## load head score file, llama-2-7b-80k for example
import json
import numpy as np
import math
import matplotlib.pyplot as plt
with open('./head_score/Llama-3.2-1B-Instruct_g=16_e=3.json') as file:
    head_list = json.loads(file.readline())
## use the average retrieval score and ranking
head_score_list = [([int(ll) for ll in l[0].split("-")],np.mean(l[1])) for l in head_list.items()]
head_score_list = sorted(head_score_list, key=lambda x: x[1], reverse=True) 
top_retrieval_heads = [[l[0],  round(np.mean(l[1]), 2)] for l in head_score_list]

# print(top_retrieval_heads)

counts = [0 for i in range(10)]
for head in top_retrieval_heads:
    rank = math.floor(head[1]*10)
    # print(head[1],rank)
    counts[rank] += 1

print(counts)
# Categories from 0 to 9
categories = list(range(0,100,10))
# categories = [10,20,30,40,50,60,70,80,90,100]

# Create the histogram
bars = plt.bar(categories, counts, color='blue', edgecolor='black', width=10)

# Add labels and title
plt.xlabel('Retrieval Score')
plt.ylabel('Count')
plt.title('Retrieval Score Distribution')
# Add count labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, str(height),
             ha='center', va='bottom', fontsize=10)
plt.savefig('results/plots/histogram_g=16_e=3.png', format='png', dpi=300)
