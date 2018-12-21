import nltk
import torch
import time
import data
import numpy as np

# 直接PIP損失を計算するプログラム

fpath="/home/kawata/ml-research/pytorch-examples/word_language_model/data/penn/train.txt"
corpus = data.Corpus(fpath)

n = corpus.num_vocab
k = 100 # trained embedding dimension
a = 0.5
use_cuda=True
sigma = 1.0
#estimated_sigma = np.linalg.norm(corpus.M1-corpus.M2)/n/2
print("sigma = {}".format(sigma))

M_oracle = corpus.M1+corpus.M2 # oracle matrix
M_noisy = np.zeros((n,n)) # 
#d = np.linalg.matrix_rank(M) # oracle embedding dimension = rank of matrix M
d = n
print("rank d = {}".format(d))

for i in range(n):
    for j in range(i,n):
        noise = np.random.normal(0,sigma)
        M_noisy[i][j] = M_oracle[i][j] + noise
        M_noisy[j][i] = M_oracle[j][i] + noise
        
# oracle embedding
start=time.time()
M_oracle = torch.from_numpy(M_oracle).cuda()
print("Cuda : {}[s]".format(time.time()-start))
start=time.time()
U,S,V = M_oracle.cuda().svd()
print("SVD : {}[s]".format(time.time()-start))
E_oracle = torch.mul(U[:,:d],torch.squeeze(S)[:d])
E_oracle = E_oracle.cpu().numpy()

# noisy embedding
start=time.time()
M_noisy = torch.from_numpy(M_noisy).cuda()
print("Cuda : {}[s]".format(time.time()-start))
start=time.time()
U,S,V = M_noisy.cuda().svd()
print("SVD : {}[s]".format(time.time()-start))
U = U.cpu().numpy()
S = S.cpu().numpy()
k_list = [(i+1)*5 for i in range(100)]
E2_oracle = np.matmul(E_oracle,E_oracle.T)
for k in k_list:
    E_noisy = np.dot(U[:,:k],S[:k])
    loss = np.linalg.norm(E2_oracle-np.matmul(E_noisy,E_noisy.T))
    print("k = {}| pip loss {}".format(k,loss))
