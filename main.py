import nltk
import torch
import time
import data
import numpy as np

#fpath = "/home/kawata/nlp-research/corpus/the-westbury-lab-wikipedia-corpus/mini-west.txt"
fpath="/home/kawata/ml-research/pytorch-examples/word_language_model/data/penn/train.txt"
#fpath="/home/kawata/nlp-research/GloVe/text8"
corpus = data.Corpus(fpath)

# これぐらいなら普通にGPUで分解できる
#M = torch.randn(10000,10000)
n = corpus.num_vocab
k = 100 # trained embedding dimension
a = 0.5
use_cuda=True

estimated_sigma = np.linalg.norm(corpus.M1-corpus.M2)/n/2
print("sigma = {}".format(estimated_sigma))

M = corpus.M1+corpus.M2
d = np.linalg.matrix_rank(M) # oracle embedding dimension = rank of matrix M
print("rank d = {}".format(d))
start=time.time()
M = torch.from_numpy(M).cuda()
print("Cuda : {}[s]".format(time.time()-start))
start=time.time()
if use_cuda:
    U,S,V = M.cuda().svd()
    S = S.cpu().numpy()
else:
    U,S,V = M.svd()
    S = S.numpy()
print("SVD : {}[s]".format(time.time()-start))

for i in range(n):
    S[i] = max(0.0,S[i]-2*estimated_sigma*np.sqrt(n))
    
#k_list = [50,100,200,300,400,500,600,700,800,900,1000,2000,3000,4000]
k_list = [(i+1)*5 for i in range(100)]
for k in k_list:
    # term1
    t1 = 0.0
    for i in range(k,d):
        t1 += S[i]**(4.*a)
    t1 = np.sqrt(t1)

    # term2
    t2 = 0.0
    for i in range(k):
        t2 += S[i]**(4.*a-2.)
    t2 = np.sqrt(t2)*2.*np.sqrt(2.*n)*a*estimated_sigma
    
    # term3
    t3 = 0.0
    for i in range(k):
        t = 0.0
        if S[i] <= 0.0 and S[i+1] <= 0.0:
            continue
        for r in range(k):
            for s in range(k):
                if r<=i and i<s and (S[r]>0.0 or S[s]>0.0):
                    t += (S[r]-S[s])**(-2)
            t3 += t*estimated_sigma*(S[i]**(2.*a)-S[i+1]**(2.*a))
    t3 = np.sqrt(2) * t3
    print("k={} | {} = {} + {} + {}".format(k,t1+t2+t3,t1,t2,t3))
