import submission
import pickle
import time
import numpy as np
# How to run your implementation for Part 1
with open('./toy_example/Data_File', 'rb') as f:
    Data_File = pickle.load(f, encoding = 'bytes')#shape: (768, 128)
with open('./toy_example/Centroids_File', 'rb') as f:
    Centroids_File = pickle.load(f, encoding = 'bytes')#shape: (2, 256, 64)
start = time.time()
codebooks, codes = submission.pq(Data_File, P=2, init_centroids=Centroids_File, max_iter = 20)
end = time.time()
time_cost_1 = end - start
print(time_cost_1)
# print(Data_File.shape,type(Data_File),"\n",Data_File)
# print(Centroids_File.shape,type(Centroids_File),"\n",Centroids_File)
# print(codebooks.shape,type(codebooks),"\n",codebooks)
# print(codes.shape,type(codes),"\n",codes)
# print(codebooks, codes)

# How to run your implementation for Part 2
with open('./toy_example/Query_File', 'rb') as f:
    queries = pickle.load(f, encoding = 'bytes')
# queries = np.concatenate((queries,np.ones((1,128))*2,np.ones((1,128))*30),axis=0)
# queries = np.concatenate((np.ones((1,64))*6,np.ones((1,64))*11),axis=1)
# print(queries)
start = time.time()
candidates = submission.query(queries,codebooks,codes,T=10)
end = time.time()
time_cost_2 = end - start
print(time_cost_2)
# testarray *= 2
# output for part 2.
print(candidates)
