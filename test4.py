import submission
import pickle
import time
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
# How to run your implementation for Part 1
with open('./example/Data_File_2', 'rb') as f:
    Data_File = pickle.load(f, encoding = 'bytes')#shape: (768, 128)
with open('./example/Centroids_File_2', 'rb') as f:
    Centroids_File = pickle.load(f, encoding = 'bytes')#shape: (2, 256, 64)
start = time.time()
codebooks, codes = submission.pq(Data_File, P=4, init_centroids=Centroids_File, max_iter = 20)
end = time.time()
time_cost_1 = end - start
print("time 1",time_cost_1)

with open('./example/Codebooks_2', 'rb') as f:
    gdcodebooks = pickle.load(f, encoding = 'bytes')#shape: (768, 128)
with open('./example/Codes_2', 'rb') as f:
    gdcodes = pickle.load(f, encoding = 'bytes')#shape: (2, 256, 64)
diffcodebooks = gdcodebooks - codebooks
diffcodes = gdcodes - codes
for i in range(len(diffcodebooks)):
	if np.sum(diffcodebooks[i]):
		# print(i,codebooks[i],gdcodebooks[i])
		print(i)
for i in range(len(diffcodes)) :
	if np.sum(diffcodes[i]) :
		print(i,codes[i],gdcodes[i])

with open('./example/Query_File_2', 'rb') as f:
    queries = pickle.load(f, encoding = 'bytes')
# queries = queries[:1]
# print(queries.shape)
start = time.time()
candidates = submission.query(queries,codebooks,codes,T=100)
end = time.time()
time_cost_2 = end - start


with open('./example/Candidates_2', 'rb') as f:
    queries = pickle.load(f, encoding = 'bytes')
# print(candidates)
print("time 2",time_cost_2)
for i in range(len(candidates)):
	# print(len(queries[:][i]),len(candidates[i]))
	diff1 = queries[:][i] - candidates[i]
	diff2 = candidates[i] - queries[:][i]
	if diff1 :
		print(i)
		print("diff1",diff1)
		print("diff2",diff2)
		print()
