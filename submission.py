import numpy as np
from scipy.spatial.distance import cdist
import heapq
# Task 1
def pq(data, P, init_centroids, max_iter):
    code_books = np.zeros(init_centroids.shape)#.astype('float32')
    codes = np.zeros([data.shape[0],P]).astype('uint8')
    # partition into P blocks
    data = np.split(data, P, axis=1)
    for i in range(P):
        # K-medians clustering
        init_centroidsNew = init_centroids[i].copy()
        for m in range(max_iter):
            # calculate L1 distance
            distance = cdist(data[i],init_centroids[i],metric='cityblock')
            # assign points to clusters
            cluster = np.argmin(distance,axis=1)
            # update centroids using median
            for c in range(init_centroids[i].shape[0]):
                tmp = data[i][np.argwhere(cluster==c)]
                if len(tmp):
                    init_centroidsNew[c] = np.median(tmp,axis=0)
            # early break
            if (init_centroids[i] == init_centroidsNew).all():
                break
            init_centroids[i] = init_centroidsNew.copy()
        # another half k-medians to obtain codes
        distance = cdist(data[i],init_centroidsNew,metric='cityblock')
        codes[:,i] = np.argmin(distance,axis=1)
        code_books[i] = init_centroidsNew
    code_books,codes = code_books.astype('float32'),codes.astype('uint8')
    return code_books,codes

#Task 2
def query(queries, codebooks, codes, T):
    P,K,_ = codebooks.shape
    Q,M = queries.shape
    if codes.shape[0] <= T:
        return [set(range(codes.shape[0])) for i in range(Q)]
    dtable = np.empty((Q,P,K))
    #partition query into p blocks
    query = np.array(np.split(queries, P, axis=1))
    for i in range(P) :
        dtable[:,i,:] = cdist(query[i],codebooks[i],metric='cityblock')
    dsort = np.argsort(dtable,axis=2)
    codesu = np.unique(codes,axis=0)#.tolist()
    dictionary = {}
    for i in range(len(codesu)):
        dictionary[tuple(codesu[i])] = set(np.where((codes==codesu[i]).all(1))[0])
    merge = [set() for i in range(Q)]
    cols = range(P)
    for q in range(Q) :
        stackp = [0 for i in cols]
        stackn = dsort[q][cols,stackp]
        # que = queue.PriorityQueue()
        h = []
        stackdis = np.sum(dtable[q][cols,stackn])
        t = (stackdis,0,stackn,stackp)
        k = 1
        # que.put(t)
        heapq.heappush(h,t)
        hashtable = {}
        hashtable[tuple(stackp)] = None
        while len(merge[q])<T :
            # least = que.get()
            least = heapq.heappop(h)
            if tuple(least[2]) in dictionary:
                merge[q] |= dictionary[tuple(least[2])]
                if len(merge[q])>=T:
                    break
            for i in cols :
                child = least[3].copy()
                child[i] += 1
                if child[i]<K and tuple(child) not in hashtable:
                    hashtable[tuple(child)] = None
                    stackn = dsort[q][cols,child]
                    stackdis = np.sum(dtable[q][cols,stackn])
                    t = (stackdis,k,stackn,child)
                    k += 1
                    # que.put(t)
                    heapq.heappush(h,t)
    return merge

def query2(queries, codebooks, codes, T):
    dtable = np.empty((queries.shape[0],codebooks.shape[0], codebooks.shape[1]))
    #partition query into p blocks
    query = np.array(np.split(queries, codebooks.shape[0], axis=1))
    merge = []
    for i in range(query.shape[0]):
        #L1 distance estimation
        dtable[:,i,:] = cdist(query[i],codebooks[i],metric='cityblock')
    for i in range(query.shape[1]) :
        merge.append(my_merge(dtable[i], codes,T))
    return merge

def my_merge(dtable, codes,T):
    if codes.shape[0]<=T:
        return set(range(codes.shape[0]))
    #compute inverted multi-index
    distance = np.sum(dtable[range(dtable.shape[0]),codes],axis=1)
    #assgin candidates into sets
    loc = set()
    while len(loc)<T:
        minloc = np.argmin(distance)
        loc |= set(np.where((codes==codes[minloc]).all(1))[0])
        distance[minloc]=np.inf
    return loc