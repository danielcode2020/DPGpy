import numpy as np
import random as r
import networkx as nx

from pandas import *
from timeit import default_timer as timer
from datetime import timedelta
import matplotlib.pyplot as plt

INT_MAX = 300


def rand():
    return r.randint(1, 200)  # 200 maximal weigth for an edge


def rnd():
    return r.random()  # random double [0,1]


def coeff_submatrix(m, size):
    count = 0
    for i in range(size):
        for j in range(i + 1, size):
            if m[i][j] != INT_MAX:
                count += 1
    count = count / ((size * (size - 1)) / 2)
    return count


def create_matrix(size, case):
    m = [[INT_MAX for i in range(size)] for j in range(size)]
    edges = 0
    while edges <= size:
        for i in range(size):
            for j in range(i + 1, size):
                if i == j:
                    m[i][j] = 0
                else:
                    if rnd() < case:
                        m[i][j] = rand()
                        edges += 1
    return m

def minDistance(dist, queue):
    # Initialize min value and min_index as -1
    minimum = float("Inf")
    min_index = -1
    # from the dist array,pick one which
    # has min value and is till in queue
    for i in range(len(dist)):
        if dist[i] < minimum and i in queue:
            minimum = dist[i]
            min_index = i
    return min_index
def pathArray(parent, j, m):
    # Base Case : If j is source
    if parent[j] == -1:
        m.append(j)
        return
    pathArray(parent, parent[j], m)
    m.append(j)
    return m

def dijkstra(graph, src, dest,n):
    row = col = n

    # array pentru pastrarea distantei dintre src si i
    dist = [float("Inf")] * row

    # array pentru shortes path tree
    parent = [-1] * row

    # distanta dintre nodul sursa cu el insusi este mereu 0
    dist[src] = 0

    # Adaugam toate varfurile in queue
    queue = []

    for i in range(row):
        queue.append(i)

    while queue:
        # alegem virful cu distanta minima din multimea de varfuri adaugate deja in queue
        u = minDistance(dist, queue)
        # scoatem elementul minim
        queue.remove(u)
        # facem update la dist si indexul parintelui dintre varfurile adiacente ale varfului ales
        # consideram doar varfurile care sunt ramase in queue
        for i in range(col):
            if graph[u][i] and i in queue:
                if dist[u] + graph[u][i] < dist[i]:
                    dist[i] = dist[u] + graph[u][i]
                    parent[i] = u
    l = []
    pathArray(parent, dest, l)
    return l


G = nx.Graph()

n = 22
m = create_matrix(n, 0.33)
sursa = 0

destinatie = n-1
sp = dijkstra(m,sursa,destinatie,n)
print(sp)

# din matricea creata adaugam in G muchiile cu parametrii necesari
for i in range(n):
    for j in range(n):
        if m[i][j] != 0 and m[i][j] != INT_MAX and i in sp and j in sp and j!=sursa and i!=destinatie:
            G.add_edge(i, j, weight=m[i][j],width = 4)
        
        elif m[i][j]!=0 and m[i][j]!= INT_MAX:
            G.add_edge(i, j, weight=m[i][j],width = 1)
            
# spring layout - tipul de layout pentru graf
layout = nx.spring_layout(G)

# adaugam grosimea muchiilor
weights = nx.get_edge_attributes(G,'width').values()

nx.draw(G, layout,with_labels = True,  width =list(weights))

# adaugam greutatea muchiilor pentru label pe fiecare muchie
edge_labels = dict([((n1, n2), d['weight'])
                    for n1, n2, d in G.edges(data=True)])

# Ddeseneaza graful folosing pos si lista de labeluri pentru edge
nx.draw_networkx_edge_labels(G, pos=layout,edge_labels=edge_labels)

plt.show()
