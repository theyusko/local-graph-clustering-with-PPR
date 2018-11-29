'''
Test code for algorithm of Cai et. al
'''
import math
import operator
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
from Code.pr_Modified_networkx_lib import pagerank_scipy as pagerankModified
from networkx import pagerank_numpy as pagerank_numpy
import Code.utilities as my_utilities

# Reads data to G graph. Change gml to read other things
G = nx.read_gml("../Data/footballTSEweb/footballTSEinput.gml")

vect = [str(x) for x in range(50)]

# Generating a copy of G graph. May be redundant.
G2 = G
pos = nx.spring_layout(G2) # Layout

## Calls page rank for graph without personalization
pr = pagerankModified(G2)
##
threshold = 0.6 # Threshold for graph coloring
prList = list(pr.values()) # This stores dict values as a list (Hope so)
print(prList)
# Colormap is a list of RGB A values. Will be used to color the graph depending on the PR value.
# A (alpha does not work ??)
colormap = [(0,1 - (prList[x] * 100)**3,1, (prList[x] * 100)) for x in range(len(pr))]


# Following draws graph nodes, labels and edges separately
nx.draw_networkx_nodes(G2, pos=pos, node_size=150, node_color=colormap)
# Pos_higher and related calculations are used to draw PR values above nodes.
pos_higher = {}
y_off = 0.02  # offset on the y axis

for k, v in pos.items():
    pos_higher[k] = (v[0], v[1]+y_off)
nx.draw_networkx_labels(G2, pos=pos_higher, font_size=8, font_color='b')
nx.draw_networkx_edges(G2, pos=pos, style='dotted', alpha=0.5)
pos_higher2 = {}

for k, v in pos.items():
    pos_higher2[k] = (v[0], v[1]+y_off*2)
nx.draw_networkx_labels(G2, pos=pos_higher2, alpha=0.7, font_color='r', labels=pr, font_size=8)
plt.show() # display subgraph


## Test code to run pagerank on each node with personalization
prS = []
counter = 0
for i in G2.nodes():
    prS.append(pagerankModified(G2, personalization={i:1}))
    # Comment in to see the max for each personalization
    #print(i + ":\t\t"+ max(prS[counter].items(), key=operator.itemgetter(1))[0])
    counter += 1

## Get medians
# Using Sorted representation of the pr dict to get items around median according ot PR values
# I don't know if this is a good choice
sorted_pr = sorted(pr.items(), key=operator.itemgetter(1))
tmp_mid =  int(math.floor(len(sorted_pr)/2))
five_nodes = [sorted_pr[tmp_mid], sorted_pr[tmp_mid + 1], sorted_pr[tmp_mid + 2], sorted_pr[tmp_mid-1], sorted_pr[tmp_mid - 2]]
print(five_nodes)

# Following should run PR personalized at each of the chosen nodes
prS = [] # Clean previous list
counter = 0
for i in five_nodes:
    prS.append(pagerank_numpy(G2, personalization={i[0]:1}))
    print(i[0] + ":\t\t" + max(prS[counter].items(), key=operator.itemgetter(1))[0])
    counter += 1

#TODO: Check this; https://www.learndatasci.com/tutorials/k-means-clustering-algorithms-python-intro/
# It shows a tutorial on karate club dataset, which I think was also in the paper
# It explains clustering part.
edge_matrix = my_utilities.graph_to_edge_matrix(G2) # TODO: IndexError. Need to fix the function.
from sklearn import cluster
k_clusters = 2
k_means_result = cluster.KMeans(n_clusters=k_clusters, n_init=200).fit(edge_matrix)

print(k_means_result)