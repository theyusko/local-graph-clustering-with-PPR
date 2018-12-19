'''
Test code for algorithm of Cai et. al
'''
import math
import operator
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
from Code.pr_Modified_networkx_lib import pagerank_scipy as pagerankModified
from networkx import pagerank_scipy as pagerank_scipy
import Code.utilities as my_utilities
import time


# Reads data to G graph. Change gml to read other things
G = nx.read_gml("../Data/footballTSEweb/footballTSEinput.gml")
# Football teams are partitioned into 12 groups (Cai)
#G = nx.karate_club_graph()
# Karate club is partitioned into 2 groups ultimately
vect = [str(x) for x in range(50)]

# Generating a copy of G graph. May be redundant.
G2 = G
pos = nx.spring_layout(G2) # Layout

## Calls page rank for graph without personalization
pr = pagerankModified(G2)
##
threshold = 0.6 # Threshold for graph coloring
prList = list(pr.values()) # This stores dict values as a list (Hope so)
#print(prList)
# Colormap is a list of RGB A values. Will be used to color the graph depending on the PR value.
# A (alpha does not work ??)
colormap = [(0,1 - (prList[x] * 100)**3,1, (prList[x] * 100)) for x in range(len(pr))]



# Following draws graph nodes, labels and edges separately
nx.draw_networkx_nodes(G2, pos=pos, node_size=150, node_color='b')
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
## Dummy
prS = []
counter = 0
for i in G2.nodes():
    prS.append(pagerank_scipy(G2, personalization={i:1}))
    # Comment in to see the max for each personalization
    #print(i + ":\t\t"+ max(prS[counter].items(), key=operator.itemgetter(1))[0])
    counter += 1
## Dummy

prS = []
counter = 0

start_time_of_standard_PR = time.clock()
for i in G2.nodes():
    prS.append(pagerank_scipy(G2, personalization={i:1}))
    # Comment in to see the max for each personalization
    #print(i + ":\t\t"+ max(prS[counter].items(), key=operator.itemgetter(1))[0])
    counter += 1
end_time_of_standard_PR = time.clock()
print("Std elapsed time = " + str(end_time_of_standard_PR - start_time_of_standard_PR))


prS_modified = []
counter = 0

start_time_of_modified_PR = time.clock()
for i in G2.nodes():
    prS_modified.append(pagerankModified(G2, personalization={i:1}))
    # Comment in to see the max for each personalization
    #print(i + ":\t\t"+ max(prS_modified[counter].items(), key=operator.itemgetter(1))[0])
    counter += 1
end_time_of_modified_PR = time.clock()


print("Modified elapsed time = " + str(end_time_of_modified_PR - start_time_of_modified_PR))
## Get medians
# Using Sorted representation of the pr dict to get items around median according ot PR values
# I don't know if this is a good choice
sorted_pr = sorted(pr.items(), key=operator.itemgetter(1))
tmp_mid =  int(math.floor(len(sorted_pr)/2))
five_nodes = [sorted_pr[tmp_mid], sorted_pr[tmp_mid + 1], sorted_pr[tmp_mid + 2], sorted_pr[tmp_mid-1], sorted_pr[tmp_mid - 2]]
print(five_nodes)



# Following should run PR personalized at each of the chosen nodes
prS5 = [] # Clean previous list
counter = 0
for i in five_nodes:
    prS5.append(pagerankModified(G2, personalization={i[0]:1}))
    print(str(i[0]) + ":\t\t" + str(max(prS5[counter].items(), key=operator.itemgetter(1))[0]))
    counter += 1




#print(prS_modified[0])
sim_threshold = 0.01
for i in prS_modified:
    for p in list(i.keys()):
        if i[p] < sim_threshold:
            del i[p]
#print(prS_modified[0])
for i in prS:
    for p in list(i.keys()):
        if i[p] < sim_threshold:
            del i[p]


cluster_merge_threshold = 0.4

tmp = len(prS_modified)
i = 0
while i < tmp:
    p = i
    while p < tmp:
        if (len(prS_modified[i].keys() & prS_modified[p].keys())
                / min(len(prS_modified[i]), len(prS_modified[p])) > cluster_merge_threshold):
            prS_modified[i].update(prS_modified[p])
            del prS_modified[p]
            tmp = tmp - 1
        p = p + 1
    i = i + 1

print(prS_modified[0])
print(len(prS_modified))

tmp = len(prS)
i = 0
while i < tmp:
    p = i
    while p < tmp:
        if (len(prS[i].keys() & prS[p].keys())
                / min(len(prS[i]), len(prS[p])) > cluster_merge_threshold):
            prS[i].update(prS[p])
            del prS[p]
            tmp = tmp - 1
        p = p + 1
    i = i + 1


import matplotlib.pyplot as plt
from matplotlib import cm as cm
#print(cm.cmap_d.keys())

#colormap = cm.get_cmap('Blues')
#colormap = [cm.jet(x) for x in range(len(prS_modified))]
#colormap = ['r', 'g', 'b', 'r', 'b', 'g']
#colormap = [cm.jet(i) for i in range(len(prS_modified))]
colormap = plt.cm.get_cmap('RdYlBu')
norm = plt.Normalize(vmin=0, vmax=max(len(prS_modified), len(prS)))
print(colormap)


print(len(prS_modified))
for i in range(len(prS_modified)):
    #colormap[i] = [colormap[i] for x in range(len(prS_modified))]
    clusteri = nx.Graph(G2.subgraph(prS_modified[i]))
    nx.draw_networkx_nodes(clusteri, pos=pos,
                           cmap=colormap(norm(i)),
                           node_color=colormap(norm(i)), alpha=0.8)
    nx.draw_networkx_edges(clusteri, pos=pos)
plt.show()

print(len(prS))
for i in range(len(prS)):
    clusteri = nx.Graph(G2.subgraph(prS[i]))
    nx.draw_networkx_nodes(clusteri, pos=pos,
                           cmap=colormap(norm(i)),
                           node_color=colormap(norm(i)), alpha=0.8)
    nx.draw_networkx_edges(clusteri, pos=pos)
plt.show()