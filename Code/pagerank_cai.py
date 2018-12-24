'''
Test code for algorithm of Cai et. al
'''
import math
import operator
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
from Code.utilities import pagerank_scipy_modified as pagerankModified
from networkx import pagerank_scipy as pagerank_scipy
import Code.utilities as my_utilities
import time


# Reads data to G graph. Change gml to read other things
G = nx.read_gml("../Data/footballTSEweb/footballTSEinput.gml")
# Football teams are partitioned into 12 groups (Cai)
#G = nx.karate_club_graph()
# Karate club is partitioned into 2 groups ultimately
#G= nx.read_edgelist("../Data/Stanford_EU_email_core/email-Eu-core.txt",
#                    nodetype=int, create_using=nx.DiGraph())
#G= nx.read_edgelist("../Data/AutonomousSystemsGraphs/as20000102.txt",
#                    nodetype=int, create_using=nx.DiGraph())
#G = nx.read_gml("../Data/PolBooks/polbooks/polbooks.gml")
vect = [str(x) for x in range(50)]
print(len(G))
# Generating a copy of G graph. May be redundant.
G2 = G
pos = nx.spring_layout(G2) # Layout

## Calls page rank for graph without personalization
pr = pagerankModified(G2, max_iter=1000)
##
threshold = 0.6 # Threshold for graph coloring
prList = list(pr.values()) # This stores dict values as a list (Hope so)
print(prList)
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
    prS.append(pagerank_scipy(G2, personalization={i:1}, max_iter=1000))
    # Comment in to see the max for each personalization
    #print(i + ":\t\t"+ max(prS[counter].items(), key=operator.itemgetter(1))[0])
    counter += 1
## Dummy

prS = []
counter = 0

start_time_of_standard_PR = time.clock()
for i in G2.nodes():
    prS.append(pagerank_scipy(G2, personalization={i:1}, max_iter=1000))
    # Comment in to see the max for each personalization
    #print(i + ":\t\t"+ max(prS[counter].items(), key=operator.itemgetter(1))[0])
    counter += 1
end_time_of_standard_PR = time.clock()
print("Std elapsed time = " + str(end_time_of_standard_PR - start_time_of_standard_PR))


prS_modified = []
counter = 0

start_time_of_modified_PR = time.clock()
for i in G2.nodes():
    prS_modified.append(pagerankModified(G2, personalization={i:1}, max_iter=1000))
    # Comment in to see the max for each personalization
    #print(i + ":\t\t"+ max(prS_modified[counter].items(), key=operator.itemgetter(1))[0])
    counter += 1
end_time_of_modified_PR = time.clock()

time_of_PR_M = end_time_of_modified_PR - start_time_of_modified_PR
time_of_PR   = end_time_of_standard_PR - start_time_of_standard_PR
print("Modified elapsed time = " + str(end_time_of_modified_PR - start_time_of_modified_PR))
## Get medians
# Using Sorted representation of the pr dict to get items around median according ot PR values
# I don't know if this is a good choice
sorted_pr = sorted(pr.items(), key=operator.itemgetter(1))
tmp_mid =  int(math.floor(len(sorted_pr)/2))
five_nodes = [sorted_pr[tmp_mid], sorted_pr[tmp_mid + 1], sorted_pr[tmp_mid + 2], sorted_pr[tmp_mid-1], sorted_pr[tmp_mid - 2]]
print(five_nodes)



#print(prS_modified[0])
sim_threshold = 0.01
start_time = time.clock()
for i in prS_modified:
    for p in list(i.keys()):
        if i[p] < sim_threshold:
            del i[p]
end_time = time.clock()
time_of_PR_M += end_time - start_time

start_time = time.clock()
#print(prS_modified[0])
for i in prS:
    for p in list(i.keys()):
        if i[p] < sim_threshold:
            del i[p]
end_time = time.clock()
time_of_PR += end_time - start_time



cluster_merge_threshold = 0.4

start_time = time.clock()
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
end_time = time.clock()
time_of_PR_M += end_time - start_time

print(prS_modified[0])
print(len(prS_modified))

start_time = time.clock()
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
end_time = time.clock()
time_of_PR += end_time - start_time



import matplotlib.pyplot as plt
from matplotlib import cm as cm
#print(cm.cmap_d.keys())

colormap = plt.cm.get_cmap('RdYlBu')
norm = plt.Normalize(vmin=0, vmax=max(len(prS_modified), len(prS)))
print(colormap)


print(len(prS_modified))
for i in range(len(prS_modified)):
    #colormap[i] = [colormap[i] for x in range(len(prS_modified))]
    clusteri = nx.Graph(G2.subgraph(prS_modified[i]))
    nx.draw_networkx_nodes(clusteri, pos=pos,
                           cmap=colormap(norm(i)),
                           node_color=colormap(norm(i)), alpha=0.5)
    nx.draw_networkx_edges(clusteri, pos=pos)
plt.show()



print(len(prS))
for i in range(len(prS)):
    clusteri = nx.Graph(G2.subgraph(prS[i]))
    nx.draw_networkx_nodes(clusteri, pos=pos,
                           cmap=colormap(norm(i)),
                           node_color=colormap(norm(i)), alpha=0.5)
    nx.draw_networkx_edges(clusteri, pos=pos)
plt.show()




import networkx.algorithms.cuts as cuts
import networkx.algorithms.community as community

# Generally, the lower value of
# conductance the better quality the cluster is (Cai 2011).
conductance_list_M = []
part_set_M         = []
for i in range(len(prS_modified)):
    p = i
    while p < len(prS_modified):
        conductance_list_M.append(
            cuts.conductance(G2, prS_modified[i],
                             prS_modified[p]))
        p += 1
conductance_list = []

for i in range(len(prS)):
    p = i + 1
    while p < len(prS):
        conductance_list.append(
            cuts.conductance(G2, prS[i],
                             prS[p]))
        p += 1

print(sum(conductance_list_M))
print(sum(conductance_list))


def draw_community(g, position):
    start_time = time.clock()
    communities_generator = community.girvan_newman(g)
    end_time = time.clock()
    top_level_communities = next(communities_generator)
    next_level_communities = next(communities_generator)
    next_level_communities2 = next(communities_generator)
    next_level_communities3 = next(communities_generator)
    # position = nx.spring_layout(g)  # calculate position for each node
    # pos is needed because we are going to draw a few nodes at a time,
    # pos fixes their positions.
    nx.draw(g, position, edge_color='k', with_labels=True,
            font_weight='light', node_size=280, width=0.9)
    colors = ['r', 'g', 'b', 'c', 'm', 'y']

    # for c in top_level_communities:
    #   nx.draw_networkx_nodes(g, position, nodelist=list(c), node_color=colors[index])
    #  index += 1
    index = 0
    for c in next_level_communities3:
        nx.draw_networkx_nodes(g, position, nodelist=list(c), node_color=colors[index])
        index += 1
    plt.show()
    print(end_time - start_time)


draw_community(G2, pos)
print(time_of_PR_M)
print(time_of_PR)


