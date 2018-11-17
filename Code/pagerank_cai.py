'''
Test code for algorithm of Cai et. al
'''
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
from Code.pr_Modified_networkx_lib import pagerank_scipy as pagerankModified
G = nx.read_gml("../Data/footballTSEweb/footballTSEinput.gml")

#pr = nx.pagerank(G)
#vect = [G.nodes(x) for x in range(100)]

vect = [str(x) for x in range(50)]

G2 = G
pos = nx.spring_layout(G2)


##
pr = pagerankModified(G2)
##
threshold = 0.6
prList = list(pr.values())
print(prList)
colormap = [(0,1 - (prList[x] * 100)**3,1, (prList[x] * 100)) for x in range(len(pr))]


#nx.draw_networkx_edges(G2, pos)
nx.draw_networkx_nodes(G2, pos=pos, node_size=150, node_color=colormap)
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

