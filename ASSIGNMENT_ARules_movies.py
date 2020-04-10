# ASSIGNMENT ASSOCIATION_RULES my_movies dataset

# importing important libraries
import pandas as pd
import numpy as np
pd.set_option('display.max_columns',500)
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

movies = pd.read_csv("E:/Association_Rules/my_movies.csv")
movies = movies.iloc[:,5:]
movies

#Quick look at the top items
movies.sum().to_frame('Frequency').sort_values('Frequency',ascending=False).plot(kind='bar',figsize=(12,8),title='Frequent Items')

# getting frequent items
frequent_items = apriori(movies,  min_support=0.2,use_colnames=True)
frequent_items.shape # 13
frequent_items.sort_values('support',ascending=False) 

# Getting rules from the frequent itemsets
rules=association_rules(frequent_items, metric='lift',min_threshold=1)
rules.shape # we have obtained 16 rules 
rules.head()

# removing redundant rules
def sort_rules(i):
    return sorted(list(i))
rules_movies = rules.antecedents.apply(sort_rules) + rules.consequents.apply(sort_rules)
rules_movies = rules_movies.apply(sorted)   
rules_sets = list(rules_movies)
unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]
index_rules = []
for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))

# getting rules without any redundancy 
rules_no_redundancy  = rules.iloc[index_rules,:]
rules.shape # we have 16 rules including duplicate rules
rules_no_redundancy.shape # now we have 6 rules after removing duplicate rules

# Sorting them with respect to lift 
rules_no_redundancy = rules_no_redundancy.sort_values('lift',ascending=False)

#### By changing the metric to confidence
rules1 = association_rules(frequent_items,metric='confidence',min_threshold=0.8)
# by changing metric to 0.8 confidence we get 8 rules and if remove duplicate rules
# we get 5 rules

#############################
%matplotlib qt
def draw_graph(rules_no_redundancy, rules_to_show):
  import networkx as nx  
  G1 = nx.DiGraph()
   
  color_map=[]
  N = 50
  colors = np.random.rand(N)    
  strs=['R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11']   
   
   
  for i in range (rules_to_show):      
    G1.add_nodes_from(["R"+str(i)])
    
     
    for a in rules_no_redundancy.iloc[i]['antecedents']:
                
        G1.add_nodes_from([a])
        
        G1.add_edge(a, "R"+str(i), color=colors[i] , weight = 2)
       
    for c in rules_no_redundancy.iloc[i]['consequents']:
             
            G1.add_nodes_from([c])
            
            G1.add_edge("R"+str(i), c, color=colors[i],  weight=2)
 
  for node in G1:
       found_a_string = False
       for item in strs: 
           if node==item:
                found_a_string = True
       if found_a_string:
            color_map.append('yellow')
       else:
            color_map.append('green')       
 
 
   
  edges = G1.edges()
  colors = [G1[u][v]['color'] for u,v in edges]
  weights = [G1[u][v]['weight'] for u,v in edges]
 
  pos = nx.spring_layout(G1, k=16, scale=1)
  nx.draw(G1, pos, edges=edges, node_color = color_map, edge_color=colors, width=weights, font_size=16, with_labels=False)            
   
  for p in pos:  # raise text positions
           pos[p][1] += 0.07
  nx.draw_networkx_labels(G1, pos)
  plt.show()
 
     
draw_graph (rules_no_redundancy, 1) 
draw_graph(rules_no_redundancy,6)
rules_no_redundancy

draw_graph(rules_no_redundancy,2)
'''
yellow circle gives the rule name.. R0 is the first rule
If arrow is away from the green circles , then that is antecedent, here LOTR1
if arrow is towards the green circle, then that is consequent, here LOTR2
'''

####### RECOMMENDATIONS
'''
1. The first rule is LOTR1 ==> LOTR2
    if anyone has watched either of the movie (not both), then we can 
    recommend them to watch the other sequel
2. Those who have watched either Green-Mile or Sixth-Sense, we can recommend
    them to watch the other movie
3. If watched one or two of the moives Sixth-Sense, Gladiator and Patriot,
    we can recommend them to watch the other movies
'''





