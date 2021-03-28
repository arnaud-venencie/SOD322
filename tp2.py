#%%
#############################
######### TD2 ###############
#############################

import time
from collections import Counter
from collections import defaultdict 
import os
dir = os.path.dirname(__file__)

###################################
### LOAD DATA ###########
###################################


def adjarray(filename) :
  
    #lecture du nombre de noeuds et d'arrêtes
    file=open(filename, 'r')
    lines =file.readlines()
    file.close()

    stanford = False
    # détection de graphes de type "stanford" 
    if lines[0][0] == '#' : 
       stanford = True
    
    graph_d = {}
    nodes = []
  
    for row in lines:
        if row[0] !='#':
            if stanford:
                key = int(row.split('\t')[0]) 
                val = int(row.split('\t')[1])
            
            else : #pour le graphe scholar
                key = int(row[:-1].split(' ')[0]) 
                val = int(row[:-1].split(' ')[1])

            try:
               graph_d[key].append(val)
            except:
               graph_d[key]=[val]
            try:
               graph_d[val].append(key)
            except:
               graph_d[val]=[key]
            
            nodes.append(key)
            nodes.append(val)
    nb_nodes= len(set(nodes))
    nb_edges = len(lines[4:])
              
    return(graph_d,nodes,nb_nodes,nb_edges) 


#############################
### Exercice 1 ###########
#############################


# IMPLEMENTATION DE LA MINHEAP ###########
# Elle contient tout le graphe sous la forme d'élements (degré, noeud)
# L'ordre sur les éléments est défini par le premier élément du 
# tuple seulement, soit par le degré. 

# On garde en mémoire l'adresse du noeud dans la minheap :
# indexes_heap[i] est l'index correspondant au noeud i dans la minheap

# On utilise l'implémentation traditionnelle d'une minheap
# de la bibliothèque heapq (https://xoutil.readthedocs.io/en/2.1.6/_modules/heapq.html) 
# que l'on modifie pour pouvoir garder trace des indexes des noeuds


def heapify(x):
    """Transform list into a heap, in-place, in O(len(x)) time."""
    n = len(x)
    # Transform bottom-up.  The largest index there's any point to looking at
    # is the largest with a child index in-range, so must have 2*i + 1 < n,
    # or i < (n-1)/2.  If n is even = 2*j, this is (2*j-1)/2 = j-1/2 so
    # j-1 is the largest, which is n//2 - 1.  If n is odd = 2*j+1, this is
    # (2*j+1-1)/2 = j so j-1 is the largest, and that's again n//2-1.
    for i in reversed(range(n//2)):
        _siftup(x, i)


def _siftdown(heap, startpos, pos):
    newitem = heap[pos]
    # Follow the path to the root, moving parents down until finding a place
    # newitem fits.
    while pos > startpos:
        parentpos = (pos - 1) >> 1
        parent = heap[parentpos]
        if newitem < parent:
            heap[pos] = parent
            ###### MODIFICATION :  écriture de l'adresse du noeud dans indexes_heap
            indexes_heap[parent[1]]=pos 
            ######
            pos = parentpos
            continue
        break
    heap[pos] = newitem
    ###### MODIFICATION :  écriture de l'adresse du noeud dans indexes_heap
    indexes_heap[newitem[1]]=pos
    ######


def _siftup(heap, pos):

    endpos = len(heap)
    startpos = pos
    newitem = heap[pos]
    # Bubble up the smaller child until hitting a leaf.
    childpos = 2*pos + 1    # leftmost child position
    while childpos < endpos:
        # Set childpos to index of smaller child.
        rightpos = childpos + 1
        if rightpos < endpos and not heap[childpos] < heap[rightpos]:
            childpos = rightpos
        # Move the smaller child up.
        heap[pos] = heap[childpos]

        ###### MODIFICATION :  écriture de l'adresse du noeud dans indexes_heap
        indexes_heap[heap[childpos][1]]=pos 
        #####

        pos = childpos
        childpos = 2*pos + 1
        ###########
    # The leaf at pos is empty now.  Put newitem there, and bubble it up
    # to its final resting place (by sifting its parents down).
    heap[pos] = newitem

    ###### MODIFICATION :  écriture de l'adresse du noeud dans indexes_heap
    indexes_heap[newitem[1]]=pos 
    #####

    _siftdown(heap, startpos, pos)

def heappop(heap):
    """Pop the smallest item off the heap, maintaining the heap invariant."""
    lastelt = heap.pop()    # raises appropriate IndexError if heap is empty
    if heap:
        returnitem = heap[0]
        heap[0] = lastelt
        ###### MODIFICATION :  écriture de l'adresse du noeud dans indexes_heap
        indexes_heap[heap[0][1]]=0 
        #####

        _siftup(heap, 0)
        return returnitem
    return lastelt


def decrease_degree(indexes_heap,h,value):
  """
    Diminue value[0] de 1 et réorganise la minheap en conséquence.
    Les modifications des indexes des noeuds sont sauvegardés dans indexes_heap
    
    attention : value est un couple (degré,noeud)
    et on décrémente seulement le premier élément
    
    --Input--
    indexes_heap (dictionnaire de taille n):  
    h (meanhip de taille n): 
    value (tuple (degré,noeud))   
  """
  
  #localisation de value dans la minheap
  index = indexes_heap[value[1]]
  #décrémentation du degré
  h[index]= (value[0] - 1,value[1])
  
  #on fait remonter l'élément tant qu'il est plus grand que ses parents
  notdone = True
  while index > 0 and notdone :
       next_index = int((index-1)/2)
       
       if value[0]-1 < h[next_index][0]:
         indexes_heap[h[index][1]] = next_index
         indexes_heap[h[next_index][1]] = index  
         h[index], h[next_index] = h[next_index] ,h[index] 
         
       else :
           notdone = False

       index = next_index


### ALGORITHME DE CORE DECOMPOSITION ###########

def core_decomposition(h,adj,degres,number_edges,indexes_heap):
    """

    --Input--
    indexes_heap (dictionnaire de taille n)
    h (minheap de taille n)
    number_edges (entier)
    degres (dictionnaire de taille n) : degrés de tous les noeuds 
    adj (dictionnaire de taille n) : adj_array du graphe

    --Output--
    C (entier) : core value du graphe
    c (dictionnaire de taille n) : core value des noeuds
    eta (dictionnaire de taille n): k-core-ordonnement des noeuds
    edge_memory (liste de taille n) : nombre de liens dans les k-préfixes pour le k-core-ordonnement
    prefix_density (liste de taille n) : densité des k-préfixes pour le k-core-ordonnement
    """
  
    # récupération du nombre de lien et de noeud
    m = number_edges
    i = len(adj.keys()) 
    # initialisation de la liste de densités des préfixes
    prefix_density = []
    edge_memory = []
    prefix_density = i*[0]
    edge_memory = i*[0]
    edge_memory[i-1] =  m
    prefix_density[i-1] = edge_memory[i-1]/i
    #initialisation de la core value du graphe et du dictionnaire contenant les cores values des noeuds 
    C = 0 
    c = {}
    # initialisation de l'ordonnement k-core
    eta = i*[0]
    while i!=0 :
        #extraction du noeud avec le plus petit degré
        dv,v = heappop(h)
        d = degres[v]
        #actualisation du graphe
        for neighbour in adj[v]:
            adj[neighbour].remove(v)
            decrease_degree(indexes_heap,h,(degres[neighbour],neighbour))
            degres[neighbour] = degres[neighbour] -1
     

        #actualisation de c 
        C = max (C, dv ) 
        #stockage de l'ordre
        c[v] = C
        eta[i-1] = v
        #calcul des densités des préfixes à rebour
        if i!=1 :
            edge_memory[i-2] = edge_memory[i-1]-d 
            prefix_density[i-2]=edge_memory[i-2]/(i-1)
        i = i-1
    return(C,c,eta,edge_memory,prefix_density)


## LOADING DATA ###########
start = time.time()

#filename = dir + r'\com-lj.ungraph\com-lj.ungraph.txt'
filename = dir + r'\com-amazon.ungraph.txt\com-amazon.ungraph.txt'
#filename =  dir + r'\com-orkut.ungraph.txt\com-orkut.ungraph.txt'

adj,nodes,number_nodes,number_edges = adjarray(filename) #tp1.adjarray(filename)

print( f'number_nodes : {number_nodes}')
print( f'number_edges : {number_edges}')

degres = Counter(nodes) #calcul du degré pour chaque noeud avec 

### INITIALISATION DE LA MINHEAP ###########

h = [(degres[noeud],noeud) for noeud in degres.keys()]

indexes_heap ={}
for i in range(len(h)):
    indexes_heap[h[i][1]] = i                

heapify(h)

### CORE DECOMPOSITION ###########

C,c,eta,edge_memory,prefix_density= core_decomposition(h, adj.copy(),degres,number_edges,indexes_heap)


print (f'core value: {C}')

size_densest = prefix_density.index(max(prefix_density))+1
edge_density = edge_memory[size_densest-1]/(size_densest*(size_densest-1)*0.5)
average_degree_density = edge_memory[size_densest-1]/size_densest


print(f'size densest: {size_densest}')
print(f'edge density: {edge_density}')
print(f'average degree density : {average_degree_density }')

end = time.time()
print(end - start)


#############################
### Exercice 2 ###########
#############################

import seaborn as sns
import random as rd
import matplotlib.pyplot as plt
import pandas as pd


filename = dir + r'\scholar.tar\scholar\scholar\net.txt'

adj,nodes,number_nodes,number_edges = adjarray(filename)

print( f'number_nodes : {number_nodes}')
print( f'number_edges : {number_edges}')

degres = Counter(nodes) #calcul du degré pour chaque noeud avec 
h = [(degres[noeud],noeud) for noeud in degres.keys()]
indexes_heap ={}
for i in range(len(h)):
    indexes_heap[h[i][1]] = i                
heapify(h)

########### CORE DECOMPOSITION
C,c,eta,edge_memory,prefix_density= core_decomposition(h, adj.copy(),degres,number_edges,indexes_heap)

########### PLOT 1 : Core en fonction du degré
degres = Counter(nodes) #calcul du degré pour chaque noeud avec 
df_degres = pd.DataFrame({'noeud' : list(degres.keys()), 'degre' : list(degres.values())})
df_cores = pd.DataFrame({'noeud' : list(c.keys()), 'core' : list(c.values())})

df = df_degres.merge(df_cores, on ='noeud')
df_counts = df.groupby(['core','degre']).count()
df_counts = df_counts.reset_index()

df_counts.loc[df_counts['core'] == C, 'extreme cluster'] = 1
df_counts.loc[df_counts['core'] != C, 'extreme cluster'] = 0

splot = sns.scatterplot(x = 'degre',y='core',hue = 'noeud',data =df_counts,palette =  sns.color_palette("viridis", as_cmap=True) )
splot.plot(df_counts['degre'].tolist(),df_counts['degre'].tolist(),color ='black')
splot.set(xscale="log",yscale="log")
plt.savefig('exo_2_2.png')



########### PLOT 2 : Cluster
splot = sns.scatterplot(x = 'degre',y='core',hue = 'extreme cluster',data =df_counts,palette =sns.color_palette("tab10")[:2])
splot.plot(df_counts['degre'].tolist(),df_counts['degre'].tolist(),color ='black')
splot.set(xscale="log",yscale="log")
plt.savefig('cluster_2.png')

###########  Identification des éléments du cluster (ceux dont la core value est maximale (= C))

df.loc[df['core'] == C, 'extreme cluster'] = 1
df.loc[df['core'] != C, 'extreme cluster'] = 0


########### Lecture du fichier des noms
filename_id = dir + r'\scholar.tar\scholar\scholar\ID.txt'
file=open(filename_id, 'r')
lines =file.readlines()
file.close()

noeuds = []
noms = []
for row in lines:
        noeud = int(row.split(' ')[0])
        print(noeud)
        longueur_chiffre = len(row.split(' ')[0])
        nom = row[longueur_chiffre +1 :]
        noeuds.append(noeud)
        noms.append(nom)

df_ID = pd.DataFrame({'noeud' : noeuds, 'nom' : noms} )
df_ID['nom'] =df_ID['nom'].apply(lambda x : x.split('\n')[0])


########## Identification des auteurs du cluster
liste = df[df['extreme cluster']==1].merge( df_ID, on ='noeud')['nom'].tolist()

for name in liste :
    print(name)


# %%
