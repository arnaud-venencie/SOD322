#############################
######### TD1 ###############
#############################


#############################
### Load function ###########
#############################

def adjarray(filename) :
    #affichage du nom du graphe
    name = filename.split('.')[0]
    print(f'Graphe {name}')
    
    #lecture du nombre de noeuds et d'arrêtes
    file=open(filename, 'r')
    lines =file.readlines()
    file.close()

    number_nodes = int(lines[2].split(' ')[2])
    number_edges = int(lines[2].split(' ')[4])
    
    graph_d = {}
    nodes = []
    for row in lines:
        if row[0] !='#':
            node1 = int(row.split('\t')[0]) 
            node2 = int(row.split('\t')[1])
            if node1 not in graph_d.keys():
                graph_d[node1] = [node2]
            else:
                graph_d[node1].append(node2)
            if node2 not in graph_d.keys():
                graph_d[node2] = [node1]
            else:
                graph_d[node2].append(node1)
        
            nodes.append(node1)
            nodes.append(node2)
    nb_nodes= len(set(nodes))
    nb_edges = len(lines[4:])
           

    #vérification nombre d'arrête:
    if nb_edges == number_edges:
        print(f'Nombre edges {number_edges}')
    else :
        print('erreur chargement')
    
    #vérification nombre de noeuds:
    if nb_nodes == number_nodes:
        print(f'Nombre nodes {number_nodes}')
    else :
        print('erreur chargement')
   
    
    return(graph_d) 

def edge_liste(filename) :

    #affichage du nom du graphe
    name = filename.split('.')[0]
    print(f'Graphe {name}')
    
    #lecture du nombre de noeuds et d'arrêtes
    file=open(filename, 'r')
    lines =file.readlines()
    file.close()

    liste = []
    for row in lines:
        if row[0] !='#': # Enleve les lignes commentaires       
            d = int(row.split('\t')[0])
            a = int(row.split('\t')[1])
            liste.append([d, a])
   
    return(liste) 




def adjmatrix(filename): 
    #affichage du nom du graphe
    name = filename.split('.')[0]
    print(f'Graphe {name}')
    
    #lecture du nombre de noeuds et d'arrêtes
    file=open(filename, 'r')
    lines =file.readlines()
    file.close()

    number_nodes = int(lines[2].split(' ')[2])
    number_edges = int(lines[2].split(' ')[4])
    
    mat = np.zeros((number_nodes,number_nodes))



def edge_liste(filename) :

    #affichage du nom du graphe
    name = filename.split('.')[0]
    print(f'Graphe {name}')
    
    #lecture du nombre de noeuds et d'arrêtes
    file=open(filename, 'r')
    lines =file.readlines()
    file.close()

    liste = []
    for row in lines:
        if row[0] !='#': # Enleve les lignes commentaires       
            d = int(row.split('\t')[0])
            a = int(row.split('\t')[1])
            liste.append([d, a])
   
    return(liste) 

###################################
##### Exercice 2 ##################
##### BFS for graph size ##########
###################################


def bfs(visited, graph, node):

    queue = []  # Create a fifo

    visited[node] = 0  #
    queue.append(node)
    while queue:
        s = queue.pop(0)  # Fifo
        for neighbour in graph[s]:
            distance = visited[s]
            if neighbour not in visited.keys():
                visited[neighbour] = distance + 1
                queue.append(neighbour)
    #print(visited)
    return visited

def lower_bound(graph, first_node, iteration):

    node = first_node
    max_distance = 0 

    for t in range(iteration):
        visited = {}
        bfs_visited = bfs(visited=visited, graph=graph, node=node)
        maximum = max(bfs_visited, key=bfs_visited.get)

        dist_from_max = bfs_visited[maximum]

        if max_distance < dist_from_max:
            max_distance = dist_from_max

        node = maximum
        print(maximum, dist_from_max)
    print(max_distance)
    return max_distance

import random

def upper_bound(graph):

    node_list = list(graph.keys())
    node =  random.choice(node_list)
    visited = {}
    upper_b = 0
    for i in range(2):
        bfs_visited = bfs(visited=visited, graph=graph, node=node)
        maximum = max(bfs_visited, key=bfs_visited.get)
        dist_from_max1 = bfs_visited[maximum]

        bfs_visited[maximum] = 0

        maximum2= max(bfs_visited, key=bfs_visited.get)
        dist_from_max2 = bfs_visited[maximum2]

        if dist_from_max1 + dist_from_max2 >upper_b:
            upper_b = dist_from_max1 + dist_from_max2
        node =  random.choice(node_list)
        
    print(upper_b)
    return dist_from_max1 + dist_from_max2

    
#############################################
######## Exercice 3 ########################
######## Listing_triangle ###################
#############################################

def triangle_count(adjarray):
    list_triangle_count = 0
    #a = []
    for node, _to_nodes in adjarray.items():
        for _to_node in _to_nodes:
            W = list(set(adjarray[node]) & set(adjarray[_to_node]))
            for w in W:
                list_triangle_count +=1
                #a.append((node,_to_node,w))
    print(list_triangle_count)
    return list_triangle_count

def triangle_count2(adjarray):

    list_triangle_count = 0
    adjarray_trunc = {}

    # for each node and its neighbors in adjarray
    for node, _to_nodes in adjarray.items():

        # If we have not created its truncated sorted list of neighbors, create it
        if node not in adjarray_trunc.keys():
            adjarray_trunc[node] = trunc_list(adjarray[node], node)

        # We take neighbors higher than node
        for _to_node in adjarray_trunc[node]:
            # If we have not created its truncated sorted list of neighbors, create it
            if _to_node not in adjarray_trunc.keys():
                adjarray_trunc[_to_node] = trunc_list(adjarray[_to_node], _to_node)
            
            
            # Calculate the intersection between the node neighbors and the neighborhood of the neighbor nodes
            W = list(set(adjarray_trunc[_to_node]) & set(adjarray_trunc[node]))
        

            for w in W:
                # the triangle is u,v,w  
                list_triangle_count +=1
            

    print(list_triangle_count)
    return list_triangle_count

def trunc_list(list_u,u):
    '''
    function that create a truncated and sorted list 
    Input : list_u a list/ u a value
    Output : return the sorted list with value higher than u
    '''

    # Sort
    list_u = sorted(list_u)
    node_index = 0
    if list_u[-1] <= u : 
        return []
    while (int(list_u[node_index]) < int(u)) and node_index < len(list_u)-1:
        node_index+=1
    return list_u[node_index:]


############## Load Data #################

#Amazon
#filename = r'..\com-amazon.ungraph.txt\com-amazon.ungraph.txt'

#Lj
filename = r'..\com-lj.ungraph\com-lj.ungraph.txt'

#orkut
#filename = r'..\com-orkut.ungraph.txt\com-orkut.ungraph.txt'

graph = {
  1: [2,3],
  3 : [1, 2,4],
  2: [1,3,4],
  4: [2,3]
  }


######################################
### Exercice 1/2 for adjarray ########
####################################

exo_1 = True

## Pickle Dump
## create a copy in pickle to go faster 
#import pickle
#with open('am.obj', 'wb') as fp:
#         pickle.dump(adjarray, fp)

if exo_1:
    import time
    tps1 = time.clock()
    #adjarray = adjarray(filename)
    tps2 = time.clock()
    print(tps2 - tps1)




######################################
### Exercice 3 Bfs ##################
####################################

exo_3 = False

#import time
if exo_3:

    lower_bound(adjarray,list(adjarray.keys())[0],5)
    upper_bound(adjarray)


######################################
### Exercice 4 list_triangle #########
####################################

exo_4 = False

if exo_4 :

    tps1 = time.clock()
    triangle_count(adjarray)
    tps2 = time.clock()
    print(tps2 - tps1)

    tps1 = time.clock()
    triangle_count2(adjarray)
    tps2 = time.clock()
    print(tps2 - tps1)

