#############################
######### TD3 ###############
#############################


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
dir = os.path.dirname(__file__)

#############################
### Exercice 1 ##############
#############################

############  LOAD FONCTION

def read_file(filename) :
    ''' lecture de la edge liste '''

    file=open(filename, 'r')
    lines =file.readlines()
    file.close()

    return(lines)

############   FONCTIONS PAGERANK

def calculate_dout(lines_file):
    '''Calcul du degré sortant et initialisation de P_t
    
    --Input--
    lines_files : liste de couple (s,t)

    --Output--
    d_out : dictionnaire de taille n contenant les degrés sortants des noeuds
    P_t : dictionnaire de taille n contenant la valeur initale des scores pageranks
    
    '''

    d_out = {}
    P_t = {}
    for row in lines_file:
        if row[0] !='#' and row != '' and row !='\n': # Enleve les lignes commentaires 
            s = int(row.split('\t')[0])
            t = int(row.split('\t')[1])

            if s not in d_out:
                d_out[s] = 1
                P_t[s] = 1/2070486
            else :
                d_out[s] = d_out[s]+ 1

            if t not in d_out:
                P_t[t] = 1/2070486

            
    print(len(list(P_t.keys())))
    return(d_out, P_t)

def calculate_din(lines_file):
    ''' calcul du degré entrant 
    
    --Input--
    lines_files : liste de couple (s,t)

    --Output--
    d_in : dictionnaire de taille n contenant les degrés entrants des noeuds    
    '''
    d_in = {}
    for row in lines_file:
        if row[0] !='#' and row != '' and row !='\n': # Enleve les lignes commentaires 
            s = int(row.split('\t')[0])
            t = int(row.split('\t')[1])
            if t not in d_in:
                d_in[t] = 1
            else :
                d_in[t] = d_in[t]+ 1

    return(d_in)

def matvectorprod(lines_file, P_t, d_out):
    ''' calcul du produit entre P_t et la matrice de transition T 
    
    --Input--
    lines_files : liste de couple (s,t)
    P_t : dictionnaire de taille n contenant la valeur des scores pageranks au temps t
    d_out : dictionnaire de taille n contenant les degrés sortants des noeuds

    --Output--
    P_t_plus_1 : dictionnaire de taille n contenant le produit T*P_t (valeur des scores pageranks au temps t+1)
    
    '''

    P_t_plus_1 = {}
    for row in lines_file:
        if row[0] !='#' and row != '' and row !='\n': 
            s = int(row.split('\t')[0])
            t = int(row.split('\t')[1])

            if t in P_t_plus_1:
                P_t_plus_1[t] =  P_t_plus_1[t] + P_t[s]/d_out[s]
            else :
                P_t_plus_1[t] = P_t[s]/d_out[s]

            if s not in P_t_plus_1:
                P_t_plus_1[s] = 0

    return(P_t_plus_1)


def page_rank(lines_file, t_range , alpha):
    ''' algorithme du page rank
    
    --Input--
    lines_files : liste de couple (s,t)
    t_range : nombre d'itération
    alpha : paramètre de la marche aléatoire

    --Output--
    P_t : dictionnaire de taille n contenant la valeur des scores pageranks au temps t_range
    
    '''
    d_out,P_t = calculate_dout(lines_file)
    for t in range(t_range):
        print ('itération',t)
        P_t = matvectorprod(lines_file, P_t, d_out)
        
        Sum_P_t = 0
        for node in P_t.keys():
            P_t[node] = (1-alpha)*P_t[node] + alpha * 1/2070486
            Sum_P_t += P_t[node]

        for node in P_t.keys():
            P_t[node] += (1-Sum_P_t)/2070486

    return(P_t)


def personalized_page_rank(lines_file, t_range , alpha, P0):
    ''' algorithme du page rank
    
    --Input--
    lines_files : liste de couple (s,t)
    t_range : nombre d'itération
    alpha : paramètre de la marche aléatoire
    P0 : personalization dictionary

    --Output--
    P_t : dictionnaire de taille n contenant la valeur des scores pageranks au temps t_range
    
    '''
    d_out,P_t = calculate_dout(lines_file)
    for t in range(t_range):
        print ('itération',t)
        P_t = matvectorprod(lines_file, P_t, d_out)
        
        Sum_P_t = 0
        for node in P_t.keys():
            P_t[node] = (1-alpha)*P_t[node] + alpha * P0
            Sum_P_t += P_t[node]

        for node in P_t.keys():
            P_t[node] += P0[node](1-Sum_P_t)

    return(P_t)


########## MAIN

##### LOADING DATA

filename = dir + r'alr21--dirLinks--enwiki-20071018.txt'

filename_namewiki = dir + r'alr21--pageNum2Name--enwiki-20071018.txt'

lines_file = read_file(filename)

##### PAGE RANK

P_t = page_rank(lines_file, t_range = 10, alpha=0.15)

#### BEST & WORST PAGES
best = sorted(P_t, key=P_t.get, reverse=True)[:5]
worst = sorted(P_t, key=P_t.get)[:5]

df = pd.read_csv(filename_namewiki,header=3,sep= '\t')
df[df['#'].isin(best)][['page title']]
df[df['#'].isin(worst)][['page title']]


#############################
### Exercice 2 ##############
#############################

## calcul degré entrant
din = calculate_din(lines_file)

## construction d'une data frame comprenant (degré entrant, degré sortant, pagerank score) pour chaque noeud
df_d_in = pd.DataFrame({'noeud' : list(din.keys()), 'd_in' : list(din.values())})
df_d_out = pd.DataFrame({'noeud' : list(d_out.keys()), 'd_out' : list(d_out.values())})
df_P_t = pd.DataFrame({'noeud' : list(P_t.keys()), 'pagerank' : list(P_t.values())})
df = df_d_out.merge(df_P_t, on ='noeud')
df = df.merge(df_d_in, on ='noeud')


########## PLOT 1
splot = sns.scatterplot(x = 'pagerank',y='d_in',data =df,palette =  sns.color_palette("viridis", as_cmap=True) )
splot.set(xscale="log",yscale="log")
plt.savefig('tp3_2.png')
print('plot2 good')


########## PLOT 2
splot = sns.scatterplot(x = 'pagerank',y='d_out',data =df,palette =  sns.color_palette("viridis", as_cmap=True) )
splot.set(xscale="log",yscale="log")
plt.savefig('tp3_1.png')

########## PLOT ALPHA

for alpha in [0.1,0.2,0.5,0.9] :
    P_t_y = page_rank(lines_file, Nit, alpha=alpha)
    df_alpha =  pd.DataFrame({'noeud' : list(P_t_y.keys()), f'pagerank_{alpha}' : list(P_t_y.values())})
    df = df.merge(df_alpha, on ='noeud')
    splot = sns.scatterplot(x = 'pagerank',y=f'pagerank_{alpha}',data =df,palette =  sns.color_palette("viridis", as_cmap=True) )
    splot.set(xscale="log",yscale="log")
    plt.savefig(f'tp3_{alpha}.png')
    

    



