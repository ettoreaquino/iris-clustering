"""Main module
"""

import numpy as np
import statistics as st
import random as rd
import plotly.express as px
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from scipy import stats

def plot_caracteristicas(iris_data):
  fig = make_subplots(rows=1, cols=2)

  fig.add_trace(go.Scatter(x=iris_data["Petal Length"],
                          y=iris_data["Petal Width"],
                          mode='markers', name='Pétalas'), row=1, col=1)
  fig.add_trace(go.Scatter(x=iris_data["Sepal Length"],
                          y=iris_data["Sepal Width"],
                          mode='markers', name='Sépalas'), row=1, col=2)
  
  # Update xaxis properties
  fig.update_xaxes(title_text="Comprimento (cm)", row=1, col=1)
  fig.update_xaxes(title_text="Comprimento (cm)", row=1, col=2)

  # Update yaxis properties
  fig.update_yaxes(title_text="Largura (cm)", row=1, col=1)
  fig.update_yaxes(title_text="Largura (cm)", row=1, col=2)

  fig.update_layout(title_text="Características")
  fig.show()


def Normal_Flower(media,cov):
    '''Calcula a normal para os dados a partir da média e covariancia'''
    
    mean       = np.array(media)
    covariance = np.array(cov)
    
    # Encontra a normal multivariável
    Var = stats.multivariate_normal(mean, covariance)
    
    return Var

def calcProb(Todos_pontos,Normal):
    ''' Calcula a probabilidade de cada ponto (linhas) fazer parte do grupo'''
    
    # transforma para array
    points = np.array(Todos_pontos)
    # conta o numero de colunas e de linhas
    num_lin,num_cols = points.shape
    
    Prob_vec = []
    
    for i in range(num_lin):
        Prob_vec.append(Normal.pdf(points[i,:]))
    
    return Prob_vec

def populacao(N_individuos,dimensao):
    # simplesmente falando que todo mundo começa randomico
    return np.random.rand(N_individuos,dimensao)

def separaDados(iris_data,iris_target,iris_dic):
    target = np.array(iris_target)
    ind = []

    for i in range(len(iris_target)-1):  # Retorna o número de itens em uma lista 
        if target[i]==iris_dic:
            ind.append(i) 
            
    #print(iris_data)
    out =  np.array(iris_data)
    
    return out[np.array(ind),:],np.array(ind)

def FOB(media,cov,Data_points):
    # Recebo a média e covariancia ( um individuo)
    
    # Calculo a normal para esses dados
    Normal = Normal_Flower(media,cov)
    
    # Calcula a probabilidade individual de cada elemento
    Probability_array = calcProb(Data_points,Normal)
    
    # Soma todas as probabilidades (quanto maior, mais otimizado estará)
    Soma = np.sum(np.array(Probability_array))
    
    return (1/Soma)

def FOB2(Dim,Data_points):
    '''Usada para otimizar a media e a covariancia'''
    #print(Dim)
    # Recebo a média e covariancia ( um individuo)
    # Separa a média
    media = Dim[0:4]
    #print(media)
    # criando uma matriz de covariancia diagonal
    Cv = Dim[4:]
    #print(Cv)
    cov = Cv#np.zeros((4,4))
    #np.fill_diagonal(cov,Cv)    
    
    # Calculo a normal para esses dados
    Normal = Normal_Flower(media,cov)
    
    # Calcula a probabilidade individual de cada elemento
    Probability_array = calcProb(Data_points,Normal)
    
    # Soma todas as probabilidades (quanto maior, mais otimizado estará)
    Soma = np.sum(np.array(Probability_array))
    
    if Soma==0:
        return 999999999999
    else:
         return (1/Soma)  # Maximização

def GWO(lb,ub,N_it,a,Num_lobos,cov,Data_flower):

    dimensao = len(lb);
    # Inicialização dos indivíduos
    wolves= np.random.rand(N_it,Num_lobos,dimensao) 
    wolves[0][:][:] = populacao(Num_lobos,dimensao)
        
    fob = [0 for row in range(Num_lobos)];
    
    for NL in range(Num_lobos):
        fob[NL] = FOB(wolves[0][NL],cov,Data_flower)

    t = 0;
    # inicializa o vetor para guardar as melhores FOBs
    V_fob = [0 for row in range(N_it)];
    
    medVAL = [0 for row in range(N_it)];
    medVAL[0] = st.mean(fob);

    # Salva a posicao do Alpha, que tem a menor fob
    alphapos = fob.index(min(fob));
    # Salva o melhor lobo (alpha)
    alpha = wolves[0][alphapos].copy();
    # Salva a fob do alpha
    alphafob = fob[alphapos];
    # Atribui um valor grande para substituir na proxima iteração
    fob[alphapos] = 10**10;
    
    # Salva a posicao do beta, que tem a menor fob já que o alpha agora tem uma fob ruim (10**10)
    betapos = fob.index(min(fob));
    # Salva o lobo beta
    beta    = wolves[0][betapos].copy();
    # Atribui um valor grande para substituir na proxima iteração
    fob[betapos] = 10**10;
    
    # Salva a posicao do delta, que tem a menor fob já que o alpha agora tem uma fob ruim (10**10)
    deltapos = fob.index(min(fob));
    # Salva o lobo delta
    delta    = wolves[0][deltapos].copy();
    
    # Salva a melhor FOB da iteração 0
    V_fob[0] = alphafob;

    while t in range(N_it-1):
        
        for m in range(Num_lobos):
            
            if m != alphapos and m != betapos and m != deltapos:
                
                for n in range(dimensao):
                    
                    # quanto mais proximo do fim das iterações, menos os lobos alpha, beta e delta, irão interferir nos demais
                    ganho_it = a - a*(t)/N_it;
                    
                    # Coeficientes dinâmico (tende a 0 no fim das iterações)
                    A1 = 2*ganho_it*rd.random()-ganho_it;
                    A2 = 2*ganho_it*rd.random()-ganho_it;
                    A3 = 2*ganho_it*rd.random()-ganho_it;
                    
                    # Outro coficiente aleatório
                    C1 = 2*rd.random();
                    C2 = 2*rd.random();
                    C3 = 2*rd.random();
                    
                    # lobo alpha
                    Dalpha = abs(C1*alpha[n] - wolves[t][m][n]);
                    X1     = alpha[n] - A1*Dalpha;
                    
                    # lobo beta
                    Dbeta  = abs(C2*beta[n] - wolves[t][m][n]);
                    X2     = beta[n] - A2*Dbeta;
                    
                    # lobo delta
                    Ddelta = abs(C3*delta[n] - wolves[t][m][n]);
                    X3     = delta[n] - A3*Ddelta;
                   
                    # Nova posição para lobos
                    wolves[t+1][m][n] = (X1 + X2 + X3) /3;   
                    
                    #Saturação das variáveis
                    if wolves[t+1][m][n] < lb[n]:
                        wolves[t+1][m][n] = lb[n]
                    elif wolves[t+1][m][n] > ub[n]:
                        wolves[t+1][m][n] = ub[n];
            else:
                wolves[t+1][m] = wolves[t][m].copy();

        for k in range(Num_lobos):
            fob[k] = FOB(wolves[t+1][k],cov,Data_flower)
            #fob[k] = FOB2(wolves[t+1][k],Data_flower)
            
        # Atualização de alpha, beta e delta
        medVAL[t+1] = st.mean(fob);
        alphapos    = fob.index(min(fob));        
        alpha       = wolves[t+1][alphapos].copy();
        alphafob    = fob[alphapos];
        #fob[alphapos]=10**10;
        
        betapos = fob.index(min(fob));
        beta    = wolves[t+1][betapos].copy();
        #fob[betapos]=10**10;
        
        deltapos= fob.index(min(fob));
        delta   = wolves[t+1][deltapos].copy();
        
        # Salva a melhor fob no vetor de fobs
        V_fob[t] = alphafob;

        t = t+1;

    X    = alpha;
    fval = alphafob;

    return X,fval,V_fob,medVAL

def classify(X,cov,Data_points):
    media = X
    
    #cov = np.zeros((4,4))
    #np.fill_diagonal(cov,Cv) 
    
    # Calculo a normal para esses dados
    Normal = Normal_Flower(media,cov)
    
    # Calcula a probabilidade individual de cada elemento
    Probability_array = calcProb(Data_points,Normal)
    return Probability_array

def plot_trainning(V_fob_setosa, V_fob_versicolor, V_fob_virginica):
  fig = make_subplots(rows=1, cols=3)

  fig.add_trace(go.Scatter(y=V_fob_setosa,mode='lines', name='Setosa'), row=1, col=1)
  fig.add_trace(go.Scatter(y=V_fob_versicolor,mode='lines', name='Versicolor'), row=1, col=2)
  fig.add_trace(go.Scatter(y=V_fob_virginica,mode='lines', name='Virginica'), row=1, col=3)

  # Update xaxis properties
  fig.update_xaxes(title_text="Iteração", row=1, col=1)
  fig.update_xaxes(title_text="Iteração", row=1, col=2)
  fig.update_xaxes(title_text="Iteração", row=1, col=3)
  fig.update_xaxes(type="log")

  # Update yaxis properties
  fig.update_yaxes(title_text="Covariância", row=1, col=1)

  fig.update_layout(title_text="Processo de Otimização")
  fig.show()
    