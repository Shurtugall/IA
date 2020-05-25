# -*- coding: utf-8 -*-
# Anaconda3 5.2.0 (Python 3.7.6)
# Esse programa executa em qualquer distribuição Python 3
# SOM para o dataset "iris_data_012.txt"

# ============================================================================

"""
Created on Sun May 24 18:22:35 2020
@author: Gabriel Righi
Universidade Federal de Santa Maria 
"""

import numpy as np
import matplotlib.pyplot as plt
#Em caso de falha tente > pip uninstall matplotlib
#e em seguida > pip install matplotlib

#Função recebe
#e retorna o índice da linha e coluna do SOM, com tamanho m_Linhas e m_Colunas
#que são as coordenadas da célula do mapa cujo vetor está mais próximo do
#item de dados em data[t].
def no_mais_proximo(data, t, map, m_Linhas, m_Colunas):
  # (linha,coluna) do nó do mapa mais próximo a data[t]
  resultado = (0,0)
  menor_distancia = 1.0e20
  for i in range(m_Linhas):
    for j in range(m_Colunas):
      ed = dist_euclidiana(map[i][j], data[t])
      if ed < menor_distancia:
        menor_distancia = ed
        resultado = (i, j)
  return resultado

#Função recebe dois vetores e retorna a distancia euclidiana entre eles.
#Lembrete: A distancia euclidiana é definida como: somatorio dos pontos
#sqrt((pi-qi)^2), onde i inicia em 1 e vai até n, sendo n definido pelo usuario.
def dist_euclidiana(v1, v2):
  return np.linalg.norm(v1 - v2)

#Recebe as coordenadas da celula 1, r1 e c1 e coordenadas da celula 2, r2 e c2.
#Retorna a distancia Manhattan entre as celulas (o valor absoluto de r1 - r2
#somado com o valor absoluto de c1 - c2).
def dist_manhattan(r1, c1, r2, c2):
  return np.abs(r1-r2) + np.abs(c1-c2)

#Recebe uma lista de valores de 0 até n
#Retorna o valor que mais aparece nessa lista. Por exemplo, considere a lista:
#[0,2,2,1,0,1,1,2,1], o número 1 aparece mais vezes, então vai retornar 1.
def mais_comum(lst, n):
  # lst is a list of values 0 . . n
  if len(lst) == 0: return -1
  counts = np.zeros(shape=n, dtype=np.int)
  for i in range(len(lst)):
    counts[lst[i]] += 1
  return np.argmax(counts)

# ============================================================================
  
def main():
  #Definição de variáveis
  np.random.seed(1)
  Dim = 4                       #dimensao
  Linhas = 30; Colunas = 30     #Tamanhos de linhas e Linhas do SOM
  AlcanceMax = Linhas + Colunas #Tamanho máximo calculado em linhas + Linhas
  AprendMax = 0.5               #Taxa inicial de aprendizagem usada na construção
  ItMax = 5000                  #Limite máximo de iterações para treinar" o SOM

  #Carregar os dados do arquivo para a memória
  print("\nCarregando dados Iris para memoria \n")
  #A variavel data_file recebe os dados do iris_data_012.txt
  data_file = "iris_data_012.txt"
  data_x = np.loadtxt(data_file, delimiter=",", usecols=range(0,4),
    dtype=np.float64)
  data_y = np.loadtxt(data_file, delimiter=",", usecols=[4],
    dtype=np.int)

  # Construção do SOM
  print("Construindo um SOM 30x30 a partir dos dados Iris")
  #a função random_sample() está gerando uma matriz 30x30 onde cada célula do
  #vetor de tamanho 4 possui valores aleatórios no intervalo [0.0 a 1.0]
  map = np.random.random_sample(size=(Linhas,Colunas,Dim))
  for s in range(ItMax):
    #printa a iteração que está sendo feita a cada 500 iterações
    if s % (ItMax/10) == 0: print("iteração = ", str(s))
    #pct_rest calcula a porcentagem de iterações que faltam para concluir, por
    #exemplo, se s = 25 e ItMax = 100, então pct_rest = 0.75
    pct_rest = 1.0 - ((s * 1.0) / ItMax)
    #alcance_atual é a distancia máxima que as células podem aceitar como próximas
    #uma das outras durante a iterção s.
    alcance_atual = (int)(pct_rest * AlcanceMax)
    taxa_atual = pct_rest * AprendMax
    
    #t recebe um dado aleatório e o melhor nó é determinado
    t = np.random.randint(len(data_x))
    (bmu_linha, bmu_coluna) = no_mais_proximo(data_x, t, map, Linhas, Colunas)
    #cada nó do SOM é examinado. Se o nó atual está próximo ao melhor
    #nó da unidade correspondente, então o vetor do nó atual é atualizado
    for i in range(Linhas):
      for j in range(Colunas):
        if dist_manhattan(bmu_linha, bmu_coluna, i, j) < alcance_atual:
          #a atualização aproxima o vetor do nó atual do item dos dados atuais 
          #usando o valor taxa_atual que diminui lentamente ao longo do tempo.
          map[i][j] = map[i][j] + taxa_atual * \
(data_x[t] - map[i][j])
  print("Construção do SOM finalizada \n")

  # Construção da U-Matrix
  print("Construindo a U-Matrix do SOM")
  u_matrix = np.zeros(shape=(Linhas,Colunas), dtype=np.float64)
  #Inicialmente cada 30x30 célula da U-Matrix possui um valor 0.0. Em seguida,
  #cada célula na U-Matrix é processada
  for i in range(Linhas):
    for j in range(Colunas):
      #v é o vetor no Som que corresponde à célula U-Matrix atual. Cada célula
      #adjacente no SOM(acima, abaixo, esquerda e direita) é processada e a soma
      #das distancias euclidianas é calculada
      v = map[i][j]
      soma_dist = 0.0; ct = 0
     
      if i-1 >= 0:    # acima
        soma_dist += dist_euclidiana(v, map[i-1][j]); ct += 1
      if i+1 <= Linhas-1:   # abaixo
        soma_dist += dist_euclidiana(v, map[i+1][j]); ct += 1
      if j-1 >= 0:   # esquerda
        soma_dist += dist_euclidiana(v, map[i][j-1]); ct += 1
      if j+1 <= Colunas-1:   # direita
        soma_dist += dist_euclidiana(v, map[i][j+1]); ct += 1
      
      u_matrix[i][j] = soma_dist / ct
  print("U-Matrix construída \n")

  # Um valor muito pequeno em uma célula da U-Matrix significa que a célula
  #correspondente no SOM está muito próxima de seus vizinhos, então as células
  #vizinhas fazem parte de um grupo semelhante.
  #A biblioteca MatPlotLib incluida no inicio do arquivo plota a U-matrix.
  plt.imshow(u_matrix, cmap='gray')  # cor preto = próximo = aglomerados
  plt.show()

  # Como os dados possuem rótulos, uma possível visualização:
  #associa cada rótulo de dados a um nó do mapa
  print("Associando cada rotulo dos dados a um nódulo do mapa")
  #Primeiro, seta uma matriz 30x30 onde cada célula contém uma lista vazia
  mapping = np.empty(shape=(Linhas,Colunas), dtype=object)
  for i in range(Linhas):
    for j in range(Colunas):
      mapping[i][j] = []     #Lista vazia

  #Cada célula é processada, e o rótulo da classe (0,1 ou 2) é associado ao item
  #de dados mais próximo da célula correspondente no SOM e é adiconado à lista
  for t in range(len(data_x)):
    (m_row, m_col) = no_mais_proximo(data_x, t, map, Linhas, Colunas)
    mapping[m_row][m_col].append(data_y[t])

  #Em seugida, o rótulo da classe mais comum é extraído da lista na célula atual
  #e colocado em uma matriz mapa_rotulos.
  mapa_rotulos = np.zeros(shape=(Linhas,Colunas), dtype=np.int)
  for i in range(Linhas):
    for j in range(Colunas):
      mapa_rotulos[i][j] = mais_comum(mapping[i][j], 3)
 
  plt.imshow(mapa_rotulos, cmap=plt.cm.get_cmap('terrain_r', 4))
  plt.colorbar()
  plt.show()
  
# ============================================================================
  
if __name__=="__main__":
  main()