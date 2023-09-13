import networkx as nx
import os
import matplotlib.pyplot as plt
import matplotlib
import matplotlib as mpl
import numpy as np
from deputy import Deputado
from graph import set_graph_inverted_weights, set_graph_normalized, write_graph_file, generate_centrality, heatmap, generate_heatmap

## ------------------------- FILTERING ---------------------------
print("O programa disponibiliza de dados de políticos e seus partidos dos anos 2002 até os anos 2022\nSelecione o ano e os partidos que deseja analisar:\n")
ano = input("Ano: ")
num_partidos = input("Numero de partidos que deseja analisar: ")

partidos = []

for n in range(int(num_partidos)):
    aux = input(f"Sigla do partido {n + 1}: ")
    partidos.append(aux.upper())

trs = input("Insira o peso mínimo para ser considerado no grafo (de 0 a 100): ")
threshold = int(trs) / 100
trs_formated = str("{:.3f}".format(round(threshold, 3)))
print(f"Filtros escolhidos:\n- Ano: {ano}\n- Partidos: {partidos}\n- Threshold: {trs_formated}\n")



## ---------------------- READING DEPUTY FILE AND ADDING DEPUTIES -----------------------
file_politicians = "politicians" + ano + ".txt"
file_graph = "graph" + ano + ".txt"

print(f"Arquivo de politicos definido: {file_politicians}")
print(f"Arquivo de grafo definido: {file_graph}\n")

cur_path = os.path.dirname(__file__)
path_politicians = '.\\datasets\\' + file_politicians
new_path_p = os.path.relpath(path_politicians , cur_path)

deputies = []
print("Definindo deputados...")

with open(new_path_p, encoding="utf8") as file_p:
    lines = file_p.read().splitlines()
    for line in lines:
        words = line.split(';')
        d = Deputado(words[0],words[1], words[2])
        deputies.append(d)
        
print(f"Deputados adicionados: {len(deputies)}\n")

## ------------------- READING GRAPH FILE -----------------------------------
graph = nx.Graph()
graph_normalized = nx.Graph()

path_graph = '.\\datasets\\' + file_graph
new_path_g = os.path.relpath(path_graph , cur_path)

with open(new_path_g, encoding="utf8") as file_g:
    lines = file_g.read().splitlines()

set_graph_inverted_weights(lines, deputies, partidos, threshold, graph)
set_graph_normalized(lines, deputies, partidos, threshold, graph_normalized)

## ----------------------- WRITING GRAPH ------------------------------

# write_graph_file(graph)
# write_graph_file(graph_normalized)

## ---------------------- CENTRALITY ----------------------

generate_centrality(graph)

## ----------------------------- HEATMAP ------------------------------------

generate_heatmap(graph_normalized)