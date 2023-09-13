from flask import Flask, jsonify, request
from flask_cors import CORS  # Importe o módulo Flask-CORS
import networkx as nx
import matplotlib.patches as mpatches
import os
import matplotlib.pyplot as plt
import matplotlib
import matplotlib as mpl
import numpy as np
from deputy import Deputado
from graph import set_graph_inverted_weights, set_graph_normalized, set_graph_normalized_no_threshold, write_graph_file, generate_centrality, heatmap, generate_heatmap, generate_plot


app = Flask(__name__)
CORS(app)  # Configure o CORS para permitir todas as origens

@app.route('/api/data', methods=['GET'])
def get_data():
    data = {'message': 'Hello from Python backend!'}
    return jsonify(data)

@app.route('/api/sendDados', methods=['POST'])
def send_data():
    data = request.json  # Os dados do formulário serão enviados como JSON
    print("Dados recebidos:", data)
    ano = data['ano']
    partidos = data['partido'].split()
    threshold = int(data['percent']) / 100
    trs_formated = str("{:.3f}".format(round(threshold, 3)))
    # Faça algo com os dados recebidos, por exemplo, imprimir no console

    print(f"Filtros escolhidos:\n- Ano: {ano}\n- Partidos: {partidos}\n- Threshold: {trs_formated}\n")



    ## -------------- READING DEPUTY FILE AND ADDING DEPUTIES ---------------
    file_politicians = "politicians" + ano + ".txt"
    file_graph = "graph" + ano + ".txt"

    print(f"Arquivo de politicos definido: {file_politicians}")
    print(f"Arquivo de grafo definido: {file_graph}\n")

    cur_path = os.path.dirname(__file__)
    path_politicians = '.\\datasets\\' + file_politicians
    new_path_p = os.path.relpath(path_politicians , cur_path)

    deputies = []
    all_parties = []
    print("Definindo deputados...")

    with open(new_path_p, encoding="utf8") as file_p:
        lines = file_p.read().splitlines()
        for line in lines:
            words = line.split(';')
            d = Deputado(words[0],words[1], words[2])
            deputies.append(d)
            if words[1] not in all_parties:
                all_parties.append(words[1])
            
    print(f"Deputados adicionados: {len(deputies)}\n")

    ## ------------------- READING GRAPH FILE ------------------------------
    graph = nx.Graph()
    graph_no_threshold = nx.Graph()
    graph_normalized = nx.Graph()

    path_graph = '.\\datasets\\' + file_graph
    new_path_g = os.path.relpath(path_graph , cur_path)

    with open(new_path_g, encoding="utf8") as file_g:
        lines = file_g.read().splitlines()

    values_no_threshold = set_graph_normalized_no_threshold(lines, deputies, partidos, graph_no_threshold)
    values_graph = set_graph_inverted_weights(lines, deputies, partidos, threshold, graph)
    values_normalized = set_graph_normalized(lines, deputies, partidos, threshold, graph_normalized)

    ## ---------------------------- TEST --------------------------------

    generate_centrality(graph, partidos)

    generate_heatmap(graph_no_threshold, partidos)

    generate_plot(graph_normalized, all_parties, values_normalized[0], values_normalized[1], partidos)
    

    return jsonify({"message": "Dados recebidos com sucesso!"})

if __name__ == '__main__':
    app.run(debug=True)