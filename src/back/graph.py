import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import matplotlib as mpl
import numpy as np
from deputy import Deputado

def set_graph_normalized(lines, deputies, parties, threshold, graph):

    print("Definindo grafo...\n")
    for line in lines:
        words = line.split(';')
        weight = int(words[2])
        flag = 0
        for d in deputies:
            if d.nome == words[0]: # checando se o nome do objeto Deputado é igual o do grafo
                if d.siglaPartido not in parties: # se o partido não estiver nos filtros definidos, quebra
                    break
                else:
                    if not graph.has_node(d): # se o nó do deputado não existir, é adicionado
                        graph.add_node(d)
        
                for n in deputies:
                    if n.nome == words[1]: # checando se o objeto Deputado possui o mesmo nome
                        if n.siglaPartido not in parties: # checando se partido é o solicitado
                            break
                        else:
                            if not graph.has_node(n): # checando se nó já existe no grafo
                                graph.add_node(n) # adiciona se não existir

                        if graph.has_node(d) and graph.has_node(n): # verifica se existem os dois nó no grafo
                            if not graph.has_edge(d, n): # verifica se já não existe uma aresta
                                wn = d.normalize(n, weight) # normaliza o peso da aresta
                                if wn < threshold: # verifica se o peso é menor que o threshold
                                    flag = 1
                                    break # se for menor, não monta no grafo e sai do comando

                                else: # se o peso for maior ou igual, cria a aresta com peso
                                    graph.add_weighted_edges_from([(d, n, wn), (n, d, wn)])
                    
                    if flag == 1:
                        break
                if flag == 1:
                    break

    print(f"Nos adicionados: {graph.number_of_nodes()}")
    print(f"Arestas adicionadas: {graph.number_of_edges()}")
    print(f"Grafo conexo: {nx.is_connected(graph)}\n")

def add_deputy(graph, deputies, parties, deputy):
    for d in deputies:
        if d.nome == deputy:
            if d.siglaPartido not in parties:
                break
            else:
                if not graph.has_node(d):
                    graph.add_node(d)

def set_graph_inverted_weights(lines, deputies, parties, threshold, graph):

    print("Definindo grafo...\n")
    for line in lines:
        words = line.split(';')
        weight = int(words[2])
        flag = 0
        for d in deputies:
            if d.nome == words[0]: # checando se o nome do objeto Deputado é igual o do grafo
                if d.siglaPartido not in parties: # se o partido não estiver nos filtros definidos, quebra
                    break
                else:
                    if not graph.has_node(d): # se o nó do deputado não existir, é adicionado
                        graph.add_node(d)
        
                for n in deputies:
                    if n.nome == words[1]: # checando se o objeto Deputado possui o mesmo nome
                        if n.siglaPartido not in parties: # checando se partido é o solicitado
                            break
                        else:
                            if not graph.has_node(n): # checando se nó já existe no grafo
                                graph.add_node(n) # adiciona se não existir

                        if graph.has_node(d) and graph.has_node(n): # verifica se existem os dois nó no grafo
                            if not graph.has_edge(d, n): # verifica se já não existe uma aresta
                                wn = d.normalize(n, weight) # normaliza o peso da aresta
                                wi = d.invert_weight(n, weight) # normaliza para criar caminho
                                if wn < threshold: # verifica se o peso é menor que o threshold
                                    flag = 1
                                    break # se for menor, não monta no grafo e sai do comando

                                else: # se o peso for maior ou igual, cria a aresta com peso
                                    graph.add_weighted_edges_from([(d, n, wi), (n, d, wi)])
                    
                    if flag == 1:
                        break
                if flag == 1:
                    break

    print(f"Nos adicionados: {graph.number_of_nodes()}")
    print(f"Arestas adicionadas: {graph.number_of_edges()}")
    print(f"Grafo conexo: {nx.is_connected(graph)}\n")


def write_graph_file(graph):
    fname = input("Insira o nome do arquivo para escrever o grafo (sem extensao): ")
    file = fname + ".txt"
    w_graph = open(file, "w", encoding="utf-8")
    line1 = str(graph.number_of_nodes()) + " " + str(graph.number_of_edges()) + "\n"
    w_graph.write(line1)
    for n, nbrs in graph.adj.items():
        for nbr, eattr in nbrs.items():
            wt = eattr['weight']
            edge = n.__str__() + " " + nbr.__str__() + " " + str("{:.3f}".format(round(wt, 3))) + "\n"
            ## edge = n.nome + " " + nbr.nome + " " + str(wt) + "\n"
            w_graph.write(edge)
    w_graph.close()
    print(f"- Grafo escrito no arquivo {file}\n")

def generate_centrality(graph):
    print("Gerando visualizacao de Centralidade do grafo...")

    fig, ax = plt.subplots()

    b_centrality = nx.betweenness_centrality(graph)

    xpoints = []
    ypoints = []
    temp = []

    for i in b_centrality.items():
        temp.append(i)

    temp.sort(key=lambda d: d[1])

    for v in temp: ## eixo y
        ypoints.append(v[1])

    for d in temp: ## eixo x
        xpoints.append(d[0].__str__())

    # for d in b_centrality.keys(): ## eixo x
        # xpoints.append(d.nome)
       

    ax.bar(xpoints, ypoints)
    labels = ax.get_xticklabels()
    plt.setp(labels, rotation=45, horizontalalignment='right', fontsize=5)

    ax.set_xlabel('Deputados')
    ax.set_ylabel('Betweennes')
    ax.set_title('Medida de Centralidade')

    plt.show()
    fig.savefig("betweenness.png")

    print("Visualização de Centralidade gerada.\n")


def heatmap(data, row_labels, col_labels, ax = None,
            cbar_kw = {}, cbarlabel = "", **kwargs):
    if not ax:
      ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax = ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation = -90, va = "bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels = col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels = row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top = True, bottom = False,
                   labeltop = True, labelbottom = False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation = -45, ha = "right",
             rotation_mode = "anchor", fontsize=6)
    plt.setp(ax.get_yticklabels(), ha = "right", fontsize=6)

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor = True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor = True)
    ax.grid(which = "minor", color = "w", linestyle = '-', linewidth = 3)
    ax.tick_params(which =  "minor", bottom = False, left = False)

    return im, cbar


def annotate_heatmap(im, data = None, valfmt="{x:.2f}",
                     textcolors = ("black", "white"),
                     threshold = None, **textkw):

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, 
    # but allow it to be overwritten by textkw.
    kw = dict(horizontalalignment = "center",
              verticalalignment = "center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def by_party(e : Deputado):
    return e.siglaPartido

def generate_heatmap(graph_normalized):

    dp_x = []
    dp_y = []
    wgs = []
    temp = []

    for d in graph_normalized.adj.keys(): ## eixo x
        temp.append(d)

    temp.sort(key=by_party)

    for n in temp:
        dp_x.append(n.__str__())
        dp_y.append(n.__str__())

    for i in temp:
        wg = []
        for d in temp:
            if graph_normalized.has_edge(i, d):
                wg.append(graph_normalized.adj[i][d]['weight'])

            elif (i == d):
                wg.append(1)
            else:
                wg.append(0)
        wgs.append(wg)


    tam = len(dp_x)

    for i in range(tam):
        if len(wgs[i]) < tam:
            for w in range( tam - len(wgs[i])):
                wgs[i].append(0)

    data = np.array(wgs)
    x_labs = dp_x
    y_labs = np.array(dp_y)

    ig, ax = plt.subplots()
    im, cbar = heatmap(data, row_labels = x_labs, col_labels = y_labs,
                   ax = ax, cmap = "YlGn", cbarlabel = "Proximidade")

    plt.show()