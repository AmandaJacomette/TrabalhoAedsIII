import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
import matplotlib as mpl
import numpy as np
from deputy import Deputado
from pylab import *

def set_graph_normalized(lines, deputies, parties, threshold, graph):
    deputados = []
    partidos = {}
    dselecionado = None
    nselecionado = None

    for d in deputies:
        if d.siglaPartido in parties:
            deputados.append(d.nome)
            partidos[d.nome] = d.siglaPartido

    print("Definindo grafo Normalizado...\n")
    for line in lines:
        words = line.split(';')

        if words[0] in deputados and words[1] in deputados:
            for d in deputies:
                if d.nome == words[0]:
                    dselecionado = d
                elif d.nome == words[1]:
                    nselecionado = d
    
            weight = int(words[2])
            wn = dselecionado.normalize(nselecionado, weight)
            if wn >= threshold:
                graph.add_weighted_edges_from([(dselecionado, nselecionado, wn), (nselecionado, dselecionado, wn)])
            else:
                graph.add_node(nselecionado)
                graph.add_node(dselecionado)


    print(f"Nos adicionados: {graph.number_of_nodes()}")
    print(f"Arestas adicionadas: {graph.number_of_edges()}")
    print(f"Grafo conexo: {nx.is_connected(graph)}\n")
    return deputados, partidos

def set_graph_normalized_no_threshold(lines, deputies, parties, graph):
    deputados = []
    partidos = {}
    dselecionado = None
    nselecionado = None

    for d in deputies:
        if d.siglaPartido in parties:
            deputados.append(d.nome)
            partidos[d.nome] = d.siglaPartido

    print("Definindo grafo Sem Threshold...\n")
    for line in lines:
        words = line.split(';')

        if words[0] in deputados and words[1] in deputados:
            for d in deputies:
                if d.nome == words[0]:
                    dselecionado = d
                elif d.nome == words[1]:
                    nselecionado = d
    
            weight = int(words[2])
            wn = dselecionado.normalize(nselecionado, weight)
            graph.add_weighted_edges_from([(dselecionado, nselecionado, wn), (nselecionado, dselecionado, wn)])


    print(f"Nos adicionados: {graph.number_of_nodes()}")
    print(f"Arestas adicionadas: {graph.number_of_edges()}")
    print(f"Grafo conexo: {nx.is_connected(graph)}\n")
    return deputados, partidos

def add_deputy(graph, deputies, parties, deputy):
    for d in deputies:
        if d.nome == deputy:
            if d.siglaPartido not in parties:
                break
            else:
                if not graph.has_node(d):
                    graph.add_node(d)

def set_graph_inverted_weights(lines, deputies, parties, threshold, graph):
    deputados = []
    partidos = {}
    dselecionado = None
    nselecionado = None

    for d in deputies:
        if d.siglaPartido in parties:
            deputados.append(d.nome)
            partidos[d.nome] = d.siglaPartido

    print("Definindo grafo com Pesos Invertidos...\n")
    for line in lines:
        words = line.split(';')

        if words[0] in deputados and words[1] in deputados:
            for d in deputies:
                if d.nome == words[0]:
                    dselecionado = d
                elif d.nome == words[1]:
                    nselecionado = d
    
            weight = int(words[2])
            wn = dselecionado.normalize(nselecionado, weight)
            wi = dselecionado.invert_weight(nselecionado, weight)
            if wn >= threshold:
                graph.add_weighted_edges_from([(dselecionado, nselecionado, wi), (nselecionado, dselecionado, wi)])
            else:
                graph.add_node(nselecionado)
                graph.add_node(dselecionado)


    print(f"Nos adicionados: {graph.number_of_nodes()}")
    print(f"Arestas adicionadas: {graph.number_of_edges()}")
    print(f"Grafo conexo: {nx.is_connected(graph)}\n")
    return deputados, partidos


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

def generate_centrality(graph, parties):
    print("Gerando visualizacao de Centralidade do grafo...")

    fig, ax = plt.subplots(figsize = (10, 10))

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
    string = "betweenness"
    for i in parties:
        string += " "
        string += i
    string += ".png"

    fig.savefig(string)

    print("Visualização de Centralidade gerada.\n")
    return string


def heatmap(data, row_labels, col_labels, ax = None,
            cbar_kw = {}, cbarlabel = "", **kwargs):
    if not ax:
      ax = plt.gca()

    im = ax.imshow(data, **kwargs)

    cbar = ax.figure.colorbar(im, ax = ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation = -90, va = "bottom")

  
    ax.set_xticks(np.arange(data.shape[1]), labels = col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels = row_labels)

    ax.tick_params(top = True, bottom = False,
                   labeltop = True, labelbottom = False)

    plt.setp(ax.get_xticklabels(), rotation = -45, ha = "right",
             rotation_mode = "anchor", fontsize=6)
    plt.setp(ax.get_yticklabels(), ha = "right", fontsize=6)

 
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

    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.


    kw = dict(horizontalalignment = "center",
              verticalalignment = "center")
    kw.update(textkw)


    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)


    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def by_party(e : Deputado):
    return e.siglaPartido

def generate_heatmap(graph_normalized, parties):
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

    ig, ax = plt.subplots() # "YlGn"
    im, cbar = heatmap(data, row_labels = x_labs, col_labels = y_labs, ax = ax, cmap = "afmhot", cbarlabel = "Proximidade")

    plt.show()
    string = "heatmap"
    for i in parties:
        string += " "
        string += i
    string += ".png"

    ig.savefig(string)
    print("Visualização de HeatMap gerada.\n")
    return string


def generate_plot(graph, parties_list, deputies, parties_graph, parties):
    cor = []
    pcor = {}
    colors = []
    # cmap = plt.get_cmap('plasma', len(parties_list))
    # plt.set_cmap(cmap)

    for i in range(len(parties_list)):
        colors.append('#%06X' % randint(0, 0xFFFFFF))

    for d in range(len(parties_list)):
       pcor[parties_list[d]] = colors[d]

    for nd in graph.adj.keys():
        if nd.nome in deputies:
            cor.append(pcor[parties_graph[nd.nome]])

    values = []
    values_parties = []

    for i in range(len(parties_list)):
        if parties_list[i] in parties_graph:
            values.append(pcor[parties_list[i]])
            values_parties.append(parties_list[i])

    fig, ax = plt.subplots(figsize=(10, 15))
    node_positions = nx.spring_layout(graph, scale = 100)
    # plt.figure(num= None, figsize=(10, 15), dpi= 140)
    # nx.draw_spring(graph, with_labels = True)
    nodes = nx.draw(graph, pos=node_positions, node_color = cor, with_labels=True, font_size=6, node_size=12, width = 0.2)
    
    legend_labels = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=pcor[party], markersize=8, label=party) for party in values_parties]
    ax.legend(handles=legend_labels, title="Partidos", loc='upper right', prop={'size': 6})
    plt.show()
    
    string = "graph"
    for i in parties:
        string += " "
        string += i
    string += ".png"
    print("Visualização de Grafo gerada.\n")
    fig.savefig(string)
    return string