import json
from json import JSONEncoder

class Deputado:
    def __init__(self, nome, partido, votacoes):
        self.nome = nome
        self.siglaPartido = partido
        self.total_votacoes = int(votacoes)

    def __repr__(self): ## função para representar o deputado em caso de print do objeto
        return '{} ({})'.format(self.nome, self.siglaPartido)
    
    def __str__(self):
        ## return '{}, do partido {} e id {}. Votacoes: {}'.format(self.nome, self.siglaPartido, self.id, self.votacoes)
        return '{} ({})'.format(self.nome, self.siglaPartido)
    
    def polls_attended(self):
        return '{}  {}\n'.format(self.nome, self.totalVotacoes)

    def votes(self):
        return self.total_votacoes
    
    def print_connection(self, node, weight):
        return '{}  {}  {}'.format(self.nome, node.nome, str(weight))
    
    def normalize(self, deputy, weight):
            weight = float(weight)
            div = min(self.votes(), deputy.votes())
            res = weight / float(div)
            return res

    def invert_weight(self, deputy, weight):
        wn = self.normalize(deputy, weight)
        weight = 1 - wn
        return weight

class EncodeDeputado(JSONEncoder): ## classe para encodificar o deputado para dicionário
    def default(self, o):
            return o.__dict__
