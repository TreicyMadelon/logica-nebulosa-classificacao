"""Fuzzy k-NN (Keller, Gray e Givens, 1985): pertinências, decisão e área cinzenta.

O EfficientNetB0 só dá os números de cada imagem. O Fuzzy k-NN faz o resto: mede a
distância da imagem nova até cada imagem do bairro, ordena, pega as k vizinhas mais
próximas, dá a cada uma o peso 1/d² (mais perto = mais peso) e calcula a pertinência:
pertinência a cachorro = pesos das vizinhas cachorro ÷ soma de todos os pesos.
As duas pertinências somam 1. Pertinência não é probabilidade: diz o quanto a
imagem pertence ao conjunto, não a chance de um evento.
"""
from decimal import ROUND_HALF_UP, Decimal

import numpy as np

import config


def fuzzy_knn(X_treino_num, y_treino, X_novo_num, k=config.K):
    """Devolve as pertinências [não cachorro, cachorro] de cada imagem nova e os
    índices das k vizinhas no bairro (da mais próxima para a mais distante)."""
    X_treino_num = np.asarray(X_treino_num, dtype=np.float64)
    y_treino = np.asarray(y_treino)
    pertinencias = []
    vizinhas = []                                         # quem são as vizinhas de cada imagem
    for x in X_novo_num:
        d = np.linalg.norm(X_treino_num - x, axis=1)     # distância a cada imagem de treino
        viz = np.argsort(d)[:k]                           # as k vizinhas mais próximas
        peso = 1 / (d[viz] ** 2 + 1e-9)                   # mais perto = mais peso
        p_cachorro = peso[y_treino[viz] == 1].sum() / peso.sum()
        pertinencias.append([1 - p_cachorro, p_cachorro])
        vizinhas.append(viz)                              # índices das k vizinhas
    return np.array(pertinencias), vizinhas               # ex.: [[0.35, 0.65], ...] e [[12, 7, 431, 88, 5], ...]


def decidir(p_nao, p_cachorro):
    """Classe = a de maior pertinência, por um if comum (sem corte).

    Exatamente 0,50 fica como não cachorro: só é cachorro quando a pertinência
    a cachorro é a maior.
    """
    if p_cachorro > p_nao:
        return 1        # é cachorro
    return 0            # não é cachorro


def na_area_cinzenta(p_cachorro):
    """Faixa do professor, lida de config.FAIXA_CINZENTA (hoje pertinência a cachorro de
    0,40 a 0,60, limites inclusive, com a classe vencedora de 0,51 a 0,60).

    Só marca a dúvida; não muda a classe.
    """
    minimo, maximo = config.FAIXA_CINZENTA
    return minimo <= p_cachorro <= maximo


def formatar_pertinencia(valor):
    """Duas casas decimais e vírgula, com o arredondamento convencional: o meio sobe.

    O f"{v:.2f}" do Python não serve aqui. Ele arredonda sobre o que o float guarda, e
    0,605 em binário é 0,60499999…, um tiquinho abaixo do meio: desceria para 0,60.
    Decimal(str(valor)) lê o número como ele é escrito, e ROUND_HALF_UP sobe o empate.
    """
    arredondado = Decimal(str(valor)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return f"{arredondado}".replace(".", ",")


def limites_area_cinzenta():
    """Os dois limites da faixa, já no formato da tela: ("0,40", "0,60").

    Todo texto que cita a faixa sai daqui, e daqui sai de config.FAIXA_CINZENTA: mudar a
    faixa no config muda a classificação e os textos juntos, sem sobrar número escrito à mão.
    """
    minimo, maximo = config.FAIXA_CINZENTA
    return formatar_pertinencia(minimo), formatar_pertinencia(maximo)


def mensagem(classe, p_nao, p_cachorro):
    """A classe e a pertinência da classe vencedora: "<classe> · pertinência <valor>".

    Pertinência, não porcentagem: o número é contínuo de 0 a 1 e diz o quanto a imagem
    pertence ao conjunto, não a chance de um evento.
    """
    texto = "É cachorro" if classe == 1 else "Não é cachorro"
    return f"{texto} · pertinência {formatar_pertinencia(max(p_nao, p_cachorro))}"


def texto_pertinencias(classe, p_nao, p_cachorro):
    """As duas pertinências, a vencedora primeiro:
    "<classe vencedora> <valor> · <outra classe> <1 − valor>".

    A perdedora sai de 1 menos a vencedora já arredondada, para as duas somarem 1,00 na tela.
    """
    nomes = {1: "cachorro", 0: "não cachorro"}
    vencedora = formatar_pertinencia(max(p_nao, p_cachorro))
    perdedora = formatar_pertinencia(1 - float(vencedora.replace(",", ".")))
    return f"{nomes[classe]} {vencedora} · {nomes[1 - classe]} {perdedora}"


def classificar(X_treino_num, y_treino, x_num, k=config.K):
    """Classifica uma imagem (os 1.280 números dela) contra o bairro."""
    pert, viz = fuzzy_knn(X_treino_num, y_treino, [x_num], k)
    p_nao, p_cachorro = float(pert[0][0]), float(pert[0][1])
    classe = decidir(p_nao, p_cachorro)
    return {
        "p_nao": p_nao,
        "p_cachorro": p_cachorro,
        "classe": classe,
        "mensagem": mensagem(classe, p_nao, p_cachorro),
        "area_cinzenta": na_area_cinzenta(p_cachorro),
        "vizinhas": [int(i) for i in viz[0]],
    }
