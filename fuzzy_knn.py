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


def formatar_pertinencia(valor):
    """Duas casas decimais e vírgula, com o arredondamento convencional: o meio sobe.

    O f"{v:.2f}" do Python não serve aqui. Ele arredonda sobre o que o float guarda, e
    0,605 em binário é 0,60499999…, um tiquinho abaixo do meio: desceria para 0,60.
    Decimal(str(valor)) lê o número como ele é escrito, e ROUND_HALF_UP sobe o empate.
    """
    arredondado = Decimal(str(valor)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return f"{arredondado}".replace(".", ",")


def pertinencia_na_tela(p_cachorro):
    """A pertinência a cachorro com duas casas: o único número de que tudo sai.

    A classe, a faixa, a cor da caixa e o que a tela mostra saem daqui. A pertinência a
    não cachorro é 1 menos este valor — as duas nunca são arredondadas em separado, senão
    poderiam não somar 1,00 na tela.
    """
    return float(formatar_pertinencia(float(p_cachorro)).replace(",", "."))


def decidir(p_cachorro):
    """Classe = a de maior pertinência, por um if comum (sem corte).

    Com a pertinência a cachorro em duas casas, a de não cachorro é 1 menos ela, então
    comparar as duas é comparar com 0,50. Exatamente 0,50 fica como não cachorro: só é
    cachorro quando a pertinência a cachorro é a maior.
    """
    p_cachorro = pertinencia_na_tela(p_cachorro)
    if p_cachorro > 1 - p_cachorro:
        return 1        # é cachorro
    return 0            # não é cachorro


def na_area_cinzenta(p_cachorro):
    """Faixa do professor, lida de config.FAIXA_CINZENTA (hoje pertinência a cachorro de
    0,40 a 0,60, limites inclusive, com a pertinência da classe vencedora de 0,50 a 0,60,
    com duas casas).

    A faixa sai do mesmo número em duas casas que decide a classe: uma pertinência de
    0,3965 conta como 0,40 e entra na faixa. Só marca a dúvida; não muda a classe.
    """
    minimo, maximo = config.FAIXA_CINZENTA
    return minimo <= pertinencia_na_tela(p_cachorro) <= maximo


def limites_area_cinzenta():
    """Os dois limites da faixa, já no formato da tela: ("0,40", "0,60").

    Todo texto que cita a faixa sai daqui, e daqui sai de config.FAIXA_CINZENTA: mudar a
    faixa no config muda a classificação e os textos juntos, sem sobrar número escrito à mão.
    """
    minimo, maximo = config.FAIXA_CINZENTA
    return formatar_pertinencia(minimo), formatar_pertinencia(maximo)


def mensagem(classe, p_cachorro):
    """A classe e a pertinência da classe vencedora: "<classe> · pertinência <valor>".

    O valor sai do mesmo número em duas casas que decidiu a classe, então nunca
    desmente a classe que vem antes dele.

    Pertinência, não porcentagem: o número é contínuo de 0 a 1 e diz o quanto a imagem
    pertence ao conjunto, não a chance de um evento.
    """
    p_cachorro = pertinencia_na_tela(p_cachorro)
    texto = "É cachorro" if classe == 1 else "Não é cachorro"
    return f"{texto} · pertinência {formatar_pertinencia(max(p_cachorro, 1 - p_cachorro))}"


def texto_pertinencias(classe, p_cachorro):
    """As duas pertinências, a vencedora primeiro:
    "<classe vencedora> <valor> · <outra classe> <1 − valor>".

    As duas saem do mesmo número em duas casas: a de não cachorro é 1 menos a de
    cachorro, então somam 1,00 na tela sempre.
    """
    nomes = {1: "cachorro", 0: "não cachorro"}
    p_cachorro = pertinencia_na_tela(p_cachorro)
    valores = {1: p_cachorro, 0: 1 - p_cachorro}
    return (f"{nomes[classe]} {formatar_pertinencia(valores[classe])} · "
            f"{nomes[1 - classe]} {formatar_pertinencia(valores[1 - classe])}")


def classificar(X_treino_num, y_treino, x_num, k=config.K):
    """Classifica uma imagem (os 1.280 números dela) contra o bairro."""
    pert, viz = fuzzy_knn(X_treino_num, y_treino, [x_num], k)
    p_nao, p_cachorro = float(pert[0][0]), float(pert[0][1])
    classe = decidir(p_cachorro)
    return {
        "p_nao": p_nao,                                  # medida crua; a tela usa a de duas casas
        "p_cachorro": p_cachorro,
        "p_cachorro_tela": pertinencia_na_tela(p_cachorro),
        "classe": classe,
        "mensagem": mensagem(classe, p_cachorro),
        "area_cinzenta": na_area_cinzenta(p_cachorro),
        "vizinhas": [int(i) for i in viz[0]],
    }
