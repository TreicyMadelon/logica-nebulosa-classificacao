"""Avaliação no teste fixo: matriz de confusão, acurácia, meta, galeria da área cinzenta e rodadas.

O sistema é avaliado só nas 80 imagens do teste fixo, que nunca entram no bairro.
O scikit-learn não classifica: só conta os acertos (confusion_matrix e accuracy_score).
"""
import shutil
from datetime import date
from decimal import ROUND_HALF_UP, Decimal

from sklearn.metrics import accuracy_score, confusion_matrix

import config
import dados
from fuzzy_knn import fuzzy_knn, limites_area_cinzenta, na_area_cinzenta

ARQ_RODADAS = config.PASTA_RESULTADOS / "rodadas.csv"
CAMPOS_RODADAS = ["rodada", "o_que_mudou", "extrator", "k", "bairro", "VP", "FN", "FP", "VN",
                  "acuracia", "area_cinzenta", "meta", "data"]
CAMPOS_RESULTADOS = ["arquivo", "classe_real", "dataset", "categoria_original", "p_nao_cachorro",
                     "p_cachorro", "resposta", "casa", "area_cinzenta", "vizinhas", "classes_vizinhas"]

# As 4 situações da galeria da área cinzenta, na ordem do PRD:
# (rótulo real, pertinência a cachorro, o modelo disse, resultado)
# Os limites da faixa vêm do config; o 0,50 é fixo porque é onde a classe vira, não um ajuste.
MINIMO_CINZENTA, MAXIMO_CINZENTA = limites_area_cinzenta()
CASAS = ("VP", "FN", "VN", "FP")
SITUACOES = {
    "VP": ("cachorro", f"mais de 0,50 até {MAXIMO_CINZENTA}", "é cachorro", "acertou (VP)"),
    "FN": ("cachorro", f"{MINIMO_CINZENTA} a 0,50", "não é cachorro", "errou (FN)"),
    "VN": ("não cachorro", f"{MINIMO_CINZENTA} a 0,50", "não é cachorro", "acertou (VN)"),
    "FP": ("não cachorro", f"mais de 0,50 até {MAXIMO_CINZENTA}", "é cachorro", "errou (FP)"),
}

# Cada sigla por extenso, para a legenda da visão admin: (nome, o que aconteceu, acertou?).
LEGENDA_CASAS = {
    "VP": ("verdadeiro positivo", "rótulo real cachorro e classe cachorro", True),
    "FN": ("falso negativo", "rótulo real cachorro e classe não cachorro", False),
    "FP": ("falso positivo", "rótulo real não cachorro e classe cachorro", False),
    "VN": ("verdadeiro negativo", "rótulo real não cachorro e classe não cachorro", True),
}

# O que significa a imagem estar nesta casa E na área cinzenta (a dúvida não muda a classe).
LEITURA_AREA_CINZENTA = {
    "VP": "Caiu na área cinzenta, mas a classe saiu certa: o sistema ficou em dúvida e "
          "ainda assim acertou que é cachorro.",
    "FN": "Caiu na área cinzenta e a classe saiu errada: era cachorro, e o sistema disse que não. "
          "O aviso de dúvida já estava na tela.",
    "VN": "Caiu na área cinzenta, mas a classe saiu certa: o sistema ficou em dúvida e "
          "ainda assim acertou que não é cachorro.",
    "FP": "Caiu na área cinzenta e a classe saiu errada: não era cachorro, e o sistema disse que era. "
          "O aviso de dúvida já estava na tela.",
}


def percentual(valor, casas_decimais=1):
    """0.9125 → "91,3%".

    Arredonda meio-para-cima. No teste fixo de 80 imagens a acurácia é múltipla de
    1,25 ponto, então metade dos valores cai exatamente no empate (73/80 = 91,25%);
    o arredondamento do Python desempataria ora para cima, ora para baixo.

    Decimal(str(valor)), e não Decimal(valor), pela mesma razão do formatar_pertinencia:
    partindo do float, 23/80 e mais duas acurácias desciam uma casa (28,7% no lugar de 28,8%).
    """
    casas = Decimal(10) ** -casas_decimais
    arredondado = (Decimal(str(valor)) * 100).quantize(casas, rounding=ROUND_HALF_UP)
    return f"{arredondado}%".replace(".", ",")


def casa(real, previsto):
    """Em que casa da matriz a imagem cai."""
    if real == 1:
        return "VP" if previsto == 1 else "FN"
    return "FP" if previsto == 1 else "VN"


def contar_casas(y_teste, y_previsto):
    """Matriz de confusão. Ordem do sklearn: [[VN, FP], [FN, VP]] (linha = rótulo, coluna = resposta)."""
    vn, fp, fn, vp = confusion_matrix(y_teste, y_previsto, labels=[0, 1]).ravel()
    return {"VP": int(vp), "FN": int(fn), "FP": int(fp), "VN": int(vn)}


def meta_atingida(acuracia, fn, fp):
    """Meta no teste fixo: acurácia ≥ 90% (72 das 80), com FN ≤ 6 e FP ≤ 6."""
    criterios = [
        (acuracia >= config.META_ACURACIA, f"acurácia ≥ {config.META_ACURACIA:.0%}"),
        (fn <= config.META_MAX_FN, f"FN ≤ {config.META_MAX_FN}"),
        (fp <= config.META_MAX_FP, f"FP ≤ {config.META_MAX_FP}"),
    ]
    return all(ok for ok, _ in criterios), criterios


def avaliar_teste_fixo(bairro, teste, k=config.K):
    """As 80 imagens passam pelo mesmo Fuzzy k-NN; a matriz compara as respostas com os rótulos."""
    X_treino_num, y_treino = bairro["X"], bairro["y"]
    X_teste_num, y_teste = teste["X"], teste["y"]          # o y_teste fica guardado; o modelo não vê
    pert, viz = fuzzy_knn(X_treino_num, y_treino, X_teste_num, k=k)   # pertinências e vizinhas das 80
    y_previsto = (pert[:, 1] > pert[:, 0]).astype(int)               # maior pertinência → 1 ou 0
    casas = contar_casas(y_teste, y_previsto)
    acuracia = accuracy_score(y_teste, y_previsto)                   # (VP + VN) ÷ total

    resultados = []
    for i, arquivo in enumerate(teste["arquivos"]):
        real, previsto = int(y_teste[i]), int(y_previsto[i])
        resultados.append({
            "arquivo": arquivo,
            "classe_real": dados.classe_do_rotulo(real),
            "dataset": teste["datasets"][i],
            "categoria_original": teste["categorias"][i],
            "p_nao_cachorro": f"{pert[i, 0]:.4f}",
            "p_cachorro": f"{pert[i, 1]:.4f}",
            "resposta": dados.classe_do_rotulo(previsto),
            "casa": casa(real, previsto),
            "area_cinzenta": "sim" if na_area_cinzenta(pert[i, 1]) else "não",
            "vizinhas": ";".join(bairro["arquivos"][j] for j in viz[i]),
            "classes_vizinhas": ";".join(dados.classe_do_rotulo(y_treino[j]) for j in viz[i]),
        })
    return {"resultados": resultados, "casas": casas, "acuracia": float(acuracia)}


def galeria_area_cinzenta(resultados):
    """As imagens de teste que caíram na faixa, separadas nas 4 situações (os erros também contam na matriz)."""
    galeria = {situacao: [] for situacao in CASAS}
    for resultado in resultados:
        if resultado["area_cinzenta"] == "sim":
            galeria[resultado["casa"]].append(resultado)
    for lista in galeria.values():
        lista.sort(key=lambda resultado: float(resultado["p_cachorro"]))
    return galeria


# ---------------------------------------------------------------- rodadas
def pasta_rodada(numero):
    return config.PASTA_RESULTADOS / f"rodada_{numero}"


def ler_rodadas():
    return sorted(dados.ler_csv(ARQ_RODADAS), key=lambda rodada: int(rodada["rodada"]))


def ler_resultados_rodada(numero):
    return dados.ler_csv(pasta_rodada(numero) / "resultados.csv")


def registrar_rodada(numero, o_que_mudou, medidas, extrator, k, tamanho_bairro):
    """Grava a linha da rodada na tabela, os resultados imagem por imagem e cópias das imagens de FP e FN."""
    pasta = pasta_rodada(numero)
    if pasta.exists():
        shutil.rmtree(pasta)          # registrar de novo a mesma rodada substitui o registro anterior
    pasta.mkdir(parents=True)
    dados.gravar_csv(pasta / "resultados.csv", medidas["resultados"], CAMPOS_RESULTADOS)

    for resultado in medidas["resultados"]:
        if resultado["casa"] in ("FP", "FN"):
            origem = dados.localizar_imagem(resultado["arquivo"])
            if origem:
                destino = pasta / resultado["casa"] / f"p{float(resultado['p_cachorro']):.2f}_{resultado['arquivo']}"
                destino.parent.mkdir(exist_ok=True)
                shutil.copy2(origem, destino)

    casas = medidas["casas"]
    atingida, _ = meta_atingida(medidas["acuracia"], casas["FN"], casas["FP"])
    linha = {
        "rodada": numero,
        "o_que_mudou": o_que_mudou,
        "extrator": extrator,
        "k": k,
        "bairro": tamanho_bairro,
        **casas,
        "acuracia": f"{medidas['acuracia']:.4f}",
        "area_cinzenta": sum(1 for resultado in medidas["resultados"] if resultado["area_cinzenta"] == "sim"),
        "meta": "sim" if atingida else "não",
        "data": date.today().isoformat(),
    }
    rodadas = [rodada for rodada in dados.ler_csv(ARQ_RODADAS) if rodada.get("rodada") != str(numero)]
    rodadas.append(linha)
    rodadas.sort(key=lambda rodada: int(rodada["rodada"]))
    dados.gravar_csv(ARQ_RODADAS, rodadas, CAMPOS_RODADAS)
