"""Avaliação no teste fixo (RF8) e registro das rodadas (RF9).

Uso:
    python avaliar.py
        só mostra a matriz, a acurácia, a meta e a área cinzenta
    python avaliar.py --rodada 1 --mudanca "ponto de partida: bairro com 320 (160 + 160)"
        registra a rodada em resultados/rodadas.csv e copia as imagens de FP e FN
        para resultados/rodada_1/
    python avaliar.py --rodada 4 --mudanca "k = 9" --k 9
        rodada que testa o k: usa o mesmo bairro, sem montar de novo

Regras das rodadas: uma mudança por rodada (dados, uma configuração ou o código),
sempre testada nas mesmas 80 imagens.
"""
import argparse
import sys

import avaliacao
import config
import dados
from avaliacao import percentual
from fuzzy_knn import limites_area_cinzenta


def imprimir_matriz(casas):
    print("Matriz de confusão")
    print(f"{'':16}{'disse cachorro':>16}{'disse não':>12}")
    print(f"{'era cachorro':16}{'VP ' + str(casas['VP']):>16}{'FN ' + str(casas['FN']):>12}")
    print(f"{'não era':16}{'FP ' + str(casas['FP']):>16}{'VN ' + str(casas['VN']):>12}")


def main():
    parser = argparse.ArgumentParser(description="Avalia no teste fixo e registra a rodada.")
    parser.add_argument("--rodada", type=int, help="número da rodada a registrar (ex.: 1)")
    parser.add_argument("--mudanca", help='o que mudou nesta rodada (ex.: "k = 9")')
    parser.add_argument("--k", type=int, default=config.K, help=f"vizinhas do Fuzzy k-NN (padrão: {config.K})")
    args = parser.parse_args()
    if args.rodada is not None and not args.mudanca:
        parser.error('--rodada precisa de --mudanca "o que mudou nesta rodada"')
    if args.k < 1:
        parser.error("--k precisa ser pelo menos 1")
    if not config.ARQ_BAIRRO.exists() or not config.ARQ_TESTE.exists():
        sys.exit("O bairro ainda não foi montado. Rode primeiro: python montar_bairro.py")

    bairro = dados.carregar_vetores(config.ARQ_BAIRRO)
    teste = dados.carregar_vetores(config.ARQ_TESTE)
    for aviso in dados.conferir_vetores(dados.ler_metadados(), bairro, teste):
        print(f"Atenção: {aviso}")

    medidas = avaliacao.avaliar_teste_fixo(bairro, teste, k=args.k)
    casas, total = medidas["casas"], len(teste["arquivos"])
    print(f"Teste fixo: {total} imagens · bairro: {len(bairro['arquivos'])} imagens · "
          f"extrator: {bairro['extrator']} · k = {args.k}\n")
    imprimir_matriz(casas)
    print(f"\nAcurácia: ({casas['VP']} + {casas['VN']}) ÷ {total} = {percentual(medidas['acuracia'])}")
    atingida, criterios = avaliacao.meta_atingida(medidas["acuracia"], casas["FN"], casas["FP"])
    marcas = " · ".join(f"{texto} {'✓' if ok else '✗'}" for ok, texto in criterios)
    print(f"Meta: {marcas} → {'atingida' if atingida else 'não atingida'}")

    galeria = avaliacao.galeria_area_cinzenta(medidas["resultados"])
    na_faixa = sum(len(lista) for lista in galeria.values())
    minimo, maximo = limites_area_cinzenta()
    print(f"\nÁrea cinzenta (pertinência a cachorro de {minimo} a {maximo}): {na_faixa} de {total} imagens")
    print("  " + " · ".join(f"{situacao} {len(galeria[situacao])}" for situacao in avaliacao.CASAS))

    if args.rodada is not None:
        avaliacao.registrar_rodada(args.rodada, args.mudanca, medidas, bairro["extrator"], args.k,
                                   len(bairro["arquivos"]))
        print(f"\nRodada {args.rodada} registrada em resultados/rodadas.csv; "
              f"imagens de FP e FN copiadas para resultados/rodada_{args.rodada}/.")
        print("Veja a matriz, as rodadas e a galeria na visão admin: streamlit run app.py")


if __name__ == "__main__":
    main()
