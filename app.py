"""Aplicação em Streamlit (Parte 2 · Etapas 7 a 11, e os resultados da Etapa 12).

Rodar:  streamlit run app.py

Visão principal (home): upload → a classe com o grau (pertinência da classe vencedora),
a caixa colorida da faixa em que a pertinência caiu (verde, amarela ou vermelha) com as
duas pertinências, e as fotos das 5 vizinhas que explicam de onde saiu o número. Fecha
com o bloco "Imagens anteriores", que fica só na sessão (st.session_state) e nunca
entra na base.

A classe, a faixa, a cor da caixa e todo número na tela saem da pertinência a cachorro
com duas casas (fuzzy_knn.pertinencia_na_tela): nada aqui arredonda por conta própria.

Visão admin: os resultados no teste fixo (rodadas, matriz de confusão, acurácia,
meta e galeria da área cinzenta), lidos do que o avaliar.py registrou.
"""
import hashlib
import io

import streamlit as st

import avaliacao
import config
import dados
from avaliacao import percentual
from extrator import carregar_extrator, extrair
from fuzzy_knn import classificar, formatar_pertinencia, limites_area_cinzenta, texto_pertinencias

st.set_page_config(page_title="Cachorro × não cachorro", page_icon="🐶", layout="wide")


# ---------------------------------------------------------------- carregamento
@st.cache_resource(show_spinner="Carregando o bairro...")
def carregar_bairro(marca):
    """marca = horário da última montagem: se o bairro for montado de novo, ele é recarregado."""
    return dados.carregar_vetores(config.ARQ_BAIRRO)


@st.cache_resource(show_spinner="Carregando o extrator (na primeira vez, o Keras baixa os pesos)...")
def carregar_modelo(nome):
    return carregar_extrator(nome)


def analisar(conteudo, bairro):
    """Etapas 8 a 10: o mesmo extrator do bairro, o Fuzzy k-NN e a decisão."""
    extrator, preparar = carregar_modelo(bairro["extrator"])
    x_num = extrair(extrator, preparar, [io.BytesIO(conteudo)])[0]    # a imagem enviada → 1.280 números
    resultado = classificar(bairro["X"], bairro["y"], x_num, k=config.K)
    resultado["vizinhas"] = [(bairro["arquivos"][i], int(bairro["y"][i])) for i in resultado["vizinhas"]]
    return resultado


# ---------------------------------------------------------------- pedaços da tela
def mostrar_imagem(caminho, legenda, largura):
    if caminho is not None:
        st.image(str(caminho), caption=legenda, width=largura)
    else:
        st.caption(f"{legenda} (imagem não encontrada)")


def mostrar_vizinhas(vizinhas, largura=130):
    """vizinhas: [(arquivo, rótulo)], da mais próxima para a mais distante."""
    for posicao, (coluna, (arquivo, rotulo)) in enumerate(zip(st.columns(len(vizinhas)), vizinhas), start=1):
        with coluna:
            mostrar_imagem(dados.localizar_imagem(arquivo), f"{posicao}ª · {dados.texto_classe(rotulo)}", largura)


def plural(n, singular, plural_):
    return f"{n} {singular if n == 1 else plural_}"


# Em qual das três faixas a pertinência a cachorro caiu: título, regra da faixa e a cor da caixa.
# A caixa só explica a decisão que o Fuzzy k-NN já tomou; ela não muda a classe.
# Os números das regras vêm do config, pelo limites_area_cinzenta: mudar a faixa lá muda estes textos.
MINIMO_CINZENTA, MAXIMO_CINZENTA = limites_area_cinzenta()
FAIXAS = {
    "cinzenta": ("⚠ Imagem na área cinzenta",
                 f"Toda imagem enviada pelo usuário que cair entre {MINIMO_CINZENTA} e {MAXIMO_CINZENTA}",
                 "warning"),
    "cachorro": ("🐶 Imagem na área é cachorro",
                 f"Toda imagem enviada pelo usuário que cair acima de {MAXIMO_CINZENTA}",
                 "success"),
    "nao_cachorro": ("❌ Imagem na área não é cachorro",
                     f"Toda imagem enviada pelo usuário que cair abaixo de {MINIMO_CINZENTA}",
                     "error"),
}


def faixa_do_resultado(resultado):
    if resultado["area_cinzenta"]:
        return "cinzenta"
    return "cachorro" if resultado["classe"] == 1 else "nao_cachorro"


def texto_caixa(resultado):
    """O título da faixa, as duas pertinências (a vencedora primeiro) e a contagem das vizinhas."""
    classe, vizinhas = resultado["classe"], resultado["vizinhas"]
    titulo, regra, _ = FAIXAS[faixa_do_resultado(resultado)]
    pertinencias = texto_pertinencias(classe, resultado["p_cachorro"])
    caes = sum(1 for _, rotulo in vizinhas if rotulo == 1)
    contagem = {1: plural(caes, "cachorro", "cachorros"),
                0: plural(len(vizinhas) - caes, "não cachorro", "não cachorros")}
    return (f"**{titulo}**  \n"
            f"{regra}  \n"
            f"pertinência da imagem: {pertinencias}  \n"
            f"parecida com: {contagem[classe]} e {contagem[1 - classe]} "
            f"entre as {len(vizinhas)} imagens mais próximas")


def mostrar_resultado(item):
    """Etapa 11: sempre a classe com o grau, a caixa da faixa e as fotos das vizinhas.

    As vizinhas aparecem nas três faixas, não só na cinzenta: são elas que explicam de onde
    saiu a pertinência, tanto numa resposta folgada quanto numa apertada. Nenhuma das caixas
    muda a classe: elas só explicam a decisão que o Fuzzy k-NN já tomou.

    O título e a caixa saem do mesmo número em duas casas que decidiu a classe e a faixa,
    então a cor da caixa nunca desmente o número que ela mostra.
    """
    resultado = item["resultado"]
    col_imagem, col_texto = st.columns([1, 2])
    with col_imagem:
        st.image(item["conteudo"], caption=item["nome"], width=300)
    with col_texto:
        st.markdown(f"## {resultado['mensagem']}")
        caixa = getattr(st, FAIXAS[faixa_do_resultado(resultado)][2])
        caixa(texto_caixa(resultado))
    st.markdown(f"**As {len(resultado['vizinhas'])} imagens do bairro mais próximas** "
                "(da mais próxima para a mais distante)")
    mostrar_vizinhas(resultado["vizinhas"])


def selecionar(ident):
    st.session_state.selecionada = ident


def mostrar_imagens_anteriores():
    historico = st.session_state.historico
    if not historico:
        return
    st.divider()
    st.subheader("Imagens anteriores")
    st.caption("Clique para rever o resultado. Elas ficam só nesta sessão: somem ao fechar ou "
               "recarregar a página e não entram na base.")
    recentes = list(reversed(historico))
    por_linha = 6
    for inicio in range(0, len(recentes), por_linha):
        for coluna, item in zip(st.columns(por_linha), recentes[inicio:inicio + por_linha]):
            with coluna:
                st.image(item["conteudo"], width=120)
                st.button(item["resultado"]["mensagem"], key=f"rever_{item['id']}", help=item["nome"],
                          on_click=selecionar, args=(item["id"],),
                          disabled=item["id"] == st.session_state.selecionada)


# ---------------------------------------------------------------- visão principal
def mostrar_pertinencia_e_classe():
    """Por que a classe sai da maior pertinência, e não de um Alfa-Corte escolhido à mão."""
    with st.container(border=True):
        st.markdown(
            "O sistema calcula a **pertinência**: o quanto a imagem pertence ao conjunto “cachorro”, "
            "de 0 a 1. A pertinência a “não cachorro” é o que falta para 1.\n\n"
            "A **classe** é uma decisão. Fica a classe de maior pertinência: se a pertinência a "
            "cachorro passa de 0,50, a resposta é “É cachorro”; se fica em 0,50 ou menos, "
            "“Não é cachorro”.\n\n"
            "Para a classe, escolhi seguir a regra do próprio Fuzzy k-NN, a de maior pertinência, "
            "em vez de um Alfa-Corte (parâmetro numérico) escolhido à mão, como 0,60. Assim, a "
            "classe é sempre aquela a que a imagem mais pertence, e a dúvida aparece na área "
            "cinzenta, sem mudar a classe.\n\n"
            "Por isso uma imagem pode ser “É cachorro” e estar na área cinzenta ao mesmo tempo "
            "(nuances da dúvida): ela pertence mais ao conjunto cachorro do que ao outro, mas "
            "por pouco.")


def visao_principal():
    st.title("Cachorro × não cachorro")
    st.caption(f"O EfficientNetB0 descreve a imagem em 1.280 números, e o Fuzzy k-NN (k = {config.K}) "
               "compara esses números com o bairro. A classe é a de maior pertinência.")
    mostrar_pertinencia_e_classe()
    if not config.ARQ_BAIRRO.exists():
        st.error("O bairro ainda não foi montado. Rode no terminal: python montar_bairro.py")
        return
    bairro = carregar_bairro(config.ARQ_BAIRRO.stat().st_mtime)

    estado = st.session_state
    if "historico" not in estado:
        estado.historico = []          # imagens enviadas nesta sessão (não entram na base)
        estado.selecionada = None
        estado.ultimo_envio = None

    enviado = st.file_uploader("Envie uma imagem", type=["jpg", "jpeg", "png", "webp"])
    if enviado is None:
        estado.ultimo_envio = None
    else:
        conteudo = enviado.getvalue()
        ident = hashlib.sha1(conteudo).hexdigest()
        if (enviado.name, ident) != estado.ultimo_envio:      # envio novo
            estado.ultimo_envio = (enviado.name, ident)
            if ident not in {item["id"] for item in estado.historico}:
                try:
                    with st.spinner("Analisando a imagem..."):
                        resultado = analisar(conteudo, bairro)
                    estado.historico.append({"id": ident, "nome": enviado.name,
                                             "conteudo": conteudo, "resultado": resultado})
                except Exception as erro:   # arquivo corrompido ou formato que o Pillow não abre
                    st.error(f"Não consegui analisar essa imagem: {erro}")
            if ident in {item["id"] for item in estado.historico}:
                estado.selecionada = ident

    selecionado = next((item for item in estado.historico if item["id"] == estado.selecionada), None)
    if selecionado is not None:
        mostrar_resultado(selecionado)
    mostrar_imagens_anteriores()


# ---------------------------------------------------------------- visão admin
def linha_da_tabela(rodada):
    return {
        "Rodada": rodada["rodada"],
        "O que mudou": rodada["o_que_mudou"],
        "Extrator": rodada["extrator"],
        "k": rodada["k"],
        "Bairro": rodada["bairro"],
        "VP": rodada["VP"],
        "FN": rodada["FN"],
        "FP": rodada["FP"],
        "VN": rodada["VN"],
        "Acurácia": percentual(float(rodada["acuracia"])),
        "Área cinzenta": rodada["area_cinzenta"],
        "Meta": "✓" if rodada["meta"] == "sim" else "✗",
    }


def texto_legenda(situacao, com_marca=False):
    """A sigla da matriz por extenso: "FP · falso positivo — rótulo real não cachorro e classe cachorro"."""
    nome, explicacao, acertou = avaliacao.LEGENDA_CASAS[situacao]
    marca = f" {'✓' if acertou else '✗'}" if com_marca else ""
    return f"**{situacao} · {nome}** — {explicacao}{marca}"


def mostrar_matriz(rodada):
    total = sum(int(rodada[situacao]) for situacao in avaliacao.CASAS)
    st.markdown("**Matriz de confusão**")
    st.caption(f"Conta as {total} imagens do teste fixo, inclusive as da área cinzenta. "
               "A galeria mais abaixo mostra só as que caíram na faixa de dúvida, "
               "então os números dela são menores que os daqui.")
    st.markdown(
        "|  | disse cachorro | disse não |\n"
        "|---|---|---|\n"
        f"| **era cachorro** | VP {rodada['VP']} | FN {rodada['FN']} |\n"
        f"| **não era** | FP {rodada['FP']} | VN {rodada['VN']} |")
    # Na ordem de leitura da matriz, não na do PRD.
    st.caption("  \n".join(texto_legenda(situacao, com_marca=True)
                           for situacao in ("VP", "FN", "FP", "VN")))


def mostrar_acuracia_e_meta(rodada):
    vp, fn, fp, vn = (int(rodada[casa]) for casa in ("VP", "FN", "FP", "VN"))
    acuracia = float(rodada["acuracia"])
    st.metric("Acurácia", percentual(acuracia), help="(VP + VN) ÷ total")
    st.caption(f"({vp} + {vn}) ÷ {vp + fn + fp + vn}")
    atingida, criterios = avaliacao.meta_atingida(acuracia, fn, fp)
    st.markdown("**Meta:** " + " · ".join(f"{texto} {'✓' if ok else '✗'}" for ok, texto in criterios))
    if atingida:
        st.success("Meta atingida")
    else:
        st.warning("Meta não atingida")


def legenda_teste(item):
    """A pertinência sai da coluna de duas casas, a mesma que decidiu a classe e a faixa.

    Rodadas antigas, gravadas antes dessa coluna existir, caem na medida crua.
    """
    categoria = dados.nome_categoria(item["dataset"], item["categoria_original"])
    pertinencia = formatar_pertinencia(float(item.get("p_cachorro_tela") or item["p_cachorro"]))
    return f"{categoria} · pertinência a cachorro {pertinencia}"


def mostrar_galeria(resultados):
    galeria = avaliacao.galeria_area_cinzenta(resultados)
    na_faixa = sum(len(lista) for lista in galeria.values())
    st.subheader(f"Galeria da área cinzenta · {na_faixa} de {len(resultados)} imagens")
    st.caption(f"Imagens do teste fixo com pertinência a cachorro de {MINIMO_CINZENTA} a "
               f"{MAXIMO_CINZENTA}. Os erros daqui também contam na matriz; as vizinhas mostram "
               "por que o modelo ficou em dúvida.")
    st.caption("Em cada bloco abaixo, três coisas diferentes: **rótulo real** é o gabarito da base, "
               "o que a imagem é de fato; **pertinência** é o número que o Fuzzy k-NN calculou, de 0 a 1; "
               "**classe** é a decisão tirada desse número — acima de 0,50, cachorro. "
               "A sigla (VP, FN, VN, FP) compara só o rótulo real com a classe.")
    for situacao in avaliacao.CASAS:
        faixa, resultado = avaliacao.SITUACOES[situacao][1], avaliacao.SITUACOES[situacao][3]
        lista = galeria[situacao]
        st.markdown(f"#### {resultado} · {len(lista)}")
        st.caption(f"{texto_legenda(situacao)}  \n"
                   f"{avaliacao.LEITURA_AREA_CINZENTA[situacao]}  \n"
                   f"pertinência a cachorro nesta galeria: {faixa}")
        if not lista:
            st.write("Nenhuma imagem nesta situação.")
            continue
        for item in lista:
            with st.container(border=True):
                col_teste, col_vizinhas = st.columns([1, 4])
                with col_teste:
                    mostrar_imagem(dados.localizar_imagem(item["arquivo"]), legenda_teste(item), 170)
                with col_vizinhas:
                    rotulos = [dados.rotulo(classe) for classe in item["classes_vizinhas"].split(";")]
                    mostrar_vizinhas(list(zip(item["vizinhas"].split(";"), rotulos)), largura=110)


def visao_admin():
    st.title("Visão admin · resultados no teste fixo")
    rodadas = avaliacao.ler_rodadas()
    if not rodadas:
        st.info("Nenhuma rodada registrada ainda. Rode no terminal:\n\n"
                'python avaliar.py --rodada 1 --mudanca "ponto de partida: bairro com 320 (160 + 160)"')
        return

    st.subheader("Rodadas")
    st.dataframe([linha_da_tabela(rodada) for rodada in rodadas], hide_index=True)

    por_numero = {int(rodada["rodada"]): rodada for rodada in rodadas}
    numeros = list(por_numero)
    numero = st.selectbox("Rodada", numeros, index=len(numeros) - 1,
                          format_func=lambda n: f"Rodada {n} · {por_numero[n]['o_que_mudou']}")
    rodada = por_numero[numero]

    col_matriz, col_metricas = st.columns(2)
    with col_matriz:
        mostrar_matriz(rodada)
    with col_metricas:
        mostrar_acuracia_e_meta(rodada)
    mostrar_galeria(avaliacao.ler_resultados_rodada(numero))


# ---------------------------------------------------------------- navegação
visao = st.sidebar.radio("Visão", ["Visão principal", "Visão admin"])
st.sidebar.caption(f"k = {config.K} · área cinzenta: pertinência a cachorro de "
                   f"{MINIMO_CINZENTA} a {MAXIMO_CINZENTA}")
if visao == "Visão principal":
    visao_principal()
else:
    visao_admin()
