# Cachorro × não cachorro · plano A

Trabalho final de Lógica Nebulosa (prof. Mardonio). O sistema recebe uma imagem e responde se é cachorro ou não, com o grau fuzzy da resposta: a classe e a pertinência da classe vencedora, um número de 0 a 1. O EfficientNetB0 descreve a imagem em 1.280 números, e o Fuzzy k-NN (k = 5, peso 1/d²) compara esses números com um bairro de 320 imagens rotuladas. A classe é a de maior pertinência; com pertinência a cachorro entre 0,40 e 0,60, a tela também mostra o aviso de área cinzenta.

## A lógica do projeto

O raciocínio segue o mapa do projeto, de cima para baixo: requisitos → classes e tipo de aprendizado → dados → treino → grau (pertinência) → classe pela maior pertinência e área cinzenta → avaliação → melhoria. Cada passo existe por causa do anterior.

### 1. Os requisitos definem o problema

Projeto fuzzy com classificação binária.

A classificação binária define as **duas classes**: cachorro (rótulo 1) e não cachorro (rótulo 0). "Não cachorro" é tudo o que não é cachorro: gatos, outros animais, pessoas, objetos e cenários.

### 2. As classes pedem aprendizado supervisionado

As imagens já vêm marcadas como cachorro ou não, então o problema é **supervisionado**: cada exemplo traz a resposta certa (o rótulo). Fuzzy k-NN na versão clássica de classificação, sem rótulo, as vizinhas não diriam nada.

### 3. As bases de origem fornecem as imagens rotuladas

As 400 imagens vêm de duas bases de origem: o conjunto de validação do **ImageNet** (objeto em destaque, muita variedade de animais, como o hamster) e o **Pascal VOC** (cenas do dia a dia, mais parecidas com as fotos que um usuário envia). Cada classe tem 100 imagens de cada base. Essa mistura simétrica impede que o estilo de uma base vire pista para separar as classes (o viés de dataset, de Torralba e Efros). Do ImageNet entram só fotos de validação, que o EfficientNetB0 nunca viu no treino.

As imagens rotuladas precisam de duas coisas:

- **Um "não X" variado**: se "não cachorro" tivesse só gatos, o sistema aprenderia "cachorro × gato". Por isso entram outros animais, canídeos selvagens (lobo, coiote, raposa, dingo, mabeco, bons candidatos à área cinzenta), pessoas, objetos e cenários.
- **Balanceamento**: 200 + 200. Com muito mais exemplos de uma classe, o sistema tenderia a responder sempre ela, e a acurácia enganaria.

A base equilibrada passa pela **triagem com YOLO**, que confere se cada foto bate com a pasta (por exemplo, um cachorro ao fundo de uma foto em `nao_cachorro/`). Rótulo errado ensina errado. A triagem vem antes do split.

### 4. O split separa o treino da prova

O **split 80/20** divide as 400 imagens: 80% vão para o treino (o bairro, 320 imagens) e 20% para o teste (80 imagens que o sistema nunca viu, como uma prova com questões inéditas).

- O **stratify** equilibra o split: sorteia dentro de cada grupo classe + dataset de origem, então o teste fica com 20 imagens de cada base em cada classe, a mesma simetria da base.
- O **random_state=42** fixa o sorteio: rodar de novo dá a mesma divisão. Se o resultado mudar, foi por algo que você alterou.
- O arquivo `base_imagens/teste_fixo.txt` guarda as 80 sorteadas. Assim, quando uma melhoria acrescenta imagens, o teste continua o mesmo e as novas vão só para o bairro.

### 5. Treinar é montar o bairro

No Fuzzy k-NN, treinar não ajusta pesos nem calcula loss: é **montar a base de referência**. O EfficientNetB0 (pré-treinado, usado como extrator de características) transforma cada uma das 320 imagens em 1.280 números, que ficam guardados com os rótulos. Esse é o bairro. Ele só é montado de novo quando mudam os dados ou o extrator.

### 6. O grau fuzzy: o Fuzzy k-NN dá a pertinência

Quando chega uma imagem nova, o mesmo EfficientNetB0 transforma a imagem em 1.280 números, e o **Fuzzy k-NN** compara esses números com o bairro:

1. mede a distância da imagem nova até cada imagem do bairro;
2. pega as 5 mais próximas (as vizinhas);
3. dá a cada vizinha o peso 1/d²: quem está mais perto pesa mais;
4. calcula a pertinência: pesos das vizinhas cachorro ÷ soma de todos os pesos.

Pela vizinhança: o KNN comum só conta as casas vizinhas ("4 cachorros e 1 gato, então é cachorro"). O Fuzzy k-NN é o vizinho detalhista, que mede a distância de cada casa e responde em grau: "sua casa pertence 0,65 à rua dos cachorros e 0,35 à rua dos gatos".

Atenção: Pertinência diz o quanto a imagem pertence ao conjunto "cachorro";

### 7. As pertinências criam as faixas, e o meio é a área cinzenta

A classe é a de **maior pertinência**, decidida por um `if` comum. A pertinência a cachorro cria três situações:

| pertinência a cachorro | o que a tela mostra |
|---|---|
| abaixo de 0,40 | "Não é cachorro", com o grau, na caixa vermelha "❌ Imagem na área não é cachorro" |
| de 0,40 a 0,60 | a classe com o grau, na caixa amarela "⚠ Imagem na área cinzenta" |
| acima de 0,60 | "É cachorro", com o grau, na caixa verde "🐶 Imagem na área é cachorro" |

A caixa aparece sempre, nas três faixas: ela traz a regra da faixa, a linha "pertinência da imagem:" com as duas pertinências e a linha "parecida com:". Abaixo dela vêm as fotos das 5 vizinhas, também nas três faixas — são elas que explicam de onde saiu a pertinência, tanto numa resposta folgada quanto numa apertada. O que só a área cinzenta tem é o aviso de que a decisão foi apertada.

A área cinzenta não muda a classe: só marca que a decisão foi apertada, e as 5 vizinhas explicam a dúvida. 

### 8. A pertinência vira a mensagem na tela

A tela junta o binário e o fuzzy ao mesmo tempo. A mensagem principal segue a regra `<classe> · pertinência <valor>`: a classe é "É cachorro" ou "Não é cachorro", e o valor é a pertinência da classe vencedora calculada para aquela imagem, de 0 a 1, com duas casas decimais. Abaixo dela vem o bloco colorido da faixa, com a linha "pertinência da imagem:", que repete a vencedora e mostra a outra classe. Nenhuma das duas frases usa porcentagem: pertinência não é probabilidade.

Arredondamento: com duas casas, uma pertinência da vencedora pouco acima de 0,50 (abaixo de 0,505) aparece como 0,50 mesmo quando a classe é cachorro. A classe vem do valor completo, não do arredondado.

As imagens enviadas ficam no bloco "Imagens anteriores" só durante a sessão e nunca entram na base.

### 9. A matriz de confusão avalia o grau fuzzy (No teste)

As 80 imagens do teste fixo passam pelo mesmo Fuzzy k-NN, e a **matriz de confusão** compara as respostas com os rótulos guardados: VP, FN, FP e VN. Da matriz saem:

- a **acurácia**: (VP + VN) ÷ total. A meta é ≥ 90% (72 das 80), com FN ≤ 6 e FP ≤ 6;
- a pergunta **FP × FN: qual pesa?** O sistema erra mais chamando não cachorro de cachorro (FP) ou deixando cachorro passar (FN)? Isso diz onde olhar.

A área cinzenta também vira uma **galeria** no teste fixo, com as 4 situações (VP, FN, VN, FP) e as vizinhas de cada imagem. Os erros dela mostram as lacunas do bairro.

### 10. Se o modelo erra muito, a matriz pede a melhoria

A melhoria monta o bairro de novo (ou ajusta o código) e testa nas **mesmas 80 imagens**, uma mudança por vez:

- **dados**: acrescentar exemplos parecidos com os erros (ex.: lobos em não cachorro), sempre na mesma quantidade em cada classe e em cada base de origem, passando pela triagem;
- **extrator**: trocar o EfficientNetB0 pelo MobileNetV2;
- **código ou configuração**: corrigir um erro silencioso ou testar outro k.

Cada volta é uma rodada, registrada numa tabela. Lida de cima para baixo, a tabela mostra o que cada melhoria fez com os acertos e com cada tipo de erro. E o ciclo volta ao passo 5.

## Arquivos

| arquivo | o que faz | etapa do Fluxo completo |
|---|---|---|
| `config.py` | k, faixa da área cinzenta, meta, pastas e composição da base | — |
| `codigos_imagenet.py` | tabela de códigos do ImageNet (ex.: n02342885 = hamster) | 2 |
| `fontes.py` | lê os XML das duas bases e sorteia as fotos | 1 a 3 |
| `montar_base.py` | monta a base de 400 imagens ou acrescenta imagens numa melhoria | 1 a 3 |
| `triagem.py` | YOLO, relatório, decisões e critério de parada | 4 |
| `extrator.py` | imagem → 1.280 números (EfficientNetB0; MobileNetV2 na rodada de troca) | 5 e 8 |
| `montar_bairro.py` | extrai os números e separa o bairro (320) do teste fixo (80) | 5 e 6 |
| `dados.py` | metadados, distribuição, split 80/20, teste fixo e vetores | 3 e 6 |
| `fuzzy_knn.py` | `fuzzy_knn`, decisão pela maior pertinência, área cinzenta e mensagem | 9 e 10 |
| `avaliacao.py` e `avaliar.py` | matriz de confusão, acurácia, meta, galeria e rodadas | 12 |
| `app.py` | Streamlit: visão principal e visão admin | 7 a 11 |
| `tests/` | testes com números inventados | — |

Pastas criadas pelos scripts:

```
base_imagens/
  cachorro/                    rótulo 1
  nao_cachorro/                rótulo 0
  metadados.csv                arquivo, classe, dataset, categoria_original
  teste_fixo.txt               as 80 do teste fixo (gravado na primeira montagem do bairro)
  distribuicao_planejada.csv   quantas por classe, dataset e categoria (a triagem confere)
triagem/                       relatório e decisões; descartadas/ guarda as imagens substituídas
vetores/                       bairro.npz (320 imagens) e teste.npz (80)
resultados/                    rodadas.csv e rodada_N/ (resultado por imagem e cópias de FP e FN)
```

## 1. Instalar

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-triagem.txt     # só para a triagem: traz o PyTorch
```

Use um Python que o TensorFlow suporte (3.10 a 3.12 é o caminho mais seguro). Na primeira vez, o Keras baixa os pesos do EfficientNetB0 (~16 MB) e a Ultralytics baixa o `yolov8n.pt` (~6 MB). Depois, tudo roda localmente.

## 2. Baixar as bases de origem

Os dois datasets de origem:

- **ImageNet (validação)**: image-net.org, com cadastro. Arquivos `ILSVRC2012_img_val.tar` (as 50.000 fotos) e `ILSVRC2012_bbox_val_v3.tgz` (as anotações em XML).
- **Pascal VOC 2012**: host.robots.ox.ac.uk/pascal/VOC/voc2012, arquivo `VOCtrainval_11-May-2012.tar`.

Coloque assim (ou mude os caminhos em `config.py`):

```
bases_origem/
  imagenet/
    imagens/        ← as fotos .JPEG de ILSVRC2012_img_val.tar
    anotacoes/      ← os .xml de ILSVRC2012_bbox_val_v3.tgz (pasta val/)
  VOCdevkit/
    VOC2012/        ← de VOCtrainval_11-May-2012.tar (Annotations/ e JPEGImages/)
```

## 3. Parte 1 · construção da base (uma vez)

1. **Montar a base**: `python montar_base.py`
   Copia as 400 fotos para `base_imagens/` e grava os metadados.

   Se você clonou este repositório, use `python montar_base.py --remontar`. As imagens não entram no Git (o ImageNet e o Pascal VOC não permitem redistribuição), mas o `metadados.csv` entra: o `--remontar` copia das bases de origem exatamente as 400 que ele lista, sem sortear de novo. A base sai idêntica à que gerou os resultados registrados, com a substituição da triagem já aplicada — então você pode pular direto para o passo 3.

   | classe | categoria real | dataset de origem | imagens |
   |---|---|---|---|
   | cachorro | raças domésticas | ImageNet (validação) | 100 |
   | cachorro | cachorro | Pascal VOC | 100 |
   | não cachorro | outros animais (com hamster e canídeos selvagens) | ImageNet (validação) | 50 |
   | não cachorro | outros animais (pássaro, gato, vaca, cavalo, ovelha) | Pascal VOC | 50 |
   | não cachorro | objetos e cenários | ImageNet (validação) | 50 |
   | não cachorro | pessoas | Pascal VOC | 50 |

2. **Triagem com YOLO** (repita até aparecer "Triagem concluída"):
   - `python triagem.py` roda o YOLO e gera o relatório com os casos suspeitos e uma amostra das imagens não sinalizadas.
   - Abra `triagem/relatorio.html` para ver as fotos. Em `triagem/relatorio.csv`, escreva na coluna `decisao` de cada linha: `manter`, `corrigir` (o rótulo), `substituir` (mesma categoria e mesmo dataset) ou `revisar` (fica pendente). Nas linhas da amostra, `manter` quer dizer "conferi e está certa".
   - `python triagem.py --aplicar` aplica as decisões.
   - Se houve substituição, rode `python triagem.py` de novo: as imagens novas também passam pela triagem.

   A triagem termina quando não há imagens sem triagem nem casos pendentes, a distribuição planejada está preservada e a amostra foi conferida. Não é preciso zerar os alertas do YOLO: uma pessoa ao lado de um gato não é erro, e um lobo detectado como "dog" continua não cachorro.

3. **Montar o bairro**: `python montar_bairro.py`
   Extrai os 1.280 números de cada imagem. Na primeira vez, o split 80/20 (estratificado por classe + dataset de origem, `random_state=42`) sorteia as 80 do teste fixo e grava a lista em `base_imagens/teste_fixo.txt`; nas próximas, o código só lê essa lista e avisa se uma imagem do teste foi removida ou renomeada. Não apague esse arquivo: a prova mudaria, e as rodadas deixariam de ser comparáveis.

4. **Avaliar a rodada 1**: `python avaliar.py --rodada 1 --mudanca "ponto de partida: bairro com 320 (160 + 160)"`
   Mostra a matriz, a acurácia, a meta e a área cinzenta; registra a rodada em `resultados/rodadas.csv` e copia as imagens de FP e FN para `resultados/rodada_1/`.

## 4. Parte 2 · aplicação

`streamlit run app.py`. Na barra lateral, escolha a visão:

- **Visão principal**: envie uma imagem e veja a classe com o grau. Abaixo vem sempre a caixa da faixa em que a pertinência caiu — verde "🐶 Imagem na área é cachorro", amarela "⚠ Imagem na área cinzenta" ou vermelha "❌ Imagem na área não é cachorro" — com a regra da faixa, as duas pertinências e a contagem das vizinhas. Depois da caixa, também sempre, as fotos das 5 vizinhas com a classe de cada uma. O bloco "Imagens anteriores" guarda as imagens enviadas só durante a sessão: somem ao fechar ou recarregar a página e não entram na base.
- **Visão admin**: tabela das rodadas e, para a rodada escolhida, a matriz de confusão, a acurácia, a meta e a galeria da área cinzenta nas 4 situações (VP, FN, VN, FP), cada imagem com as suas 5 vizinhas.

Meta no teste fixo: acurácia ≥ 90% (72 das 80), com FN ≤ 6 e FP ≤ 6.

## 5. Rodadas de melhoria

Uma mudança por rodada (dados, uma configuração ou o código), sempre testada nas mesmas 80 imagens. O bairro só é montado de novo quando mudam os dados ou o extrator.

| rodada | o que muda | comandos |
|---|---|---|
| 2 · exemplo | + 80 imagens (20 de cada dataset em cada classe), após a triagem | `python montar_base.py --acrescentar 20` → triagem (passo 2) → `python montar_bairro.py` → `python avaliar.py --rodada 2 --mudanca "+ 80 imagens (20 de cada dataset em cada classe)"` |
| 3 · exemplo | troca do extrator: EfficientNetB0 → MobileNetV2 (com o `preprocess_input` dele) | `python montar_bairro.py --extrator mobilenetv2` → `python avaliar.py --rodada 3 --mudanca "MobileNetV2"` |
| 4 · exemplo | ajuste de código | mude o código → `python montar_bairro.py` (só se a mudança mexer na extração ou nos dados) → `python avaliar.py --rodada 4 --mudanca "o que mudou"` |
| teste do k | k = 5 × k = 9 (mesmo bairro) | `python avaliar.py --rodada N --mudanca "k = 9" --k 9` |

- Para direcionar o acréscimo ao que o modelo erra, use `--categorias` com códigos do ImageNet ou nomes do Pascal VOC: `python montar_base.py --acrescentar 20 --categorias n02114367,n02114548` sorteia os não cachorros do ImageNet só entre lobos; as outras partes seguem o sorteio normal, mantendo a simetria.
- O app usa sempre o extrator com que o bairro foi montado. Para voltar ao EfficientNetB0 depois da rodada 3, rode `python montar_bairro.py` sem `--extrator`.
- Para o app usar outro k, mude `K` em `config.py`.

## Onde cada requisito está

| requisito | onde |
|---|---|
| RF1 · upload | `app.py` (`visao_principal`): a imagem é analisada sozinha, sem remontar o bairro |
| RF2 · mesmo EfficientNetB0 | `extrator.py`: só redimensiona para 224 × 224, sem normalizar à mão; o app usa o extrator gravado com o bairro |
| RF3 · pertinências | `fuzzy_knn.py` (`fuzzy_knn`): 5 vizinhas, peso 1/d², as duas pertinências somam 1 |
| RF4 · decisão | `fuzzy_knn.py` (`decidir`): maior pertinência por um `if` comum; exatamente 0,50 → não cachorro |
| RF5 · resultado | `fuzzy_knn.py` (`mensagem`): classe + pertinência da classe vencedora |
| RF6 · área cinzenta | `fuzzy_knn.py` (`na_area_cinzenta`) e a caixa amarela do `app.py`: o aviso de que a decisão foi apertada |
| RF7 · imagens anteriores | `app.py`, em `st.session_state` |
| RF8 · avaliar no teste fixo | `avaliar.py` e `avaliacao.py`; visão admin |
| RF9 · rodadas | `resultados/rodadas.csv`; visão admin |
| RF10 · faixa da pertinência | `app.py` (`FAIXAS`, `faixa_do_resultado`, `texto_caixa`): toda imagem recebe a caixa da sua faixa — verde acima de 0,60, amarela de 0,40 a 0,60, vermelha abaixo de 0,40 — com a linha "pertinência da imagem:", a vencedora primeiro, e as fotos das 5 vizinhas abaixo |

## Escolhas que o PRD não fixa

Estão em `config.py` e `codigos_imagenet.py` e podem ser trocadas.

1. **Sorteio das fotos**: em rodízio entre as categorias, com semente fixa (42). Os 50 outros animais do Pascal VOC saem 10 de cada (pássaro, gato, vaca, cavalo, ovelha), e os 100 cachorros do ImageNet saem de raças diferentes.
2. **Categorias do ImageNet**: o PRD fixa o hamster e os canídeos selvagens (lobos, coiote, dingo, mabeco e raposas). As outras espécies e os objetos e cenários (25 categorias em cada grupo, 2 fotos por categoria) estão listados em `codigos_imagenet.py`.
3. **Regra de inclusão no Pascal VOC**: já no sorteio, a foto de cachorro precisa ter o cão como maior objeto anotado (e não marcado como difícil); a de não cachorro não pode ter nenhum cachorro anotado. A "categoria original" de uma foto do Pascal VOC é o maior objeto anotado.
4. **Triagem**: detecções do YOLO abaixo de 0,25 de confiança são ignoradas (padrão da Ultralytics); cachorro abaixo de 0,50 conta como detecção incerta. A amostra das não sinalizadas é 5% de cada lote triado.
5. **Imagem substituída** não é apagada: vai para `triagem/descartadas/` e não volta a ser sorteada.
6. **Acréscimo**: N fotos de cada dataset em cada classe; em não cachorro, divididas igualmente entre as duas categorias reais de cada dataset (ou só nas categorias de `--categorias`).
7. **Tabela de rodadas**: além das colunas do PRD (o que mudou, VP, FN, FP, VN e acurácia), guarda o extrator, o k, o tamanho do bairro, quantas imagens caíram na área cinzenta, se a meta foi atingida e a data.
8. **Imagem**: redimensionamento bilinear para 224 × 224, a mesma função para bairro, teste e upload, com correção da orientação de foto de celular. O upload aceita jpg, png e webp; foto HEIC do iPhone precisa ser exportada como JPG.
9. **Visão admin** sem senha, escolhida na barra lateral.
10. **`--ignorar-triagem`** em `montar_bairro.py` monta o bairro mesmo com a triagem incompleta. Existe só como saída de emergência; o PRD pede a triagem antes da extração.

## Testes

`python -m pytest tests`

Usam números inventados: não precisam das imagens, do TensorFlow nem do YOLO. Cobrem o `fuzzy_knn` (incluindo o exemplo de 0,65 do card Grau fuzzy), a decisão, a área cinzenta, a mensagem, o split estratificado, o teste fixo gravado, a leitura dos XML, o sorteio e a matriz.
