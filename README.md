# Detecção de Sonolência em Motoristas — Sistemas Embarcados

Sistema de visão computacional que monitora o motorista de um caminhão por uma
câmera montada no **espelho retrovisor interno** e dispara um **alarme sonoro
crescente** quando detecta sinais de sono/fadiga (olhos fechados ou
semicerrados por tempo prolongado, bocejos).

É a evolução do projeto de *eye tracking* para estudos: lá a câmera ficava de
frente para o usuário e media **para onde** ele olhava (atenção/foco). Aqui o
objetivo é diferente — detectar **sonolência**, usando as métricas padrão da
indústria automotiva de fadiga: **EAR** e **PERCLOS**.

## Como funciona

| Etapa | Técnica |
|-------|---------|
| Detecção facial | MediaPipe FaceLandmarker (478 pontos + blendshapes + pose) |
| Abertura dos olhos | **EAR** — Eye Aspect Ratio (6 pontos por olho) |
| Bocejo | **MAR** — Mouth Aspect Ratio |
| Fadiga acumulada | **PERCLOS** — % de fecho ocular numa janela de 60 s |
| Microssono | olhos fechados continuamente ≥ 2 s (evento contado) |
| Piscadas | **taxa de piscadas/min** e duração média (indicadores de fadiga) |
| Atenção / distração | **head pose** (yaw): detecta "olhos fora da via" |
| Decisão | Pontuação de sonolência 0–100 que **sobe** com olho fechado/semicerrado/distração e **desce** quando volta a abrir |
| Atuação | Alarme sonoro que **escala** em 3 níveis + latência medida |

### Calibração automática (3 s)
No início (e a qualquer momento com a tecla `c`), o sistema mede o EAR de
"olho aberto" do próprio motorista. Todos os limiares passam a ser **relativos**
a esse valor. É isso que torna o sistema robusto ao **ângulo do retrovisor** e
às diferenças entre pessoas — não há um número fixo "mágico" de EAR.

### Pontuação e níveis de alerta
A pontuação evolui no tempo (exatamente o "ir aumentando" pedido):

- Olho **fechado** → sobe rápido (+45/s)
- Olho **semicerrado** → sobe devagar (+18/s)
- Olho **aberto** → desce (−25/s)
- **Bocejo** → +8 instantâneo

| Pontuação | Nível | Som |
|-----------|-------|-----|
| 0–29  | NORMAL  | silêncio |
| 30–59 | ATENÇÃO | beep suave, espaçado (880 Hz) |
| 60–84 | ALERTA  | beep agudo e frequente (1320 Hz) |
| 85–100| PERIGO  | alarme quase contínuo e estridente (2000 Hz) + moldura vermelha |

## Instalação

Cada máquina/SO deve criar seu **próprio ambiente virtual** (a pasta `.venv` não
é versionada — veja [Solução de problemas](#solução-de-problemas) se você
encontrar referências a uma `.venv` antiga).

```powershell
python -m venv .venv
.venv\Scripts\activate          # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

Compatível com Python 3.10 a 3.13.

## Execução

Há duas formas de rodar:

### 1) Dashboard web (recomendado para a apresentação)

```powershell
python dashboard.py            # câmera 0, porta 5000
python dashboard.py 1          # outra câmera (ex.: webcam USB no retrovisor)
python dashboard.py 0 5001     # outra porta (se a 5000 já estiver em uso)
```

Depois abra **http://127.0.0.1:5000** no navegador. Mostra, ao vivo:
vídeo com os olhos marcados, pontuação de sonolência com medidor e **gráfico
histórico**, nível de alerta, EAR, PERCLOS, FPS, nº de bocejos e tempo de
sessão — além de botões de **Recalibrar** e **ligar/desligar som**.

> Tecnologia: Flask puro (vídeo via stream MJPEG + métricas via polling JSON).
> O gráfico é desenhado em canvas vanilla, **sem CDN** — funciona offline.

### 2) Janela standalone (OpenCV)

```powershell
python drowsiness_detector.py        # usa a câmera 0 (padrão)
python drowsiness_detector.py 1      # usa outra câmera (ex.: webcam no retrovisor)
```

### Controles
| Tecla | Ação |
|-------|------|
| `q` / `ESC` | sair |
| `c` | recalibrar |
| `m` | liga/desliga o som |

## Posicionamento da câmera

A câmera vai junto ao espelho retrovisor interno, apontada para o rosto do
motorista (ver imagens de referência). A calibração automática compensa o
ângulo. Para ângulos muito extremos, mantenha o rosto razoavelmente enquadrado
e refaça a calibração com `c`.

## Ajuste fino

Os parâmetros de comportamento ficam no topo de `drowsiness_detector.py`
(seção 2): limiares de olho, velocidade da pontuação, limiares dos níveis e
janela do PERCLOS. Ajuste conforme os testes em campo.

## Arquivos do projeto

| Arquivo | Função |
|---------|--------|
| `dashboard.py` | dashboard web em tempo real (apresentação) |
| `templates/index.html` | interface do dashboard |
| `drowsiness_detector.py` | detector standalone + toda a lógica (EAR/PERCLOS/alarme/CSV) |
| `evaluate.py` | avaliação quantitativa: compara EAR × blendshape, gera figuras/tabela |
| `artigo/main.tex` | rascunho do artigo (modelo SBrT, em inglês) |
| `face_landmarker.task` | modelo do MediaPipe (3,6 MB, já incluído) |
| `logs/` | CSV das sessões gravadas (gerado em runtime) |
| `figures/` | figuras e tabela LaTeX geradas pelo `evaluate.py` |
| `requirements.txt` | dependências (MediaPipe + Flask + matplotlib) |
| `README.md` | este documento |

## Customizações sobre o software padrão (Atividade 3)

**Software padrão utilizado:** [MediaPipe](https://ai.google.dev/edge/mediapipe)
(Google) — biblioteca de visão computacional que fornece o modelo
*Face Landmarker* (478 pontos faciais + *blendshapes*). Base original do grupo:
o projeto de *eye tracking* para estudos (EyeTrax + Kalman + MediaPipe).

**Customizações/esforços de desenvolvimento do grupo:**

1. **Mudança de objetivo:** de *gaze tracking* (para onde o usuário olha) para
   **detecção de sonolência** (quão fechados os olhos estão ao longo do tempo).
2. **Reposicionamento da câmera:** adaptação da câmera frontal para o
   **espelho retrovisor** (visão de ângulo), com **calibração automática de 3s**
   que torna os limiares relativos ao olho aberto de cada motorista — compensando
   o ângulo e diferenças entre pessoas.
3. **Métricas de fadiga:** implementação de **EAR**, **PERCLOS** (janela de 60s)
   e **MAR** (bocejo), em vez do índice de atenção (IAF) do projeto original.
4. **Pontuação de sonolência + alarme escalonado:** score contínuo 0–100 que
   sobe/desce no tempo e um **alarme sonoro de 3 níveis** que escala.
5. **Migração para a Tasks API:** o build do MediaPipe para Python 3.13 não tem a
   API antiga `solutions.FaceMesh`; reescrevemos com `FaceLandmarker`.
6. **Correção de carregamento com caminho acentuado:** o loader nativo não abre
   pastas com acento no Windows; carregamos o modelo via `model_asset_buffer`.
7. **Dashboard IoT próprio:** servidor Flask (MJPEG + métricas JSON) com gráfico
   histórico em canvas, no lugar do dashboard do projeto de estudos.
8. **Comparação de 2 métodos + instrumentação:** registro em CSV com *ground
   truth* e o `evaluate.py`, comparando **EAR (geométrico)** vs **blendshape
   eyeBlink (ML)** com matriz de confusão e métricas (Atividade 4).
9. **Monitoramento de atenção (distração):** estimativa de **head pose** (yaw)
   pela matriz de transformação facial, com linha de base na calibração, para
   detectar "olhos fora da via" — cobre o *Attention Monitoring* do título.
10. **Métricas de fadiga adicionais:** detecção de **microssono** (olhos fechados
    ≥2s), **taxa de piscadas/min**, duração média de piscada e **latência do
    alerta** — números objetivos para a seção de resultados do artigo.

> Repositório: inserir a URL aqui e adicionar o professor **rigelfernandes** como
> membro (Atividade 1). A mesma URL deve entrar como referência no artigo.

## Coleta de dados e avaliação quantitativa

### Modo automático (recomendado) — um comando só

```powershell
python coletar_e_avaliar.py        # ou: python coletar_e_avaliar.py 1
```

Conduz o protocolo sozinho (fases cronometradas alternando **OLHOS ABERTOS** e
**FECHE OS OLHOS**, marcando o *ground truth* automaticamente), grava o CSV, e ao
final **roda o `evaluate.py` sozinho** + imprime **FPS médio** e **latência média
do alarme**. Gera tudo de uma vez: tabela, figuras, FPS e latência. Basta seguir
as instruções na tela (~2 min). Tecla `q` aborta (mesmo assim avalia o coletado).

### Modo manual

Para gerar os números do artigo:

1. Rode o dashboard (ou o standalone) e deixe **calibrar**.
2. Ligue **Gravar CSV** (botão no dashboard, ou tecla `r` no standalone).
3. Alterne o **Rótulo** (botão, ou tecla `g`): mantenha em *sonolento* **enquanto
   fecha os olhos** e em *alerta* com os olhos abertos. Faça isso por ~1–2 min.
4. Pare a gravação. O CSV fica em `logs/`.
5. Rode a avaliação:

   ```powershell
   python evaluate.py            # usa o CSV mais recente
   ```

   Gera em `figures/`: matrizes de confusão, gráfico de métricas, varredura de
   limiar (PNG **e** PDF) e `metrics_table.tex` (tabela pronta para o LaTeX).

O rascunho do artigo (modelo SBrT, em inglês) está em [`artigo/main.tex`](artigo/main.tex).

## Solução de problemas

- **`.venv` não funciona / aponta para outro usuário ou Python**: a pasta
  `.venv` é local de cada máquina e não deve ser versionada. Apague a pasta
  `.venv` e recrie com `python -m venv .venv` (veja [Instalação](#instalação)).
- **"Address already in use" / porta 5000 ocupada**: já existe um
  `dashboard.py` rodando (nesta ou em outra janela). Acesse
  `http://127.0.0.1:5000` direto, feche a instância antiga, ou rode com outra
  porta: `python dashboard.py 0 5001`.
- **"ERRO: SEM CAMERA" / câmera não abre**: outro processo (outra instância do
  dashboard, Teams, etc.) pode estar usando a câmera. Feche-o e recarregue a
  página, ou indique outro índice de câmera (`python dashboard.py 1`).

## Observações

- Usa a **Tasks API** do MediaPipe (`FaceLandmarker`) — não a antiga `solutions`,
  que não existe no build para Python 3.13. O modelo `.task` já vem no projeto.
- O modelo é carregado em **bytes** (`model_asset_buffer`), e não por caminho,
  porque o loader nativo do MediaPipe não abre pastas com **acentos** no Windows
  (e esta pasta tem: *Duty Cosméticos / Área de Trabalho*).
- O alarme usa `winsound` (nativo do Windows). Em outro SO, vira alerta visual
  + bell do terminal.
- Para um produto embarcado real (ex.: Raspberry Pi), trocar o `winsound` por
  um buzzer GPIO e o `cv2.imshow` por log/telemetria.
