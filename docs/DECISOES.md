# Registro de Decisões Metodológicas e Arquiteturais (ADRs)

Este documento registra formalmente as decisões de projeto, metodologia de pesquisa, formulações matemáticas rigorosas, alternativas avaliadas e os motivos de descarte, garantindo transparência científica e reprodutibilidade no trabalho de Mineração de Dados (Pós-Graduação, UNESP).

---

## ADR-001: Estrutura do Repositório e Ambiente de Execução

- **Data**: 2026-10-20 (Atualizado em 2026-11-10)
- **Status**: Aprovado / Implementado
- **Decisão**: 
  - Adotar layout de pacote Python modular (`src/cpd/` com subpacotes `generators`, `detectors`, `evaluation`, `experiments`, `benchmarks`).
  - Utilizar `pyproject.toml` e `requirements.txt` com pinning estrito.
  - Fixação de seeds pseudo-aleatórias (via `numpy.random.Generator` com `PCG64`) em todas as etapas geradoras e experimentais.

---

## ADR-002: Avaliação do Pacote `signal-processing-algorithms` (MongoDB) e Implementação Própria

- **Data**: 2026-10-20
- **Status**: Decisão Tomada (Implementação Própria do E-Divisive e CUSUM)
- **Contexto**: Daly et al. (ICPE 2020) utilizam o algoritmo *E-Divisive with Means* em produção na MongoDB via biblioteca `signal-processing-algorithms` (versão 2.1.6).
- **Avaliação Empírica**: Falha crítica de importação no Python 3.12+ decorrente da dependência de extensão C não empacotada (`_e_divisive.so`) e bugs no fallback em Python puro.
- **Decisão**: Implementar os detectores de forma nativa e vetorizada com NumPy.

---

## ADR-003: Taxonomia e Modelagem de Ruído em Séries Sintéticas de CI

- **Data**: 2026-10-20
- **Status**: Aprovado / Implementado na Fase 1
- **Decisão**: O gerador sintético (`src/cpd/generators/synthetic.py`) suporta 5 perfis:
  1. `gaussian`: $N(0, \sigma^2)$ (baseline ideal).
  2. `ar1`: $e_t = \phi e_{t-1} + \sqrt{1 - \phi^2} \sigma w_t$, mantendo $\text{Var}(e_t) = \sigma^2$.
  3. `student_t`: $t_\nu$ reescalonado com $\nu > 2$ para $\text{Var}(e_t) = \sigma^2$.
  4. `outliers`: ruído base + impulsos com probabilidade $p$ e magnitude $M\sigma$.
  5. `bimodal`: mistura bimodal de hardware heterogêneo de runners de CI.

---

## ADR-004: Correção Rigorosa de Atribuição: E-Div-Means (Daly et al. 2020) vs E-Div-Medians (James et al. 2014)

- **Data**: 2026-11-10
- **Status**: Aprovado / Retificado
- **Contexto e Retificação Histórica**:
  - **Correção da Proposta Original**: A proposta inicial do trabalho atribuía erroneamente a variante "E-Divisive with Medians" ao artigo de Daly et al. (ICPE 2020, arXiv:2003.00584). Uma inspeção rigorosa do artigo e do código-fonte da MongoDB confirma que Daly et al. utilizam o **E-Divisive with Means** (introduzido originariamente por Matteson & James, JASA 2014), no qual a estatística de divergência de energia é fundamentada na média amostral e na hipótese de primeiro momento finito. No sistema da MongoDB, as medianas são calculadas estritamente a posteriori como estatísticas descritivas para exibição nos dashboards de engenharia e para estimativa do Hazard Level.
  - **Variante com Medianas (E-Div-Medians)**: Trata-se de uma proposta metodológica distinta, publicada por **Nicholas A. James, Arun Kejariwal e David S. Matteson** no artigo *"Leveraging Cloud Data to Mitigate User Experience from 'Breaking Bad'"* (arXiv:1411.7955, 2014; apresentado no IEEE Big Data, 2016), e implementada no pacote Twitter `BreakoutDetection`.
  - **Nomenclatura Padronizada no Projeto**:
    1. `E-Div-Means`: Implementação do algoritmo de Matteson & James (2014) / Daly et al. (2020).
    2. `E-Div-Medians`: Variante robusta com medianas de James, Kejariwal & Matteson (arXiv 2014 / IEEE 2016).
    *A sigla isolada EDM é expressamente evitada para prevenir ambiguidades.*
- **Formulação Matemática Exata**:
  1. **E-Div-Means**:
     $$\mathcal{E}(X, Y) = \frac{2}{nm} \sum_{i=1}^n \sum_{j=1}^m \|x_i - y_j\| - \frac{1}{n^2} \sum_{i=1}^n \sum_{i'=1}^n \|x_i - x_{i'}\| - \frac{1}{m^2} \sum_{j=1}^m \sum_{j'=1}^m \|y_j - y_{j'}\|$$
     $$Q(\tau) = \frac{n \cdot m}{n + m} \mathcal{E}(X, Y)$$
  2. **E-Div-Medians**:
     $$m_{XY} = \text{median}\left\{ |x_i - y_j| : 1 \le i \le n, 1 \le j \le m \right\}$$
     $$m_{XX} = \text{median}\left\{ |x_i - x_{i'}| : 1 \le i < i' \le n \right\}, \quad m_{YY} = \text{median}\left\{ |y_j - y_{j'}| : 1 \le j < j' \le m \right\}$$
     $$\mathcal{E}_{med}(X, Y) = 2 m_{XY} - m_{XX} - m_{YY}$$
- **Hipótese de Pesquisa Registrada**:
  - *Hipótese H1*: O algoritmo `E-Div-Medians` exibirá desempenho superior em Precision e F1-score comparado ao `E-Div-Means` em cenários com contaminação por outliers esporádicos e distribuições de cauda pesada (t-Student), enquanto o `E-Div-Means` terá maior sensibilidade e poder estatístico em desvios sutis sob ruído Gaussiano puro.

---

## ADR-005: CUSUM Robusto com Janela Inicial, Período de Guarda e Reestimação de Baseline

- **Data**: 2026-11-10
- **Status**: Aprovado
- **Decisão**:
  1. **Janela Inicial**: $\hat{\mu}_0 = \text{mediana}(X_{1:w_{init}})$ e $\hat{\sigma}_0 = 1.4826 \times \text{MAD}(X_{1:w_{init}})$ com $w_{init} = 30$. Em caso de $\text{MAD} = 0$, adota-se $\sigma_{fallback} = \max(\text{std}(X_{1:w_{init}}), 10^{-6})$.
  2. **Reset e Reestimação**: Ao disparar um alarme no índice $t$, o CUSUM zera as somas acumuladas ($S_t^+ = S_t^- = 0$) e reestima a média basal $\hat{\mu}_0$ utilizando as observações do novo regime subsequente ao alarme ($X_{t:t+w_{init}}$), impedindo que o nível antigo continue acumulando deriva.
  3. **Guarda Temporal**: Alarmes subsequentes são suprimidos por $g = 15$ commits.
  4. **Alarme vs Início**: As detecções primárias são reportadas pelo instante do alarme $t_{alarm}$, enquanto o início estimado $\tau_{start} = \max\{s \le t : S_s = 0\} + 1$ é preservado nos metadados.

---

## ADR-006: Protocolo de Calibração com Sementes Disjuntas

- **Data**: 2026-11-10
- **Status**: Aprovado
- **Decisão**:
  - Calibração realizada em 200 séries estacionárias independentes geradas com a faixa de sementes 10.000 a 10.200 (estritamente disjunta das sementes de avaliação 1 a 50).
  - Alvo primário: $FAR_{1k} \le 1.0$ alarme por 1000 commits.

---

## ADR-007: Dossiê de Fontes de Dados Reais de CI com Verificação de Fatos

- **Data**: 2026-11-10
- **Status**: Registrado com Status de Verificação Explícito (vide ADR-007 anterior).

---

## ADR-008: Convenção Unificada de Indexação de Change Points

- **Data**: 2026-11-10
- **Status**: Aprovado
- **Decisão**: O índice $\tau$ é o primeiro ponto pertencente ao novo regime. O Ruptures filtra o breakpoint final $n$.

---

## ADR-009: Taxonomia de Falsos Alarmes e Matching Bipartido com Custo Finito

- **Data**: 2026-11-10
- **Status**: Aprovado
- **Decisão**: Emprego de $FAR_{series}$ e $FAR_{1k}$. Resolução de matching 1-para-1 ótimo via `linear_sum_assignment` com penalidade finita $M > |G| \cdot (k_{max} + 1)$ para maximizar True Positives antes de minimizar a distância temporal.

---

## ADR-010: Comparação Estatística Pareada e Esquema Parquet

- **Data**: 2026-11-10
- **Status**: Aprovado
- **Decisão**: Teste pareado de Wilcoxon com rank-biserial $r$, bootstrap de séries completas ($B=1000$) e armazenamento particionado em Parquet.

---

## ADR-011: Regra de Posição de Change Points e Decisão sobre $k_{true}=2$ em $n=100$

- **Data**: 2026-11-10
- **Status**: Aprovado
- **Contexto**: Para uma série de tamanho $n=100$, a exigência de margem de estabilidade inicial ($w_{init} \ge 30$, margem de segurança $\ge 40$), margem de borda final ($\ge 25$) e distância mínima entre regimes ($\ge 30$) torna a existência de dois change points ($k_{true}=2$) matematicamente hiper-restringida:
  - Ponto 1: mínimo $t = 40$.
  - Distância mínima: 30 $\implies$ Ponto 2: mínimo $t = 70$.
  - Margem final: $n - 25 = 75$.
  - Isso restringe o Ponto 2 ao intervalo minúsculo $[70, 75]$ (apenas 6 posições possíveis), eliminando qualquer variabilidade estocástica do benchmark.
- **Decisão Metodológica**:
  1. Para $n=100$, adota-se apenas $k_{true} \in \{0, 1\}$.
  2. Para $n \in \{500, 1000\}$, adota-se $k_{true} \in \{0, 1, 2\}$.
  3. A direção de cada transição de nível é sorteada aleatoriamente por semente (regressões positivas e otimizações negativas com probabilidade 0.5 cada).
  4. As sementes dos cenários são derivadas estritamente via `hashlib.sha256(f"{cenario_id}_{semente}".encode())`, imunes ao processo Python ou sistema operacional.
