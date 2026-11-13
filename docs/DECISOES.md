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


---

## ADR-012: Rejeição da Aproximação por Subamostragem e Adoção do `E-Div-Medians` Exato via Numba (Árvore de Fenwick)

- **Data**: 2026-11-13
- **Status**: Aprovado / Implementado (Detector Oficial)
- **Contexto**: No commit `607088a`, o método `_calculate_t_stats_medians` de `EDivisiveDetector` / `EDivisiveMediansDetector` (`src/cpd/detectors/edivisive.py`) havia sido alterado para acelerar o cálculo de medianas em séries médias e longas por meio de uma **aproximação por subamostragem determinística** em dois níveis:
  1. **Subamostragem da grade temporal (`tau`)**: `step = 1` para $n \le 150$, `step = 2` para $150 < n \le 300$, e `step = 5` para $n > 300$ (além do ponto final $n - \text{min\_size}$). Candidatos intermediários recebiam $Q(\tau) = 0$.
  2. **Subamostragem determinística de pares (`np.linspace`)**: 300 pares intra-segmento (quando $>500$) e 500 pares inter-segmentos (quando $>1000$).
- **Justificativa Histórica — Tabela de Erros da Aproximação (passo único de $Q(\tau)$, `min_size=25`, `seed=42`)**:
  - `n100_gauss_cp50`: `cp_ex=50`, `cp_ap=50` (erro = 0 commits), erro relativo em $\max Q(\tau)$ = **0.20%**, speedup **1.0x** (`9.8 ms` vs `9.8 ms`).
  - `n300_gauss_cp150`: `cp_ex=150`, `cp_ap=155` (erro = 5 commits), erro relativo em $\max Q(\tau)$ = **9.26%**, speedup **4.8x** (`178.0 ms` vs `36.9 ms`).
  - `n500_outliers_cp250`: `cp_ex=247`, `cp_ap=240` (erro = 7 commits), erro relativo em $\max Q(\tau)$ = **2.04%**, speedup **19.9x** (`883.3 ms` vs `44.4 ms`).
  - `n500_student_t_cp250`: `cp_ex=250`, `cp_ap=270` (erro = 20 commits), erro relativo em $\max Q(\tau)$ = **3.25%**, speedup **19.9x** (`822.0 ms` vs `41.3 ms`).
  - `n1000_gauss_cp500`: `cp_ex=500`, `cp_ap=510` (erro = 10 commits), erro relativo em $\max Q(\tau)$ = **3.65%**, speedup **23.6x** (`7555.6 ms` vs `320.0 ms`).
- **Impacto na Hipótese H1**:
  - Como as tolerâncias de avaliação em `src/cpd/evaluation/metrics.py` são $k \in \{1, 3, 5, 10\}$ commits, o deslocamento de localização induzido pela subamostragem (ex.: 7 commits em `outliers` e 20 commits em `student_t` para $n=500$) comprometia diretamente Precision, Recall e F1-score nos perfis exatos onde H1 prevê superioridade do `E-Div-Medians` sobre o `E-Div-Means`.
- **Decisão**:
  - Adotar como **detector oficial** a implementação **100% exata** em $O(N^2 \log N)$ baseada em compressão de ranks e Árvore de Fenwick (BIT) compilada via `numba.njit(cache=True)` (`exact_medians: bool = True` por padrão).
  - Em testes unitários (`n=60, 90, 100`, incluindo `outliers` e `student_t`), a versão Numba BIT coincide bit-a-bit (`atol=1e-12`) com a referência ingênua exata tanto em $Q(\tau)$ quanto nos p-valores e change points das permutações.
  - Em benchmark limpo com $B=100, \alpha=0.20$, a versão exata Numba é **5,4× a 43× mais rápida** que a antiga aproximação (`31,4 ms` em `n=100`, `1,63 s` em `n=500`, `8,74 s` em `n=1000`), viabilizando o experimento completo sem aproximações.
  - Manter `exact_medians=False` apenas como opção legada/comparativa **não utilizada nos resultados principais**, registrando `"exact_medians"` nos metadados de `DetectionResult`.


---

## ADR-013: Saturação Bidirecional na Calibração (`saturado_inferior` e `saturado_superior`)

- **Data**: 2026-11-13
- **Status**: Aprovado
- **Decisão**: O manifesto de calibração registra explicitamente duas flags de saturação por `(detector, perfil, n, alvo)`:
  1. **`saturado_inferior`**: o `FAR_1k` realizado na calibração permanece **acima** do alvo mesmo no parâmetro mais restritivo da grade (`h=20.0` no CUSUM, `pen=1600/800` no PELT/Binseg, `alpha=0.01` — mínimo alcançável com $B=100$ permutações — no `E-Div`).
     - *Consequência para comparação a FAR igualado*: o detector opera com excesso de falsos alarmes em relação ao alvo (vantagem artificial de Recall e penalização de Precision); comparações diretas contra detectores que atingiram o alvo devem sinalizar explicitamente essa violação do teto de FAR.
  2. **`saturado_superior`**: o `FAR_1k` realizado na calibração permanece **abaixo** do alvo mesmo no parâmetro mais permissivo da grade (`h=4.0` no CUSUM, `pen=50/30` no PELT/Binseg, `alpha=0.20` base no `E-Div`).
     - *Consequência para comparação a FAR igualado*: o detector opera de forma mais conservadora que o orçamento de falsos alarmes permitido (menor Recall potencial em desvios sutis), não conseguindo "gastar" a cota de FAR até o alvo.

---

## ADR-014: Tamanhos de Calibração por `n`, Intervalos de Confiança de Poisson e Critério de Validação

- **Data**: 2026-11-13
- **Status**: Aprovado
- **Decisão**:
  1. **Tamanhos por `n`** (disjuntos das sementes de avaliação `1..50`):
     - $n = 100$: **2.000** séries de calibração (`10000..11999`) e **1.000** séries de validação (`20000..20999`) por perfil (exposição de $200\text{k}$ e $100\text{k}$ commits, garantindo $\approx 40$ e $\approx 20$ alarmes esperados no alvo $FAR_{1k}=0{,}2$).
     - $n \in \{500, 1000\}$: **200** séries de calibração (`10000..10199`) e **100** séries de validação (`20000..20099`) por perfil.
  2. **IC de Poisson de 95%**: Todas as tabelas de FAR reportam o número observado e esperado de alarmes e o intervalo de confiança exato de Poisson de 95% (via distribuição $\chi^2$).
  3. **Critério de Validação**: A calibração de uma célula é aceita se o **limite inferior do IC de 95% do `FAR_1k` realizado na validação não excede o alvo** ($\text{IC}_{95\%,\text{inf}} \le \text{alvo}$). Em caso de falha, registra-se `validacao_falhou: true` e **nunca** se recalibra usando o conjunto de validação.

---

## ADR-015: Padronização de `jump=1` no `ruptures`

- **Data**: 2026-11-13
- **Status**: Aprovado
- **Decisão**: Todas as chamadas a `ruptures.Pelt` e `ruptures.Binseg` — tanto na calibração (`calibrate_penalty_for_fpr` e `calibration.py`) quanto na avaliação (`RupturesDetector`) — devem especificar explicitamente `jump=1` (evitando o padrão `jump=5` da biblioteca), garantindo resolução exata de localização em commits e coerência entre calibração e avaliação.
