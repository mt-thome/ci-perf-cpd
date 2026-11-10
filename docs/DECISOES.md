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
- **Contexto**: Daly et al. (ICPE 2020) utilizam o algoritmo *E-Divisive with Means* (EDM) em produção na MongoDB via biblioteca `signal-processing-algorithms` (versão 2.1.6).
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

## ADR-004: Correção e Esclarecimento Terminológico: E-Divisive with Means vs E-Divisive with Medians

- **Data**: 2026-11-10
- **Status**: Aprovado
- **Contexto**: A literatura de Change Point Detection possui duas ramificações baseadas em estatísticas de energia com a sigla "EDM":
  1. **E-Divisive with Means (Matteson & James, 2014; Daly et al., ICPE 2020)**:
     - No artigo seminal de Daly et al. (ICPE 2020, Seção 3 e 4), o método implementado e avaliado na infraestrutura da MongoDB é o **E-Divisive with Means** (introduzido por Matteson & James, JASA 2014).
     - **Formulação Matemática Exata**:
       Para duas partições contíguas de dados $X = \{x_1, \dots, x_n\}$ e $Y = \{y_1, \dots, y_m\}$, a divergência de energia entre as amostras é dada por:
       $$\mathcal{E}(X, Y) = \frac{2}{nm} \sum_{i=1}^n \sum_{j=1}^m \|x_i - y_j\| - \frac{1}{n^2} \sum_{i=1}^n \sum_{i'=1}^n \|x_i - x_{i'}\| - \frac{1}{m^2} \sum_{j=1}^m \sum_{j'=1}^m \|y_j - y_{j'}\|$$
       A estatística de teste para a partição no ponto candidato $\tau$ (onde $n = \tau$ e $m = T - \tau$) pondera essa divergência pelo tamanho amostral relativo:
       $$Q(\tau) = \frac{n \cdot m}{n + m} \mathcal{E}(X, Y)$$
       O ponto de corte ótimo na iteração divisiva é:
       $$\hat{\tau} = \arg\max_\tau Q(\tau)$$
       A significância estatística é obtida por teste de permutação de Monte Carlo com $B$ permutações aleatórias, rejeitando a hipótese nula $H_0$ se $p = \frac{1 + \sum_{b=1}^B \mathbb{I}(Q_{perm, b}^* \ge Q_{obs}^*)}{B + 1} \le \alpha$.
  2. **E-Divisive with Medians (James, Kejariwal & Matteson, 2014 - Twitter BreakoutDetection)**:
     - Em sistemas com outliers severos ("spikes"), James et al. propuseram substituir as médias aritméticas das distâncias pareadas pelas suas **medianas**:
       $$m_{XY} = \text{median}\left\{ |x_i - y_j| : 1 \le i \le n, 1 \le j \le m \right\}$$
       $$m_{XX} = \text{median}\left\{ |x_i - x_{i'}| : 1 \le i < i' \le n \right\}, \quad m_{YY} = \text{median}\left\{ |y_j - y_{j'}| : 1 \le j < j' \le m \right\}$$
       $$\mathcal{E}_{med}(X, Y) = 2 m_{XY} - m_{XX} - m_{YY}$$
- **Decisão Metodológica**:
  - Implementamos **ambas as métricas de agregação** no detector E-Divisive (`EDivisiveDetector` com parâmetro `metric="means"` e `metric="medians"`).
  - Por padrão para o confronto com Daly et al. (2020), utilizaremos `metric="means"`, mas permitiremos alternar para `"medians"` na análise empírica do perfil com `outliers`, documentando rigorosamente a distinção.
  - Registramos a restrição de tamanho mínimo de segmento (`min_size`, padrão 30) para evitar partições degeneradas nas bordas da série.

---

## ADR-005: CUSUM Robusto com Janela Inicial, Período de Guarda e Ponto de Alarme vs Ponto de Início

- **Data**: 2026-11-10
- **Status**: Aprovado
- **Contexto**: O algoritmo CUSUM clássico requer parâmetros de referência basal $(\mu_0, \sigma_0)$. Em séries de CI sujeitas a regressões de desempenho, estimar a média e dispersão sobre a série temporal inteira contamina severamente os estimadores (uma regressão que dobra o tempo de execução inflaria a média global e o desvio padrão, mascarando a própria detecção).
- **Decisão**:
  1. **Janela Inicial de Calibração Local**: Os parâmetros de referência são estimados estritamente nas primeiras $w_{init}$ observações (padrão $w_{init} = 30$ commits):
     $$\hat{\mu}_0 = \text{mediana}(X_{1:w_{init}})$$
     $$\hat{\sigma}_0 = 1.4826 \times \text{MAD}(X_{1:w_{init}}) = 1.4826 \times \text{mediana}(|X_{1:w_{init}} - \hat{\mu}_0|)$$
     *Nunca* é utilizado o $\sigma$ paramétrico verdadeiro do gerador sintético. Em casos degenerados onde $\text{MAD} = 0$, adota-se $\sigma_{fallback} = \max(\text{std}(X_{1:w_{init}}), 10^{-6})$ para prevenir divisão por zero.
  2. **Reset Pós-Detecção**: Ao disparar um alarme no índice $t$, as somas acumuladas $S_t^+$ e $S_t^-$ são zeradas ($S_t^+ = S_t^- = 0$). O nível de referência $\hat{\mu}_0$ é opcionalmente atualizado para a mediana do novo regime após a transição.
  3. **Período de Guarda (`guard_window`)**: Após a detecção no commit $t$, o detector suprime novos alarmes nos commits $t+1, \dots, t + g$ (padrão $g = 15$ commits). Isso impede que a cauda de uma transição de nível contínua gere múltiplos alarmes redundantes para o mesmo evento de regressão.
  4. **Ponto de Alarme vs Ponto Estimado de Início**:
     - No CUSUM, a soma cumulativa acumula resíduos a partir de um instante inicial $\tau_{start} = \max\{s \le t : S_s = 0\} + 1$, disparando o alarme no instante $t_{alarm} \ge \tau_{start}$ quando $S_{t_{alarm}} > h\hat{\sigma}$.
     - O detector expõe ambos os valores: `change_points` registra o instante do alarme $t_{alarm}$ (compatível com pipelines de monitoramento em tempo real), enquanto `metadata["estimated_start_points"]` registra $\tau_{start}$.
     - As métricas de avaliação e matching são calculadas por padrão sobre o instante de alarme $t_{alarm}$, refletindo o momento real em que o pipeline de CI recebe o alerta.

---

## ADR-006: Calibração Pareada com Sementes Disjuntas e Desacoplamento da Tolerância Temporal

- **Data**: 2026-11-10
- **Status**: Aprovado
- **Decisão**:
  1. **Sem vazamento de dados (Data Leakage)**: A calibração de limiares ($h^*$ no CUSUM, penalidade $\beta^*$ no PELT/Binseg) para fixar $\text{FPR} \le 1\%$ é executada em séries sem regressão ($k_{true} = 0$) geradas com sementes estritamente disjuntas das sementes de teste do grid de avaliação.
  2. **Execução Pareada**: No grid fatorial, cada semente pseudo-aleatória $s \in \{1, \dots, R\}$ ($R=50$) gera uma instância exata de série temporal sintética que é avaliada identicamente por todos os detectores em paralelo. Intervalos de confiança de 95% serão calculados via Bootstrap empírico não-paramétrico.
  3. **Desacoplamento de $k_{tol}$**: Os detectores registram apenas os vetores brutos de índices de commits detectados. O matching temporal sob janela de tolerância $\pm k$ commits ($k \in \{1, 3, 5, 10\}$) é realizado inteiramente em etapa de pós-processamento, economizando custo computacional e eliminando reexecuções desnecessárias dos detectores.

---

## ADR-007: Dossiê de Fontes de Dados Reais de CI com Verificação de Fatos

- **Data**: 2026-11-10
- **Status**: Registrado com Status de Verificação Explícito
- **Classificação**:
  - `[VERIFICADO]`: Confirmado diretamente em repositório oficial, publicação acadêmica ou registro DOI.
  - `[HIPÓTESE / A VALIDAR EMPIRICAMENTE]`: Inferência plausível da literatura, mas que requer download ou inspeção prévia de commits reais antes da Fase 6.

### Tabela de Verificação:

1. **Opção A: Faster CPython / PyPerformance**
   - URL Oficial: `https://github.com/faster-cpython/benchmarking-public` e `https://github.com/python/pyperformance` `[VERIFICADO]`
   - Licença: MIT / PSF License `[VERIFICADO]`
   - Formato dos Dados: Arquivos `.json` de execuções do `pyperformance` por commit na branch principal `[VERIFICADO]`
   - Histórico por Commit: Sim, histórico cronológico nos branches e releases do CPython `[VERIFICADO]`
   - Ground Truth de Regressões: Parcial. A existência de issues com label `type-performance` é confirmada no CPython, mas o mapeamento exato de commit de regressão requer seleção manual na Fase 6 `[HIPÓTESE / A VALIDAR]`

2. **Opção B: MongoDB Evergreen Performance Dataset**
   - URL Oficial: Zenodo DOI `10.5281/zenodo.5138516` (Registro: "MongoDB Performance Test Result Dataset", por David Daly, 26 de Julho de 2021) `[VERIFICADO]`
   - Licença: Creative Commons Attribution 4.0 International (CC-BY 4.0) `[VERIFICADO]`
   - Formato dos Dados: Arquivo compactado `mongodb_performance_results.tgz` (~3.1 GB) contendo resultados brutos de testes do Evergreen `[VERIFICADO]`
   - Histórico por Commit: Sim, dados gerados pelo framework interno de CI `[VERIFICADO]`
   - "Centenas de Regressões Anotadas": **Correção**. O dataset de 3.1 GB contém os resultados brutos dos testes. Embora Daly et al. (2020) descrevam 383 testes e triagens na MongoDB, as anotações explícitas de ground truth humano de cada commit precisam ser conferidas dentro do arquivo extraído ou do pacote de replicação de Costa et al. (Zenodo `18889836`), não devendo ser presumidas como trivialmente rotuladas sem inspeção direta `[HIPÓTESE / A VALIDAR]`

3. **Opção C: Rust Compiler (`rustc-perf`)**
   - URL Oficial: `https://github.com/rust-lang/rustc-perf` e painel `https://perf.rust-lang.org` `[VERIFICADO]`
   - Licença: MIT / Apache 2.0 dual `[VERIFICADO]`
   - Formato dos Dados: Dados históricos disponíveis via API JSON pública do site e suporte a banco de dados relacional SQLite (`collector.sqlite`) em execuções locais de teste `[VERIFICADO]`
   - Histórico por Commit: Sim, mapeado commit a commit via automação do Bors `[VERIFICADO]`
   - Ground Truth de Regressões: Elevado. Os relatórios semanais de triagem no repositório `rust-lang/rust` e no Zulip documentam abertamente PRs causadores de regressões severas (ex: flags de reversão rápida) `[VERIFICADO]`

---

## ADR-008: Convenção de Indexação de Change Points (Início do Novo Regime)

- **Data**: 2026-11-10
- **Status**: Aprovado
- **Decisão**:
  - Adota-se como convenção matemática única que o índice de um change point $\tau$ representa o **primeiro ponto pertencente ao novo regime** ($x_\tau$ já é governado pela nova distribuição/nível).
  - No gerador sintético, a transição é aplicada em `signal[cp:]`.
  - Nos detectores E-Divisive, PELT e Binseg sobre degrau determinístico sem ruído, a detecção retorna $\tau = 100$ exato (sem *off-by-one*). O wrapper do Ruptures elimina sistematicamente o último breakpoint formal retornado pela biblioteca ($n$), garantindo indexação puramente interna.

---

## ADR-009: Taxonomia de Métricas em Séries Estacionárias ($k_{true}=0$) e Casos Degenerados

- **Data**: 2026-11-10
- **Status**: Aprovado
- **Decisão**:
  1. **Não utilizar o termo "FPR" em séries temporais de CI**:
     - Em detecção de pontos de mudança, classificar commits negativos verdadeiros ($TN$) não tem significado físico claro e distorce a métrica proporcionalmente a $n$.
     - Adota-se rigorosamente:
       a) **Proporção de Séries com Alarme Falso ($FAR_{series}$)**: Proporção de séries estacionárias que apresentaram $\ge 1$ detecção espúria:
          $$FAR_{series} = \frac{1}{R} \sum_{i=1}^R \mathbb{I}(N_{det, i} > 0)$$
       b) **Falsos Alarmes por 1000 Commits ($FAR_{1k}$)**: Frequência normalizada de alarmes por commit:
          $$FAR_{1k} = \frac{\sum_{i=1}^R N_{det, i}}{\sum_{i=1}^R n_i} \times 1000$$
  2. **Tratamento de Casos Degenerados**:
     - Sem detecções ($N_{det} = 0, N_{true} > 0$): $TP=0, FP=0, FN=N_{true}$, $\text{Precision}=\text{NaN}, \text{Recall}=0.0, F1=0.0$.
     - Sem ground truth com alarmes ($N_{det} > 0, N_{true} = 0$): $TP=0, FP=N_{det}, FN=0$, $\text{Precision}=0.0, \text{Recall}=\text{NaN}, F1=0.0$.
     - Sem ground truth e sem alarmes ($N_{det} = 0, N_{true} = 0$): $TP=0, FP=0, FN=0$, $\text{Precision}=\text{NaN}, \text{Recall}=\text{NaN}, F1=\text{NaN}$.
     - Na agregação Macro, valores $\text{NaN}$ são tratados explicitamente e o número de séries válidas é reportado. Séries puramente estacionárias **nunca** são misturadas com séries com regressão no cálculo de Precision, Recall e F1.
  3. **Matching Ótimo Bipartido com Custo Finito**:
     - Para viabilizar a resolução via `scipy.optimize.linear_sum_assignment` sem falhas numéricas causadas por infinitos, utiliza-se penalidade de desacoplamento finita $M > |G| \times (k_{max} + 1)$. A matriz de custo define $C_{i,j} = |g_i - d_j|$ se dentro da janela e $M$ caso contrário.
     - Essa formulação garante a maximização estrita de True Positives (1 match a custo $< M$ é sempre preferido a qualquer conjunto de emparelhamentos espúrios), minimizando subsidiariamente a distância temporal total.

---

## ADR-010: Comparação Estatística Pareada e Esquema de Persistência em Parquet

- **Data**: 2026-11-10
- **Status**: Aprovado
- **Decisão**:
  1. **Teste Pareado de Wilcoxon com Holm e Tamanho de Efeito**:
     - Para comparar dois detectores nas mesmas $R$ sementes, aplica-se o teste dos postos sinalizados de Wilcoxon sobre o F1 por série ($D_i = F1_{A, i} - F1_{B, i}$).
     - Descarta-se pares onde $D_i = \text{NaN}$. Os postos são atribuídos a $|D_i|$ para diferenças não nulas ($n_{diff} \le R$).
     - Calcula-se explicitamente a soma dos postos positivos $W^+$ e negativos $W^-$.
     - O tamanho de efeito rank-biserial é calculado como:
       $$r = \frac{W^+ - W^-}{\frac{n_{diff}(n_{diff}+1)}{2}}$$
     - Em comparações múltiplas entre detectores, aplica-se a correção sequencial de Bonferroni-Holm.
  2. **Bootstrap Não-Paramétrico de Séries**:
     - O bootstrap reamostra **séries completas** (com reposição), recalculando as métricas agregadas (Micro-F1) em cada iteração ($B=1000$), gerando o intervalo de confiança percentil a 95%.
  3. **Esquema de Dados Brutos em Parquet**:
     - As execuções brutas são armazenadas desacopladas de $k$ com as colunas: `cenario_id`, `split`, `perfil_ruido`, `params_ruido`, `n`, `magnitude`, `semente`, `detector`, `parametros`, `versao_codigo`, `ground_truth`, `indices_detectados`, `scores_detectados`, `tempo_cpu_ms`.
