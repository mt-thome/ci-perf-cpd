# Registro de Decisões Metodológicas e Arquiteturais (ADRs)

Este documento registra formalmente as decisões de projeto, metodologia de pesquisa, alternativas avaliadas e os motivos de descarte, garantindo transparência científica e reprodutibilidade no trabalho de Mineração de Dados (Pós-Graduação, UNESP).

---

## ADR-001: Estrutura do Repositório e Ambiente de Execução

- **Data**: 2026-10-20
- **Status**: Aprovado / Implementado
- **Contexto**: O projeto necessita de um pipeline modular para benchmark e comparação empírica de múltiplos algoritmos de detecção de pontos de mudança (CPD), capaz de rodar testes unitários, notebooks exploratórios e grids de experimentos controlados de forma determinística.
- **Decisão**: 
  - Adotar layout padrão de pacote Python (`src/cpd/` com subpacotes `generators`, `detectors`, `evaluation`, `experiments`, `benchmarks`).
  - Utilizar `pyproject.toml` com versões mínimas e `requirements.txt` com pinning estrito.
  - Fixação de seeds pseudo-aleatórias (via `numpy.random.Generator` com `PCG64`) em todas as etapas geradoras e experimentais.
- **Alternativas Descartadas**:
  - Script monolítico único: Descartado por inviabilizar testes unitários automatizados e reutilização em cenários combinatórios.
  - Notebooks como código principal: Descartado para garantir rastreabilidade, isolamento de escopo e execução via CLI determinística.

---

## ADR-002: Avaliação do Pacote `signal-processing-algorithms` (MongoDB) e Decisão de Implementação Própria do E-Divisive

- **Data**: 2026-10-20
- **Status**: Decisão Tomada (Implementação Própria do E-Divisive com Medianas)
- **Contexto**: Daly et al. (ICPE 2020) utilizam o algoritmo *E-Divisive with Medians* (EDM) em produção na MongoDB. A biblioteca oficial disponibilizada publicamente é `signal-processing-algorithms` (versão 2.1.6).
- **Avaliação Empírica do Pacote**:
  - A instalação via PyPI é concluída, porém ao importar o módulo `signal_processing_algorithms.energy_statistics`:
    ```
    OSError: no file with expected extension
    ImportError: cannot import name 'calculate_distance_matrix' from 'signal_processing_algorithms.energy_statistics.cext_calculator'
    ```
  - **Causa Raiz**: O módulo `cext_calculator.py` depende de uma biblioteca C compilada nativa (`_e_divisive.so`). As wheels publicadas no PyPI não distribuem binários pré-compilados compatíveis com distribuições Linux modernas e Python 3.12+, e o fallback em Python puro contém um bug de importação estrutural (as funções só são definidas no escopo interno do bloco `try`).
- **Decisão**:
  - **Implementar o E-Divisive com Medianas (EDM)** em Python puro e vetorizado com NumPy no módulo `src/cpd/detectors/edivisive.py`.
  - A nossa implementação seguirá rigorosamente a formulação matemática de Matteson & James (2014) e as adaptações de Daly et al. (ICPE 2020) com substituição de médias amostrais por medianas e distâncias euclidianas robustas.
  - Implementar validação e testes cruzados contra instâncias canônicas conhecidas para garantir corretude teórica.
- **Alternativas Descartadas**:
  - Tentar forçar a compilação C do pacote antigo da MongoDB via build manual: Descartado por adicionar dependências de compilação C frágeis, que prejudicam a portabilidade e a reprodutibilidade por outros avaliadores e pesquisadores.
  - Usar apenas implementações de E-Divisive em R via rpy2: Descartado por adicionar overhead de IPC e dependência externa de runtime R.

---

## ADR-003: Taxonomia e Modelagem de Ruído em Séries Sintéticas de CI

- **Data**: 2026-10-20
- **Status**: Aprovado / Implementado na Fase 1
- **Contexto**: Métricas de benchmark em pipelines de CI (ex.: tempo de execução de testes de ponta a ponta ou microbenchmarks) apresentam padrões de perturbação muito específicos, que diferem de ruído branco ideal.
- **Decisão**:
  O gerador sintético (`src/cpd/generators/synthetic.py`) deve suportar 5 perfis estocásticos fundamentais:
  1. `gaussian`: Ruído i.i.d. gaussiano $N(0, \sigma^2)$ (linha de base ideal).
  2. `ar1`: Ruído autocorrelacionado AR(1) ($e_t = \phi e_{t-1} + w_t$), modelando dependência temporal (ex.: aquecimento de JVM, caching progressivo, fragmentação de memória).
  3. `student_t`: Distribuição t-Student com graus de liberdade baixos ($\nu=3, 4$), modelando ruídos com caudas pesadas e flutuações com variância finita mas eventos extremos frequentes.
  4. `outliers`: Ruído base + impulsos pontuais esporádicos com probabilidade $p$ e magnitude $M$, modelando gargalos transitórios de I/O de rede ou garbage collection pauses (que NÃO são regressões permanentes).
  5. `bimodal`: Mistura de distribuições com duas médias distintas, simulando alocação heterogênea de nós de build em ambientes de nuvem elástica (ex.: instâncias com frequências de clock distintas ou estados de C-state/Turbo Boost variados).
- **Magnitude de Regressão**: Definida em termos de múltiplos de desvio-padrão equivalente ($\delta \in \{0.5\sigma, 1.0\sigma, 2.0\sigma, 5.0\sigma\}$) para desacoplar a escala do valor absoluto e permitir comparações padronizadas entre cenários.

