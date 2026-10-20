# Detecção de Regressões de Desempenho em Pipelines de Integração Contínua via Mineração de Séries Temporais

Projeto acadêmico para o curso de **Mineração de Dados** (Pós-Graduação, UNESP).

## 1. Visão Geral e Contexto Científico

Em pipelines de Integração Contínua (CI), benchmarks de desempenho são executados repetidamente a cada commit para assegurar que mudanças no código-fonte não introduzam lentidão de execução ou aumento anômalo de consumo de recursos.

Dada uma série temporal univariada $X = (x_1, x_2, \dots, x_n)$ representando uma métrica de desempenho (ex.: latência média de requisição ou tempo de execução de um microbenchmark) ao longo de $n$ commits sucessivos, o desafio consiste em **detectar automaticamente os *change points* (mudanças de nível de média/mediana)** que caracterizam regressões reais de desempenho, diferenciando-os de ruído de medição, oscilações de infraestrutura e outliers isolados.

Este projeto compara empiricamente algoritmos de detecção de pontos de mudança (*Change Point Detection* - CPD) em termos de:
- **Precisão, Recall e F1-Score** com tolerância temporal configurável ($\pm k$ commits).
- **Taxa de Falsos Positivos (FPR / False Alarms)** em séries estritamente estacionárias (sem regressões).
- **Custo Computacional**: Complexidade de tempo e memória em função de $n$.
- **Robustez contra perfis de ruído**: Gaussiano, Autocorrelacionado AR(1), Caudas Pesadas (t-Student), Outliers esporádicos e Bimodalidade (heterogeneidade de runners).

### Referências Teóricas Centrais
1. **Daly et al. (ICPE 2020)**: *"The Use of Change Point Detection to Identify Software Performance Regressions in a Continuous Integration System"*, arXiv:2003.00584. Introduz o algoritmo *E-Divisive with Medians* (EDM) em larga escala na MongoDB.
2. **Apache Otava (2025)**: *"8 Years of Optimizing Apache Otava"*, arXiv:2505.06758. Evolução algorítmica de detecção de pontos de mudança de complexidade $O(n^3)$ para tempo amortizado constante por nova observação em streaming.
3. **Matteson & James (2014)**: *"A Nonparametric Approach to Change Point Detection with Manifold Applications"*, JASA. Base estatística do método E-Divisive.
4. **Truong et al. (2020)**: *"Selective review of offline change point detection methods"*, Signal Processing (biblioteca `ruptures`).

---

## 2. Estrutura do Repositório

```text
ci-perf-cpd/
├── docs/
│   └── DECISOES.md           # Architecture Decision Records (ADRs) metodológicos
├── data/
│   ├── raw/                  # Datasets brutos (séries reais de projetos open-source)
│   └── processed/            # Séries tratadas e preparadas para benchmarks
├── notebooks/
│   └── 01_synthetic_generator_exploration.ipynb # Demonstração e visualização da Fase 1
├── reports/                  # Relatórios gerados, figuras e tabelas finais
├── results/                  # Métricas e logs experimentais estruturados (JSON/CSV)
├── src/
│   └── cpd/
│       ├── __init__.py
│       ├── generators/       # Geradores de séries sintéticas e simuladores de ruído
│       │   ├── __init__.py
│       │   └── synthetic.py
│       ├── detectors/        # Algoritmos de CPD (CUSUM, E-Divisive, Baselines)
│       │   ├── __init__.py
│       │   ├── base.py       # Interface comum Detector: detect(series) -> list[int]
│       │   ├── cusum.py
│       │   ├── edivisive.py
│       │   └── baseline.py
│       ├── evaluation/       # Matching com janela de tolerância e cálculo de métricas
│       │   ├── __init__.py
│       │   └── metrics.py
│       ├── experiments/      # Runners de grade fatorial e orquestração de cenários
│       │   ├── __init__.py
│       │   └── runner.py
│       └── benchmarks/       # Profiling de memória e tempo de execução
│           ├── __init__.py
│           └── profiling.py
├── tests/                    # Bateria de testes unitários automatizados (pytest)
│   └── test_generators.py
├── pyproject.toml            # Metadados do pacote Python e dependências
├── requirements.txt          # Versões exatas fixadas para reprodutibilidade estrita
└── README.md
```

---

## 3. Instalação e Ambiente

O projeto requer **Python 3.11+** (recomendado Python 3.12).

### Opção 1: Usando `uv` (Recomendado - Mais rápido)
```bash
# Criar o ambiente virtual com Python 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate

# Instalar dependências fixadas e o pacote em modo editável
uv pip install -r requirements.txt
uv pip install -e .
```

### Opção 2: Usando `venv` e `pip` padrão
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

---

## 4. Executando os Testes e Verificações

Para validar a integridade dos módulos implementados:

```bash
# Executar a suíte de testes com pytest
pytest tests/ -v
```

---

## 5. Metodologia de Reprodutibilidade

Todas as gerações de séries sintéticas, perturbações estocásticas e particionamentos utilizam geradores pseudo-aleatórios explícitos (`numpy.random.default_rng(seed)`), garantindo que os mesmos commits e pontos de mudança sejam reconstruídos de forma exata e idêntica entre diferentes execuções.

