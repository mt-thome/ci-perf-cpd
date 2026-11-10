# ==============================================================================
# Script de Validação Cruzada: Comparação ecp::e.divisive (R) vs E-Div-Means (Python)
# Projeto: Detecção de Regressões de Desempenho em CI (Mineração de Dados, UNESP)
# ==============================================================================

# Requisitos:
# install.packages("ecp")
# install.packages("jsonlite")

library(ecp)
library(jsonlite)

# Diretório dos dados
data_dir <- "data/reference_ecp"
manifest_path <- file.path(data_dir, "manifest_series.json")

if (!file.exists(manifest_path)) {
  stop("Manifesto não encontrado! Execute primeiro scripts/export_ecp_validation_set.py.")
}

manifest <- fromJSON(manifest_path)
results <- list()

cat("=== INICIANDO VALIDAÇÃO CRUZADA R (ecp) vs PYTHON (E-Div-Means) ===\n\n")

for (i in 1:nrow(manifest)) {
  row <- manifest[i, ]
  csv_file <- file.path(data_dir, row$filename)
  df <- read.csv(csv_file)
  
  # Preparar matriz para o ecp (ecp espera matriz n x d)
  X <- matrix(df$value, ncol = 1)
  
  # Executar e.divisive com mesmos parâmetros (alpha=0.05, R=99 permutações, min.size=25)
  set.seed(42)
  fit <- e.divisive(X, sig.lvl = 0.05, R = 99, min.size = 25, alpha = 1)
  
  # A biblioteca ecp retorna estimativas de breakpoints com 1 e (n+1) como bordas.
  # Em indexação base-1 do R, o ponto k corresponde ao índice (k-1) na convenção Python base-0.
  raw_cps_r <- fit$estimates
  internal_cps_r <- raw_cps_r[raw_cps_r > 1 & raw_cps_r <= (nrow(X))]
  r_cps_py_indexing <- sort(as.integer(internal_cps_r - 1))
  
  py_cps <- sort(as.integer(unlist(row$py_ediv_means_cps)))
  gt_cps <- sort(as.integer(unlist(row$ground_truth_cps)))
  
  match_exact <- identical(r_cps_py_indexing, py_cps)
  
  cat(sprintf("[Série %02d | %s | n=%d]\n", row$series_id, row$noise_type, row$n))
  cat(sprintf("  Ground Truth      : %s\n", paste(gt_cps, collapse=", ")))
  cat(sprintf("  Python E-Div-Means: %s\n", paste(py_cps, collapse=", ")))
  cat(sprintf("  R ecp::e.divisive : %s\n", paste(r_cps_py_indexing, collapse=", ")))
  cat(sprintf("  Concordância Exata: %s\n\n", ifelse(match_exact, "SIM (100%)", "DIVERGÊNCIA")))
  
  results[[i]] <- list(
    series_id = row$series_id,
    noise_type = row$noise_type,
    n = row$n,
    ground_truth = gt_cps,
    python_cps = py_cps,
    r_cps = r_cps_py_indexing,
    exact_match = match_exact
  )
}

# Salvar relatório consolidado
out_json <- file.path(data_dir, "r_ecp_validation_results.json")
write_json(results, out_json, pretty = TRUE)
cat(sprintf("Validação concluída! Resultados salvos em: %s\n", out_json))

