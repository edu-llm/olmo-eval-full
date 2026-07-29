#!/usr/bin/env Rscript
# Mean/Sigma linking across custom IRT chunks → irt_item_parameters_combined.csv
#
# Usage (from ATLAS repo root):
#   Rscript scripts/02_link_chunks_custom.r \
#     --outdir=arc_0p5_7b \
#     --chunk_ends=106,211,...,831

parse_args <- function() {
  out <- list(outdir = "arc_0p5_7b")
  for (a in commandArgs(trailingOnly = TRUE)) {
    if      (grepl("^--outdir=", a))     out$outdir     <- sub("^--outdir=", "", a)
    else if (grepl("^--chunk_ends=", a)) out$chunk_ends <- as.integer(strsplit(sub("^--chunk_ends=", "", a), ",")[[1]])
  }
  out
}
args <- parse_args()
stopifnot(!is.null(args$chunk_ends))
outdir        <- args$outdir
chunk_indices <- args$chunk_ends

scores_list <- lapply(chunk_indices, function(i) {
  read.csv(file.path(outdir, paste0("irt_person_scores_", i, ".csv")))
})
reference_scores <- scores_list[[1]]
cat("Reference scores columns:", colnames(reference_scores), "\n")

linked_params_list <- list()
linked_params_list[[1]] <- read.csv(
  file.path(outdir, paste0("irt_item_parameters_", chunk_indices[1], ".csv")))

for (j in 2:length(scores_list)) {
  i        <- chunk_indices[j]
  params_j <- read.csv(file.path(outdir, paste0("irt_item_parameters_", i, ".csv")))

  sc_ref <- as.numeric(reference_scores$F1)
  sc_j   <- as.numeric(scores_list[[j]]$F1)

  A <- sd(sc_ref, na.rm = TRUE) / sd(sc_j, na.rm = TRUE)
  B <- mean(sc_ref, na.rm = TRUE) - A * mean(sc_j, na.rm = TRUE)
  cat("Chunk", i, "linking: A=", A, "B=", B, "\n")

  params_j_star    <- params_j
  params_j_star$a1 <- params_j$a1 / A
  params_j_star$d  <- A * params_j$d + B * params_j$a1
  if ("g" %in% colnames(params_j)) params_j_star$g <- params_j$g

  linked_params_list[[j]] <- params_j_star
  cat("Correlation after linking:",
      cor(sc_ref, A * sc_j + B, use = "pairwise.complete.obs"), "\n")
}

item_params_combined <- do.call(rbind, linked_params_list)
out_csv <- file.path(outdir, "irt_item_parameters_combined.csv")
write.csv(item_params_combined, out_csv, row.names = FALSE)
cat("Saved", nrow(item_params_combined), "linked items to", out_csv, "\n")
