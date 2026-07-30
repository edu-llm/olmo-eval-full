
args <- commandArgs(trailingOnly = TRUE)
kv <- list()
for (a in args) {
  if (grepl("^--", a)) {
    parts <- strsplit(sub("^--", "", a), "=", fixed = TRUE)[[1]]
    kv[[parts[1]]] <- parts[2]
  }
}
outdir <- kv$outdir
chunk_indices <- as.integer(strsplit(kv$chunk_ends, ",")[[1]])
scores_list <- lapply(chunk_indices, function(i)
  read.csv(file.path(outdir, paste0("irt_person_scores_", i, ".csv"))))
reference_scores <- scores_list[[1]]
linked <- list(read.csv(file.path(outdir, paste0("irt_item_parameters_", chunk_indices[1], ".csv"))))

for (j in 2:length(scores_list)) {
  i <- chunk_indices[j]
  params_j <- read.csv(file.path(outdir, paste0("irt_item_parameters_", i, ".csv")))
  sc_ref <- as.numeric(reference_scores$F1)
  sc_j <- as.numeric(scores_list[[j]]$F1)
  A <- sd(sc_ref, na.rm = TRUE) / sd(sc_j, na.rm = TRUE)
  B <- mean(sc_ref, na.rm = TRUE) - A * mean(sc_j, na.rm = TRUE)
  r <- cor(sc_ref, A * sc_j + B, use = "pairwise.complete.obs")
  if (!is.na(r) && r < 0) {
    A <- -A
    B <- mean(sc_ref, na.rm = TRUE) - A * mean(sc_j, na.rm = TRUE)
    r2 <- cor(sc_ref, A * sc_j + B, use = "pairwise.complete.obs")
    cat("Chunk", i, "FLIPPED A=", A, "B=", B, "cor", r, "->", r2, "\n")
  } else {
    cat("Chunk", i, "A=", A, "B=", B, "cor", r, "\n")
  }
  a_orig <- params_j$a1
  params_j$a1 <- a_orig / A
  params_j$d <- A * params_j$d + B * a_orig
  if ("g" %in% colnames(params_j)) params_j$g <- params_j$g
  linked[[j]] <- params_j
}
combined <- do.call(rbind, linked)
out_csv <- file.path(outdir, "irt_item_parameters_combined.csv")
write.csv(combined, out_csv, row.names = FALSE)
cat("Saved", nrow(combined), "linked items to", out_csv, "\n")
