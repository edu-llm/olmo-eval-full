#!/usr/bin/env Rscript
# Recalibrate ATLAS-style ARC 3PL on a custom (row-subsampled) train matrix.
#
# Merges ATLAS's `01_fit_irt_custom.r` (chunked mirt 3PL EM) and
# `02_link_chunks_custom.r` (mean/sigma linking) into one call, with chunk
# boundaries computed dynamically from the surviving item columns so it works
# for small, size-balanced pools (where more columns go constant than in the
# full 1686-model pool). All IRT conventions are identical to ATLAS: single
# factor 3PL, EM, mean-sigma linking on EAP F1 person scores, params (a1,d,g).
#
# Usage:
#   Rscript recal_fit_link.r --data_file=<subset.csv> --out=<combined.csv> \
#       [--chunk_size=105]

suppressMessages(library(mirt))

parse_args <- function() {
  out <- list(chunk_size = 105L)
  for (a in commandArgs(trailingOnly = TRUE)) {
    if      (grepl("^--data_file=", a)) out$data_file  <- sub("^--data_file=", "", a)
    else if (grepl("^--out=", a))       out$out        <- sub("^--out=", "", a)
    else if (grepl("^--chunk_size=", a))out$chunk_size <- as.integer(sub("^--chunk_size=", "", a))
  }
  out
}
args <- parse_args()
stopifnot(!is.null(args$data_file), !is.null(args$out))

data <- read.csv(args$data_file, check.names = TRUE)
if ("avg_score" %in% colnames(data)) data$avg_score <- NULL

data_clean <- na.omit(data)
data       <- data_clean[, colSums(is.na(data_clean)) == 0]

constant_cols <- apply(data, 2, function(x) length(unique(x)) == 1)
clean_data    <- data[, !constant_cols]
cat("Dropped", sum(constant_cols), "constant columns.\n")

constant_rows <- apply(clean_data, 1, function(x) length(unique(x)) == 1)
clean_data    <- clean_data[!constant_rows, ]
cat("Cleaned dims:", dim(clean_data), " (col 1 = model id)\n")

ncol_all <- ncol(clean_data)          # col 1 = id, cols 2..ncol_all = items
last_item <- ncol_all
# Dynamic chunk ends over item columns 2..last_item in steps of chunk_size.
ends <- seq(1L + args$chunk_size, last_item, by = args$chunk_size)
if (length(ends) == 0L || ends[length(ends)] != last_item) ends <- c(ends, last_item)
cat("Item columns 2..", last_item, "; chunk_ends:", paste(ends, collapse = ","), "\n")

fit_chunk <- function(start_col, end_col) {
  dat <- clean_data[, start_col:end_col, drop = FALSE]
  # Guard: drop any columns that are constant within this subsample.
  keep <- apply(dat, 2, function(x) length(unique(x)) > 1)
  dat <- dat[, keep, drop = FALSE]
  if (ncol(dat) < 2) return(NULL)
  model <- tryCatch(
    mirt(dat, 1, itemtype = "3PL", method = "EM",
         technical = list(NCYCLES = 100000), verbose = FALSE),
    error = function(e) { cat("  chunk fit error:", conditionMessage(e), "\n"); NULL }
  )
  if (is.null(model)) return(NULL)
  fs <- tryCatch(
    fscores(model, method = "EAP", full.scores = TRUE,
            full.scores.SE = TRUE, quadpts = 61),
    error = function(e) NULL
  )
  if (is.null(fs)) return(NULL)
  ip <- coef(model, simplify = TRUE)$items
  list(F1 = as.numeric(fs[, "F1"]), items = as.data.frame(ip))
}

chunks <- list()
prev <- 1L
for (e in ends) {
  sc <- prev + 1L
  cat("Fitting chunk cols", sc, "to", e, "(", e - sc + 1L, "items )\n")
  chunks[[length(chunks) + 1L]] <- list(res = fit_chunk(sc, e))
  prev <- e
}
chunks <- Filter(function(z) !is.null(z$res), chunks)
if (length(chunks) == 0L) stop("no chunk fit succeeded")

ref_F1 <- chunks[[1]]$res$F1
linked <- list()
for (j in seq_along(chunks)) {
  ip <- chunks[[j]]$res$items
  ip$X <- rownames(ip)
  if (j == 1L) {
    linked[[j]] <- ip
  } else {
    sc_j <- chunks[[j]]$res$F1
    A <- sd(ref_F1, na.rm = TRUE) / sd(sc_j, na.rm = TRUE)
    B <- mean(ref_F1, na.rm = TRUE) - A * mean(sc_j, na.rm = TRUE)
    if (!is.finite(A) || A == 0) { A <- 1; B <- 0 }
    ip$a1 <- ip$a1 / A
    ip$d  <- A * ip$d + B * ip$a1
    linked[[j]] <- ip
    cat("Chunk", j, "link: A=", round(A, 3), "B=", round(B, 3), "\n")
  }
}

combined <- do.call(rbind, linked)
cols <- c("X", "a1", "d", "g", "u")
for (cc in cols) if (!cc %in% colnames(combined)) combined[[cc]] <- NA
combined <- combined[, cols]
write.csv(combined, args$out, row.names = FALSE)
cat("Saved", nrow(combined), "linked items to", args$out, "\n")
