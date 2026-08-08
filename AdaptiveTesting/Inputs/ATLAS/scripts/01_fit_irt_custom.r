#!/usr/bin/env Rscript
# Fit IRT for one column chunk on a custom train matrix / chunk_ends.
#
# Usage (from ATLAS repo root):
#   Rscript scripts/01_fit_irt_custom.r \
#     --data_file=data/..._0p5_7b.csv \
#     --chunk_end=106 \
#     --chunk_ends=106,211,...,831 \
#     --outdir=arc_0p5_7b \
#     --itemtype=3PL

library(mirt)

parse_args <- function() {
  out <- list(itemtype = "3PL", outdir = "arc_0p5_7b")
  for (a in commandArgs(trailingOnly = TRUE)) {
    if      (grepl("^--data_file=", a))  out$data_file  <- sub("^--data_file=", "", a)
    else if (grepl("^--chunk_end=", a))  out$chunk_end  <- as.integer(sub("^--chunk_end=", "", a))
    else if (grepl("^--chunk_ends=", a)) out$chunk_ends <- as.integer(strsplit(sub("^--chunk_ends=", "", a), ",")[[1]])
    else if (grepl("^--itemtype=", a))   out$itemtype   <- sub("^--itemtype=", "", a)
    else if (grepl("^--outdir=", a))     out$outdir     <- sub("^--outdir=", "", a)
  }
  out
}
args <- parse_args()
stopifnot(!is.null(args$data_file), !is.null(args$chunk_end), !is.null(args$chunk_ends))

chunk_end <- args$chunk_end
itemtype  <- args$itemtype
outdir    <- args$outdir
chunk_ends <- args$chunk_ends

if (!chunk_end %in% chunk_ends)
  stop("chunk_end=", chunk_end, " not in chunk_ends")

cat("Loading:", args$data_file, "\n")
data <- read.csv(args$data_file)
# Drop leaderboard avg_score so it is never treated as an IRT item
if ("avg_score" %in% colnames(data)) {
  data$avg_score <- NULL
  cat("Dropped avg_score column.\n")
}
cat("Dimensions of original data:", dim(data), "\n")

data_clean <- na.omit(data)
data       <- data_clean[, colSums(is.na(data_clean)) == 0]

constant_cols <- apply(data, 2, function(x) length(unique(x)) == 1)
clean_data    <- data[, !constant_cols]
cat("Dropped", sum(constant_cols), "constant columns.\n")

constant_rows <- apply(clean_data, 1, function(x) length(unique(x)) == 1)
clean_data    <- clean_data[!constant_rows, ]
cat("Dropped", sum(constant_rows), "constant rows.\n")
cat("Dimensions of cleaned data:", dim(clean_data), "\n")

chunk_idx <- which(chunk_ends == chunk_end)
start_col <- if (chunk_idx == 1L) 2L else chunk_ends[chunk_idx - 1L] + 1L
if (chunk_end > ncol(clean_data))
  stop("chunk_end=", chunk_end, " > ncol(clean_data)=", ncol(clean_data))
dat <- clean_data[, start_col:chunk_end]
cat("Chunk", chunk_idx, "of", length(chunk_ends),
    ": columns", start_col, "to", chunk_end, "(", ncol(dat), "items)\n")

cat("Fitting", itemtype, "model...\n")
model <- mirt(dat, 1, itemtype = itemtype, method = "EM",
              technical = list(NCYCLES = 100000))
print(model)

m2           <- M2(model)
theta_scores <- fscores(model, method = "EAP", full.scores = TRUE,
                        full.scores.SE = TRUE, quadpts = 61)
item_params  <- coef(model, simplify = TRUE)$items
print(item_params)

dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
tag <- as.character(chunk_end)

write.csv(theta_scores, file.path(outdir, paste0("irt_person_scores_",    tag, ".csv")), row.names = FALSE)
write.csv(item_params,  file.path(outdir, paste0("irt_item_parameters_",  tag, ".csv")), row.names = TRUE)
write.csv(m2,           file.path(outdir, paste0("m2_",                   tag, ".csv")), row.names = TRUE)
cat("Saved outputs to", outdir, "\n")
