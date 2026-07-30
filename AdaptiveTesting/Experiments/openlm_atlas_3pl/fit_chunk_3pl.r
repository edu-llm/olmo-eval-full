
library(mirt)
args <- commandArgs(trailingOnly = TRUE)
kv <- list()
for (a in args) {
  if (grepl("^--", a)) {
    parts <- strsplit(sub("^--", "", a), "=", fixed = TRUE)[[1]]
    kv[[parts[1]]] <- parts[2]
  }
}
data_file <- kv$data_file
chunk_end <- as.integer(kv$chunk_end)
chunk_ends <- as.integer(strsplit(kv$chunk_ends, ",")[[1]])
outdir <- kv$outdir
ncycles <- as.integer(if (!is.null(kv$ncycles)) kv$ncycles else "500")

data <- read.csv(data_file)
if ("avg_score" %in% colnames(data)) data$avg_score <- NULL
data_clean <- na.omit(data)
data <- data_clean[, colSums(is.na(data_clean)) == 0]
constant_cols <- apply(data, 2, function(x) length(unique(x)) == 1)
clean_data <- data[, !constant_cols]
constant_rows <- apply(clean_data, 1, function(x) length(unique(x)) == 1)
clean_data <- clean_data[!constant_rows, ]

chunk_idx <- which(chunk_ends == chunk_end)
start_col <- if (chunk_idx == 1L) 2L else chunk_ends[chunk_idx - 1L] + 1L
dat <- clean_data[, start_col:chunk_end]
cat("Fitting 3PL chunk", chunk_end, "items", ncol(dat), "ncycles", ncycles, "\n")
model <- mirt(dat, 1, itemtype = "3PL", method = "EM",
              technical = list(NCYCLES = ncycles))
theta_scores <- fscores(model, method = "EAP", full.scores = TRUE,
                        full.scores.SE = TRUE, quadpts = 61)
item_params <- coef(model, simplify = TRUE)$items
m2 <- tryCatch(M2(model), error = function(e) data.frame(error = as.character(e)))
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
tag <- as.character(chunk_end)
write.csv(theta_scores, file.path(outdir, paste0("irt_person_scores_", tag, ".csv")), row.names = FALSE)
write.csv(item_params, file.path(outdir, paste0("irt_item_parameters_", tag, ".csv")), row.names = TRUE)
write.csv(m2, file.path(outdir, paste0("m2_", tag, ".csv")), row.names = TRUE)
cat("Saved 3PL chunk", tag, "\n")
