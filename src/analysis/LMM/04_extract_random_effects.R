#!/usr/bin/env Rscript

# 04_extract_random_effects.R
# Extracts intercepts (random effects) for Words and Models from saved LMM models (.rds).
# Usage: Rscript src/analysis/LMM/04_extract_random_effects.R --norm "aoa_kuperman"

options(width = 150)
suppressPackageStartupMessages({
    library(data.table)
    library(lme4)
})

# --- Args ---
args <- commandArgs(trailingOnly = TRUE)
target_norm <- NULL
output_dir_arg <- NULL

if (length(args) > 0) {
    for (i in seq_along(args)) {
        if (args[i] == "--norm" && i < length(args)) target_norm <- args[i + 1]
        if (args[i] == "--output_dir" && i < length(args)) output_dir_arg <- args[i + 1]
    }
}

if (is.null(target_norm)) stop("❌ Must provide --norm argument!")

# --- Config ---
INPUT_DIR <- if (!is.null(output_dir_arg)) output_dir_arg else "outputs/results/LMM_Full"
MODEL_FILE <- file.path(INPUT_DIR, paste0("model_", target_norm, ".rds"))
OUTPUT_WORD <- file.path(INPUT_DIR, paste0("u_word_", target_norm, ".csv"))
OUTPUT_MODEL <- file.path(INPUT_DIR, paste0("u_model_", target_norm, ".csv"))

message(paste("▶ Extracting Random Effects for:", target_norm))

if (!file.exists(MODEL_FILE)) {
    stop(paste("❌ Model file not found:", MODEL_FILE))
}

# --- Load Model ---
message("  Loading model (this might take a minute)...")
model <- readRDS(MODEL_FILE)

# --- Extract RE ---
message("  Extracting Random Effects...")
# condVar=FALSE for speed
re <- ranef(model, condVar = FALSE)

# 1. Word Effects
u_word <- re$word
dt_word <- data.table(word = rownames(u_word), effect = u_word$`(Intercept)`)
fwrite(dt_word, OUTPUT_WORD)
message(paste("  ✅ Saved Word Effects:", nrow(dt_word), "rows ->", OUTPUT_WORD))

# 2. Model Effects
u_model <- re$model
dt_model <- data.table(model = rownames(u_model), effect = u_model$`(Intercept)`)
fwrite(dt_model, OUTPUT_MODEL)
message(paste("  ✅ Saved Model Effects:", nrow(dt_model), "rows ->", OUTPUT_MODEL))

message("Done.")
