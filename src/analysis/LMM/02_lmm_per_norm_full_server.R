#!/usr/bin/env Rscript

# 02_lme_per_norm_full_server.R
# Runs LMM Variance Partitioning for a SINGLE norm provided as argument.
# Usage: Rscript 02_lme_per_norm_full_server.R --norm "aoa_kuperman"
# CONFIGURATION: FULL DATA (No subsampling), Extract Interaction BLUPs.
# DESIGNED FOR: High-RAM Server (500GB+)

options(width = 150)
suppressPackageStartupMessages({
    library(data.table)
    library(lme4)
})

# Parse Args
args <- commandArgs(trailingOnly = TRUE)
target_norm <- NULL
subsample_n <- NULL
nrows_arg <- -1
output_dir_arg <- NULL

if (length(args) > 0) {
    for (i in seq_along(args)) {
        if (args[i] == "--norm" && i < length(args)) {
            target_norm <- args[i + 1]
        }
        if (args[i] == "--subsample" && i < length(args)) {
            subsample_n <- as.integer(args[i + 1])
        }
        if (args[i] == "--nrows" && i < length(args)) {
            nrows_arg <- as.integer(args[i + 1])
        }
        if (args[i] == "--output_dir" && i < length(args)) {
            output_dir_arg <- args[i + 1]
        }
    }
}

if (is.null(target_norm)) {
    stop("Must provide --norm argument!")
}

# --- Configuration ---
INPUT_DIR <- "outputs/raw_behavior/model_norms_clean/stochastic"
OUTPUT_DIR <- if (!is.null(output_dir_arg)) output_dir_arg else "outputs/results/LMM_Full"
MASTER_DATA_DIR <- file.path(OUTPUT_DIR, "data_used")
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(MASTER_DATA_DIR, recursive = TRUE, showWarnings = FALSE)
message(paste("▶ Output directory:", OUTPUT_DIR))

FILE_GLOBAL <- file.path(OUTPUT_DIR, "lme_variance_partitioning_global.csv")
FILE_SPECIFIC <- file.path(OUTPUT_DIR, "lme_variance_partitioning_per_model.csv")

# --- 1. Data Loading ---
message(paste("▶ Loading data for:", target_norm))

INPUT_PARQUET <- file.path("outputs/raw_behavior/model_norms_arrow", paste0(target_norm, ".parquet"))

if (!file.exists(INPUT_PARQUET)) {
    stop(paste("❌ Pre-sharded parquet file not found for", target_norm, ". Did you run 01_prepare_arrow_shards.py first? Path:", INPUT_PARQUET))
}

# Load the single pre-filtered, compressed shard instantly
full_data <- as.data.table(arrow::read_parquet(INPUT_PARQUET))

if (nrows_arg > 0) {
    message(paste("ℹ Reading only first", nrows_arg, "rows (--nrows)"))
    full_data <- head(full_data, nrows_arg)
}

# Filter Valid Rows Only
initial_rows <- nrow(full_data)
full_data <- full_data[is_effective_valid == TRUE & !is.na(rating_val)]
final_rows <- nrow(full_data)
dropped_rows <- initial_rows - final_rows

message(paste("Filtered valid rows:", final_rows, "| Dropped:", dropped_rows))

# --- Subsampling Logic ---
if (!is.null(subsample_n) && subsample_n > 0) {
    message(paste("Subsampling requested. Target words:", subsample_n))

    unique_words <- unique(full_data$word)
    if (length(unique_words) > subsample_n) {
        set.seed(42) # For reproducibility
        sampled_words <- sample(unique_words, subsample_n)
        full_data <- full_data[word %in% sampled_words]
        message(paste("Subsampled to", length(sampled_words), "words."))
        message(paste("Rows after subsampling:", format(nrow(full_data), big.mark = ",")))
    } else {
        message(paste("Subsample N (", subsample_n, ") >= Total Unique Words (", length(unique_words), "). Skipped subsampling."))
    }
}

# Ensure strict factor levels
full_data[, word := as.factor(word)]
full_data[, model := as.factor(model)]

if (nrow(full_data) < 100) {
    message("Too few observations. Exiting.")
    quit(save = "no", status = 0)
}

# Save Master Data File
master_file <- file.path(MASTER_DATA_DIR, paste0("master_data_", target_norm, ".csv"))
fwrite(full_data, master_file)
message(paste("Saved master data used for LMM to:", master_file))

message(paste("Total rows available for", target_norm, ":", format(nrow(full_data), big.mark = ",")))

# --- 2. LMM Analysis (FULL DATA) ---
message(paste("\n--- Analyzing Norm (Full Data):", target_norm, "---"))

tryCatch(
    {
        start_time <- Sys.time()

        # Fit Model
        # Note: On full data (~1M rows), this might take 10-20 mins.
        # Optimizer: bobyqa is usually robust.
        # Calculation of calc.derivs=FALSE speeds up the end phase if we don't need strict convergence diagnostics.
        model <- lmer(rating_val ~ 1 + (1 | word) + (1 | model) + (1 | word:model),
            data = full_data,
            control = lmerControl(optimizer = "bobyqa", calc.derivs = FALSE)
        )
        message(paste("  Fit complete in", round(difftime(Sys.time(), start_time, units = "secs"), 2), "s"))

        # Extract Global Components
        vc <- as.data.frame(VarCorr(model))
        sigma_sq <- attr(VarCorr(model), "sc")^2
        if (is.null(sigma_sq)) sigma_sq <- 0

        resid_row <- vc[vc$grp == "Residual", ]
        var_residual <- if (nrow(resid_row) > 0) resid_row$vcov else sigma_sq

        sum_vc <- sum(vc$vcov[!is.na(vc$vcov) & vc$grp != "Residual"])
        total_var_check <- sum_vc + var_residual

        safe_get <- function(grp) {
            val <- vc[vc$grp == grp, "vcov"]
            if (length(val) == 0) 0 else val
        }

        var_trait <- safe_get("word")
        var_method <- safe_get("model")
        var_interaction <- safe_get("word:model")

        res_global <- data.table(
            norm = target_norm,
            n_obs = nrow(full_data),
            var_trait = var_trait,
            var_method = var_method,
            var_interaction = var_interaction,
            var_residual = var_residual,
            total_var = total_var_check,
            prop_trait = var_trait / total_var_check,
            prop_method = var_method / total_var_check,
            prop_interaction = var_interaction / total_var_check,
            prop_residual = var_residual / total_var_check,
            intercept = fixef(model)[[1]]
        )

        # Extract Specific
        message("  Extracting Random Effects...")
        # condVar=FALSE is MANDATORY for 1M rows/400k groups to fit in reasonable memory/time.
        # We only need point estimates.
        re <- ranef(model, condVar = FALSE)

        # 1. Bias
        bias_vals <- re$model
        bias_dt <- data.table(model = rownames(bias_vals), bias_shift = bias_vals$`(Intercept)`)

        # 2. Idiosyncrasy (Interaction)
        inter_vals <- re$`word:model`
        inter_dt <- data.table(id = rownames(inter_vals), val = inter_vals$`(Intercept)`)

        # Split IDs
        inter_dt[, c("p1", "p2") := tstrsplit(id, ":", fixed = TRUE)]
        models_in_data <- unique(as.character(full_data$model))
        if (all(inter_dt$p1 %in% models_in_data)) {
            inter_dt[, model_col := p1]
        } else if (all(inter_dt$p2 %in% models_in_data)) {
            inter_dt[, model_col := p2]
        } else {
            inter_dt[, model_col := ifelse(p1 %in% models_in_data, p1, p2)]
        }

        # Calculate Variance of Idiosyncrasy PER MODEL
        idiosyncrasy_dt <- inter_dt[, .(var_idiosyncrasy = var(val)), by = model_col]

        # --- Save BLUPs for Embedding Prediction ---
        blup_file <- file.path(OUTPUT_DIR, paste0("blups_", target_norm, ".csv"))
        blup_dt <- inter_dt[, .(model = model_col, word = ifelse(p1 %in% models_in_data, p2, p1), idiosyncrasy = val)]
        fwrite(blup_dt, blup_file)
        message(paste("  Saved BLUPs to:", blup_file))

        # Merge
        res_specific <- merge(bias_dt, idiosyncrasy_dt, by.x = "model", by.y = "model_col", all = TRUE)
        res_specific[, norm := target_norm]
        res_specific[, squared_bias := bias_shift^2]

        # Save (Append with File Locking Logic ideally, but simplified here)
        file_g_norm <- file.path(OUTPUT_DIR, paste0("global_", target_norm, ".csv"))
        file_s_norm <- file.path(OUTPUT_DIR, paste0("specific_", target_norm, ".csv"))

        fwrite(res_global, file_g_norm)
        fwrite(res_specific, file_s_norm)

        # Save Model Object and Summary
        file_model <- file.path(OUTPUT_DIR, paste0("model_", target_norm, ".rds"))
        saveRDS(model, file_model)

        file_summary <- file.path(OUTPUT_DIR, paste0("summary_", target_norm, ".txt"))
        capture.output(summary(model), file = file_summary)

        message(paste("  Saved results, model, and summary for", target_norm))
    },
    error = function(e) {
        message(paste("  Error:", e$message))
        quit(save = "no", status = 1)
    }
)
