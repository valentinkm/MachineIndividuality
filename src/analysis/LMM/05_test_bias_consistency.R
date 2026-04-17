#!/usr/bin/env Rscript
# src/analysis/LMM/05_test_bias_consistency.R

suppressPackageStartupMessages({
    library(data.table)
    library(stats)
})

# Parse Args
args <- commandArgs(trailingOnly = TRUE)
output_dir_arg <- NULL
if (length(args) > 0) {
    for (i in seq_along(args)) {
        if (args[i] == "--output_dir" && i < length(args)) output_dir_arg <- args[i + 1]
    }
}

# Config
INPUT_DIR <- if (!is.null(output_dir_arg)) output_dir_arg else "outputs/results/LMM_Full"
OUTPUT_FILE <- file.path(INPUT_DIR, "bias_consistency_test.csv")

FILE_SPECIFIC <- file.path(INPUT_DIR, "lme_variance_partitioning_per_model.csv")
FILE_GLOBAL <- file.path(INPUT_DIR, "lme_variance_partitioning_global.csv")

# 1. Load Data
message("▶ Loading data...")

if (!file.exists(FILE_SPECIFIC) || !file.exists(FILE_GLOBAL)) {
    stop("❌ Missing Input Files. Please run LMM analysis first.")
}

df_specific <- fread(FILE_SPECIFIC) # Columns: model, bias_shift, norm, ...
df_global <- fread(FILE_GLOBAL) # Columns: norm, total_var, ...

# 2. Preprocessing
message("ℹ Preprocessing and Standardizing...")

# Merge specific and global to get total variance for each observation
# We need total_var to standardize the bias_shift
df_merged <- merge(df_specific, df_global[, .(norm, total_var)], by = "norm")

# Calculate Standardized Bias (Effect Size)
# Z = Bias / SD_total
df_merged[, z_bias := bias_shift / sqrt(total_var)]

# Filter out AoA norms (different properties, often excluded)
df_analysis <- df_merged[!grepl("aoa", norm, ignore.case = TRUE)]
message(paste("ℹ Analysis Norms:", paste(unique(df_analysis$norm), collapse = ", ")))

# 3. Statistical Testing
message("▶ Running Linear Model (One-Sample t-test per model)...")

# Model: z_bias ~ 0 + model
# "0 +" forces distinct intercepts for each model, effectively testing H0: mean(z_bias) = 0
lm_model <- lm(z_bias ~ 0 + model, data = df_analysis)
lm_summary <- summary(lm_model)

# Extract Results
coefs <- as.data.frame(lm_summary$coefficients)
# Structure of coefs: "Estimate", "Std. Error", "t value", "Pr(>|t|)"

# Format Output Table
results_dt <- data.table(
    model = gsub("model", "", rownames(coefs)), # Clean "modelGemma-2" -> "Gemma-2"
    estimate_mean_z = coefs[, "Estimate"],
    std_error = coefs[, "Std. Error"],
    t_statistic = coefs[, "t value"],
    p_value = coefs[, "Pr(>|t|)"]
)

# 4. Correction and Significance
# FDR Adjustment for multiple comparisons (many models)
results_dt[, p_adj := p.adjust(p_value, method = "fdr")]

# Add helpful stars
results_dt[, signif := fcase(
    p_adj < 0.001, "***",
    p_adj < 0.01, "**",
    p_adj < 0.05, "*",
    p_adj < 0.1, ".",
    default = ""
)]

# Add Readable Consistency Label
results_dt[, consistency := fcase(
    p_adj < 0.05 & estimate_mean_z > 0, "Consistently POSITIVE",
    p_adj < 0.05 & estimate_mean_z < 0, "Consistently NEGATIVE",
    default = "Inconsistent / Neutral"
)]

# Sort by Estimate
setorder(results_dt, -estimate_mean_z)

# 5. Save and Print
fwrite(results_dt, OUTPUT_FILE)
message(paste("✅ Saved consistency test results to:", OUTPUT_FILE))

# Terminal Output (Top Significant)
message("\n📊 Significant Results (FDR < 0.05):")
print(results_dt[p_adj < 0.05, .(model, estimate = round(estimate_mean_z, 3), p_adj = format.pval(p_adj, digits = 2), consistency)])
