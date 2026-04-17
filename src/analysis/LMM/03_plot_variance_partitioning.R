#!/usr/bin/env Rscript
# src/analysis/LMM/03_plot_variance_partitioning.R

suppressPackageStartupMessages({
    library(ggplot2)
    library(data.table)
    library(dplyr)
    library(tidyr)
    library(viridis)
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
OUTPUT_DIR <- file.path(INPUT_DIR, "plots")
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

FILE_GLOBAL <- file.path(INPUT_DIR, "lme_variance_partitioning_global.csv")
FILE_SPECIFIC <- file.path(INPUT_DIR, "lme_variance_partitioning_per_model.csv")

# 0. Consolidate Individual Result Files (if present)
message("▶ Checking for individual result files to merge...")
global_files <- list.files(INPUT_DIR, pattern = "global_.*\\.csv", full.names = TRUE)
specific_files <- list.files(INPUT_DIR, pattern = "specific_.*\\.csv", full.names = TRUE)

if (length(global_files) > 0) {
    message(paste("  Found", length(global_files), "individual global files. Merging..."))
    df_global_all <- rbindlist(lapply(global_files, fread), fill = TRUE)
    fwrite(df_global_all, FILE_GLOBAL)
    message(paste("  ✅ Merged to:", FILE_GLOBAL))
    file.remove(global_files)
    message("  🗑️  Deleted individual global files.")
}

if (length(specific_files) > 0) {
    message(paste("  Found", length(specific_files), "individual specific files. Merging..."))
    df_specific_all <- rbindlist(lapply(specific_files, fread), fill = TRUE)
    fwrite(df_specific_all, FILE_SPECIFIC)
    message(paste("  ✅ Merged to:", FILE_SPECIFIC))
    file.remove(specific_files)
    message("  🗑️  Deleted individual specific files.")
}

# 1. Load Consolidated Data
message("▶ Loading data...")
if (!file.exists(FILE_GLOBAL) || !file.exists(FILE_SPECIFIC)) {
    stop("❌ Missing result files (and no individual files found to merge).")
}

df_global <- fread(FILE_GLOBAL)
df_specific <- fread(FILE_SPECIFIC)

# 2. Prepare for Plotting
# We want to stack: Trait + Residual (Global) + Bias^2 + Idiosyncrasy (Specific)
# Note: This is an *approximate* decomposition because Bias^2 + Idiosyncrasy != Total Method+Interaction Variance exactly,
# but it represents the model's specific contribution.

# Join
df_plot <- merge(df_specific, df_global, by = "norm", suffixes = c("_spec", "_glob"))

# Calculate components for the stack
# We use the raw variances
df_plot[, comp_trait := var_trait] # Global signal
df_plot[, comp_residual := var_residual] # Global noise
df_plot[, comp_bias := bias_shift^2] # Model systematic error
df_plot[, comp_idiosyncrasy := var_idiosyncrasy] # Model unique error

df_plot[is.na(comp_idiosyncrasy), comp_idiosyncrasy := 0] # Fallback

# The total variance might differ slightly from the global total due to the decomposition approximation.
# Let's calculate a "Model Total Variance" = Trait + Resid + Bias + Idio
df_plot[, model_total_var := comp_trait + comp_residual + comp_bias + comp_idiosyncrasy]

# Calculate proportions
df_plot[, prop_trait := comp_trait / model_total_var]
df_plot[, prop_residual := comp_residual / model_total_var]
df_plot[, prop_bias := comp_bias / model_total_var]
df_plot[, prop_idiosyncrasy := comp_idiosyncrasy / model_total_var]

# Reshape to Long for ggplot
df_long <- melt(df_plot,
    id.vars = c("norm", "model"),
    measure.vars = c("prop_trait", "prop_bias", "prop_idiosyncrasy", "prop_residual"),
    variable.name = "component",
    value.name = "proportion"
)

# Rename levels for prettier legend
levels(df_long$component) <- c("Trait (Signal)", "Bias (Systematic)", "Idiosyncrasy (Unique)", "Residual (Noise)")

# 3. Plotting
message("▶ Generating plots...")

# A. Stacked Bar Chart (Faceted by Norm)
p1 <- ggplot(df_long, aes(x = model, y = proportion, fill = component)) +
    geom_bar(stat = "identity", width = 0.8) +
    facet_wrap(~norm, scales = "free_x", ncol = 3) + # 3 columns for 15 norms = 5 rows
    scale_fill_manual(values = c(
        "Trait (Signal)" = "#2ca02c",
        "Bias (Systematic)" = "#1f77b4",
        "Idiosyncrasy (Unique)" = "#ff7f0e",
        "Residual (Noise)" = "#d62728"
    )) +
    labs(
        title = "Variance Partitioning by Model and Norm",
        subtitle = "Decomposition of Model Ratings into Signal (Trait), Bias, Idiosyncrasy, and Noise",
        x = NULL,
        y = "Proportion of Variance",
        fill = "Variance Component"
    ) +
    theme_minimal(base_size = 14) +
    theme(
        axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
        legend.position = "top",
        strip.text = element_text(face = "bold", size = 10)
    )

file_p1 <- file.path(OUTPUT_DIR, "variance_partitioning_stacked_by_norm.pdf")
ggsave(file_p1, p1, width = 12, height = 15)
message(paste("  ✅ Saved:", file_p1))


# B. Summary Aggregated Plot
df_agg <- df_long[, .(mean_prop = mean(proportion)), by = .(model, component)]

p2 <- ggplot(df_agg, aes(x = model, y = mean_prop, fill = component)) +
    geom_bar(stat = "identity", width = 0.6) +
    scale_fill_manual(values = c(
        "Trait (Signal)" = "#2ca02c",
        "Bias (Systematic)" = "#1f77b4",
        "Idiosyncrasy (Unique)" = "#ff7f0e",
        "Residual (Noise)" = "#d62728"
    )) +
    labs(
        title = "Average Variance Partitioning across All Norms",
        x = NULL,
        y = "Average Proportion",
        fill = "Component"
    ) +
    theme_minimal(base_size = 14) +
    theme(
        legend.position = "right",
        axis.text.x = element_text(angle = 45, hjust = 1)
    )

file_p2 <- file.path(OUTPUT_DIR, "variance_partitioning_summary.pdf")
ggsave(file_p2, p2, width = 12, height = 6)
message(paste("  ✅ Saved:", file_p2))

message("✅ Visualization Complete.")
