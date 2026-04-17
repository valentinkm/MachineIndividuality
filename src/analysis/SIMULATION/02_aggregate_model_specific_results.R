#!/usr/bin/env Rscript

# 02_aggregate_model_specific_results.R
# Aggregates Conditional Simulation results and calculates Model-Specific P-Values.
# Compares:
#   Null Distribution: "var_idio_sim" from output/results/LMM_Simulation/sim_results_*.csv
#   Real Value: "var_idiosyncrasy" from LMM_OUTPUT_DIR/lme_variance_partitioning_per_model.csv

suppressPackageStartupMessages({
    library(data.table)
})

# --- Config ---
SIM_DIR <- "outputs/results/LMM_Simulation"
REAL_FILE <- file.path(Sys.getenv("LMM_OUTPUT_DIR", "outputs/results/LMM_Full_filtered"), "lme_variance_partitioning_per_model.csv")
OUT_DIR <- "outputs/results/LMM_Simulation/analysis"
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

# --- 1. Load Data ---
message("▶ Loading Real Model-Specific Variances...")
if (!file.exists(REAL_FILE)) stop("❌ Missing Real LMM Results File")
df_real <- fread(REAL_FILE)
# We need: norm, model, var_idiosyncrasy (Real Value)
real_vals <- df_real[, .(norm, model, var_idiosyncrasy)]

message("▶ Loading Simulation Results...")
sim_files <- list.files(SIM_DIR, pattern = "sim_results_.*\\.csv", full.names = TRUE)
sim_list <- lapply(sim_files, function(f) {
    d <- fread(f)
    # Extract norm from filename "sim_results_normname.csv"
    bn <- basename(f)
    extracted_norm <- sub("sim_results_(.*)\\.csv", "\\1", bn)
    d[, norm := extracted_norm]
    # Keep ensuring columns exist
    if (!"var_idio_sim" %in% names(d)) {
        return(NULL)
    }
    return(d)
})
df_sim <- rbindlist(sim_list, fill = TRUE)

message(paste("ℹ Loaded", nrow(df_sim), "simulation rows."))

# --- 2. Calculate P-Values ---
message("▶ Calculating P-Values per Model/Norm...")

# Join Real values into Sim data for easy comparison
# (Left join on Sim ensures we match correct Norm+Model)
df_analysis <- merge(df_sim, real_vals, by = c("norm", "model"), all.x = TRUE)

# Clean up any potential NAs (if model mismatch)
missing_matches <- df_analysis[is.na(var_idiosyncrasy)]
if (nrow(missing_matches) > 0) {
    message(paste("⚠ Warning:", nrow(missing_matches), "simulation rows matched no real model variance."))
    # Usually caused by minor naming diffs or incomplete real file
}
df_analysis <- df_analysis[!is.na(var_idiosyncrasy)]

# Calculate P-Value
# P = (Count(Sim >= Real) + 1) / (N_Sim + 1)
# Groups: Norm, Model
stats <- df_analysis[, .(
    real_idio_var = first(var_idiosyncrasy),
    null_idio_mean = mean(var_idio_sim, na.rm = TRUE),
    null_idio_std = sd(var_idio_sim, na.rm = TRUE),
    null_idio_max = max(var_idio_sim, na.rm = TRUE),
    n_sims = .N,
    n_exceed = sum(var_idio_sim >= var_idiosyncrasy, na.rm = TRUE)
), by = .(norm, model)]

stats[, p_value := (n_exceed + 1) / (n_sims + 1)]
stats[, z_score := (real_idio_var - null_idio_mean) / null_idio_std]

# Handle infinite Z (if null sd is 0)
stats[is.infinite(z_score), z_score := NA]

# --- 3. Save Summary ---
out_file <- file.path(OUT_DIR, "model_specific_p_values.csv")
fwrite(stats, out_file)
message(paste("✅ Saved P-Values to:", out_file))

# --- 4. Identify Non-Significant Models ---
non_sig <- stats[p_value > 0.05]
if (nrow(non_sig) > 0) {
    message("\n⚠ Found Models with Non-Significant Idiosyncrasy (p > 0.05):")
    print(non_sig[, .(norm, model, p_value, real_idio_var, null_idio_max)])
} else {
    message("\n✅ ALL tested model idiosyncrasies are Significant (p < 0.05)!")
}
message("✅ Aggregation Complete.")
