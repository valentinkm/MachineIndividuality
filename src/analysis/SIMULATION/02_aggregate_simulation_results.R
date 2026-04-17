#!/usr/bin/env Rscript

# 02_aggregate_simulation_results.R
# Aggregates results from `sim_results_*.csv` files and calculates p-values.
# Usage: Rscript src/analysis/SIMULATION/02_aggregate_simulation_results.R

suppressPackageStartupMessages({
    library(data.table)
})

# --- Config ---
SIM_RESULTS_DIR <- "outputs/results/LMM_Simulation"
PARAM_FILE <- file.path(Sys.getenv("LMM_OUTPUT_DIR", "outputs/results/LMM_Full_filtered"), "lme_variance_partitioning_global.csv")
OUTPUT_FILE <- file.path(SIM_RESULTS_DIR, "simulation_p_values.csv")

message("======== SIMULATION RESULTS AGGREGATION ========")

# 1. Load Real Parameters (to get the observed variance)
if (!file.exists(PARAM_FILE)) stop("Parameter file not found!")
real_params <- fread(PARAM_FILE)
setkey(real_params, norm)

# 2. Find Simulation Files
sim_files <- list.files(SIM_RESULTS_DIR, pattern = "sim_results_.*\\.csv", full.names = TRUE)

if (length(sim_files) == 0) {
    stop("No simulation result files found in ", SIM_RESULTS_DIR)
}

message(paste("Found", length(sim_files), "simulation files."))

final_stats <- data.table()

for (f in sim_files) {
    # Extract norm name from filename: sim_results_NORMNAME.csv
    norm_name <- sub("sim_results_(.*)\\.csv", "\\1", basename(f))

    # Load Sim Data
    sim_data <- fread(f)
    n_sims <- nrow(sim_data)

    # Get Real Variance
    if (!norm_name %in% real_params$norm) {
        warning(paste("Norm", norm_name, "not found in real parameters. Skipping."))
        next
    }

    real_interaction <- real_params[norm == norm_name, var_interaction]

    # Calculate Statistics
    # Null Hypothesis: True Interaction Variance is 0.
    # We compare Observed Interaction Variance to the distribution of Simulated Interaction Variances (generated with 0 interaction).
    #
    # The sim results have one row per (model × sim_id), with global_inter_sim
    # repeated for each model within a sim. Get unique values per sim_id.
    null_vars <- unique(sim_data[, .(sim_id, global_inter_sim)])$global_inter_sim
    n_sims <- length(null_vars)

    # P-value: Proportion of simulated variances >= real variance
    # add 1 to both numerator and denominator (Davison-Hinkley)
    n_exceed <- sum(null_vars >= real_interaction)
    p_value <- (n_exceed + 1) / (n_sims + 1)

    # Summary Stats of Null Distribution
    mean_null <- mean(null_vars)
    max_null <- max(null_vars)
    sd_null <- sd(null_vars)

    # Z-score (Effect Size approx)
    z_score <- (real_interaction - mean_null) / sd_null

    final_stats <- rbind(final_stats, data.table(
        norm = norm_name,
        n_sim = n_sims,
        real_var_interaction = real_interaction,
        null_mean_var = mean_null,
        null_max_var = max_null,
        p_value = p_value,
        z_score = z_score
    ))
}

# 3. Save
fwrite(final_stats, OUTPUT_FILE)

message("\n--- Aggregation Complete ---")
print(final_stats)
message(paste("\nSaved to:", OUTPUT_FILE))
