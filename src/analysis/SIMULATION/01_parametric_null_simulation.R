#!/usr/bin/env Rscript

# 01_parametric_null_simulation.R (FULL PARAMETRIC BOOTSTRAP)
# Tests if Model-Specific Idiosyncrasy is significant.
# Logic:
# 1. Load Real Data (design structure only — which word-model pairs exist).
# 2. Load estimated variance components from original LMM fit.
# 3. For each simulation:
#    a. Draw FRESH u_word ~ N(0, σ²_word) for each word
#    b. Draw FRESH u_model ~ N(0, σ²_model) for each model
#    c. Draw ε ~ N(0, σ²_resid) for each observation
#    d. Construct y_sim = intercept + u_word + u_model + ε  (interaction = 0 under H₀)
# 4. Refit LMM and extract interaction variance → null distribution.

options(width = 150)
suppressPackageStartupMessages({
    library(data.table)
    library(lme4)
    library(methods)
})

# --- Args ---
args <- commandArgs(trailingOnly = TRUE)
target_norm <- NULL
n_sim <- 100
nrows_arg <- -1
n_cores <- 1

if (length(args) > 0) {
    for (i in seq_along(args)) {
        if (args[i] == "--norm" && i < length(args)) target_norm <- args[i + 1]
        if (args[i] == "--n_sim" && i < length(args)) n_sim <- as.integer(args[i + 1])
        if (args[i] == "--nrows" && i < length(args)) nrows_arg <- as.integer(args[i + 1])
        if (args[i] == "--n_cores" && i < length(args)) n_cores <- as.integer(args[i + 1])
    }
}

if (is.null(target_norm)) stop("❌ Must provide --norm argument!")

# --- Config ---
OUTPUT_DIR <- "outputs/results/LMM_Simulation"
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
OUT_FILE <- file.path(OUTPUT_DIR, paste0("sim_results_", target_norm, ".csv"))

PARAM_DIR <- Sys.getenv("LMM_OUTPUT_DIR", "outputs/results/LMM_Full_filtered")
PARAM_FILE <- file.path(PARAM_DIR, "lme_variance_partitioning_global.csv")

message(paste("▶ FULL PARAMETRIC BOOTSTRAP:", target_norm))

# --- 1. LOAD DATA (Design Structure) ---
clean_dir <- "outputs/results/LMM_Simulation/data_clean"
clean_file <- file.path(clean_dir, paste0(target_norm, "_sim_ready.csv"))

if (!file.exists(clean_file)) {
    stop(paste("❌ Clean data file not found:", clean_file, "\nPlease run 00_prepare_simulation_data.R first."))
}

message(paste("   Loading design structure:", clean_file))
sim_data <- fread(clean_file)

# Basic validation
if (!all(c("model", "norm", "word", "rating_val") %in% names(sim_data))) {
    stop("❌ Loaded data missing required columns (model, norm, word, rating_val)")
}

message(paste("   ✅ Loaded:", nrow(sim_data), "rows."))
message(paste("   Unique Models:", length(unique(sim_data$model))))

# Ensure types
sim_data[, word := as.character(word)]
sim_data[, model := as.character(model)]
sim_data[, rating_val := as.numeric(rating_val)]

# Filter NA ratings just in case
sim_data <- sim_data[!is.na(rating_val)]

# Subsample if requested (by WORDS, not rows, to preserve crossed design)
if (nrows_arg > 0 && nrows_arg < nrow(sim_data)) {
    all_words <- unique(sim_data$word)
    n_words_sample <- min(nrows_arg, length(all_words))
    set.seed(42)
    sampled_words <- sample(all_words, n_words_sample)
    sim_data <- sim_data[word %in% sampled_words]
    message(paste("   Subsampled to", n_words_sample, "words →", nrow(sim_data), "rows"))
}

# --- 2. LOAD VARIANCE COMPONENTS (from original LMM) ---
if (!file.exists(PARAM_FILE)) stop(paste("❌ Global params not found:", PARAM_FILE))

params_df <- fread(PARAM_FILE)
norm_params <- params_df[norm == target_norm]

if (nrow(norm_params) == 0) stop(paste("❌ Norm", target_norm, "not found in", PARAM_FILE))

intercept_real <- norm_params$intercept
var_word_real <- norm_params$var_trait
var_model_real <- norm_params$var_method
var_resid_real <- norm_params$var_residual
var_inter_real <- norm_params$var_interaction # For reference (observed value to beat)

sd_word <- sqrt(var_word_real)
sd_model <- sqrt(var_model_real)
sd_resid <- sqrt(var_resid_real)

message(paste("   Variance components (from original LMM):"))
message(paste(
    "     σ²_word =", round(var_word_real, 6),
    "  σ²_model =", round(var_model_real, 6),
    "  σ²_resid =", round(var_resid_real, 6)
))
message(paste("     Observed σ²_interaction =", round(var_inter_real, 6), "(target to beat)"))

# --- 3. PREPARE DESIGN ---
# Convert to plain data.frame for safe forking
n_obs <- nrow(sim_data)
sim_df <- as.data.frame(sim_data)
sim_df$word <- as.factor(sim_df$word)
sim_df$model <- as.factor(sim_df$model)

unique_words <- levels(sim_df$word)
unique_models <- levels(sim_df$model)
n_words <- length(unique_words)
n_models <- length(unique_models)

# Pre-compute integer indices for fast lookup
word_idx <- as.integer(sim_df$word) # integer index into unique_words
model_idx <- as.integer(sim_df$model) # integer index into unique_models

# Drop columns not needed for simulation to minimize forked memory
sim_df$rating_val <- NULL
sim_df$norm <- NULL
invisible(gc())

message(paste("   Design:", n_words, "words ×", n_models, "models =", n_obs, "observations"))

# --- 4. PARALLEL SIMULATION (BATCHED) ---
# data.table's internal threading deadlocks under fork(); disable before mclapply
if (n_cores > 1) data.table::setDTthreads(1)

# Batch size = n_cores (one sim per core per batch)
batch_size <- n_cores
n_batches <- ceiling(n_sim / batch_size)

message(paste("\n▶ Starting", n_sim, "Full Parametric Bootstrap iterations (", n_cores, "cores,", n_batches, "batches )..."))

run_one_sim <- function(i) {
    tryCatch(
        {
            set.seed(42 + i)

            # A. Draw FRESH random effects (the key difference from conditional sim)
            u_word_new <- rnorm(n_words, mean = 0, sd = sd_word)
            u_model_new <- rnorm(n_models, mean = 0, sd = sd_model)

            # B. Generate data: y = intercept + u_word + u_model + ε
            #    (interaction = 0 under H₀)
            epsilon <- rnorm(n_obs, mean = 0, sd = sd_resid)

            # Build y_simulated directly — sim_df already has word & model factors
            local_df <- sim_df
            local_df$y_simulated <- intercept_real +
                u_word_new[word_idx] +
                u_model_new[model_idx] +
                epsilon

            # C. Fit Model
            fit <- lmer(y_simulated ~ 1 + (1 | word) + (1 | model) + (1 | word:model),
                data = local_df,
                control = lmerControl(optimizer = "bobyqa", calc.derivs = FALSE)
            )

            # D. Extract Variance Components
            vc <- as.data.frame(VarCorr(fit))
            v_i_global <- vc[vc$grp == "word:model", "vcov"]
            if (length(v_i_global) == 0) v_i_global <- 0

            # E. Model-Specific BLUP Variance — pure base R, no data.table
            re <- ranef(fit, condVar = FALSE)
            inter_vals <- re$`word:model`
            ids <- rownames(inter_vals)
            vals <- inter_vals$`(Intercept)`

            # Split "word:model" IDs to extract model name
            parts <- strsplit(ids, ":", fixed = TRUE)
            p1 <- vapply(parts, `[`, character(1), 1)
            p2 <- vapply(parts, `[`, character(1), 2)
            models <- ifelse(p1 %in% unique_models, p1, p2)

            # Variance per model (base R aggregate)
            inter_df <- data.frame(model = models, val = vals, stringsAsFactors = FALSE)
            model_vars <- aggregate(val ~ model, data = inter_df, FUN = var)
            names(model_vars) <- c("model", "var_idio_sim")
            model_vars$sim_id <- i
            model_vars$global_inter_sim <- v_i_global

            return(model_vars)
        },
        error = function(e) {
            message(paste("  ❌ Sim", i, "error:", conditionMessage(e)))
            return(NULL)
        }
    )
}

all_results <- list()

if (n_cores > 1) {
    for (b in seq_len(n_batches)) {
        start_i <- (b - 1) * batch_size + 1
        end_i <- min(b * batch_size, n_sim)
        sim_ids <- start_i:end_i
        message(paste("  Batch", b, "/", n_batches, ": sims", start_i, "-", end_i))
        batch_res <- parallel::mclapply(sim_ids, run_one_sim, mc.cores = n_cores)
        all_results <- c(all_results, batch_res)
        invisible(gc())
    }
} else {
    for (i in 1:n_sim) {
        res <- run_one_sim(i)
        all_results <- c(all_results, list(res))
        if (i %% 10 == 0) message(paste("  Sim", i, "/", n_sim))
    }
}

all_results <- all_results[!sapply(all_results, is.null)]
final_results <- do.call(rbind, all_results)
write.csv(final_results, OUT_FILE, row.names = FALSE)
message(paste("\n✅ Saved detailed simulation results:", OUT_FILE))
