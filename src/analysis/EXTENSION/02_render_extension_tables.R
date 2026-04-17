#!/usr/bin/env Rscript
# ═══════════════════════════════════════════════════════════════════════
# 02_render_extension_tables.R
# Renders EXTENSION analysis results as publication-ready PDF tables
# ═══════════════════════════════════════════════════════════════════════

suppressPackageStartupMessages({
  library(data.table)
  library(grid)
  library(gridExtra)
})

# ── Paths ──
args <- commandArgs(trailingOnly = FALSE)
script_dir <- dirname(sub("--file=", "", args[grep("--file=", args)]))
REPO_ROOT <- normalizePath(file.path(script_dir, "../../.."))
RESULTS_DIR <- file.path(REPO_ROOT, "outputs", "results", "EXTENSION")
PUB_DIR     <- file.path(REPO_ROOT, "publication_plots")
dir.create(PUB_DIR, showWarnings = FALSE, recursive = TRUE)

# ── Display name maps ──
DISPLAY_MODELS <- c(
  "qwen32b" = "Qwen3-32B",
  "qwen3_235b_it" = "Qwen3-235B-A22B",
  "mistral24b" = "Mistral-Small-24B",
  "gemma27b" = "gemma-3-27b-it",
  "gptoss_20b" = "gpt-oss-20b",
  "gptoss_120b" = "gpt-oss-120b",
  "olmo3_32b_it" = "OLMo-3.1-32B",
  "falcon_h1_34b_it" = "Falcon-H1-34B",
  "granite_4_small" = "granite-4.0-h-small",
  "phi_4" = "phi-4"
)

DISPLAY_NORMS <- c(
  "arousal_warriner" = "Arousal",
  "concreteness_brysbaert" = "Concreteness",
  "valence_mohammad" = "Valence",
  "visual_lancaster" = "Visual",
  "auditory_lancaster" = "Auditory",
  "gustatory_lancaster" = "Gustatory",
  "olfactory_lancaster" = "Olfactory",
  "haptic_lancaster" = "Haptic",
  "aoa_kuperman" = "AoA (Kuperman)",
  "aoa_brysbaert" = "AoA (Brysbaert)",
  "morality_troche" = "Morality",
  "gender_association_glasgow" = "Gender Assoc.",
  "humor_engelthaler" = "Humor",
  "socialness_diveica" = "Socialness"
)

# ═══════════════════════════════════════════════════════════════════════
# TABLE 1: Human Correlation Matrix (Model × Norm)
# ═══════════════════════════════════════════════════════════════════════
cat("▶ Rendering Human Correlation Table...\n")

corr <- fread(file.path(RESULTS_DIR, "mode_human_alignment.csv"))

# Map display names
corr[, model_display := DISPLAY_MODELS[model]]
corr[, norm_display  := DISPLAY_NORMS[norm]]

# Pivot wide: rows = models, columns = norms
# Use mean_human_corr (stochastic mean correlation)
wide <- dcast(corr, model_display ~ norm_display, value.var = "mean_human_corr")

# Compute row-wise mean (Fisher-z)
norm_cols <- names(wide)[names(wide) != "model_display"]
wide[, `Mean (Fisher z)` := {
  z_vals <- atanh(as.numeric(unlist(.SD)))
  tanh(mean(z_vals, na.rm = TRUE))
}, by = model_display, .SDcols = norm_cols]

# Sort by mean descending
setorder(wide, -`Mean (Fisher z)`)

# Rename model column
setnames(wide, "model_display", "Model")

# Format to 3 decimal places
for (col in names(wide)[-1]) {
  wide[[col]] <- sprintf("%.3f", wide[[col]])
}

# Render to PDF
tt <- tableGrob(wide, rows = NULL,
  theme = ttheme_minimal(
    base_size = 8,
    core = list(fg_params = list(fontface = "plain", fontsize = 7)),
    colhead = list(fg_params = list(fontface = "bold", fontsize = 7))
  )
)

pdf(file.path(PUB_DIR, "human_correlation_table.pdf"), width = 14, height = 5)
grid.newpage()
grid.text("Human–Model Correlation (Stochastic Mean r)", y = 0.97,
          gp = gpar(fontsize = 11, fontface = "bold"))
pushViewport(viewport(y = 0.45, height = 0.85))
grid.draw(tt)
popViewport()
dev.off()
cat("  ✅ Saved: publication_plots/human_correlation_table.pdf\n")


# ═══════════════════════════════════════════════════════════════════════
# TABLE 2: Stochastic vs Deterministic Comparison
# ═══════════════════════════════════════════════════════════════════════
cat("▶ Rendering Stochastic vs Deterministic Table...\n")

# Pivot: stochastic mean, deterministic, and delta
comp <- corr[, .(
  model_display, norm_display,
  stoch_mean = mean_human_corr,
  determ     = temp0_human_corr,
  mode       = mode_human_corr,
  delta      = mean_human_corr - temp0_human_corr
)]

# Wide pivot for delta
delta_wide <- dcast(comp, model_display ~ norm_display, value.var = "delta")
setnames(delta_wide, "model_display", "Model")

# Compute mean delta per model
delta_cols <- names(delta_wide)[names(delta_wide) != "Model"]
delta_wide[, `Mean Δr` := {
  vals <- as.numeric(unlist(.SD))
  mean(vals, na.rm = TRUE)
}, by = Model, .SDcols = delta_cols]

setorder(delta_wide, -`Mean Δr`)

# Format
for (col in names(delta_wide)[-1]) {
  delta_wide[[col]] <- sprintf("%+.3f", as.numeric(delta_wide[[col]]))
}

tt2 <- tableGrob(delta_wide, rows = NULL,
  theme = ttheme_minimal(
    base_size = 8,
    core = list(fg_params = list(fontface = "plain", fontsize = 7)),
    colhead = list(fg_params = list(fontface = "bold", fontsize = 7))
  )
)

pdf(file.path(PUB_DIR, "stochastic_advantage_table.pdf"), width = 14, height = 5)
grid.newpage()
grid.text("Stochastic Advantage: Δr = r(stochastic mean) − r(deterministic)", y = 0.97,
          gp = gpar(fontsize = 11, fontface = "bold"))
pushViewport(viewport(y = 0.45, height = 0.85))
grid.draw(tt2)
popViewport()
dev.off()
cat("  ✅ Saved: publication_plots/stochastic_advantage_table.pdf\n")


# ═══════════════════════════════════════════════════════════════════════
# TABLE 3: Model Ranking Summary
# ═══════════════════════════════════════════════════════════════════════
cat("▶ Rendering Model Ranking Table...\n")

rank_dt <- corr[, .(
  stoch_mean_r = {z <- atanh(mean_human_corr); tanh(mean(z, na.rm=TRUE))},
  determ_r     = {z <- atanh(temp0_human_corr); tanh(mean(z, na.rm=TRUE))},
  mode_r       = {z <- atanh(mode_human_corr);  tanh(mean(z, na.rm=TRUE))},
  n_norms      = .N,
  mean_n_words = round(mean(n_words, na.rm = TRUE))
), by = model_display]

rank_dt[, delta_r := stoch_mean_r - determ_r]
setorder(rank_dt, -stoch_mean_r)
setnames(rank_dt, "model_display", "Model")

# Format
rank_dt[, stoch_mean_r := sprintf("%.3f", stoch_mean_r)]
rank_dt[, determ_r     := sprintf("%.3f", determ_r)]
rank_dt[, mode_r       := sprintf("%.3f", mode_r)]
rank_dt[, delta_r      := sprintf("%+.3f", as.numeric(delta_r))]
setnames(rank_dt, c("Model", "Stoch. Mean r̄", "Determ. r̄", "Mode r̄", "N Norms", "Mean Words", "Δr̄"))

tt3 <- tableGrob(rank_dt, rows = NULL,
  theme = ttheme_minimal(
    base_size = 9,
    core = list(fg_params = list(fontface = "plain", fontsize = 9)),
    colhead = list(fg_params = list(fontface = "bold", fontsize = 9))
  )
)

pdf(file.path(PUB_DIR, "model_ranking_table.pdf"), width = 10, height = 5)
grid.newpage()
grid.text("Model Human Alignment Ranking (Fisher-z Aggregated)", y = 0.97,
          gp = gpar(fontsize = 12, fontface = "bold"))
pushViewport(viewport(y = 0.45, height = 0.85))
grid.draw(tt3)
popViewport()
dev.off()
cat("  ✅ Saved: publication_plots/model_ranking_table.pdf\n")

cat("\n✅ All extension tables rendered.\n")
