#!/usr/bin/env Rscript
# src/analysis/LMM/06_combined_figure_publication.R
#
# Combined PNAS-compliant figure: Panel A (variance partitioning) + Panel B (specificity ratios)
# Outputs a single PDF at 18.0 cm width with Helvetica font.
#
# Usage:
#   Rscript src/analysis/LMM/06_combined_figure_publication.R [--short]

suppressPackageStartupMessages({
    library(ggplot2)
    library(data.table)
    library(dplyr)
    library(tidyr)
    library(stringr)
    library(patchwork)
})

set.seed(42) # Reproducibility for jitter

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)
is_short_format <- "--short" %in% args || "-s" %in% args

# ── Config ──
REPO_ROOT <- getwd()
LMM_DIR <- file.path(REPO_ROOT, "outputs", "results", "LMM_Full_filtered")
SPEC_DIR <- file.path(REPO_ROOT, "outputs", "results", "SPECIFICITY")
PUB_DIR <- file.path(REPO_ROOT, "publication_plots")
dir.create(PUB_DIR, recursive = TRUE, showWarnings = FALSE)

# Model display names — accurate HuggingFace IDs (org/model),
# dropping -Instruct / version suffixes (all models are instruction-tuned)
model_mapping <- c(
    "qwen32b"           = "Qwen3-32B",
    "qwen3_235b_it"     = "Qwen3-235B-A22B",
    "mistral24b"        = "Mistral-Small-24B",
    "gemma27b"          = "gemma-3-27b-it",
    "gptoss_20b"        = "gpt-oss-20b",
    "gptoss_120b"       = "gpt-oss-120b",
    "olmo3_32b_it"      = "OLMo-3.1-32B",
    "falcon_h1_34b_it"  = "Falcon-H1-34B",
    "granite_4_small"   = "granite-4.0-h-small",
    "phi_4"             = "phi-4"
)

# Model display order (short names first for better margin fit)
model_display_order <- c(
    "Qwen3-32B",
    "Qwen3-235B-A22B",
    "phi-4",
    "OLMo-3.1-32B",
    "Mistral-Small-24B",
    "granite-4.0-h-small",
    "gpt-oss-20b",
    "gpt-oss-120b",
    "gemma-3-27b-it",
    "Falcon-H1-34B"
)

# ════════════════════════════════════════════════════════════
#  PANEL A — Variance Partitioning (faceted bar chart)
# ════════════════════════════════════════════════════════════
message("▶ Building Panel A: Variance Partitioning...")

df_global <- fread(file.path(LMM_DIR, "lme_variance_partitioning_global.csv"))
df_specific <- fread(file.path(LMM_DIR, "lme_variance_partitioning_per_model.csv"))

# Apply model display names
df_specific[, model := sapply(model, function(x) {
    if (x %in% names(model_mapping)) model_mapping[x] else gsub("_", "-", x)
})]

# Prepare data
df_plot <- merge(df_specific, df_global, by = "norm", suffixes = c("_spec", "_glob"))
df_plot[, comp_trait := var_trait]
df_plot[, comp_residual := var_residual]
df_plot[, comp_bias := bias_shift^2]
df_plot[, comp_idiosyncrasy := var_idiosyncrasy]
df_plot[is.na(comp_idiosyncrasy), comp_idiosyncrasy := 0]
df_plot[, model_total_var := comp_trait + comp_residual + comp_bias + comp_idiosyncrasy]
df_plot[, prop_trait := comp_trait / model_total_var]
df_plot[, prop_residual := comp_residual / model_total_var]
df_plot[, prop_bias := comp_bias / model_total_var]
df_plot[, prop_idiosyncrasy := comp_idiosyncrasy / model_total_var]

# Norm aggregation
get_norm_display_name <- function(n) {
    case_when(
        n %in% c(
            "auditory_lancaster", "gustatory_lancaster", "haptic_lancaster",
            "olfactory_lancaster", "visual_lancaster"
        ) ~ "Sensory Norms",
        n == "valence_mohammad" ~ "Valence",
        n %in% c("aoa_brysbaert", "aoa_kuperman") ~ "Age of Acquisition",
        n == "arousal_warriner" ~ "Arousal",
        n == "concreteness_brysbaert" ~ "Concreteness",
        n == "gender_association_glasgow" ~ "Gender Association",
        n == "humor_engelthaler" ~ "Humor",
        n == "morality_troche" ~ "Morality",
        n == "socialness_diveica" ~ "Socialness",
        TRUE ~ str_to_title(gsub("_", " ", n))
    )
}
df_plot[, norm_display := get_norm_display_name(norm), by = norm]

df_agg <- df_plot[, .(
    prop_trait = mean(prop_trait),
    prop_residual = mean(prop_residual),
    prop_bias = mean(prop_bias),
    prop_idiosyncrasy = mean(prop_idiosyncrasy)
), by = .(model, norm_display)]

df_long <- melt(df_agg,
    id.vars = c("norm_display", "model"),
    measure.vars = c("prop_trait", "prop_bias", "prop_idiosyncrasy", "prop_residual"),
    variable.name = "component",
    value.name = "proportion"
)
levels(df_long$component) <- c("Trait (Consensus)", "Bias (Model Offset)", "Idiosyncrasy (Model \u00d7 Word)", "Residual (Noise)")
df_long$model <- factor(df_long$model, levels = model_display_order)

# Colors
cols_a <- c(
    "Trait (Consensus)" = "#2ca02c",
    "Bias (Model Offset)" = "#1f77b4",
    "Idiosyncrasy (Model \u00d7 Word)" = "#ff7f0e",
    "Residual (Noise)" = "#d62728"
)

p_a <- ggplot(df_long, aes(x = model, y = proportion, fill = component)) +
    geom_bar(stat = "identity", width = 0.85) +
    facet_wrap(~norm_display, ncol = 3) +
    scale_fill_manual(values = cols_a) +
    scale_y_continuous(breaks = seq(0, 1, 0.5), expand = expansion(mult = c(0, 0.02))) +
    labs(x = NULL, y = "Proportion of Variance", fill = NULL) +
    theme_minimal(base_size = 8, base_family = "Helvetica") +
    theme(
        axis.text.x = element_text(angle = 30, hjust = 1, vjust = 1, size = 6, margin = margin(t = -8)),
        axis.text.y = element_text(size = 6),
        axis.title.y = element_text(size = 7),
        legend.position = "top",
        legend.text = element_text(size = 6.5),
        legend.margin = margin(t = 0, b = -2),
        legend.key.size = unit(0.3, "cm"),
        strip.text = element_text(face = "bold", size = 7.5, margin = margin(b = 1, t = 1)),
        panel.grid.major.x = element_blank(),
        panel.spacing.y = unit(0.2, "lines"),
        panel.spacing.x = unit(0.3, "lines"),
        plot.margin = margin(t = 1, r = 2, b = 0, l = 2)
    )


# ════════════════════════════════════════════════════════════
#  PANEL B — Specificity Ratios (box + scatter)
# ════════════════════════════════════════════════════════════
message("▶ Building Panel B: Specificity Ratios...")

# Load data
df_idio <- fread(file.path(SPEC_DIR, "idiosyncrasy_specificity.csv"))
df_rate <- fread(file.path(SPEC_DIR, "ratings_specificity.csv"))

spec_col <- "spec_vs_pairwise"

# Apply model names
apply_model_name <- function(x) {
    ifelse(x %in% names(model_mapping), model_mapping[x], gsub("_", "-", x))
}
df_idio[, model := apply_model_name(target_model)]
df_rate[, model := apply_model_name(target_model)]

# Norm categories (Sensory vs Other)
sensory_norms <- c(
    "auditory_lancaster", "gustatory_lancaster", "haptic_lancaster",
    "olfactory_lancaster", "visual_lancaster"
)
df_idio[, norm_cat := ifelse(target_norm %in% sensory_norms, "Sensory", "Other")]
df_rate[, norm_cat := ifelse(target_norm %in% sensory_norms, "Sensory", "Other")]

# Tag analysis type and combine
df_idio[, analysis := "Idiosyncrasy"]
df_rate[, analysis := "Ratings"]
df_idio[, spec_val := get(spec_col)]
df_rate[, spec_val := get(spec_col)]

# Mean per model per analysis
mean_idio <- df_idio[, .(mean_spec = mean(spec_val)), by = model]
mean_rate <- df_rate[, .(mean_spec = mean(spec_val)), by = model]

# Create dodge position data
offset <- 0.18
box_w <- 0.28

# Use same model order as Panel A
model_levels_b <- model_display_order
n_models <- length(model_levels_b)
model_num <- setNames(seq_len(n_models), model_levels_b)

# --- Idiosyncrasy data ---
df_idio_plot <- df_idio[, .(model, target_norm, spec_val, norm_cat, analysis)]
df_idio_plot[, x := model_num[model] - offset]
df_idio_plot[, jitter_x := x + runif(.N, -box_w * 0.35, box_w * 0.35)]

# --- Ratings data ---
df_rate_plot <- df_rate[, .(model, target_norm, spec_val, norm_cat, analysis)]
df_rate_plot[, x := model_num[model] + offset]
df_rate_plot[, jitter_x := x + runif(.N, -box_w * 0.35, box_w * 0.35)]

# Combined data
df_b <- rbindlist(list(df_idio_plot, df_rate_plot))

# Analysis colors
analysis_cols <- c("Idiosyncrasy" = "#ff7f0e", "Ratings" = "#9a9fb0")

# Mean markers
mean_data <- rbind(
    mean_idio[, .(model, mean_spec, analysis = "Idiosyncrasy")],
    mean_rate[, .(model, mean_spec, analysis = "Ratings")]
)
mean_data[, x := ifelse(analysis == "Idiosyncrasy",
    model_num[model] - offset,
    model_num[model] + offset
)]

# Build plot using base graphics-style approach with ggplot
p_b <- ggplot() +
    # --- Boxplots for each analysis ---
    # Idiosyncrasy boxes
    geom_boxplot(
        data = df_idio_plot,
        aes(x = x, y = spec_val, group = interaction(model, analysis)),
        width = box_w,
        fill = alpha("#ff7f0e", 0.18),
        color = alpha("#ff7f0e", 0.6),
        outlier.shape = NA,
        coef = 1.5
    ) +
    # Ratings boxes
    geom_boxplot(
        data = df_rate_plot,
        aes(x = x, y = spec_val, group = interaction(model, analysis)),
        width = box_w,
        fill = alpha("#9a9fb0", 0.18),
        color = alpha("#9a9fb0", 0.6),
        outlier.shape = NA,
        coef = 1.5
    ) +
    # --- Jittered individual dots colored by analysis ---
    geom_point(
        data = df_idio_plot,
        aes(x = jitter_x, y = spec_val),
        size = 0.9, alpha = 0.55, color = "#ff7f0e"
    ) +
    geom_point(
        data = df_rate_plot,
        aes(x = jitter_x, y = spec_val),
        size = 0.9, alpha = 0.30, color = "#9a9fb0"
    ) +
    # --- Mean markers (small) ---
    # Idiosyncrasy means (circles)
    geom_point(
        data = mean_data[analysis == "Idiosyncrasy"],
        aes(x = x, y = mean_spec),
        shape = 22, size = 1.8, fill = "#ff7f0e", color = "white", stroke = 0.4
    ) +
    # Ratings means (squares)
    geom_point(
        data = mean_data[analysis == "Ratings"],
        aes(x = x, y = mean_spec),
        shape = 22, size = 1.8, fill = "#9a9fb0", color = "white", stroke = 0.4
    ) +
    # Reference line at 1.0
    geom_hline(yintercept = 1.0, linetype = "dotted", color = "black", linewidth = 0.5) +
    # Axis — wrap long model names across two lines for compact horizontal labels
    scale_x_continuous(
        breaks = seq_len(n_models),
        labels = sapply(model_levels_b, function(lab) {
            if (nchar(lab) > 12) {
                # Split at the last hyphen before or near the midpoint
                mid <- nchar(lab) %/% 2
                # Find all hyphen positions
                hpos <- gregexpr("-", lab)[[1]]
                if (any(hpos > 0)) {
                    # Pick the hyphen closest to the midpoint
                    best <- hpos[which.min(abs(hpos - mid))]
                    paste0(substr(lab, 1, best), "\n", substr(lab, best + 1, nchar(lab)))
                } else {
                    lab
                }
            } else {
                lab
            }
        }),
        expand = c(0.03, 0.03)
    ) +
    labs(
        x = NULL,
        y = "Specificity Ratio"
    ) +
    theme_minimal(base_size = 8, base_family = "Helvetica") +
    theme(
        axis.text.x = element_text(angle = 0, hjust = 0.5, vjust = 1, size = 5.5, lineheight = 0.85, margin = margin(t = -2)),
        axis.text.y = element_text(size = 6),
        axis.title.y = element_text(size = 7),
        panel.grid.major.x = element_blank(),
        legend.text = element_text(size = 6.5),
        legend.key.size = unit(0.3, "cm"),
        legend.position = "top",
        legend.margin = margin(t = 0, b = -4),
        plot.margin = margin(t = 0, r = 2, b = 0, l = 2)
    )

# Add Idiosyncrasy/Ratings legend entries via invisible dummy points
legend_data <- data.frame(
    x = c(-10, -10),
    y = c(-10, -10),
    Analysis = factor(c("Idiosyncrasy", "Ratings"), levels = c("Idiosyncrasy", "Ratings"))
)

p_b <- p_b +
    geom_point(
        data = legend_data,
        aes(x = x, y = y, shape = Analysis, fill = Analysis),
        size = 1.8, color = "white", stroke = 0.4
    ) +
    scale_shape_manual(values = c("Idiosyncrasy" = 22, "Ratings" = 22), name = NULL) +
    scale_fill_manual(
        values = c("Idiosyncrasy" = "#ff7f0e", "Ratings" = "#9a9fb0"),
        name = NULL
    ) +
    guides(
        fill = guide_legend(order = 1, override.aes = list(size = 1.8, shape = c(22, 22), color = "white")),
        shape = "none"
    ) +
    coord_cartesian(xlim = c(0.5, n_models + 0.5), ylim = c(-0.5, 7.5))


# ════════════════════════════════════════════════════════════
#  COMBINE PANELS
# ════════════════════════════════════════════════════════════
message("▶ Combining panels...")

if (is_short_format) {
    # Panel A compressed, Panel B gets more vertical space
    combined <- p_a / p_b +
        plot_layout(heights = c(4.5, 5)) +
        plot_annotation(
            tag_levels = "A",
            theme = theme(
                plot.tag = element_text(size = 10, face = "bold", family = "Helvetica")
            )
        )

    # PNAS 2-column width: 18.0 cm = 7.087 in
    output_file <- file.path(PUB_DIR, "figure1_combined_short.pdf")
    ggsave(output_file, combined,
        width = 7.087, height = 5.0,
        units = "in", device = "pdf"
    )

    message(paste("  ✅ Saved combined figure to:", output_file))
    message(paste("  📐 Width: 18.0 cm (7.09 in) | Height: 12.7 cm (5.0 in)"))
} else {
    combined <- p_a / p_b +
        plot_layout(heights = c(5, 4)) +
        plot_annotation(
            tag_levels = "A",
            theme = theme(
                plot.tag = element_text(size = 14, face = "bold", family = "Helvetica")
            )
        )

    output_file <- file.path(PUB_DIR, "figure1_combined.pdf")
    ggsave(output_file, combined,
        width = 10, height = 12,
        units = "in", device = "pdf"
    )
    message(paste("  ✅ Saved combined figure to:", output_file))
}

# ════════════════════════════════════════════════════════════
#  VARIANCE COMPONENT TABLES (for manuscript reporting)
# ════════════════════════════════════════════════════════════
message("▶ Computing variance component summary tables...")

TABLE_DIR <- file.path(REPO_ROOT, "outputs", "results", "LMM_Full_filtered")

# ── Table 1: Global LMM variance proportions per individual norm ──
tbl_per_norm <- df_global[, .(
    norm,
    prop_trait     = round(prop_trait * 100, 1),
    prop_bias      = round(prop_method * 100, 1),
    prop_idiosyncrasy = round(prop_interaction * 100, 1),
    prop_residual  = round(prop_residual * 100, 1)
)]
tbl_per_norm <- tbl_per_norm[order(prop_idiosyncrasy)]

# Add summary rows
summary_mean <- data.table(
    norm = "MEAN",
    prop_trait     = round(mean(df_global$prop_trait) * 100, 1),
    prop_bias      = round(mean(df_global$prop_method) * 100, 1),
    prop_idiosyncrasy = round(mean(df_global$prop_interaction) * 100, 1),
    prop_residual  = round(mean(df_global$prop_residual) * 100, 1)
)
summary_median <- data.table(
    norm = "MEDIAN",
    prop_trait     = round(median(df_global$prop_trait) * 100, 1),
    prop_bias      = round(median(df_global$prop_method) * 100, 1),
    prop_idiosyncrasy = round(median(df_global$prop_interaction) * 100, 1),
    prop_residual  = round(median(df_global$prop_residual) * 100, 1)
)
tbl_per_norm <- rbindlist(list(tbl_per_norm, summary_mean, summary_median))

out_per_norm <- file.path(TABLE_DIR, "variance_proportions_per_norm.csv")
fwrite(tbl_per_norm, out_per_norm)
message(paste("  ✅ Per-norm table saved to:", out_per_norm))

# ── Table 2: Aggregated variance proportions (matching figure facets) ──
# Use the same norm_display grouping as Panel A
df_global[, norm_display := get_norm_display_name(norm), by = norm]

tbl_agg <- df_global[, .(
    prop_trait     = round(mean(prop_trait) * 100, 1),
    prop_bias      = round(mean(prop_method) * 100, 1),
    prop_idiosyncrasy = round(mean(prop_interaction) * 100, 1),
    prop_residual  = round(mean(prop_residual) * 100, 1),
    n_norms        = .N
), by = .(dimension = norm_display)]
tbl_agg <- tbl_agg[order(prop_idiosyncrasy)]

# Add summary rows
agg_mean <- data.table(
    dimension = "MEAN",
    prop_trait     = round(mean(tbl_agg$prop_trait), 1),
    prop_bias      = round(mean(tbl_agg$prop_bias), 1),
    prop_idiosyncrasy = round(mean(tbl_agg$prop_idiosyncrasy), 1),
    prop_residual  = round(mean(tbl_agg$prop_residual), 1),
    n_norms        = sum(tbl_agg$n_norms)
)
agg_median <- data.table(
    dimension = "MEDIAN",
    prop_trait     = round(median(tbl_agg$prop_trait), 1),
    prop_bias      = round(median(tbl_agg$prop_bias), 1),
    prop_idiosyncrasy = round(median(tbl_agg$prop_idiosyncrasy), 1),
    prop_residual  = round(median(tbl_agg$prop_residual), 1),
    n_norms        = NA_integer_
)
tbl_agg <- rbindlist(list(tbl_agg, agg_mean, agg_median))

out_agg <- file.path(TABLE_DIR, "variance_proportions_aggregated.csv")
fwrite(tbl_agg, out_agg)
message(paste("  ✅ Aggregated table saved to:", out_agg))

# ── Dataset export: Per-norm variance proportions (for manuscript verification) ──
DATASET_DIR <- file.path(REPO_ROOT, "outputs", "datasets")
dir.create(DATASET_DIR, recursive = TRUE, showWarnings = FALSE)

pernorm_props <- copy(tbl_per_norm)
fwrite(pernorm_props, file.path(DATASET_DIR, "variance_proportions_per_norm.csv"))
message(paste("  ✅ Per-norm proportions saved to:", file.path(DATASET_DIR, "variance_proportions_per_norm.csv")))

# ── Print tables to console ──
message("\n══ Per-Norm Variance Proportions (%) ══")
message(sprintf("%-35s %7s %7s %7s %7s", "Norm", "Trait", "Bias", "Idio", "Resid"))
message(strrep("-", 75))
for (i in seq_len(nrow(tbl_per_norm))) {
    row <- tbl_per_norm[i]
    message(sprintf("%-35s %6.1f%% %6.1f%% %6.1f%% %6.1f%%",
        row$norm, row$prop_trait, row$prop_bias,
        row$prop_idiosyncrasy, row$prop_residual))
}

message("\n══ Aggregated Variance Proportions (%) — matching figure facets ══")
message(sprintf("%-25s %7s %7s %7s %7s %7s", "Dimension", "Trait", "Bias", "Idio", "Resid", "Norms"))
message(strrep("-", 75))
for (i in seq_len(nrow(tbl_agg))) {
    row <- tbl_agg[i]
    n_str <- ifelse(is.na(row$n_norms), "", as.character(row$n_norms))
    message(sprintf("%-25s %6.1f%% %6.1f%% %6.1f%% %6.1f%% %5s",
        row$dimension, row$prop_trait, row$prop_bias,
        row$prop_idiosyncrasy, row$prop_residual, n_str))
}
message("")

# ── Render tables as formatted PDF in publication_plots/ ──
message("▶ Rendering variance component tables as PDF...")

suppressPackageStartupMessages({
    library(grid)
    library(gridExtra)
})

# Helper: format a data.table for display with "%" suffixes
format_pct_table <- function(dt, label_col) {
    out <- copy(dt)
    pct_cols <- setdiff(names(out), c(label_col, "n_norms"))
    for (col in pct_cols) {
        out[[col]] <- sprintf("%.1f%%", out[[col]])
    }
    out
}

# ── Prepare Table 1 (per-norm) ──
tbl1 <- format_pct_table(tbl_per_norm, "norm")
# Pretty column names
setnames(tbl1, c("Norm", "Trait (%)", "Bias (%)", "Idiosyncrasy (%)", "Residual (%)"))

n_data_1 <- nrow(tbl_per_norm) - 2  # data rows (excl. MEAN/MEDIAN)

# Row styling: alternating fills for data rows, grey for summary
fill_1 <- rep(c("grey97", "white"), length.out = n_data_1)
fill_1 <- c(fill_1, "grey85", "grey85")  # MEAN, MEDIAN rows

bold_1 <- rep("plain", nrow(tbl1))
bold_1[(n_data_1 + 1):nrow(tbl1)] <- "bold"

theme_1 <- ttheme_minimal(
    base_size = 8,
    base_family = "Helvetica",
    core = list(
        bg_params = list(fill = fill_1),
        fg_params = list(fontface = bold_1, hjust = 0, x = 0.05)
    ),
    colhead = list(
        fg_params = list(fontface = "bold", hjust = 0, x = 0.05),
        bg_params = list(fill = "grey70")
    )
)
grob_1 <- tableGrob(tbl1, rows = NULL, theme = theme_1)

# ── Prepare Table 2 (aggregated) ──
tbl2 <- copy(tbl_agg)
tbl2[is.na(n_norms), n_norms := NA_integer_]
tbl2_fmt <- format_pct_table(tbl2, "dimension")
# Format n_norms column
tbl2_fmt[, n_norms := ifelse(is.na(tbl2$n_norms), "", as.character(tbl2$n_norms))]
setnames(tbl2_fmt, c("Dimension", "Trait (%)", "Bias (%)", "Idiosyncrasy (%)", "Residual (%)", "N Norms"))

n_data_2 <- nrow(tbl_agg) - 2
fill_2 <- rep(c("grey97", "white"), length.out = n_data_2)
fill_2 <- c(fill_2, "grey85", "grey85")

bold_2 <- rep("plain", nrow(tbl2_fmt))
bold_2[(n_data_2 + 1):nrow(tbl2_fmt)] <- "bold"

theme_2 <- ttheme_minimal(
    base_size = 8,
    base_family = "Helvetica",
    core = list(
        bg_params = list(fill = fill_2),
        fg_params = list(fontface = bold_2, hjust = 0, x = 0.05)
    ),
    colhead = list(
        fg_params = list(fontface = "bold", hjust = 0, x = 0.05),
        bg_params = list(fill = "grey70")
    )
)
grob_2 <- tableGrob(tbl2_fmt, rows = NULL, theme = theme_2)

# ── Titles ──
title_1 <- textGrob("Table 1: Variance Proportions per Norm (14 norms)",
    gp = gpar(fontface = "bold", fontsize = 10, fontfamily = "Helvetica"),
    hjust = 0, x = 0.02
)
title_2 <- textGrob("Table 2: Variance Proportions per Aggregated Dimension (9 dimensions)",
    gp = gpar(fontface = "bold", fontsize = 10, fontfamily = "Helvetica"),
    hjust = 0, x = 0.02
)

# ── Prepare Table 3 (specificity ratios per model) ──
message("▶ Computing specificity ratio summary table...")

# Average specificity per model
spec_idio_model <- df_idio[, .(
    mean_spec  = round(mean(spec_val), 2),
    median_spec = round(median(spec_val), 2),
    min_spec   = round(min(spec_val), 2),
    max_spec   = round(max(spec_val), 2)
), by = model]

spec_rate_model <- df_rate[, .(
    mean_spec  = round(mean(spec_val), 2),
    median_spec = round(median(spec_val), 2),
    min_spec   = round(min(spec_val), 2),
    max_spec   = round(max(spec_val), 2)
), by = model]

# Merge side-by-side
tbl_spec <- merge(spec_idio_model, spec_rate_model, by = "model", suffixes = c("_idio", "_rate"))
tbl_spec <- tbl_spec[order(-mean_spec_idio)]

# Add grand summary
spec_grand <- data.table(
    model = "MEAN",
    mean_spec_idio   = round(mean(spec_idio_model$mean_spec), 2),
    median_spec_idio = round(median(spec_idio_model$median_spec), 2),
    min_spec_idio    = round(min(spec_idio_model$min_spec), 2),
    max_spec_idio    = round(max(spec_idio_model$max_spec), 2),
    mean_spec_rate   = round(mean(spec_rate_model$mean_spec), 2),
    median_spec_rate = round(median(spec_rate_model$median_spec), 2),
    min_spec_rate    = round(min(spec_rate_model$min_spec), 2),
    max_spec_rate    = round(max(spec_rate_model$max_spec), 2)
)
tbl_spec <- rbindlist(list(tbl_spec, spec_grand))

# Save CSV
out_spec <- file.path(TABLE_DIR, "specificity_ratios_per_model.csv")
fwrite(tbl_spec, out_spec)
message(paste("  ✅ Specificity table saved to:", out_spec))

# Print to console
message("\n══ Specificity Ratios per Model ══")
message(sprintf("%-22s %8s %8s %8s %8s │ %8s %8s %8s %8s",
    "Model", "I.Mean", "I.Med", "I.Min", "I.Max", "R.Mean", "R.Med", "R.Min", "R.Max"))
message(strrep("-", 105))
for (i in seq_len(nrow(tbl_spec))) {
    row <- tbl_spec[i]
    message(sprintf("%-22s %8.2f %8.2f %8.2f %8.2f │ %8.2f %8.2f %8.2f %8.2f",
        row$model, row$mean_spec_idio, row$median_spec_idio, row$min_spec_idio, row$max_spec_idio,
        row$mean_spec_rate, row$median_spec_rate, row$min_spec_rate, row$max_spec_rate))
}

# Format for PDF table
tbl3_fmt <- copy(tbl_spec)
setnames(tbl3_fmt, c("Model",
    "Mean", "Median", "Min", "Max",
    "Mean ", "Median ", "Min ", "Max "))

n_data_3 <- nrow(tbl_spec) - 1  # data rows excl. MEAN
fill_3 <- rep(c("grey97", "white"), length.out = n_data_3)
fill_3 <- c(fill_3, "grey85")

bold_3 <- rep("plain", nrow(tbl3_fmt))
bold_3[nrow(tbl3_fmt)] <- "bold"

theme_3 <- ttheme_minimal(
    base_size = 7,
    base_family = "Helvetica",
    core = list(
        bg_params = list(fill = fill_3),
        fg_params = list(fontface = bold_3, hjust = 0, x = 0.05)
    ),
    colhead = list(
        fg_params = list(fontface = "bold", hjust = 0, x = 0.05),
        bg_params = list(fill = "grey70")
    )
)
grob_3 <- tableGrob(tbl3_fmt, rows = NULL, theme = theme_3)

# Add column group labels above the header
header_idio <- textGrob("Idiosyncrasy", gp = gpar(fontface = "bold", fontsize = 8, fontfamily = "Helvetica"))
header_rate <- textGrob("Ratings", gp = gpar(fontface = "bold", fontsize = 8, fontfamily = "Helvetica"))

# ── Combine and save ──
title_3 <- textGrob("Table 3: Specificity Ratios per Model (Idiosyncrasy vs Ratings)",
    gp = gpar(fontface = "bold", fontsize = 10, fontfamily = "Helvetica"),
    hjust = 0, x = 0.02
)

table_out <- file.path(PUB_DIR, "variance_proportions_table.pdf")
pdf(table_out, width = 7.087, height = 13.5, family = "Helvetica")
grid.arrange(
    title_1, grob_1,
    textGrob("", gp = gpar(fontsize = 4)),  # spacer
    title_2, grob_2,
    textGrob("", gp = gpar(fontsize = 4)),  # spacer
    title_3, grob_3,
    ncol = 1,
    heights = c(0.5, n_data_1 + 3, 0.3, 0.5, n_data_2 + 3, 0.3, 0.5, n_data_3 + 3)
)
dev.off()
message(paste("  ✅ Table PDF saved to:", table_out))
