set.seed(1)

folder <- "results/"
models <- c("meta-llama/Meta-Llama-3-8B")
models_folders_prefix <- c("ml3")
generation_methods <- c("gumbel")
cpts <- c(3, 4, 6, 8, 9, 12)
texts <- seq_len(5)
# generation_methods <- c("kirchenbauer")
# cpts <- c(3)
# texts <- seq_len(1)
experiment_settings <- NULL
for (cpt_index in seq_along(cpts)) {
  for (text_index in seq_along(texts)) {
    experiment_settings <- c(
      experiment_settings,
      paste0("random-", cpts[cpt_index], "cpts-text", texts[text_index])
    )
  }
}
experiment_settings <- c(
  experiment_settings,
  "concat-text5",
  "concat-french-text5"
)

pvalue_files_templates <- NULL
for (model_index in seq_along(models)) {
  for (experiment_index in seq_along(experiment_settings)) {
    for (generation_index in seq_along(generation_methods)) {
      pvalue_files_templates <- c(pvalue_files_templates, paste0(
        folder,
        models_folders_prefix[model_index],
        "-",
        experiment_settings[experiment_index],
        "-",
        generation_methods[generation_index],
        ".p-detect/XXX-",
        generation_methods[generation_index],
        "-YYY.csv"
      ))
    }
  }
}

# filenames_to_settings <- list(
#   "B-20-T-99",
#   "B-20-T-249",
#   "B-20-T-499",
#   "B-20-T-749",
#   "B-20-T-999",
#   "B-10-T-999",
#   "B-30-T-999",
#   "B-40-T-999",
#   "B-50-T-999"
# )
# filenames_to_settings <- rep(filenames_to_settings, length(models))
# names(filenames_to_settings) <- pvalue_files_templates
# true_change_points_list <- list(
#   c(100, 200, 300, 400),
#   c(100, 200, 300, 400),
#   c(100, 200, 300, 400),
#   c(100, 200, 300, 400),
#   c(100, 200, 300, 400),
#   c(100, 200, 300, 400),
#   c(100, 200, 300, 400),
#   c(100, 200, 300, 400),
#   c(100, 200, 300, 400)
# )
# true_change_points_list <- rep(true_change_points_list, length(models))
# names(true_change_points_list) <- pvalue_files_templates

significance_permutation_count <- 999

get_seeded_intervals <- function(n, decay = sqrt(2), unique.int = FALSE) {
  n <- as.integer(n)
  depth <- log(n, base = decay)
  depth <- ceiling(depth)

  boundary_mtx <- matrix(NA, ncol = 2)
  colnames(boundary_mtx) <- c("st", "end")
  boundary_mtx[1, ] <- c(1, n)

  depth <- log(n, base = decay)
  depth <- ceiling(depth)


  for (i in 2:depth) {
    int_length <- n * (1 / decay)^(i - 1)

    n_int <- ceiling(round(n / int_length, 14)) * 2 - 1

    boundary_mtx <- rbind(
      boundary_mtx,
      cbind(
        floor(seq(1, n - int_length, length.out = (n_int))),
        ceiling(seq(int_length, n, length.out = (n_int)))
      )
    )
  }

  if (unique.int) {
    return(unique(boundary_mtx))
  }
  boundary_mtx
}

ks_statistic <- function(pvalues) {
  result <- rep(NA, length(pvalues) - 1)
  for (k in seq_len(length(pvalues) - 1)) {
    segment_before <- pvalues[1:k]
    segment_after <- pvalues[(k + 1):length(pvalues)]
    ks_test_stat <- ks.test(segment_before, segment_after)$statistic
    result[k] <-
      k * (length(pvalues) - k) / (length(pvalues))^(3 / 2) * ks_test_stat
  }
  c(which.max(result), max(result))
}

permute_pvalues <- function(pvalues, block_size = 1) {
  pvalue_indices <- seq_len(length(pvalues) - block_size + 1)
  sampled_indices <-
    sample(pvalue_indices, size = ceiling(length(pvalues) / block_size))
  permuted_pvalues <- NULL
  for (sampled_index in sampled_indices) {
    permuted_pvalues <- c(
      permuted_pvalues, pvalues[sampled_index:(sampled_index + block_size - 1)]
    )
  }
  permuted_pvalues[seq_len(length(pvalues))]
}

segment_significance <- function(pvalues) {
  original_ks_statistic <- ks_statistic(pvalues)
  p_tilde <- c(1)
  for (t_prime in seq_len(significance_permutation_count)) {
    pvalues_permuted <- permute_pvalues(pvalues, block_size = 10)
    ks_statistic_permuted <- ks_statistic(pvalues_permuted)
    p_tilde <- c(p_tilde, original_ks_statistic[2] <= ks_statistic_permuted[2])
  }
  c(original_ks_statistic[1], mean(p_tilde))
}

prompt_count <- 1

seeded_intervals_results <- list()
pvalue_matrices <- list()
for (template_index in seq_along(pvalue_files_templates)) {  # nolint
  # The parameter `k` used in `textgen`
  segment_length <- 20
  seeded_intervals_minimum <- 50
  token_count <- 500
  seeded_intervals <- get_seeded_intervals(
    token_count - segment_length,
    decay = sqrt(2), unique.int = TRUE
  )
  segment_length_cutoff <-
    seeded_intervals[, 2] - seeded_intervals[, 1] >= seeded_intervals_minimum
  seeded_intervals <- seeded_intervals[segment_length_cutoff, ]
  seeded_intervals <- seeded_intervals + segment_length / 2

  # The above code should be the same as the one in `ablation-seedbs.R`.

  seeded_intervals_results[[
    pvalue_files_templates[[template_index]]
  ]] <- list()
  for (prompt_index in seq_len(prompt_count)) {
    seeded_intervals_df <- data.frame(
      from = seeded_intervals[, 1],
      to = seeded_intervals[, 2],
      segment_length = seeded_intervals[, 2] - seeded_intervals[, 1],
      index_within_segment = rep(NA, nrow(seeded_intervals)),
      significance = rep(NA, nrow(seeded_intervals))
    )
    for (seeded_interval_index in seq_len(nrow(seeded_intervals))) {
      filename <- sub(
        "XXX",
        prompt_index - 1,
        pvalue_files_templates[template_index]
      )
      filename <-
        sub("YYY", paste0("SeedBS-", seeded_interval_index), filename)
      seeded_intervals_df[
        seeded_interval_index, c("index_within_segment", "significance")
      ] <- c(read.csv(filename, header = TRUE))$x
    }
    seeded_intervals_df$change_point_index <-
      seeded_intervals_df$index_within_segment + seeded_intervals_df$from - 1
    seeded_intervals_results[[
      pvalue_files_templates[[template_index]]
    ]][[prompt_index]] <- seeded_intervals_df
  }

  pvalue_matrices[[pvalue_files_templates[[template_index]]]] <-
    matrix(
      NA,
      nrow = prompt_count,
      ncol = token_count
    )
  for (prompt_index in seq_len(prompt_count)) {
    pvalue_vector <- rep(NA, token_count)
    for (i in seq_len(token_count)) {
      filename <- sub(
        "XXX",
        prompt_index - 1,
        pvalue_files_templates[template_index]
      )
      filename <- sub("YYY", i - 1, filename)
      pvalue_vector[i] <- unlist(read.csv(filename, header = FALSE))
    }
    pvalue_matrices[[
      pvalue_files_templates[[template_index]]
    ]][prompt_index, ] <- pvalue_vector
  }
}

model_index <- 1
rand_index_df <- data.frame(matrix(NA, 0, 3))
for (not_threshold in c(0.05, 0.01, 0.005, 0.001)) { # nolint
  for (
    pvalue_files_template in pvalue_files_templates[
      length(filenames_to_settings) * (model_index - 1) +
        seq_along(filenames_to_settings)
    ]
  ) {
    for (
      prompt_index in seq_len(
        nrow(pvalue_matrices[[pvalue_files_template]])
      )
    ) {
      seeded_intervals_with_ptilde <-
        seeded_intervals_results[[pvalue_files_template]][[prompt_index]]
      significant_indices <-
        seeded_intervals_with_ptilde$significance <= not_threshold
      potential_change_points <-
        seeded_intervals_with_ptilde[significant_indices, ]
      result_df <- potential_change_points[FALSE, ]
      while (nrow(potential_change_points) > 0) {
        min_segment_index <-
          which.min(potential_change_points$segment_length)
        change_point <- potential_change_points[min_segment_index, ]
        result_df <- rbind(result_df, change_point)
        rest_potential <-
          potential_change_points$from > change_point$change_point_index
        rest_potential <- rest_potential | (
          potential_change_points$to < change_point$change_point_index
        )
        potential_change_points <- potential_change_points[rest_potential, ]
      }
      result_df <- result_df[order(result_df$change_point_index), ]

      rand_index_df <- rbind.data.frame(
        rand_index_df,
        cbind(
          fossil::rand.index(
            rep(
              seq_len(length(result_df$change_point_index) + 1),
              times = diff(c(
                0,
                result_df$change_point_index,
                ncol(pvalue_matrices[[pvalue_files_template]])
              ))
            ),
            rep(
              seq_len(
                length(true_change_points_list[[pvalue_files_template]]) + 1
              ),
              times = diff(c(
                0,
                true_change_points_list[[pvalue_files_template]],
                ncol(pvalue_matrices[[pvalue_files_template]])
              ))
            )
          ),
          filenames_to_settings[[pvalue_files_template]],
          not_threshold
        )
      )
    }
  }
}
colnames(rand_index_df) <- c(
  "RandIndex",
  "Setting",
  "Threshold"
)
rand_index_df$RandIndex <- as.numeric(rand_index_df$RandIndex)

ggplot2::ggplot(rand_index_df) +
  ggplot2::geom_boxplot(
    ggplot2::aes(
      x = Threshold,
      y = RandIndex,
      fill = Setting
    )
  ) +
  ggplot2::labs(
    x = "Threshold",
    y = "Rand Index",
    fill = "Setting"
  ) +
  ggplot2::theme_minimal()
ggplot2::ggsave(
  paste0("results/", models_folders_prefix[model_index], "-rand_index.pdf"),
  width = 10,
  height = 4
)

################################################################################

ggplot2::ggplot() +
  ggplot2::geom_line(
    data = data.frame(
      x = seq_len(250),
      y = pvalue_matrices[["results/ml3-concat-text5-gumbel.p-detect/XXX-gumbel-YYY.csv"]][1, seq_len(250)]
    ),
    ggplot2::aes(x = x, y = y)
  ) +
  ggplot2::theme_minimal() +
  ggplot2::labs(x = "Token Index", y = "P-value")
ggplot2::ggsave("results/ml3-concat-text5-gumbel.p-pvalue.pdf", width = 10, height = 4)

seeded_intervals_df$index <- rev(seq_len(nrow(seeded_intervals_df)))

ggplot2::ggplot() +
  ggplot2::geom_segment(
    data = seeded_intervals_df,
    ggplot2::aes(
      x = from,
      xend = to,
      y = index,
      yend = index
    )
  ) +
  ggplot2::theme_minimal() +
  ggplot2::labs(
    x = "Token Index",
    y = "Segment Index"
  )
ggplot2::ggsave("results/ml3-concat-text5-gumbel.p-segments.pdf", width = 10, height = 4)

seeded_intervals_df$ltthreshold <- seeded_intervals_df$significance <= 0.005

ggplot2::ggplot() +
  ggplot2::geom_segment(
    data = seeded_intervals_df,
    ggplot2::aes(
      x = from,
      xend = to,
      y = index,
      yend = index,
      color = ltthreshold
    )
  ) +
  ggplot2::scale_color_manual(
    values = c("black", "red"),
    labels = c("> 0.005", "<= 0.005")
  ) +
  ggplot2::geom_point(
    data = seeded_intervals_df[seeded_intervals_df$ltthreshold, ],
    ggplot2::aes(x = change_point_index, y = index),
    color = "red"
  ) +
  ggplot2::theme_minimal() +
  ggplot2::labs(
    x = "Token Index",
    y = "Segment Index"
  ) +
  ggplot2::guides(color = ggplot2::guide_legend(title = "Significance"))
ggplot2::ggsave("results/ml3-concat-text5-gumbel.p-segments-significance.pdf", width = 10, height = 4)

seeded_intervals_df$selected <- FALSE
seeded_intervals_df$selected[c(36, 41, 13, 18)] <- TRUE

ggplot2::ggplot() +
  ggplot2::geom_segment(
    data = seeded_intervals_df,
    ggplot2::aes(
      x = from,
      xend = to,
      y = index,
      yend = index,
      color = ltthreshold
    )
  ) +
  ggplot2::scale_color_manual(
    values = c("black", "red"),
    labels = c("> 0.005", "<= 0.005")
  ) +
  ggplot2::geom_point(
    data = seeded_intervals_df[seeded_intervals_df$selected, ],
    ggplot2::aes(x = change_point_index, y = index),
    color = "red"
  ) +
  ggplot2::theme_minimal() +
  ggplot2::labs(
    x = "Token Index",
    y = "Segment Index"
  ) +
  ggplot2::guides(color = ggplot2::guide_legend(title = "Significance"))
ggplot2::ggsave("results/ml3-concat-text5-gumbel.p-segments-selected.pdf", width = 10, height = 4)
