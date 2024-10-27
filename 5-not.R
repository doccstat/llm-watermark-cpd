set.seed(1)

folder <- "results/"
watermark_key_length <- 1000
changepoints <- c(0, 1, 2, 4)
changepoint_locations <- list()
changepoint_locations[["0"]] <- c()
changepoint_locations[["1"]] <- c(250)
changepoint_locations[["2"]] <- c(200, 300)
changepoint_locations[["4"]] <- c(100, 200, 300, 400)
rolling_window_size <- 20
permutation_count <- 999
token_count <- 500
seeded_intervals_minimum <- 50
models <- c("meta-llama/Meta-Llama-3-8B")
models_folders_prefix <- c("ml3")
generation_methods <- c("gumbel", "transform")
# block_size_permutation_pair <- matrix(
#   c(
#     20, 99,
#     20, 249,
#     20, 499,
#     20, 749,
#     20, 999,
#     10, 999,
#     30, 999,
#     40, 999,
#     50, 999
#   ), ncol = 2, byrow = TRUE
# )

pvalue_files_template_matrix <- NULL
pvalue_files_templates <- NULL
for (model_index in seq_along(models)) {
  for (generation_index in seq_along(generation_methods)) {
    for (changepoint_count in changepoints) {
      pvalue_files_template_matrix <- rbind(
        pvalue_files_template_matrix,
        c(
          folder,
          models_folders_prefix[model_index],
          "-",
          generation_methods[generation_index],
          "-",
          watermark_key_length,
          "-",
          changepoint_count,
          "-",
          rolling_window_size,
          "-",
          permutation_count,
          "-detect/XXX-YYY.csv"
        )
      )
      pvalue_files_templates <- c(pvalue_files_templates, paste0(
        folder,
        models_folders_prefix[model_index],
        "-",
        generation_methods[generation_index],
        "-",
        watermark_key_length,
        "-",
        changepoint_count,
        "-",
        rolling_window_size,
        "-",
        permutation_count,
        "-detect/XXX-YYY.csv"
      ))
    }
  }
}

filename <- sub("YYY", 0, sub("XXX", 0, pvalue_files_templates[1]))
metric_count <- ncol(read.csv(filename, header = FALSE))

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

seeded_intervals_list <- list()
seeded_intervals <- get_seeded_intervals(
  token_count - rolling_window_size,
  decay = sqrt(2), unique.int = TRUE
)
segment_length_cutoff <-
  seeded_intervals[, 2] - seeded_intervals[, 1] >= seeded_intervals_minimum
seeded_intervals <- seeded_intervals[segment_length_cutoff, ]
seeded_intervals <- seeded_intervals + rolling_window_size / 2
seeded_intervals_list[[20]] <- seeded_intervals

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

significance_permutation_count <- 999
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

seeded_intervals <- get_seeded_intervals(
  token_count - rolling_window_size,
  decay = sqrt(2), unique.int = TRUE
)
segment_length_cutoff <-
  seeded_intervals[, 2] - seeded_intervals[, 1] >= seeded_intervals_minimum
seeded_intervals <- seeded_intervals[segment_length_cutoff, ]
seeded_intervals <- seeded_intervals + rolling_window_size / 2

prompt_count <- 100
# The above code should be the same as the one in `seedbs.R`.
# pvalue_files <- NULL


# for (pvalue_files_template in pvalue_files_templates) {
#   for (prompt_index in seq_len(prompt_count)) {
#     # Python index in the file name
#     filename <- sub("XXX", prompt_index - 1, pvalue_files_template)
#     pvalue_files <- c(pvalue_files, filename)
#   }
# }

seeded_intervals_results <- list()
pvalue_matrices <- list()
time_matrices <- list()
clusters <- parallel::makeCluster(parallel::detectCores())
doParallel::registerDoParallel(clusters)
for (pvalue_files_template in pvalue_files_templates) {
  seeded_intervals_result <-
    foreach::`%dopar%`(
      foreach::`%:%`(
        foreach::foreach(
          prompt_index = seq_len(prompt_count),
          .combine = rbind
        ),
        foreach::foreach(
          seeded_interval_index = seq_len(nrow(seeded_intervals)),
          .combine = rbind
        )
      ),
      {
        filename <- sub("XXX", prompt_index - 1, pvalue_files_template)
        filename <-
          sub("YYY", paste0("SeedBS-", seeded_interval_index), filename)
        cbind(
          prompt_index,
          seeded_intervals[seeded_interval_index, 1],
          seeded_intervals[seeded_interval_index, 2],
          seeded_intervals[seeded_interval_index, 2] -
            seeded_intervals[seeded_interval_index, 1],
          seq_len(metric_count),
          t(read.csv(filename, header = FALSE))
        )
      }
    )
  pvalue_matrix <- foreach::`%dopar%`(
    foreach::`%:%`(
      foreach::foreach(
        prompt_index = seq_len(prompt_count),
        .combine = rbind
      ),
      foreach::foreach(
        token_index = seq_len(token_count),
        .combine = rbind
      )
    ),
    {
      filename <- sub("XXX", prompt_index - 1, pvalue_files_template)
      filename <- sub("YYY", token_index - 1, filename)
      cbind(
        prompt_index,
        token_index,
        seq_len(metric_count),
        t(read.csv(filename, header = FALSE))
      )
    }
  )
  time_matrix <- foreach::`%dopar%`(
    foreach::`%:%`(
      foreach::foreach(
        prompt_index = seq_len(prompt_count),
        .combine = rbind
      ),
      foreach::foreach(
        token_index = seq_len(token_count),
        .combine = rbind
      )
    ),
    {
      filename <- sub("XXX", prompt_index - 1, pvalue_files_template)
      filename <- sub("YYY", token_index - 1, filename)
      filename <- sub(".csv", "-time.txt", filename)
      c(
        prompt_index,
        token_index,
        as.numeric(read.csv(filename, header = FALSE))
      )
    }
  )

  seeded_intervals_result <- as.data.frame(seeded_intervals_result)
  names(seeded_intervals_result) <- c(
    "prompt_index",
    "from",
    "to",
    "segment_length",
    "metric",
    "index_within_segment",
    "significance"
  )
  seeded_intervals_result$change_point_index <-
    seeded_intervals_result$index_within_segment +
    seeded_intervals_result$from - 1
  seeded_intervals_results[[pvalue_files_template]] <- seeded_intervals_result

  pvalue_matrix <- as.data.frame(pvalue_matrix)
  names(pvalue_matrix) <- c(
    "prompt_index",
    "token_index",
    "metric",
    "pvalue"
  )
  pvalue_matrices[[pvalue_files_template]] <- pvalue_matrix

  time_matrix <- as.data.frame(time_matrix)
  names(time_matrix) <- c("prompt_index", "token_index", "time")
  time_matrices[[pvalue_files_template]] <- time_matrix
}
parallel::stopCluster(clusters)

pvalue_matrix <- NULL
for (pvalue_files_template_index in seq_along(pvalue_files_templates)) {
  pvalue_files_template <- pvalue_files_templates[pvalue_files_template_index]
  pvalue_matrix <- rbind(
    pvalue_matrix,
    cbind(
      paste0(
        "Setting ",
        c(1, 2, 3, NA, 4)[
          as.numeric(
            pvalue_files_template_matrix[pvalue_files_template_index, 8]
          ) + 1
        ]
      ),
      paste0(
        c("EMS", "ITS")[
          (
            pvalue_files_template_matrix[
              pvalue_files_template_index, 4
            ] == "transform"
          ) + 1
        ],
        c("", "L")[
          (pvalue_matrices[[pvalue_files_template]]$metric == "1") + 1
        ]
      ),
      pvalue_matrices[[pvalue_files_template]]
    )[pvalue_matrices[[pvalue_files_template]]$prompt_index <= 10, ]
  )
}
names(pvalue_matrix) <-
  c("Setting", "Method", "prompt_index", "token_index", "metric", "pvalue")
pvalue_matrix$prompt_index <- paste("Prompt", pvalue_matrix$prompt_index)
ggplot2::ggplot(
  pvalue_matrix, ggplot2::aes(
    x = token_index, y = pvalue, color = Setting
  )
) +
  ggplot2::geom_line() +
  ggplot2::facet_grid(Setting + Method ~ prompt_index) +
  ggplot2::theme_minimal() +
  ggplot2::labs(x = "Index", y = "P-Value") +
  ggplot2::scale_x_continuous(breaks = seq(0, token_count, by = 100)) +
  ggplot2::scale_y_continuous(breaks = seq(0, 1, by = 0.25)) +
  ggplot2::theme(legend.position = "none")
ggplot2::ggsave(
  paste0("results/", models_folders_prefix[1], "-pvalue.pdf"),
  width = 15,
  height = 15
)

for (model_index in seq_along(models)) { # nolint
  false_positive_df <- data.frame(matrix(NA, 0, 4))
  rand_index_df <- data.frame(matrix(NA, 0, 5))
  for (not_threshold in c(0.05, 0.01, 0.005, 0.001)) {
    for (pvalue_files_template in pvalue_files_templates[
      pvalue_files_template_matrix[, 2] == models_folders_prefix[model_index]
    ]) {
      for (metric in seq_len(metric_count)) {
        for (prompt_index in seq_len(prompt_count)) {
          seeded_intervals_result <-
            seeded_intervals_results[[pvalue_files_template]]
          significant_indices <-
            seeded_intervals_result$significance <= not_threshold
          prompt_indices <- seeded_intervals_result$prompt_index == prompt_index
          metric_indices <- seeded_intervals_result$metric == metric
          potential_change_points <-
            seeded_intervals_result[
              significant_indices & prompt_indices & metric_indices,
            ]
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

          if (pvalue_files_template %in% pvalue_files_templates[
            pvalue_files_template_matrix[, 8] == 0
          ]) {
            false_positive_df <- rbind.data.frame(
              false_positive_df,
              cbind(
                nrow(result_df),
                pvalue_files_template_matrix[
                  pvalue_files_templates == pvalue_files_template, 4
                ],
                metric,
                not_threshold
              )
            )
          } else {
            rand_index_df <- rbind.data.frame(
              rand_index_df,
              cbind(
                fossil::rand.index(
                  rep(
                    seq_len(length(result_df$change_point_index) + 1),
                    times = diff(c(
                      0,
                      result_df$change_point_index,
                      token_count
                    ))
                  ),
                  rep(
                    seq_len(
                      length(changepoint_locations[[pvalue_files_template_matrix[pvalue_files_templates == pvalue_files_template, 8]]]) + 1
                    ),
                    times = diff(c(
                      0,
                      changepoint_locations[[pvalue_files_template_matrix[pvalue_files_templates == pvalue_files_template, 8]]],
                      token_count
                    ))
                  )
                ),
                pvalue_files_template_matrix[
                  pvalue_files_templates == pvalue_files_template, 4
                ],
                pvalue_files_template_matrix[
                  pvalue_files_templates == pvalue_files_template, 8
                ],
                metric,
                not_threshold
              )
            )
          }
        }
      }
    }
  }
  colnames(false_positive_df) <- c(
    "ChangePointCount",
    "GenerationMethod",
    "Metric",
    "Threshold"
  )
  colnames(rand_index_df) <- c(
    "RandIndex",
    "GenerationMethod",
    "TrueChangePointCount",
    "Metric",
    "Threshold"
  )
  false_positive_df$ChangePointCount <-
    as.integer(false_positive_df$ChangePointCount)
  rand_index_df$RandIndex <- as.numeric(rand_index_df$RandIndex)

  false_positive_df$label <- paste0(
    "Setting 1, ",
    c("EMS", "ITS")[1 + (false_positive_df$GenerationMethod == "transform")],
    c("", "L")[1 + (false_positive_df$Metric == 1)]
  )
  rand_index_df$label <- paste0(
    "Setting ",
    c("2", "3", NA, "4")[as.numeric(rand_index_df$TrueChangePointCount)],
    ", ",
    c("EMS", "ITS")[1 + (rand_index_df$GenerationMethod == "transform")],
    c("", "L")[1 + (rand_index_df$Metric == 1)]
  )

  ggplot2::ggplot(false_positive_df) +
    ggplot2::geom_boxplot(
      ggplot2::aes(
        x = Threshold,
        y = ChangePointCount,
        fill = label
      )
    ) +
    ggplot2::labs(
      x = "Threshold",
      y = "Change Point Count",
      fill = "Setting"
    ) +
    ggplot2::theme_minimal() +
    ggplot2::theme(
      legend.position = "inside",
      legend.position.inside = c(0.5, 0.9),
      legend.direction = "horizontal",
      legend.title = ggplot2::element_blank()
    )
  ggplot2::ggsave(
    paste0(
      "results/", models_folders_prefix[model_index], "-false_positives.pdf"
    ),
    width = 8,
    height = 4
  )

  ggplot2::ggplot(rand_index_df) +
    ggplot2::geom_boxplot(
      ggplot2::aes(
        x = Threshold,
        y = RandIndex,
        fill = label
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
}
