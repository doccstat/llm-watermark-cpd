set.seed(1)

folder <- "results/"
experiment_settings <- paste0(c(0, 1, 2, 4, 9, 19), "changepoints")
# experiment_settings <- c("comment")
models <- c("meta-llama/Meta-Llama-3-8B")
models_folders_prefix <- c("ml3")
generation_methods <- c("gumbel", "transform")
block_size_permutation_pair <- matrix(
  c(
    20, 99,
    20, 249,
    20, 499,
    20, 749,
    20, 999,
    10, 999,
    30, 999,
    40, 999,
    50, 999
  ), ncol = 2, byrow = TRUE
)

filenames_template <- c("", "-edit")
pvalue_files_templates <- NULL
for (model_index in seq_along(models)) {
  for (experiment_index in seq_along(experiment_settings)) {
    for (generation_index in seq_along(generation_methods)) {
      for (distance_index in seq_along(filenames_template)) {
        pvalue_files_templates <- c(pvalue_files_templates, paste0(
          folder,
          models_folders_prefix[model_index],
          "-",
          experiment_settings[experiment_index],
          "-",
          generation_methods[generation_index],
          ".p-detect/XXX-",
          generation_methods[generation_index],
          filenames_template[distance_index],
          "-YYY.csv"
        ))
      }
    }
  }
}

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

prompt_count <- 10
pvalue_files <- NULL

for (pvalue_files_template in pvalue_files_templates) {
  for (prompt_index in seq_len(prompt_count)) {
    # Python index in the file name
    filename <- sub("XXX", prompt_index - 1, pvalue_files_template)
    pvalue_files <- c(pvalue_files, filename)
  }
}

args <- commandArgs(trailingOnly = TRUE)
template_index <- as.integer(args[1])  # Start from 1
prompt_index <- as.integer(args[2])  # Start from 0
seeded_interval_index <- as.integer(args[3])  # Start from 1 to 148 for 1300 tokens

# The parameter `k` used in `textgen`
segment_length <- 20
# as.integer(gsub('^.*B-|-T.*$', '', pvalue_files_templates[template_index]))
seeded_intervals_minimum <- 50
token_count <- 500
# token_count <- 1300
seeded_intervals <- get_seeded_intervals(
  token_count - segment_length,
  decay = sqrt(2), unique.int = TRUE
)
segment_length_cutoff <-
  seeded_intervals[, 2] - seeded_intervals[, 1] >= seeded_intervals_minimum
seeded_intervals <- seeded_intervals[segment_length_cutoff, ]
seeded_intervals <- seeded_intervals + segment_length / 2

filename <- sub("XXX", prompt_index, pvalue_files_templates[template_index])
filename <- sub("YYY", paste0("SeedBS-", seeded_interval_index), filename)
if (!file.exists(filename)) {
  pvalue_vector <- rep(
    NA,
    seeded_intervals[seeded_interval_index, 2] -
      seeded_intervals[seeded_interval_index, 1] + 1
  )
  for (i in seq_len(length(pvalue_vector))) {
    filename <- sub("XXX", prompt_index, pvalue_files_templates[template_index])
    filename <- sub(
      "YYY", seeded_intervals[seeded_interval_index, 1] + i - 1 - 1, filename
    )
    pvalue_vector[i] <- unlist(read.csv(filename, header = FALSE))
  }
  index_p_tilde <- segment_significance(pvalue_vector)
  filename <- sub("XXX", prompt_index, pvalue_files_templates[template_index])
  filename <- sub("YYY", paste0("SeedBS-", seeded_interval_index), filename)
  write.csv(index_p_tilde, filename, row.names = FALSE)
}
