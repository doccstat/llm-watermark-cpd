from numpy import ceil as np_ceil, floor, isnan, nan
from numpy.random import seed as np_random_seed
from random import sample, seed

from pandas import DataFrame

from math import ceil, log, sqrt
from numpy import array, linspace, mean, unique, full
from os import makedirs
from os.path import dirname, exists
from pandas import read_csv
from scipy.stats import ks_2samp

import sys

# Set seed for reproducibility
seed(1)
np_random_seed(1)

# Initialize variables
folder = "results/"
watermark_key_length = 1000
experiment_settings = [0, 1, 2, 4]
rolling_window_size = 20
permutation_count = 999
models = ["meta-llama/Meta-Llama-3-8B"]
models_folders_prefix = ["ml3"]
generation_methods = ["gumbel", "transform"]

# Generate pvalue_files_templates
pvalue_files_templates = []
for model_prefix in models_folders_prefix:
    for gen_method in generation_methods:
        for experiment in experiment_settings:
            template = (
                f"{folder}{model_prefix}-"
                f"{gen_method}-"
                f"{watermark_key_length}-"
                f"{experiment}-"
                f"{rolling_window_size}-"
                f"{permutation_count}-detect/XXX-YYY.csv"
            )
            pvalue_files_templates.append(template)

# Replace 'XXX' and 'YYY' with '0' for the first filename
first_filename = pvalue_files_templates[0].replace(
    "XXX", "0").replace("YYY", "0")

# Read CSV to determine metric_count
if not exists(first_filename):
    raise FileNotFoundError(f"File {first_filename} does not exist.")
metric_count = read_csv(first_filename, header=None).shape[1]


def get_seeded_intervals(n, decay=sqrt(2), unique_int=False):
    n = int(n)
    depth = log(n, decay)
    depth = ceil(depth)

    boundary_mtx = []
    boundary_mtx.append((1, n))

    for i in range(2, depth + 1):
        int_length = n * (1 / decay) ** (i - 1)
        n_int = ceil(round(n / int_length, 14)) * 2 - 1

        starts = floor(linspace(
            1, n - int_length, int(n_int))).astype(int)
        ends = np_ceil(linspace(int_length, n, int(n_int))).astype(int)
        for st, end in zip(starts, ends):
            boundary_mtx.append((st, end))

    boundary_mtx = array(boundary_mtx)

    if unique_int:
        boundary_mtx = unique(boundary_mtx, axis=0)

    return boundary_mtx


def ks_statistic(pvalues):
    result = []
    n = len(pvalues)
    for k in range(1, n):
        segment_before = pvalues[:k]
        segment_after = pvalues[k:]
        if len(segment_before) == 0 or len(segment_after) == 0:
            continue
        ks_test_stat = ks_2samp(segment_before, segment_after).statistic
        value = k * (n - k) / (n ** 1.5) * ks_test_stat
        result.append((k, value))
    if not result:
        return (None, None)
    max_k, max_val = max(result, key=lambda x: x[1])
    return (max_k, max_val)


def permute_pvalues(pvalues, block_size=1):
    n = len(pvalues)
    pvalue_indices = list(range(n - block_size + 1))
    sampled_size = ceil(n / block_size)
    sampled_indices = sample(pvalue_indices, k=sampled_size)

    permuted_pvalues = []
    for idx in sampled_indices:
        permuted_pvalues.extend(pvalues[idx:idx + block_size])

    # Truncate to original length
    return permuted_pvalues[:n]


significance_permutation_count = 999


def segment_significance(pvalues):
    original_ks_statistic = ks_statistic(pvalues)
    if original_ks_statistic[1] is None:
        return (None, None)
    p_tilde = [1]
    for _ in range(significance_permutation_count):
        pvalues_permuted = permute_pvalues(pvalues, block_size=10)
        ks_statistic_permuted = ks_statistic(pvalues_permuted)
        if ks_statistic_permuted[1] is None:
            p_tilde.append(0)
        else:
            p_tilde.append(
                int(original_ks_statistic[1] <= ks_statistic_permuted[1]))
    return (original_ks_statistic[0], mean(p_tilde))


prompt_count = 100
pvalue_files = []

for template in pvalue_files_templates:
    for prompt_idx in range(prompt_count):
        filename = template.replace("XXX", str(prompt_idx))
        pvalue_files.append(filename)

# Parse command line arguments
if len(sys.argv) < 4:
    print("Usage: script.py " +
          "<template_index> <prompt_index> <seeded_interval_index>")
    sys.exit(1)

try:
    template_index = int(sys.argv[1])  # 1 to len(pvalue_files_templates)
    prompt_index = int(sys.argv[2])    # 0 to 99
    seeded_interval_index = int(sys.argv[3])  # 1-based index
except ValueError:
    print("Arguments must be integers.")
    sys.exit(1)

# Adjust indices to 0-based for Python
template_index -= 1
seeded_interval_index -= 1

if not (0 <= template_index < len(pvalue_files_templates)):
    raise IndexError(
        f"template_index must be between 1 and {len(pvalue_files_templates)}."
    )
if not (0 <= prompt_index < prompt_count):
    raise IndexError(f"prompt_index must be between 0 and {prompt_count - 1}.")

seeded_intervals_minimum = 50
token_count = 500

seeded_intervals = get_seeded_intervals(
    token_count - rolling_window_size, decay=sqrt(2), unique_int=True)

# Apply segment length cutoff
segment_length = seeded_intervals[:, 1] - seeded_intervals[:, 0]
segment_length_cutoff = segment_length >= seeded_intervals_minimum
seeded_intervals = seeded_intervals[segment_length_cutoff, :]

# Adjust intervals by adding half the rolling window size
seeded_intervals = seeded_intervals + (rolling_window_size / 2)

# Construct seedbs_filename
seedbs_filename = pvalue_files_templates[template_index].replace(
    "XXX", str(prompt_index))
seedbs_filename = seedbs_filename.replace(
    "YYY", f"SeedBSpy-{seeded_interval_index}")

# Check if the file already exists
if not exists(seedbs_filename):
    interval = seeded_intervals[seeded_interval_index]
    start = int(interval[0])
    end = int(interval[1])

    # Initialize pvalue_matrix with NaNs
    pvalue_matrix = full((end - start + 1, metric_count), nan)

    for i in range(pvalue_matrix.shape[0]):
        pvalue_filename = pvalue_files_templates[template_index].replace(
            "XXX", str(prompt_index))
        current_yyy = start + i - 1  # Adjust for 0-based index
        pvalue_filename = pvalue_filename.replace("YYY", str(current_yyy))

        if not exists(pvalue_filename):
            raise FileNotFoundError(
                f"P-value file {pvalue_filename} does not exist.")

        # Read the CSV and flatten to a list
        pvalues = read_csv(pvalue_filename, header=None).values.flatten()
        pvalue_matrix[i, :] = pvalues

    # Apply segment_significance to each column
    index_p_tilde = []
    for col in range(pvalue_matrix.shape[1]):
        pvalues = pvalue_matrix[:, col]
        if isnan(pvalues).any():
            index_p_tilde.append([None, None])
        else:
            significance = segment_significance(pvalues.tolist())
            index_p_tilde.append(significance)

    # Convert to DataFrame for saving
    index_p_tilde_df = DataFrame(index_p_tilde)

    # Save to CSV without headers and index
    seedbs_dir = dirname(seedbs_filename)
    if not exists(seedbs_dir):
        makedirs(seedbs_dir)

    index_p_tilde_df.to_csv(seedbs_filename, sep=",",
                            header=False, index=False)
