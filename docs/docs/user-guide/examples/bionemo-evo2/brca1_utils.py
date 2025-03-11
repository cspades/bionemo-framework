# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Arc Institute. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Michael Poli. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Stanford University. All rights reserved
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import gzip
import math
import os
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from sklearn.metrics import auc, roc_curve


# Constants
WINDOW_SIZE = 8192
DEFAULT_COMMIT_HASH = "3819474bee6c24938016614411f1fa025e542bbe"  # pragma: allowlist secret


# NVIDIA Light Theme Colors
NVIDIA_GREEN = "#76B900"
BACKGROUND_COLOR = "#F8F8F8"  # Light background
GRID_COLOR = "#DDDDDD"
FONT_COLOR = "#333333"


def plot_roc_curve(df):
    """Plots an ROC curve using Altair with a light NVIDIA-themed design.

    The function assumes:
    - `class` column as the true labels (binary, 'LOF' = 1, else 0).
    - `evo2_delta_score` as the prediction score.

    Parameters:
    - df (pd.DataFrame): DataFrame containing `class` and `evo2_delta_score`.

    Returns:
    - Altair Chart: ROC Curve Visualization.
    """
    # NVIDIA theme colors
    nvidia_green = "#76B900"
    background_color = "#F8F8F8"
    grid_color = "#DDDDDD"
    font_color = "#333333"

    # Validate required columns
    if "class" not in df.columns or "evo2_delta_score" not in df.columns:
        raise ValueError("DataFrame must contain 'class' and 'evo2_delta_score' columns.")

    # Convert 'class' to binary labels: Assume 'LOF' = 1, anything else = 0
    y_true = (df["class"] == "LOF").astype(int)

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_true, -df["evo2_delta_score"])  # Negative to align with previous logic
    roc_auc = auc(fpr, tpr)

    # Create DataFrame for plotting
    roc_df = pd.DataFrame({"False Positive Rate": fpr, "True Positive Rate": tpr})

    # Create the ROC curve plot
    roc_chart = (
        alt.Chart(roc_df)
        .mark_line(color=nvidia_green, strokeWidth=3)
        .encode(
            x=alt.X(
                "False Positive Rate",
                title="False Positive Rate",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(gridColor=grid_color, labelColor=font_color, titleColor=font_color),
            ),
            y=alt.Y(
                "True Positive Rate",
                title="True Positive Rate",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(gridColor=grid_color, labelColor=font_color, titleColor=font_color),
            ),
            tooltip=["False Positive Rate", "True Positive Rate"],
        )
    )

    # Add diagonal reference line for random guessing
    diagonal = (
        alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
        .mark_line(color="gray", strokeDash=[5, 5])
        .encode(x="x", y="y")
    )

    # Combine plots with NVIDIA-themed styling and left-aligned title & subtitle
    final_chart = (
        (roc_chart + diagonal)
        .properties(
            title=alt.TitleParams(
                text=f"Zeroshot ROC Curve (AUROC = {roc_auc:.2f})",
                subtitle="Evaluating the discriminative performance of Evo 2 predictions.",
                anchor="start",
                fontSize=16,
                subtitleFontSize=14,
                color=font_color,
            ),
            width=500,
            height=400,
            background=background_color,
        )
        .configure_axis(grid=True, gridColor=grid_color)
    )

    return final_chart


def plot_strip_with_means(df, x_col="evo2_delta_score", class_col="class"):
    """Creates a strip plot with jittered points and vertical mean lines for each class.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing data.
    - x_col (str): The column name representing the x-axis values (e.g., evo2_delta_score).
    - class_col (str): The column name representing the class labels.

    Returns:
    - Altair Chart: Strip plot with mean indicators.
    """
    # NVIDIA theme colors
    nvidia_green = "#76B900"
    background_color = "#F8F8F8"
    grid_color = "#DDDDDD"
    font_color = "#333333"

    # Define class mapping for y-axis positions
    unique_classes = df[class_col].unique()
    y_positions = {cls: i for i, cls in enumerate(unique_classes)}

    # Add jitter manually to the y-axis for strip plot
    df = df.copy()  # Avoid modifying the original dataframe
    df["jitter"] = df[class_col].map(y_positions) + (np.random.rand(len(df)) - 0.5) * 0.3

    # Compute mean values for each class
    mean_scores = df.groupby(class_col)[x_col].mean().reset_index()
    mean_scores["y_start"] = mean_scores[class_col].map(y_positions) - 0.2
    mean_scores["y_end"] = mean_scores[class_col].map(y_positions) + 0.2

    # Create strip plot with jittered y values
    strip_plot = (
        alt.Chart(df)
        .mark_circle(size=20, opacity=0.6)
        .encode(
            x=alt.X(
                x_col,
                title="Delta Likelihood Score, Evo 2",
                axis=alt.Axis(gridColor=grid_color, labelColor=font_color, titleColor=font_color),
            ),
            y=alt.Y("jitter", title="BRCA1 SNV Class", axis=None),
            color=alt.Color(
                class_col,
                scale=alt.Scale(domain=list(y_positions.keys()), range=["red", nvidia_green]),
            ),
            tooltip=[x_col, class_col],
        )
    )

    # Create vertical mean lines that are limited to each class group
    mean_lines = (
        alt.Chart(mean_scores)
        .mark_rule(strokeWidth=4, opacity=0.8)
        .encode(
            x=alt.X(f"{x_col}:Q"),
            y=alt.Y("y_start:Q"),
            y2="y_end:Q",
            color=alt.value("black"),  # NVIDIA green for mean indicators
        )
    )

    # Combine strip plot and class-specific mean lines with left-aligned title & subtitle
    final_chart = (
        (strip_plot + mean_lines)
        .properties(
            title=alt.TitleParams(
                text="Distribution of Delta Likelihoods Scores",
                subtitle="Comparing Evo 2 likelihood scores for different BRCA1 SNV classes.",
                anchor="start",
                fontSize=16,
                subtitleFontSize=14,
                color=font_color,
            ),
            width=450,
            height=250,
            background=background_color,
        )
        .configure_axis(grid=True, gridColor=grid_color)
    )

    return final_chart


# Check if FP8 is supported on the current GPU
def check_fp8_support():
    """Check if FP8 is supported on the current GPU.

    FP8 requires compute capability 8.9+ (Ada Lovelace/Hopper architecture or newer).
    """
    if not torch.cuda.is_available():
        return False, "CUDA not available"

    device_props = torch.cuda.get_device_properties(0)
    compute_capability = f"{device_props.major}.{device_props.minor}"
    device_name = device_props.name

    # FP8 is supported on compute capability 8.9+ (Ada Lovelace/Hopper architecture)
    is_supported = (device_props.major > 8) or (device_props.major == 8 and device_props.minor >= 9)

    return is_supported, f"Device: {device_name}, Compute Capability: {compute_capability}"


def download_data(data_dir="brca1", commit_hash="3819474bee6c24938016614411f1fa025e542bbe"):
    """Download required data files if they don't exist locally.

    Parameters:
    -----------
    data_dir : str
        Directory to store downloaded files
    commit_hash : str
        GitHub commit hash for data version
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    excel_path = os.path.join(data_dir, "41586_2018_461_MOESM3_ESM.xlsx")
    genome_path = os.path.join(data_dir, "GRCh37.p13_chr17.fna.gz")

    if not os.path.exists(excel_path):
        os.system(
            f"wget https://github.com/ArcInstitute/evo2/raw/{commit_hash}/notebooks/brca1/41586_2018_461_MOESM3_ESM.xlsx -O {excel_path}"
        )

    if not os.path.exists(genome_path):
        os.system(
            f"wget https://github.com/ArcInstitute/evo2/raw/{commit_hash}/notebooks/brca1/GRCh37.p13_chr17.fna.gz -O {genome_path}"
        )

    return excel_path, genome_path


def load_brca1_data(excel_path):
    """Load and preprocess BRCA1 data from Excel file.

    Parameters:
    -----------
    excel_path : str
        Path to the Excel file

    Returns:
    --------
    pandas.DataFrame
        Processed BRCA1 dataframe
    """
    # Load the dataframe
    brca1_df = pd.read_excel(excel_path, header=2)

    # Select and rename columns
    brca1_df = brca1_df[
        [
            "chromosome",
            "position (hg19)",
            "reference",
            "alt",
            "function.score.mean",
            "func.class",
        ]
    ]

    brca1_df.rename(
        columns={
            "chromosome": "chrom",
            "position (hg19)": "pos",
            "reference": "ref",
            "alt": "alt",
            "function.score.mean": "score",
            "func.class": "class",
        },
        inplace=True,
    )

    # Convert to two-class system
    brca1_df["class"] = brca1_df["class"].replace(["FUNC", "INT"], "FUNC/INT")

    return brca1_df


def load_genome_sequence(genome_path):
    """Load genome sequence from FASTA file.

    Parameters:
    -----------
    genome_path : str
        Path to the genome FASTA file

    Returns:
    --------
    str
        Genome sequence string
    """
    with gzip.open(genome_path, "rt") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            return str(record.seq)

    raise ValueError("Failed to parse genome sequence")


def parse_sequences(pos, ref, alt, seq_chr17, window_size=WINDOW_SIZE):
    """Parse reference and variant sequences from the reference genome sequence.

    Parameters:
    -----------
    pos : int
        Position (1-indexed)
    ref : str
        Reference base
    alt : str
        Alternate base
    seq_chr17 : str
        Full chromosome 17 sequence
    window_size : int
        Size of the sequence window to extract

    Returns:
    --------
    tuple
        (reference_sequence, variant_sequence)
    """
    p = pos - 1  # Convert to 0-indexed position
    full_seq = seq_chr17

    ref_seq_start = max(0, p - window_size // 2)
    ref_seq_end = min(len(full_seq), p + window_size // 2)
    ref_seq = seq_chr17[ref_seq_start:ref_seq_end]
    snv_pos_in_ref = min(window_size // 2, p)
    var_seq = ref_seq[:snv_pos_in_ref] + alt + ref_seq[snv_pos_in_ref + 1 :]

    # Sanity checks
    assert len(var_seq) == len(ref_seq)
    assert ref_seq[snv_pos_in_ref] == ref
    assert var_seq[snv_pos_in_ref] == alt

    return ref_seq, var_seq


def sample_data(df, sample_frac=1.0, balanced=True, disable=False, random_state=42):
    """Sample dataframe, optionally with balanced classes.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    sample_frac : float
        Fraction of data to sample
    balanced : bool
        Whether to balance classes
    disable : bool
        Whether to disable sampling
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    pandas.DataFrame
        Sampled dataframe
    """
    if disable:
        return df

    if balanced:
        # Get the number of rows in the dataframe
        num_rows_minor_class = math.ceil(len(df[df["class"] == "LOF"]) * sample_frac)
        return (
            pd.concat(
                [
                    df[df["class"] == "LOF"].sample(n=num_rows_minor_class, random_state=random_state),
                    df[df["class"] == "FUNC/INT"].sample(n=num_rows_minor_class, random_state=random_state),
                ]
            )
            .sample(frac=1.0, random_state=random_state)
            .reset_index(drop=True)
        )
    else:
        # Calculate the number of rows to sample
        return df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)


def generate_fasta_files(df, seq_chr17, output_dir="brca1_fasta_files", window_size=WINDOW_SIZE):
    """Generate FASTA files for reference and variant sequences.

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe with variant information
    seq_chr17 : str
        Chromosome 17 sequence
    output_dir : str
        Output directory for FASTA files
    window_size : int
        Size of sequence window

    Returns:
    --------
    pandas.DataFrame
        Dataframe with added columns for FASTA names
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Paths for output files
    ref_fasta_path = output_dir / "brca1_reference_sequences.fasta"
    var_fasta_path = output_dir / "brca1_variant_sequences.fasta"

    # Track unique sequences
    ref_sequences = set()
    var_sequences = set()
    ref_seq_to_name = {}

    # Store unique sequences with metadata for writing
    ref_entries = []
    var_entries = []
    ref_names = []
    var_names = []

    # Collect unique reference and variant sequences
    for idx, row in df.iterrows():
        ref_seq, var_seq = parse_sequences(row["pos"], row["ref"], row["alt"], seq_chr17, window_size)

        # Add to sets to ensure uniqueness
        if ref_seq not in ref_sequences:
            ref_sequences.add(ref_seq)
            ref_name = f"BRCA1_ref_pos_{row['pos']}_{row['ref']}_class_{row['class']}"

            ref_entries.append(f">{ref_name}\n{ref_seq}\n")
            ref_names.append(ref_name)
            ref_seq_to_name[ref_seq] = ref_name
        else:
            ref_name = ref_seq_to_name[ref_seq]
            ref_names.append(ref_name)

        if var_seq not in var_sequences:
            var_sequences.add(var_seq)
            var_name = f"BRCA1_var_pos_{row['pos']}_{row['ref']}to{row['alt']}_class_{row['class']}"

            var_entries.append(f">{var_name}\n{var_seq}\n")
            var_names.append(var_name)
        else:
            assert False, "Duplicate variant sequence"

    # Write unique sequences to FASTA files
    with open(ref_fasta_path, "w") as f:
        f.writelines(ref_entries)

    with open(var_fasta_path, "w") as f:
        f.writelines(var_entries)

    # Add FASTA names to dataframe
    df_with_names = df.copy()
    df_with_names["ref_fasta_name"] = ref_names
    df_with_names["var_fasta_name"] = var_names

    print(f"Total unique reference sequences: {len(ref_sequences)}")
    print(f"Total unique variant sequences: {len(var_sequences)}")

    return df_with_names
