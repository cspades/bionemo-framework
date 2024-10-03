# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from abc import ABC
from pathlib import Path

import polars as pl
import pysam
import torch
from pyfaidx import Fasta
from torch.utils.data import Dataset

from bionemo.dnadl.tools.genome_editing import create_personal_sequence
from bionemo.dnadl.tools.genome_interval import GenomeInterval
from bionemo.dnadl.tools.vcf import read_variants_in_interval


class DNATokenizer(ABC):
    """Abstract class for DNA tokenizers."""

    def __call__(self, seq: str) -> torch.tensor:
        """Tokenize a DNA sequence."""
        ...


class Genome:
    """Class that creates a genome object from a fasta file. It can be used to extract sequences from the genome."""

    def __init__(self, fasta_file: str):
        """Instantiate the class."""
        fasta_file = Path(fasta_file)
        assert fasta_file.exists(), "path to fasta file must exist"
        self.seqs = Fasta(str(fasta_file))

    def _adjust_interval(self, interval: GenomeInterval, context_length: int) -> GenomeInterval:
        """Used to extend an interval by a fixed amount on both sides."""
        start, end = interval.start, interval.end
        interval_length = interval.end - interval.start
        chromosome_length = len(self.seqs[interval.chromosome])
        if interval_length < context_length:
            extra_seq = context_length - interval_length
            extra_left_seq = extra_seq // 2
            extra_right_seq = extra_seq - extra_left_seq

            start -= extra_left_seq
            end += extra_right_seq

            if start <= 0:
                start = 1

            if end > chromosome_length:
                end = chromosome_length

            return GenomeInterval(interval.chromosome, start, end)
        return interval

    def __call__(
        self,
        genome_interval: GenomeInterval,
    ) -> str:
        """Extract a sequence from the genome."""
        return str(self.seqs[genome_interval.chromosome][genome_interval.start : genome_interval.end])


class GenomeIntervalDataset(Dataset):
    """Datasets that extract sequences from a genome based on a bed file."""

    def __init__(
        self,
        genome: Genome,
        tokenizer: DNATokenizer,
        bed_file: Path,
        context_length: int | None = None,
        filter_df_fn=lambda x: x,
        chr_bed_to_fasta_map: dict = {},
    ):
        """Instantiate the class."""
        super().__init__()
        bed_path = Path(bed_file)
        assert bed_path.exists(), "path to .bed file must exist"

        df = pl.read_csv(str(bed_path), separator=" ", has_header=False)
        df = filter_df_fn(df)
        self.df = df

        # if the chromosome name in the bed file is different than the keyname in the fasta
        # can remap on the fly
        self.chr_bed_to_fasta_map = chr_bed_to_fasta_map
        self.context_length = context_length
        self.genome = genome

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.df)

    def _get_interval(self, ind: int) -> GenomeInterval:
        """Get the interval from the bed file."""
        chrom = self.chr_bed_to_fasta_map.get(self.df.row(ind)[0], self.df.row(ind)[0])
        genome_interval = GenomeInterval(chrom, self.df.row(ind)[1], self.df.row(ind)[2])
        if self.context_length is not None:
            genome_interval = self.genome._adjust_interval(genome_interval, self.context_length)
        return genome_interval

    def _get_dna_sequence(self, ind: int) -> str:
        """Get the DNA sequence from the genome."""
        genome_interval = self._get_interval(ind)
        return self.genome(genome_interval)

    def __getitem__(self, ind: int) -> torch.tensor:
        """Get an item from the dataset."""
        sequence = self._get_dna_sequence(ind)
        return self.tokenizer(sequence)


class VCFDataset(GenomeIntervalDataset):
    """Dataset that creates sequences based on a VCF. It needs a genome and a bed file to specify the windows."""

    def __init__(self, vcf_file: str, sample_ids=None, **kwargs):
        """Initialize the class."""
        super().__init__(**kwargs)

        self.vcf_file = Path(vcf_file)
        assert self.vcf_file.exists(), "path to VCF file must exist"

        if sample_ids is None:
            self._populate_sample_ids_from_vcf()
        else:
            self.sample_ids = sample_ids

    def _populate_sample_ids_from_vcf(self) -> None:
        with pysam.VariantFile(str(self.vcf_file)) as vcf_file:
            self.sample_ids = list(vcf_file.header.samples)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return super().__len__() * len(self.sample_ids)

    def _get_dna_sequence(self, ind: int) -> str:
        """Get an item from the dataset."""
        interval_index = ind // len(self.sample_ids)
        sample_index = ind % len(self.sample_ids)

        ref_sequence = super()._get_dna_sequence(interval_index)
        interval = super()._get_interval(interval_index)
        sample_variants = read_variants_in_interval(interval, self.vcf_file, self.sample_ids)[
            self.sample_ids[sample_index]
        ]
        return create_personal_sequence(ref_sequence, interval, sample_variants)
