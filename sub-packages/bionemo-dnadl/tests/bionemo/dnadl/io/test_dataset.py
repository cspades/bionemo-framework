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


import pytest

from bionemo.dnadl.io.dataset import Genome, GenomeIntervalDataset, VCFDataset
from bionemo.dnadl.tools.genome_interval import GenomeInterval


def test_genome(temp_fasta_file):
    genome = Genome(temp_fasta_file)

    seq_1 = genome(GenomeInterval("chr1", 10, 15))
    assert len(seq_1) == 5
    assert set(seq_1) == {"A"}

    seq_2 = genome(GenomeInterval("chr2", 1, 15))
    assert len(seq_2) == 14
    assert set(seq_2) == {"T"}

    with pytest.raises(KeyError):
        genome(GenomeInterval("chr3", 10, 15))


def test_genome_interval(temp_fasta_file, temp_bed_file):
    genome = Genome(temp_fasta_file)
    genome_interval = GenomeIntervalDataset(genome, None, temp_bed_file, context_length=20)

    assert len(genome_interval) == 2
    assert genome_interval._get_dna_sequence(0) == "AAAAAAAAAAAAAAAAAAAA"
    assert genome_interval._get_dna_sequence(1) == "TTTTTTTTTTTTTTT"


def test_vcf_file(temp_fasta_file, temp_bed_file, temp_vcf_file):
    genome = Genome(temp_fasta_file)
    vcf_dataset = VCFDataset(
        vcf_file=temp_vcf_file,
        sample_ids=["sample_1", "sample_2"],
        genome=genome,
        tokenizer=None,
        bed_file=temp_bed_file,
        context_length=20,
    )

    assert len(vcf_dataset) == 4

    assert vcf_dataset._get_dna_sequence(0) == "AGAAAAAAAAAAAAAAAAAA"
    assert vcf_dataset._get_dna_sequence(2) == "TTTTTTTTATTTTTT"
