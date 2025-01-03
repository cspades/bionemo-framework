# Data package
## Preprocess
### Equivalence with arc implementation
To test equivalence with the reference implementation we first downloaded processed megatrion IndexedDataset
files from Arc institute for their promotors dataset:

```bash
$ mkdir tmp_goldstandard
$ cd tmp_goldstandard
$ scp login-eos:/lustre/fsw/healthcareeng_bionemo/arc_evo2/data/promoters/pretraining_data_promoters/data_promoters_*_text_CharLevelTokenizer_document.* ./
```

```bash
$ ls -lah
-rwxr-xr-x  1 bionemo bionemo 1.2M Dec  4 00:56 data_promoters_test_text_CharLevelTokenizer_document.bin
-rwxr-xr-x  1 bionemo bionemo  20K Dec  4 00:56 data_promoters_test_text_CharLevelTokenizer_document.idx
-rwxr-xr-x  1 bionemo bionemo 392M Dec  4 00:56 data_promoters_train_text_CharLevelTokenizer_document.bin
-rwxr-xr-x  1 bionemo bionemo 6.6M Dec  4 00:56 data_promoters_train_text_CharLevelTokenizer_document.idx
-rwxr-xr-x  1 bionemo bionemo 1.2M Dec  4 00:56 data_promoters_valid_text_CharLevelTokenizer_document.bin
-rwxr-xr-x  1 bionemo bionemo  20K Dec  4 00:56 data_promoters_valid_text_CharLevelTokenizer_document.idx
```

Next we acquired the `fasta` file that was used to generate this and placed it into the tests/data folder of this sub-package.

```yaml
- datapaths: ["/workspace/bionemo2/data/mmseqs_results_rep_seq_distinct.fasta"]
  output_dir: "/workspace/bionemo2/data"
  output_prefix: promoters_ab_test_noodles_uint8_distinct
  # Datasplit
  train_split: 1.0  # because they do manual splits of first 1000 for validation, 2nd 1000 for test, and leftover for training
  valid_split: 0.0
  test_split: 0.0
  # Overwrite existing binaries. Otherwise, skip already preprocessed datasets.
  overwrite: True
  # Raw Preprocessing Transforms
  embed_reverse_complement: true
  random_reverse_complement: 0.5
  random_lineage_dropout: 0.1
  include_sequence_id: false
  transcribe: "back_transcribe"
  force_uppercase: true
  indexed_dataset_dtype: "uint8"
  # Tokenizer
  tokenizer_type: "Byte-Level"
  vocab_file: null
  vocab_size: null
  merges_file: null
  # Either a named pretrained tokenizer model, or a path to a SentencePiece tokenizer.
  pretrained_tokenizer_model: null
  special_tokens: null
  fast_hf_tokenizer: true
  append_eod: true
  enforce_sample_length: null
  ftfy: false
  # Compute
  workers: 1
  preproc_concurrency: 100000
  chunksize: 25
  # Filters
  drop_empty_sequences: true
  nnn_filter: true
  # RNG
  seed: 42
  # StipedHyena2 Taxonomic Lineage Tags
  taxonomy_data:
    FP002272:
      kingdom: KINGDOM
      phylum: PHYLUM
      clazz: CLASS
      order: ORDER
      family: FAMILY
      genus: GENUS
      species: SPECIES
```

Finally we generated our own bin/idx file for in this case everything going into the training set.
```bash
$ python sub-packages/bionemo-evo2/src/bionemo/evo2/data/preprocess.py -c sub-packages/bionemo-evo2/tests/config/mmseqs_promotors_config.yaml
```

Next to check equivalence, we were not attempting to replicate the exact ordering of the datset, we just wanted to verify
that we get the same elements out of our processed dataset as the original.

```python
>>> from megatron.core.datasets.indexed_dataset import IndexedDataset
>>> ds_train_ref = IndexedDataset("./data_promoters_train_text_CharLevelTokenizer_document")
>>> ds_val_ref = IndexedDataset("./data_promoters_valid_text_CharLevelTokenizer_document")
>>> ds_test_ref = IndexedDataset("./data_promoters_test_text_CharLevelTokenizer_document")
>>> ds_train_ours = IndexedDataset("../sub-packages/bionemo-evo2/tests/data/promoters_ab_test_byte-level_train")
>>> len(ds_train_ours) == len(ds_train_ref) + len(ds_test_ref) + len(ds_val_ref)
True
>>>  # Example of what one of these set elements looks like, it's just a string representation of the token list for an
>>>  #  element of the training dataset. We can then compare all of these to make sure that the two datasets have the
>>>  #  same set of samples.
>>> ','.join([str(t) for t in ds_train_ref[0]])
'67,84,71,71,65,71,67,67,84,71,65,67,67,65,84,65,65,71,84,65,71,84,71,71,67,84,65,84,65,65,67,71,65,71,71,65,65,71,65,65,71,65,84,71,65,65,71,65,71,65,84,84,65,71,65,71,65,65,65,65,84,71,65,65,84,71,84,84,67,84,84,71,65,65,71,84,65,71,67,67,65,84,84,71,84,84,71,84,65,71,84,84,71,84,84,71,84,71,84,71,84,71,84,65,84,71,84,84,71,65,71,65,84,71,84,84,84,84,71,71,71,71,84,84,84,71,84,84,65,84,65,84,65,71,65,71,65,71,65,71,65,84,71,84,65,71,84,84,84,71,71,84,71,65,65,71,65,71,84,65,71,71,65,84,84,67,84,67,84,84,65,67,84,65,71,84,71,84,71,65,65,71,65,84,84,65,84,84,65,67,84,65,71,71,84,65,65,67,84,65,65,65,84,71,65,71,65,84,84,67,84,65,84,67,65,65,67,84,65,65,71,84,67,65,84,84,65,71,65,71,65,84,84,71,71,65,65,65,84,71,84,84,84,67,84,84,84,84,65,71,71,84,84,84,65,65,84,65,65,65,71,84,84,84,71,84,84,84,71,65,65,84,84,71,65,71,65,65,65,71,65,71,65,71,65,71,71,65,71,65,71,65,67,65,84,84,71,67,84,84,84,71,65,65,71,71,71,65,71,65,71,84,84,84,71,71,71,84,71,71,71,84,71,65,71,71,65,84,84,71,65,65,65,65,84,71,65,65,65,65,65,84,71,65,65,67,84,71,65,65,65,65,65,71,71,84,71,84,84,65,84,65,71,84,71,65,67,67,84,71,84,67,65,65,65,65,65,65,71,67,84,71,84,71,65,65,71,65,65,71,84,71,84,84,65,84,67,67,65,65,71,65,65,65,84,65,84,71,71,65,84,84,71,67,84,65,65,84,67,65,84,65,67,84,65,67,84,71,84,84,67,65,84,84,65,84,71,65,84,84,84,84,65,84,71,84,71,84,67,65,84,71,84,71,84,71,84,71,67,67,84,65,84,67,65,84,67,65,84,84,67,67,84,84,65,84,65,84,84,84,84,65,71,84,84,71,71,67,65,65,65,65,65,65,65,65,65,65,65,71,65,67,84,84,71,71,65,65,71,84,65,84,84,71,65,65,65,65,67,67,65,65,65,84,67,84,71,65,84,67,84,67,65,65,67,67,84,65,71,65,67,65,65,71,84,67,71,65,84,84,65,65,65,71,67,84,65,65,65,67,67,71,65,65,65,65,67,67,71,65,65,84,67,67,67,71,65,67,67,71,71,84,84,65,65,84,84,71,65,65,65,65,67,67,71,65,84,67,67,65,0'
>>> # Create a set of all of these elements:
>>> all_ref_data = {','.join([str(t) for t in rec]) for ds in [ds_train_ref, ds_val_ref, ds_test_ref] for rec in ds}
>>> # Verify that there is no redundancy so we can do set equality safely
>>> len(all_ref_data) == len(ds_train_ours)
True
>>> len(all_ref_data)
343504
>>> all_our_data = {','.join([str(t) for t in rec]) for ds in [ds_train_ours] for rec in ds}
>>> len(all_our_data)
343504
>>> # Verify set equality to show that we have processed an identical dataset
>>> #  (ignoring shuffling order and train/test/val splits)
>>> all_our_data == all_ref_data
True
```
