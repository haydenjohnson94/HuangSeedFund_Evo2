## We acknowledge Jeff Penn of OSU for his support in running the following code


```python
##Fine-tuning tutorial for Evo2
This tutorial goes through a toy fine-tuning example end to end starting with a fasta and continuing training a hugging
face checkpoint on this user defined dataset.
```

```python
# Clean up any prior runs
!rm -rf preprocessed_data
!rm -rf preatraining_demo
!rm -rf nemo2_evo2_1b_8k
!rm -rf pretraining_demo
!rm -rf training_data_config.yaml
!rm -rf preprocess_config.yaml
!rm -f chr17.fa.gz
!rm -f chr18.fa.gz
!rm -f chr21.fa.gz
!rm -f chr17.fa
!rm -f chr18.fa
!rm -f chr21.fa
!rm -f chr17_18_21.fa

```


```python
import os
concat_path = "chr17_18_21.fa"
if not os.path.exists(concat_path):
    !wget https://hgdownload.soe.ucsc.edu/goldenpath/hg38/chromosomes/chr17.fa.gz
    !wget https://hgdownload.soe.ucsc.edu/goldenpath/hg38/chromosomes/chr18.fa.gz
    !wget https://hgdownload.soe.ucsc.edu/goldenpath/hg38/chromosomes/chr21.fa.gz
    !zcat chr17.fa.gz > chr17.fa
    !zcat chr18.fa.gz > chr18.fa
    !zcat chr21.fa.gz > chr21.fa
    !cat chr17.fa chr18.fa chr21.fa > chr17_18_21.fa

```

    --2025-06-18 15:55:03--  https://hgdownload.soe.ucsc.edu/goldenpath/hg38/chromosomes/chr17.fa.gz
    128.114.119.163nload.soe.ucsc.edu (hgdownload.soe.ucsc.edu)... 
    Connecting to hgdownload.soe.ucsc.edu (hgdownload.soe.ucsc.edu)|128.114.119.163|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 25930986 (25M) [application/x-gzip]
    Saving to: ‘chr17.fa.gz’
    
    chr17.fa.gz         100%[===================>]  24.73M  60.3MB/s    in 0.4s    
    
    2025-06-18 15:55:04 (60.3 MB/s) - ‘chr17.fa.gz’ saved [25930986/25930986]
    
    --2025-06-18 15:55:05--  https://hgdownload.soe.ucsc.edu/goldenpath/hg38/chromosomes/chr18.fa.gz
    Resolving hgdownload.soe.ucsc.edu (hgdownload.soe.ucsc.edu)... 128.114.119.163
    connected. to hgdownload.soe.ucsc.edu (hgdownload.soe.ucsc.edu)|128.114.119.163|:443... 
    200 OKequest sent, awaiting response... 
    Length: 25154367 (24M) [application/x-gzip]
    Saving to: ‘chr18.fa.gz’
    
    chr18.fa.gz         100%[===================>]  23.99M  60.2MB/s    in 0.4s    
    
    2025-06-18 15:55:05 (60.2 MB/s) - ‘chr18.fa.gz’ saved [25154367/25154367]
    
    --2025-06-18 15:55:06--  https://hgdownload.soe.ucsc.edu/goldenpath/hg38/chromosomes/chr21.fa.gz
    Resolving hgdownload.soe.ucsc.edu (hgdownload.soe.ucsc.edu)... 128.114.119.163
    Connecting to hgdownload.soe.ucsc.edu (hgdownload.soe.ucsc.edu)|128.114.119.163|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 12709705 (12M) [application/x-gzip]
    Saving to: ‘chr21.fa.gz’
    
    chr21.fa.gz         100%[===================>]  12.12M  47.0MB/s    in 0.3s    
    
    2025-06-18 15:55:06 (47.0 MB/s) - ‘chr21.fa.gz’ saved [12709705/12709705]
    



```python
full_fasta_path = os.path.abspath(concat_path)
output_dir = os.path.abspath("preprocessed_data")
output_yaml = f"""
- datapaths: ["{full_fasta_path}"]
  output_dir: "{output_dir}"
  output_prefix: chr17_18_21_uint8_distinct
  train_split: 0.9
  valid_split: 0.05
  test_split: 0.05
  overwrite: True
  embed_reverse_complement: true
  random_reverse_complement: 0.0
  random_lineage_dropout: 0.0
  include_sequence_id: false
  transcribe: "back_transcribe"
  force_uppercase: false
  indexed_dataset_dtype: "uint8"
  tokenizer_type: "Byte-Level"
  vocab_file: null
  vocab_size: null
  merges_file: null
  pretrained_tokenizer_model: null
  special_tokens: null
  fast_hf_tokenizer: true
  append_eod: true
  enforce_sample_length: null
  ftfy: false
  workers: 1
  preproc_concurrency: 100000
  chunksize: 25
  drop_empty_sequences: true
  nnn_filter: false  # If you split your fasta on NNN (in human these are contigs), then you should set this to true.
  seed: 12342  # Not relevant because we are not using random reverse complement or lineage dropout.
"""
with open("preprocess_config.yaml", "w") as f:
    print(output_yaml, file=f)

```


```python
!preprocess_evo2 --config preprocess_config.yaml
```

    [NeMo I 2025-06-18 15:55:18 nemo_logging:393] Using byte-level tokenization
    [NeMo I 2025-06-18 15:55:18 nemo_logging:393] Created temporary binary datasets: /workspace/bionemo2/data/preprocessed_data/chr17_18_21_uint8_distinct_byte-level_train.bin.tmp /workspace/bionemo2/data/preprocessed_data/chr17_18_21_uint8_distinct_byte-level_val.bin.tmp /workspace/bionemo2/data/preprocessed_data/chr17_18_21_uint8_distinct_byte-level_test.bin.tmp
    [NeMo I 2025-06-18 15:55:46 nemo_logging:393] Average preprocessing time per sequence: 1.2736938794453938
    [NeMo I 2025-06-18 15:55:46 nemo_logging:393] Average indexing time per sequence: 3.839247544606527
    [NeMo I 2025-06-18 15:55:46 nemo_logging:393] Number of sequences processed: 6
    [NeMo I 2025-06-18 15:55:46 nemo_logging:393] Finished preprocessing chr17_18_21_uint8_distinct ([PosixPath('/workspace/bionemo2/data/chr17_18_21.fa')]) in 28.441 seconds with 1 workers.



```python
!ls -lh preprocessed_data/
```

    total 323M
    -rw-r--r-- 1 pennjef domain users  90M Jun 18 15:55 chr17_18_21_uint8_distinct_byte-level_test.bin
    -rw-r--r-- 1 pennjef domain users   82 Jun 18 15:55 chr17_18_21_uint8_distinct_byte-level_test.idx
    -rw-r--r-- 1 pennjef domain users 159M Jun 18 15:55 chr17_18_21_uint8_distinct_byte-level_train.bin
    -rw-r--r-- 1 pennjef domain users   82 Jun 18 15:55 chr17_18_21_uint8_distinct_byte-level_train.idx
    -rw-r--r-- 1 pennjef domain users 154M Jun 18 15:55 chr17_18_21_uint8_distinct_byte-level_val.bin
    -rw-r--r-- 1 pennjef domain users   82 Jun 18 15:55 chr17_18_21_uint8_distinct_byte-level_val.idx



```python
!evo2_convert_to_nemo2 \
  --model-path hf://arcinstitute/savanna_evo2_1b_base \
  --model-size 1b --output-dir nemo2_evo2_1b_8k
```

    [NeMo I 2025-06-18 16:01:44 nemo_logging:393] Using byte-level tokenization
    GPU available: True (cuda), used: False
    TPU available: False, using: 0 TPU cores
    HPU available: False, using: 0 HPUs
    [NeMo W 2025-06-18 16:01:44 nemo_logging:405] /usr/local/lib/python3.12/dist-packages/lightning/pytorch/trainer/setup.py:177: GPU available but not used. You can set it by doing `Trainer(accelerator='gpu')`.
        
    [NeMo I 2025-06-18 16:01:44 nemo_logging:393] Fixing mis-match between ddp-config & mcore-optimizer config
    [NeMo I 2025-06-18 16:01:44 nemo_logging:393] Rank 0 has data parallel group : [0]
    [NeMo I 2025-06-18 16:01:44 nemo_logging:393] Rank 0 has combined group of data parallel and context parallel : [0]
    [NeMo I 2025-06-18 16:01:44 nemo_logging:393] All data parallel group ranks with context parallel combined: [[0]]
    [NeMo I 2025-06-18 16:01:44 nemo_logging:393] Ranks 0 has data parallel rank: 0
    [NeMo I 2025-06-18 16:01:44 nemo_logging:393] Rank 0 has context parallel group: [0]
    [NeMo I 2025-06-18 16:01:44 nemo_logging:393] All context parallel group ranks: [[0]]
    [NeMo I 2025-06-18 16:01:44 nemo_logging:393] Ranks 0 has context parallel rank: 0
    [NeMo I 2025-06-18 16:01:44 nemo_logging:393] Rank 0 has model parallel group: [0]
    [NeMo I 2025-06-18 16:01:44 nemo_logging:393] All model parallel group ranks: [[0]]
    [NeMo I 2025-06-18 16:01:44 nemo_logging:393] Rank 0 has tensor model parallel group: [0]
    [NeMo I 2025-06-18 16:01:44 nemo_logging:393] All tensor model parallel group ranks: [[0]]
    [NeMo I 2025-06-18 16:01:44 nemo_logging:393] Rank 0 has tensor model parallel rank: 0
    [NeMo I 2025-06-18 16:01:44 nemo_logging:393] Rank 0 has pipeline model parallel group: [0]
    [NeMo I 2025-06-18 16:01:44 nemo_logging:393] Rank 0 has embedding group: [0]
    [NeMo I 2025-06-18 16:01:44 nemo_logging:393] All pipeline model parallel group ranks: [[0]]
    [NeMo I 2025-06-18 16:01:44 nemo_logging:393] Rank 0 has pipeline model parallel rank 0
    [NeMo I 2025-06-18 16:01:44 nemo_logging:393] All embedding group ranks: [[0]]
    [NeMo I 2025-06-18 16:01:44 nemo_logging:393] Rank 0 has embedding rank: 0
    ----------------------------------------------------------------------------------------------------
    distributed_backend=gloo
    All distributed processes registered. Starting with 1 processes
    ----------------------------------------------------------------------------------------------------
    
    [NeMo W 2025-06-18 16:01:44 nemo_logging:405] /usr/local/lib/python3.12/dist-packages/lightning/pytorch/trainer/trainer.py:1090: `trainer.init_module` cannot fully support proper instantiation of your model with the `MegatronStrategy` strategy. Please instantiate your model inside the`LightningModule.configure_model` hook instead
        
    [NeMo I 2025-06-18 16:01:44 nemo_logging:393] Padded vocab_size: 512, original vocab_size: 512, dummy tokens: 0.
    CPU RNG state changed within GPU RNG context
    CPU RNG state changed within GPU RNG context
    CPU RNG state changed within GPU RNG context
    CPU RNG state changed within GPU RNG context
    CPU RNG state changed within GPU RNG context
    CPU RNG state changed within GPU RNG context
    CPU RNG state changed within GPU RNG context
    CPU RNG state changed within GPU RNG context
    CPU RNG state changed within GPU RNG context
    CPU RNG state changed within GPU RNG context
    CPU RNG state changed within GPU RNG context
    CPU RNG state changed within GPU RNG context
    CPU RNG state changed within GPU RNG context
    CPU RNG state changed within GPU RNG context
    [NeMo W 2025-06-18 16:01:47 nemo_logging:405] Could not copy Trainer's 'max_steps' to LR scheduler's 'max_steps'. If you are not using an LR scheduler, this warning can safely be ignored.
    [NeMo I 2025-06-18 16:01:47 nemo_logging:393] Using FullyParallelSaveStrategyWrapper(torch_dist, 1) dist-ckpt save strategy.
    [NeMo I 2025-06-18 16:01:59 nemo_logging:393] Global Checkpoint Save : Rank: 0 : Iteration: 0 : Start time: 1750287707.653s : Save duration: 11.378s
    [NeMo I 2025-06-18 16:01:59 nemo_logging:393] Converted Hyena model to Nemo, model saved to nemo2_evo2_1b_8k



```python
from pathlib import Path
output_pfx = str(Path(os.path.abspath("preprocessed_data"))/"chr17_18_21_uint8_distinct_byte-level")
output_yaml = f"""
- dataset_prefix: {output_pfx}_train
  dataset_split: train
  dataset_weight: 1.0
- dataset_prefix: {output_pfx}_val
  dataset_split: validation
  dataset_weight: 1.0
- dataset_prefix: {output_pfx}_test
  dataset_split: test
  dataset_weight: 1.0
"""
with open("training_data_config.yaml", "w") as f:
    print(output_yaml, file=f)
```


```python
import os

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29530'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'  
```

    29530
    [NeMo I 2025-06-18 16:46:47 nemo_logging:393] Using byte-level tokenization
    [NeMo W 2025-06-18 16:46:47 nemo_logging:405] WandB is currently turned off.
    [NeMo W 2025-06-18 16:46:47 nemo_logging:405] User-set tensorboard is currently turned off. Internally one may still be set by NeMo2.
    Trainer already configured with model summary callbacks: [<class 'lightning.pytorch.callbacks.rich_model_summary.RichModelSummary'>]. Skipping setting a default `ModelSummary` callback.
    GPU available: True (cuda), used: True
    TPU available: False, using: 0 TPU cores
    HPU available: False, using: 0 HPUs
    [NeMo I 2025-06-18 16:46:47 nemo_logging:393] Experiments will be logged at results/evo2/dev
    [NeMo W 2025-06-18 16:46:47 nemo_logging:405] There were no checkpoints found in checkpoint_dir or no checkpoint folder at checkpoint_dir :results/evo2/checkpoints. Training from scratch.
    [NeMo I 2025-06-18 16:46:47 nemo_logging:393] Fixing mis-match between ddp-config & mcore-optimizer config
    [NeMo I 2025-06-18 16:46:47 nemo_logging:393] Rank 0 has data parallel group : [0, 1]
    [NeMo I 2025-06-18 16:46:47 nemo_logging:393] Rank 0 has combined group of data parallel and context parallel : [0, 1]
    [NeMo I 2025-06-18 16:46:47 nemo_logging:393] All data parallel group ranks with context parallel combined: [[0, 1]]
    [NeMo I 2025-06-18 16:46:47 nemo_logging:393] Ranks 0 has data parallel rank: 0
    [NeMo I 2025-06-18 16:46:47 nemo_logging:393] Rank 0 has context parallel group: [0]
    [NeMo I 2025-06-18 16:46:47 nemo_logging:393] All context parallel group ranks: [[0], [1]]
    [NeMo I 2025-06-18 16:46:47 nemo_logging:393] Ranks 0 has context parallel rank: 0
    [NeMo I 2025-06-18 16:46:47 nemo_logging:393] Rank 0 has model parallel group: [0]
    [NeMo I 2025-06-18 16:46:47 nemo_logging:393] All model parallel group ranks: [[0], [1]]
    [NeMo I 2025-06-18 16:46:47 nemo_logging:393] Rank 0 has tensor model parallel group: [0]
    [NeMo I 2025-06-18 16:46:47 nemo_logging:393] All tensor model parallel group ranks: [[0], [1]]
    [NeMo I 2025-06-18 16:46:47 nemo_logging:393] Rank 0 has tensor model parallel rank: 0
    [NeMo I 2025-06-18 16:46:47 nemo_logging:393] Rank 0 has pipeline model parallel group: [0]
    [NeMo I 2025-06-18 16:46:47 nemo_logging:393] Rank 0 has embedding group: [0]
    [NeMo I 2025-06-18 16:46:47 nemo_logging:393] All pipeline model parallel group ranks: [[0], [1]]
    [NeMo I 2025-06-18 16:46:47 nemo_logging:393] Rank 0 has pipeline model parallel rank 0
    [NeMo I 2025-06-18 16:46:47 nemo_logging:393] All embedding group ranks: [[0], [1]]
    [NeMo I 2025-06-18 16:46:47 nemo_logging:393] Rank 0 has embedding rank: 0
    ^C



```python
!torchrun --nproc_per_node=2 --master_port=29502 /usr/local/bin/train_evo2 -d training_data_config.yaml --dataset-dir preprocessed_data --model-size 1b --devices 2 --num-nodes 1 --seq-length 1024 --micro-batch-size 2 --lr 0.0001 --warmup-steps 5 --max-steps 100 --ckpt-dir nemo2_evo2_1b_8k --clip-grad 1 --wd 0.01 --activation-checkpoint-recompute-num-layers 1 --val-check-interval 50 --ckpt-async-save
```

    W0618 16:51:14.969000 1112889 torch/distributed/run.py:792] 
    W0618 16:51:14.969000 1112889 torch/distributed/run.py:792] *****************************************
    W0618 16:51:14.969000 1112889 torch/distributed/run.py:792] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
    W0618 16:51:14.969000 1112889 torch/distributed/run.py:792] *****************************************
    [NeMo I 2025-06-18 16:51:35 nemo_logging:393] Using byte-level tokenization
    [NeMo W 2025-06-18 16:51:35 nemo_logging:405] WandB is currently turned off.
    [NeMo W 2025-06-18 16:51:35 nemo_logging:405] User-set tensorboard is currently turned off. Internally one may still be set by NeMo2.
    Trainer already configured with model summary callbacks: [<class 'lightning.pytorch.callbacks.rich_model_summary.RichModelSummary'>]. Skipping setting a default `ModelSummary` callback.
    GPU available: True (cuda), used: True
    TPU available: False, using: 0 TPU cores
    HPU available: False, using: 0 HPUs
    [NeMo I 2025-06-18 16:51:35 nemo_logging:393] Experiments will be logged at results/evo2/dev
    [NeMo W 2025-06-18 16:51:35 nemo_logging:405] There were no checkpoints found in checkpoint_dir or no checkpoint folder at checkpoint_dir :results/evo2/checkpoints. Training from scratch.
    [NeMo I 2025-06-18 16:51:35 nemo_logging:393] Fixing mis-match between ddp-config & mcore-optimizer config
    [NeMo I 2025-06-18 16:51:35 nemo_logging:393] Rank 0 has data parallel group : [0, 1]
    [NeMo I 2025-06-18 16:51:35 nemo_logging:393] Rank 0 has combined group of data parallel and context parallel : [0, 1]
    [NeMo I 2025-06-18 16:51:35 nemo_logging:393] All data parallel group ranks with context parallel combined: [[0, 1]]
    [NeMo I 2025-06-18 16:51:35 nemo_logging:393] Ranks 0 has data parallel rank: 0
    [NeMo I 2025-06-18 16:51:35 nemo_logging:393] Rank 0 has context parallel group: [0]
    [NeMo I 2025-06-18 16:51:35 nemo_logging:393] All context parallel group ranks: [[0], [1]]
    [NeMo I 2025-06-18 16:51:35 nemo_logging:393] Ranks 0 has context parallel rank: 0
    [NeMo I 2025-06-18 16:51:35 nemo_logging:393] Rank 0 has model parallel group: [0]
    [NeMo I 2025-06-18 16:51:35 nemo_logging:393] All model parallel group ranks: [[0], [1]]
    [NeMo I 2025-06-18 16:51:35 nemo_logging:393] Rank 0 has tensor model parallel group: [0]
    [NeMo I 2025-06-18 16:51:35 nemo_logging:393] All tensor model parallel group ranks: [[0], [1]]
    [NeMo I 2025-06-18 16:51:35 nemo_logging:393] Rank 0 has tensor model parallel rank: 0
    [NeMo I 2025-06-18 16:51:35 nemo_logging:393] Rank 0 has pipeline model parallel group: [0]
    [NeMo I 2025-06-18 16:51:35 nemo_logging:393] Rank 0 has embedding group: [0]
    [NeMo I 2025-06-18 16:51:35 nemo_logging:393] All pipeline model parallel group ranks: [[0], [1]]
    [NeMo I 2025-06-18 16:51:35 nemo_logging:393] Rank 0 has pipeline model parallel rank 0
    [NeMo I 2025-06-18 16:51:35 nemo_logging:393] All embedding group ranks: [[0], [1]]
    [NeMo I 2025-06-18 16:51:35 nemo_logging:393] Rank 0 has embedding rank: 0
    ----------------------------------------------------------------------------------------------------
    distributed_backend=nccl
    All distributed processes registered. Starting with 2 processes
    ----------------------------------------------------------------------------------------------------
    
    [NeMo I 2025-06-18 16:51:39 utils:302] Building Evo2Dataset splits with sizes=[400, 240, 4] and config=GPTDatasetConfig(random_seed=1234, sequence_length=1024, blend=None, blend_per_split=[(['/workspace/bionemo2/data/preprocessed_data/chr17_18_21_uint8_distinct_byte-level_train'], [1.0]), (['/workspace/bionemo2/data/preprocessed_data/chr17_18_21_uint8_distinct_byte-level_val'], [1.0]), (['/workspace/bionemo2/data/preprocessed_data/chr17_18_21_uint8_distinct_byte-level_test'], [1.0])], split=None, split_matrix=None, num_dataset_builder_threads=1, path_to_cache=None, mmap_bin_files=True, mock=False, tokenizer=<nemo.collections.common.tokenizers.bytelevel_tokenizers.ByteLevelTokenizer object at 0x149b30b99ca0>, reset_position_ids=False, reset_attention_mask=False, eod_mask_loss=False, create_attention_mask=False, drop_last_partial_validation_sequence=True, add_extra_token_to_sequence=True, s3_cache_path=None)
    [NeMo I 2025-06-18 16:51:39 utils:302] Load the _IndexReader from /workspace/bionemo2/data/preprocessed_data/chr17_18_21_uint8_distinct_byte-level_train.idx
    [NeMo I 2025-06-18 16:51:39 utils:302] 	Extract the sequence lengths
    [NeMo I 2025-06-18 16:51:39 utils:302] 	Extract the sequence pointers
    [NeMo I 2025-06-18 16:51:39 utils:302] 	Extract the document indices
    [NeMo I 2025-06-18 16:51:39 utils:302] > total number of sequences: 2
    [NeMo I 2025-06-18 16:51:39 utils:302] > total number of documents: 2
    [NeMo I 2025-06-18 16:51:39 utils:302] Build and save the Evo2Dataset train indices
    [NeMo I 2025-06-18 16:51:39 utils:302] > total number of samples: 156979
    [NeMo I 2025-06-18 16:51:39 utils:302] > total number of epochs: 1
    [NeMo I 2025-06-18 16:51:39 utils:302] Load the _IndexReader from /workspace/bionemo2/data/preprocessed_data/chr17_18_21_uint8_distinct_byte-level_val.idx
    [NeMo I 2025-06-18 16:51:39 utils:302] 	Extract the sequence lengths
    [NeMo I 2025-06-18 16:51:39 utils:302] 	Extract the sequence pointers
    [NeMo I 2025-06-18 16:51:39 utils:302] 	Extract the document indices
    [NeMo I 2025-06-18 16:51:39 utils:302] > total number of sequences: 2
    [NeMo I 2025-06-18 16:51:39 utils:302] > total number of documents: 2
    [NeMo I 2025-06-18 16:51:39 utils:302] Build and save the Evo2Dataset valid indices
    [NeMo I 2025-06-18 16:51:39 utils:302] > total number of samples: 91230
    [NeMo I 2025-06-18 16:51:39 utils:302] > total number of epochs: 1
    [NeMo I 2025-06-18 16:51:39 utils:302] Load the _IndexReader from /workspace/bionemo2/data/preprocessed_data/chr17_18_21_uint8_distinct_byte-level_test.idx
    [NeMo I 2025-06-18 16:51:39 utils:302] 	Extract the sequence lengths
    [NeMo I 2025-06-18 16:51:39 utils:302] 	Extract the sequence pointers
    [NeMo I 2025-06-18 16:51:39 utils:302] 	Extract the document indices
    [NeMo I 2025-06-18 16:51:39 utils:302] > total number of sequences: 2
    [NeMo I 2025-06-18 16:51:39 utils:302] > total number of documents: 2
    [NeMo I 2025-06-18 16:51:39 utils:302] Build and save the Evo2Dataset test indices
    [NeMo I 2025-06-18 16:51:39 utils:302] > total number of samples: 162612
    [NeMo I 2025-06-18 16:51:39 utils:302] > total number of epochs: 1
    [NeMo I 2025-06-18 16:51:39 nemo_logging:393] Padded vocab_size: 512, original vocab_size: 512, dummy tokens: 0.
    [NeMo W 2025-06-18 16:51:39 random:220] CPU RNG state changed within GPU RNG context
    [NeMo W 2025-06-18 16:51:39 random:220] CPU RNG state changed within GPU RNG context
    [NeMo W 2025-06-18 16:51:39 random:220] CPU RNG state changed within GPU RNG context
    [NeMo W 2025-06-18 16:51:39 random:220] CPU RNG state changed within GPU RNG context
    [NeMo W 2025-06-18 16:51:39 random:220] CPU RNG state changed within GPU RNG context
    [NeMo W 2025-06-18 16:51:39 random:220] CPU RNG state changed within GPU RNG context
    [NeMo W 2025-06-18 16:51:39 random:220] CPU RNG state changed within GPU RNG context
    [NeMo W 2025-06-18 16:51:39 random:220] CPU RNG state changed within GPU RNG context
    [NeMo W 2025-06-18 16:51:39 random:220] CPU RNG state changed within GPU RNG context
    [NeMo W 2025-06-18 16:51:39 random:220] CPU RNG state changed within GPU RNG context
    [NeMo W 2025-06-18 16:51:39 random:220] CPU RNG state changed within GPU RNG context
    [NeMo W 2025-06-18 16:51:39 random:220] CPU RNG state changed within GPU RNG context
    [NeMo W 2025-06-18 16:51:39 random:220] CPU RNG state changed within GPU RNG context
    [NeMo W 2025-06-18 16:51:39 random:220] CPU RNG state changed within GPU RNG context
    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1]
    [NeMo I 2025-06-18 16:51:39 nemo_logging:393] Copying Trainer's 'max_steps' (100) to LR scheduler's 'max_steps'.
    LOCAL_RANK: 1 - CUDA_VISIBLE_DEVICES: [0,1]
    [NeMo I 2025-06-18 16:51:39 num_microbatches_calculator:228] setting number of microbatches to constant 1
    [NeMo I 2025-06-18 16:51:39 nemo_logging:393]  > number of parameters on (tensor, pipeline) model parallel rank (0 ,0): 1108204800
    [NeMo I 2025-06-18 16:51:39 utils:302] Setting up DistributedDataParallel with config DistributedDataParallelConfig(grad_reduce_in_fp32=False, overlap_grad_reduce=False, overlap_param_gather=False, align_param_gather=False, use_distributed_optimizer=True, num_distributed_optimizer_instances=1, check_for_nan_in_grad=True, check_for_large_grads=False, bucket_size=None, average_in_collective=True, fp8_param_gather=False)
    [NeMo I 2025-06-18 16:51:39 utils:323] Number of buckets for gradient all-reduce / reduce-scatter: 1
        Params for bucket 1 (1108204800 elements):
        	module.decoder.layers.24.mlp.linear_fc1.layer_norm_weight
        	module.decoder.layers.22.mlp.linear_fc1.layer_norm_weight
        	module.decoder.layers.19.mixer.dense.weight
        	module.decoder.layers.16.mixer.dense_projection.weight
        	module.decoder.layers.13.mlp.linear_fc2.weight
        	module.decoder.layers.9.mixer.hyena_proj_conv.short_conv_weight
        	module.decoder.layers.6.mixer.dense.weight
        	module.decoder.layers.4.mixer.mixer.short_conv.short_conv_weight
        	module.embedding.word_embeddings.weight
        	module.decoder.layers.22.mlp.linear_fc1.weight
        	module.decoder.layers.20.mlp.linear_fc1.layer_norm_weight
        	module.decoder.layers.14.mlp.linear_fc1.weight
        	module.decoder.layers.9.mlp.linear_fc1.weight
        	module.decoder.layers.8.mixer.dense_projection.layer_norm_weight
        	module.decoder.layers.5.mixer.mixer.filter.h
        	module.decoder.layers.0.mlp.linear_fc2.weight
        	module.decoder.layers.23.mixer.dense.weight
        	module.decoder.layers.21.mlp.linear_fc1.layer_norm_weight
        	module.decoder.layers.19.mixer.dense_projection.weight
        	module.decoder.layers.15.mlp.linear_fc2.weight
        	module.decoder.layers.13.mixer.mixer.filter.p
        	module.decoder.layers.11.mixer.hyena_proj_conv.short_conv_weight
        	module.decoder.layers.8.mixer.dense.weight
        	module.decoder.layers.6.mixer.dense.bias
        	module.decoder.layers.1.mixer.dense.weight
        	module.decoder.layers.0.mixer.dense_projection.weight
        	module.decoder.layers.24.mlp.linear_fc1.weight
        	module.decoder.layers.22.mixer.mixer.filter.h
        	module.decoder.layers.20.mixer.mixer.filter.R
        	module.decoder.layers.18.mixer.dense_projection.layer_norm_weight
        	module.decoder.layers.17.self_attention.linear_proj.weight
        	module.decoder.layers.14.mixer.dense.weight
        	module.decoder.layers.13.mixer.mixer.conv_bias
        	module.decoder.layers.9.mlp.linear_fc1.layer_norm_weight
        	module.decoder.layers.0.mixer.dense.weight
        	module.decoder.layers.18.mlp.linear_fc2.weight
        	module.decoder.layers.13.mixer.hyena_proj_conv.short_conv_weight
        	module.decoder.layers.10.self_attention.linear_qkv.weight
        	module.decoder.layers.8.mixer.dense_projection.weight
        	module.decoder.layers.6.mixer.mixer.filter.gamma
        	module.decoder.layers.0.mixer.dense.bias
        	module.decoder.layers.2.mixer.mixer.filter.R
        	module.decoder.layers.24.self_attention.linear_qkv.layer_norm_weight
        	module.decoder.layers.22.mixer.mixer.conv_bias
        	module.decoder.layers.20.mixer.dense_projection.layer_norm_weight
        	module.decoder.layers.13.mlp.linear_fc1.weight
        	module.decoder.layers.12.mixer.dense_projection.layer_norm_weight
        	module.decoder.layers.9.mixer.mixer.filter.R
        	module.decoder.layers.7.mixer.dense_projection.layer_norm_weight
        	module.decoder.layers.2.mixer.hyena_proj_conv.short_conv_weight
        	module.decoder.layers.23.mixer.mixer.filter.gamma
        	module.decoder.layers.18.mixer.dense_projection.weight
        	module.decoder.layers.15.mixer.mixer.conv_bias
        	module.decoder.layers.12.mixer.dense.weight
        	module.decoder.layers.7.mlp.linear_fc2.weight
        	module.decoder.layers.0.mlp.linear_fc1.weight
        	module.decoder.layers.24.self_attention.linear_proj.weight
        	module.decoder.layers.21.mixer.mixer.short_conv.short_conv_weight
        	module.decoder.layers.15.mlp.linear_fc1.weight
        	module.decoder.layers.13.mlp.linear_fc1.layer_norm_weight
        	module.decoder.layers.9.mixer.dense_projection.layer_norm_weight
        	module.decoder.layers.4.mixer.hyena_proj_conv.short_conv_weight
        	module.decoder.layers.2.mixer.dense_projection.layer_norm_weight
        	module.decoder.layers.20.mixer.dense_projection.weight
        	module.decoder.layers.17.mlp.linear_fc2.weight
        	module.decoder.layers.17.self_attention.linear_proj.bias
        	module.decoder.layers.16.mixer.dense.weight
        	module.decoder.layers.14.mlp.linear_fc1.layer_norm_weight
        	module.decoder.layers.12.mixer.dense_projection.weight
        	module.decoder.layers.7.mixer.dense_projection.weight
        	module.decoder.layers.5.mixer.hyena_proj_conv.short_conv_weight
        	module.decoder.layers.23.mixer.mixer.conv_bias
        	module.decoder.layers.18.mlp.linear_fc1.weight
        	module.decoder.layers.15.mixer.mixer.filter.h
        	module.decoder.layers.13.mixer.mixer.filter.R
        	module.decoder.layers.11.mixer.dense_projection.layer_norm_weight
        	module.decoder.layers.8.mixer.dense.bias
        	module.decoder.layers.3.self_attention.linear_qkv.weight
        	module.decoder.layers.3.self_attention.linear_proj.bias
        	module.decoder.layers.22.mixer.hyena_proj_conv.short_conv_weight
        	module.decoder.layers.19.mlp.linear_fc2.weight
        	module.decoder.layers.16.mixer.dense.bias
        	module.decoder.layers.14.mixer.dense.bias
        	module.decoder.layers.11.mlp.linear_fc2.weight
        	module.decoder.layers.9.mixer.dense_projection.weight
        	module.decoder.layers.6.mlp.linear_fc2.weight
        	module.decoder.layers.0.mixer.dense_projection.layer_norm_weight
        	module.decoder.layers.18.mixer.dense.weight
        	module.decoder.layers.13.mixer.dense_projection.layer_norm_weight
        	module.decoder.layers.7.mlp.linear_fc1.weight
        	module.decoder.layers.1.mixer.mixer.conv_bias
        	module.decoder.final_norm.weight
        	module.decoder.layers.23.mlp.linear_fc2.weight
        	module.decoder.layers.16.mixer.mixer.filter.gamma
        	module.decoder.layers.11.mixer.dense_projection.weight
        	module.decoder.layers.8.mlp.linear_fc2.weight
        	module.decoder.layers.6.mixer.mixer.filter.p
        	module.decoder.layers.4.mixer.dense.weight
        	module.decoder.layers.1.mlp.linear_fc1.layer_norm_weight
        	module.decoder.layers.20.mixer.dense.bias
        	module.decoder.layers.17.mlp.linear_fc1.weight
        	module.decoder.layers.14.mixer.mixer.short_conv.short_conv_weight
        	module.decoder.layers.12.mixer.dense.bias
        	module.decoder.layers.7.mixer.dense.weight
        	module.decoder.layers.6.mixer.mixer.conv_bias
        	module.decoder.layers.2.mixer.dense.bias
        	module.decoder.layers.0.mixer.mixer.short_conv.short_conv_weight
        	module.decoder.layers.23.mixer.mixer.filter.p
        	module.decoder.layers.21.mixer.hyena_proj_conv.short_conv_weight
        	module.decoder.layers.19.mlp.linear_fc1.layer_norm_weight
        	module.decoder.layers.13.mixer.dense_projection.weight
        	module.decoder.layers.10.mlp.linear_fc2.weight
        	module.decoder.layers.6.mixer.hyena_proj_conv.short_conv_weight
        	module.decoder.layers.4.mixer.dense_projection.layer_norm_weight
        	module.decoder.layers.1.mixer.mixer.filter.h
        	module.decoder.layers.22.mixer.dense.bias
        	module.decoder.layers.19.mlp.linear_fc1.weight
        	module.decoder.layers.17.self_attention.linear_qkv.layer_norm_weight
        	module.decoder.layers.12.mlp.linear_fc1.layer_norm_weight
        	module.decoder.layers.11.mlp.linear_fc1.weight
        	module.decoder.layers.10.self_attention.linear_proj.bias
        	module.decoder.layers.6.mlp.linear_fc1.weight
        	module.decoder.layers.5.mixer.dense_projection.layer_norm_weight
        	module.decoder.layers.2.mixer.mixer.conv_bias
        	module.decoder.layers.2.mixer.dense_projection.weight
        	module.decoder.layers.23.mixer.hyena_proj_conv.short_conv_weight
        	module.decoder.layers.20.mixer.dense.weight
        	module.decoder.layers.18.mlp.linear_fc1.layer_norm_weight
        	module.decoder.layers.15.mixer.hyena_proj_conv.short_conv_weight
        	module.decoder.layers.12.mlp.linear_fc2.weight
        	module.decoder.layers.10.mlp.linear_fc1.layer_norm_weight
        	module.decoder.layers.8.mixer.mixer.conv_bias
        	module.decoder.layers.5.mixer.dense.weight
        	module.decoder.layers.3.self_attention.linear_proj.weight
        	module.decoder.layers.23.mlp.linear_fc1.weight
        	module.decoder.layers.22.mixer.dense_projection.layer_norm_weight
        	module.decoder.layers.19.mixer.mixer.filter.h
        	module.decoder.layers.11.mixer.dense.weight
        	module.decoder.layers.8.mlp.linear_fc1.weight
        	module.decoder.layers.6.mlp.linear_fc1.layer_norm_weight
        	module.decoder.layers.4.mixer.dense_projection.weight
        	module.decoder.layers.1.mlp.linear_fc2.weight
        	module.decoder.layers.22.mixer.dense.weight
        	module.decoder.layers.16.mlp.linear_fc2.weight
        	module.decoder.layers.9.mixer.dense.weight
        	module.decoder.layers.7.mlp.linear_fc1.layer_norm_weight
        	module.decoder.layers.5.mixer.dense_projection.weight
        	module.decoder.layers.23.mlp.linear_fc1.layer_norm_weight
        	module.decoder.layers.19.mixer.mixer.conv_bias
        	module.decoder.layers.15.mlp.linear_fc1.layer_norm_weight
        	module.decoder.layers.10.mlp.linear_fc1.weight
        	module.decoder.layers.8.mixer.mixer.filter.h
        	module.decoder.layers.6.mixer.mixer.filter.R
        	module.decoder.layers.3.mlp.linear_fc2.weight
        	module.decoder.layers.24.self_attention.linear_qkv.weight
        	module.decoder.layers.22.mixer.dense_projection.weight
        	module.decoder.layers.20.mixer.mixer.filter.gamma
        	module.decoder.layers.17.mlp.linear_fc1.layer_norm_weight
        	module.decoder.layers.16.mixer.mixer.filter.p
        	module.decoder.layers.14.mixer.hyena_proj_conv.short_conv_weight
        	module.decoder.layers.12.mixer.mixer.conv_bias
        	module.decoder.layers.9.mixer.dense.bias
        	module.decoder.layers.7.mixer.dense.bias
        	module.decoder.layers.4.mlp.linear_fc2.weight
        	module.decoder.layers.23.mixer.mixer.filter.R
        	module.decoder.layers.21.mixer.dense_projection.layer_norm_weight
        	module.decoder.layers.18.mixer.mixer.short_conv.short_conv_weight
        	module.decoder.layers.16.mixer.mixer.conv_bias
        	module.decoder.layers.12.mlp.linear_fc1.weight
        	module.decoder.layers.10.self_attention.linear_qkv.layer_norm_weight
        	module.decoder.layers.6.mixer.dense_projection.layer_norm_weight
        	module.decoder.layers.3.mlp.linear_fc1.layer_norm_weight
        	module.decoder.layers.1.mixer.hyena_proj_conv.short_conv_weight
        	module.decoder.layers.21.mlp.linear_fc2.weight
        	module.decoder.layers.16.mixer.hyena_proj_conv.short_conv_weight
        	module.decoder.layers.13.mixer.dense.weight
        	module.decoder.layers.11.mlp.linear_fc1.layer_norm_weight
        	module.decoder.layers.9.mixer.mixer.filter.gamma
        	module.decoder.layers.4.mlp.linear_fc1.layer_norm_weight
        	module.decoder.layers.23.mixer.dense_projection.layer_norm_weight
        	module.decoder.layers.20.mixer.mixer.conv_bias
        	module.decoder.layers.16.mlp.linear_fc1.weight
        	module.decoder.layers.15.mixer.dense_projection.layer_norm_weight
        	module.decoder.layers.12.mixer.mixer.filter.h
        	module.decoder.layers.10.self_attention.linear_proj.weight
        	module.decoder.layers.7.mixer.mixer.short_conv.short_conv_weight
        	module.decoder.layers.5.mixer.dense.bias
        	module.decoder.layers.2.mlp.linear_fc2.weight
        	module.decoder.layers.2.mixer.dense.weight
        	module.decoder.layers.21.mixer.dense_projection.weight
        	module.decoder.layers.19.mixer.hyena_proj_conv.short_conv_weight
        	module.decoder.layers.15.mixer.dense.weight
        	module.decoder.layers.13.mixer.dense.bias
        	module.decoder.layers.11.mixer.dense.bias
        	module.decoder.layers.6.mixer.dense_projection.weight
        	module.decoder.layers.3.mlp.linear_fc1.weight
        	module.decoder.layers.21.mixer.dense.bias
        	module.decoder.layers.16.mlp.linear_fc1.layer_norm_weight
        	module.decoder.layers.4.mlp.linear_fc1.weight
        	module.decoder.layers.2.mixer.mixer.filter.p
        	module.decoder.layers.0.mixer.hyena_proj_conv.short_conv_weight
        	module.decoder.layers.23.mixer.dense_projection.weight
        	module.decoder.layers.20.mlp.linear_fc2.weight
        	module.decoder.layers.15.mixer.dense_projection.weight
        	module.decoder.layers.13.mixer.mixer.filter.gamma
        	module.decoder.layers.8.mixer.hyena_proj_conv.short_conv_weight
        	module.decoder.layers.5.mlp.linear_fc2.weight
        	module.decoder.layers.3.self_attention.linear_qkv.layer_norm_weight
        	module.decoder.layers.1.mixer.dense.bias
        	module.decoder.layers.1.mlp.linear_fc1.weight
        	module.decoder.layers.24.self_attention.linear_proj.bias
        	module.decoder.layers.21.mlp.linear_fc1.weight
        	module.decoder.layers.16.mixer.mixer.filter.R
        	module.decoder.layers.14.mixer.dense_projection.layer_norm_weight
        	module.decoder.layers.11.mixer.mixer.short_conv.short_conv_weight
        	module.decoder.layers.5.mlp.linear_fc1.layer_norm_weight
        	module.decoder.layers.2.mixer.mixer.filter.gamma
        	module.decoder.layers.17.self_attention.linear_qkv.weight
        	module.decoder.layers.22.mlp.linear_fc2.weight
        	module.decoder.layers.20.mixer.mixer.filter.p
        	module.decoder.layers.18.mixer.hyena_proj_conv.short_conv_weight
        	module.decoder.layers.14.mlp.linear_fc2.weight
        	module.decoder.layers.9.mlp.linear_fc2.weight
        	module.decoder.layers.2.mlp.linear_fc1.weight
        	module.decoder.layers.1.mixer.dense_projection.layer_norm_weight
        	module.decoder.layers.21.mixer.dense.weight
        	module.decoder.layers.19.mixer.dense.bias
        	module.decoder.layers.16.mixer.dense_projection.layer_norm_weight
        	module.decoder.layers.8.mlp.linear_fc1.layer_norm_weight
        	module.decoder.layers.4.mixer.dense.bias
        	module.decoder.layers.24.mlp.linear_fc2.weight
        	module.decoder.layers.20.mixer.hyena_proj_conv.short_conv_weight
        	module.decoder.layers.18.mixer.dense.bias
        	module.decoder.layers.14.mixer.dense_projection.weight
        	module.decoder.layers.12.mixer.hyena_proj_conv.short_conv_weight
        	module.decoder.layers.9.mixer.mixer.filter.p
        	module.decoder.layers.7.mixer.hyena_proj_conv.short_conv_weight
        	module.decoder.layers.5.mixer.mixer.conv_bias
        	module.decoder.layers.2.mlp.linear_fc1.layer_norm_weight
        	module.decoder.layers.0.mlp.linear_fc1.layer_norm_weight
        	module.decoder.layers.23.mixer.dense.bias
        	module.decoder.layers.20.mlp.linear_fc1.weight
        	module.decoder.layers.19.mixer.dense_projection.layer_norm_weight
        	module.decoder.layers.15.mixer.dense.bias
        	module.decoder.layers.9.mixer.mixer.conv_bias
        	module.decoder.layers.5.mlp.linear_fc1.weight
        	module.decoder.layers.1.mixer.dense_projection.weight
    [NeMo I 2025-06-18 16:51:39 utils:302] Setting up optimizer with config OptimizerConfig(optimizer='adam', lr=0.0001, min_lr=None, decoupled_lr=None, decoupled_min_lr=None, weight_decay=0.01, fp16=False, bf16=True, params_dtype=torch.bfloat16, use_precision_aware_optimizer=False, main_grads_dtype=torch.float32, main_params_dtype=torch.float32, exp_avg_dtype=torch.float32, exp_avg_sq_dtype=torch.float32, loss_scale=None, initial_loss_scale=4294967296, min_loss_scale=1.0, loss_scale_window=1000, hysteresis=2, adam_beta1=0.9, adam_beta2=0.95, adam_eps=1e-08, sgd_momentum=0.9, use_distributed_optimizer=True, overlap_param_gather_with_optimizer_step=False, optimizer_cpu_offload=False, optimizer_offload_fraction=0.0, use_torch_optimizer_for_cpu_offload=False, overlap_cpu_optimizer_d2h_h2d=False, pin_cpu_grads=True, pin_cpu_params=True, clip_grad=1.0, log_num_zeros_in_grad=False, barrier_with_L1_time=False, timers=None, config_logger_dir='')
    [NeMo I 2025-06-18 16:51:39 nemo_logging:393] Doing selective restore from RestoreConfig(path='nemo2_evo2_1b_8k', adapter_path=None, load_model_state=True, load_optim_state=False, load_artifacts=True)
    [NeMo I 2025-06-18 16:51:39 nemo_logging:393] Using <megatron.core.dist_checkpointing.strategies.fully_parallel.FullyParallelLoadStrategyWrapper object at 0x149b30b285c0> dist-ckpt load strategy.
    [NeMo I 2025-06-18 16:51:41 nemo_logging:393] Global Checkpoint Load : Rank : 0 : Start time : 1750290699.753s : Time spent in load_checkpoint: 2.181s
    [NeMo I 2025-06-18 16:51:41 nemo_logging:393] Restoring model weights from RestoreConfig(path='nemo2_evo2_1b_8k', adapter_path=None, load_model_state=True, load_optim_state=False, load_artifacts=True)
    [NeMo I 2025-06-18 16:51:41 nemo_logging:393] Finished restoring from RestoreConfig(path='nemo2_evo2_1b_8k', adapter_path=None, load_model_state=True, load_optim_state=False, load_artifacts=True), cleaning up.
    ┏━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┓
    ┃[1;35m [0m[1;35m [0m[1;35m [0m┃[1;35m [0m[1;35mName                               [0m[1;35m [0m┃[1;35m [0m[1;35mType             [0m[1;35m [0m┃[1;35m [0m[1;35mParams[0m[1;35m [0m┃[1;35m [0m[1;35mMode [0m[1;35m [0m┃
    ┡━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━┩
    │[2m [0m[2m0[0m[2m [0m│ module                              │ DDP               │  1.1 B │ train │
    │[2m [0m[2m1[0m[2m [0m│ module.module                       │ Float16Module     │  1.1 B │ train │
    │[2m [0m[2m2[0m[2m [0m│ module.module.module                │ HyenaModel        │  1.1 B │ train │
    │[2m [0m[2m3[0m[2m [0m│ module.module.module.embedding      │ LanguageModelEmb… │  983 K │ train │
    │[2m [0m[2m4[0m[2m [0m│ module.module.module.rotary_pos_emb │ RotaryEmbedding   │      0 │ train │
    │[2m [0m[2m5[0m[2m [0m│ module.module.module.decoder        │ HyenaStack        │  1.1 B │ train │
    │[2m [0m[2m6[0m[2m [0m│ module.module.module.output_layer   │ ColumnParallelLi… │      0 │ train │
    └───┴─────────────────────────────────────┴───────────────────┴────────┴───────┘
    [1mTrainable params[0m: 1.1 B                                                         
    [1mNon-trainable params[0m: 0                                                         
    [1mTotal params[0m: 1.1 B                                                             
    [1mTotal estimated model params size (MB)[0m: 4.4 K                                   
    [1mModules in train mode[0m: 356                                                      
    [1mModules in eval mode[0m: 0                                                         
    [WARNING  | DotProductAttention]: flash-attn v3 may provide important feature support or performance improvement. Please install flash-attn v3 by 
    (1) git clone https://github.com/Dao-AILab/flash-attention.git
    (2) cd flash-attention/ && git checkout 27f501d && cd hopper/ && python setup.py install
    (3) python_path=`python -c "import site; print(site.getsitepackages()[0])"`
    (4) mkdir -p $python_path/flash_attn_3
    (5) wget -P $python_path/flash_attn_3 https://raw.githubusercontent.com/Dao-AILab/flash-attention/27f501dbe011f4371bff938fe7e09311ab3002fa/hopper/flash_attn_interface.py
    WARNING:DotProductAttention:flash-attn v3 may provide important feature support or performance improvement. Please install flash-attn v3 by 
    (1) git clone https://github.com/Dao-AILab/flash-attention.git
    (2) cd flash-attention/ && git checkout 27f501d && cd hopper/ && python setup.py install
    (3) python_path=`python -c "import site; print(site.getsitepackages()[0])"`
    (4) mkdir -p $python_path/flash_attn_3
    (5) wget -P $python_path/flash_attn_3 https://raw.githubusercontent.com/Dao-AILab/flash-attention/27f501dbe011f4371bff938fe7e09311ab3002fa/hopper/flash_attn_interface.py
    [NeMo W 2025-06-18 16:52:05 rerun_state_machine:1264] Implicit initialization of Rerun State Machine!
    [NeMo W 2025-06-18 16:52:05 rerun_state_machine:239] RerunStateMachine initialized in mode RerunMode.DISABLED
    Training epoch 0, iteration 0/99 | lr: 0 | global_batch_size: 4 | global_step: 0 | reduced_train_loss: 1.285 | train_step_timing in s: 23.52
    Training epoch 0, iteration 1/99 | lr: 2e-05 | global_batch_size: 4 | global_step: 1 | reduced_train_loss: 1.31 | train_step_timing in s: 2.262 | consumed_samples: 8
    Training epoch 0, iteration 2/99 | lr: 4e-05 | global_batch_size: 4 | global_step: 2 | reduced_train_loss: 1.304 | train_step_timing in s: 0.2453 | consumed_samples: 12
    Training epoch 0, iteration 3/99 | lr: 6e-05 | global_batch_size: 4 | global_step: 3 | reduced_train_loss: 1.143 | train_step_timing in s: 0.2391 | consumed_samples: 16
    Training epoch 0, iteration 4/99 | lr: 8e-05 | global_batch_size: 4 | global_step: 4 | reduced_train_loss: 1.269 | train_step_timing in s: 0.2375 | consumed_samples: 20
    Training epoch 0, iteration 5/99 | lr: 0.0001 | global_batch_size: 4 | global_step: 5 | reduced_train_loss: 1.174 | train_step_timing in s: 0.2486 | consumed_samples: 24
    Training epoch 0, iteration 6/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 6 | reduced_train_loss: 1.269 | train_step_timing in s: 0.2475 | consumed_samples: 28
    Training epoch 0, iteration 7/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 7 | reduced_train_loss: 1.294 | train_step_timing in s: 0.2514 | consumed_samples: 32
    Training epoch 0, iteration 8/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 8 | reduced_train_loss: 1.273 | train_step_timing in s: 0.2539 | consumed_samples: 36
    Training epoch 0, iteration 9/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 9 | reduced_train_loss: 1.277 | train_step_timing in s: 0.248 | consumed_samples: 40
    Training epoch 0, iteration 10/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 10 | reduced_train_loss: 0.8843 | train_step_timing in s: 0.2607 | consumed_samples: 44
    Training epoch 0, iteration 11/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 11 | reduced_train_loss: 1.288 | train_step_timing in s: 0.2315 | consumed_samples: 48
    Training epoch 0, iteration 12/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 12 | reduced_train_loss: 1.269 | train_step_timing in s: 0.2592 | consumed_samples: 52
    Training epoch 0, iteration 13/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 13 | reduced_train_loss: 1.302 | train_step_timing in s: 0.2388 | consumed_samples: 56
    Training epoch 0, iteration 14/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 14 | reduced_train_loss: 1.318 | train_step_timing in s: 0.2464 | consumed_samples: 60
    Training epoch 0, iteration 15/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 15 | reduced_train_loss: 1.008 | train_step_timing in s: 0.2594 | consumed_samples: 64
    Training epoch 0, iteration 16/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 16 | reduced_train_loss: 1.295 | train_step_timing in s: 0.2416 | consumed_samples: 68
    Training epoch 0, iteration 17/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 17 | reduced_train_loss: 1.255 | train_step_timing in s: 0.2392 | consumed_samples: 72
    Training epoch 0, iteration 18/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 18 | reduced_train_loss: 1.299 | train_step_timing in s: 0.2422 | consumed_samples: 76
    Training epoch 0, iteration 19/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 19 | reduced_train_loss: 1.29 | train_step_timing in s: 0.2487 | consumed_samples: 80
    Training epoch 0, iteration 20/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 20 | reduced_train_loss: 1.181 | train_step_timing in s: 0.2456 | consumed_samples: 84
    Training epoch 0, iteration 21/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 21 | reduced_train_loss: 1.277 | train_step_timing in s: 0.2462 | consumed_samples: 88
    Training epoch 0, iteration 22/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 22 | reduced_train_loss: 1.3 | train_step_timing in s: 0.2392 | consumed_samples: 92
    Training epoch 0, iteration 23/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 23 | reduced_train_loss: 1.296 | train_step_timing in s: 0.2415 | consumed_samples: 96
    Training epoch 0, iteration 24/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 24 | reduced_train_loss: 1.295 | train_step_timing in s: 0.2475 | consumed_samples: 100
    Training epoch 0, iteration 25/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 25 | reduced_train_loss: 1.229 | train_step_timing in s: 0.2484 | consumed_samples: 104
    Training epoch 0, iteration 26/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 26 | reduced_train_loss: 1.277 | train_step_timing in s: 0.2429 | consumed_samples: 108
    Training epoch 0, iteration 27/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 27 | reduced_train_loss: 1.315 | train_step_timing in s: 0.2663 | consumed_samples: 112
    Training epoch 0, iteration 28/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 28 | reduced_train_loss: 1.272 | train_step_timing in s: 0.252 | consumed_samples: 116
    Training epoch 0, iteration 29/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 29 | reduced_train_loss: 1.004 | train_step_timing in s: 0.2367 | consumed_samples: 120
    Training epoch 0, iteration 30/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 30 | reduced_train_loss: 1.299 | train_step_timing in s: 0.2364 | consumed_samples: 124
    Training epoch 0, iteration 31/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 31 | reduced_train_loss: 1.291 | train_step_timing in s: 0.2305 | consumed_samples: 128
    Training epoch 0, iteration 32/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 32 | reduced_train_loss: 1.237 | train_step_timing in s: 0.2381 | consumed_samples: 132
    Training epoch 0, iteration 33/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 33 | reduced_train_loss: 1.302 | train_step_timing in s: 0.2467 | consumed_samples: 136
    Training epoch 0, iteration 34/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 34 | reduced_train_loss: 1.301 | train_step_timing in s: 0.2445 | consumed_samples: 140
    Training epoch 0, iteration 35/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 35 | reduced_train_loss: 1.281 | train_step_timing in s: 0.2461 | consumed_samples: 144
    Training epoch 0, iteration 36/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 36 | reduced_train_loss: 1.041 | train_step_timing in s: 0.2423 | consumed_samples: 148
    Training epoch 0, iteration 37/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 37 | reduced_train_loss: 1.254 | train_step_timing in s: 0.247 | consumed_samples: 152
    Training epoch 0, iteration 38/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 38 | reduced_train_loss: 1.304 | train_step_timing in s: 0.2332 | consumed_samples: 156
    Training epoch 0, iteration 39/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 39 | reduced_train_loss: 1.309 | train_step_timing in s: 0.2522 | consumed_samples: 160
    Training epoch 0, iteration 40/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 40 | reduced_train_loss: 1.321 | train_step_timing in s: 0.2458 | consumed_samples: 164
    Training epoch 0, iteration 41/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 41 | reduced_train_loss: 1.273 | train_step_timing in s: 0.2432 | consumed_samples: 168
    Training epoch 0, iteration 42/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 42 | reduced_train_loss: 1.32 | train_step_timing in s: 0.2609 | consumed_samples: 172
    Training epoch 0, iteration 43/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 43 | reduced_train_loss: 1.245 | train_step_timing in s: 0.2405 | consumed_samples: 176
    Training epoch 0, iteration 44/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 44 | reduced_train_loss: 1.313 | train_step_timing in s: 0.2571 | consumed_samples: 180
    Training epoch 0, iteration 45/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 45 | reduced_train_loss: 1.288 | train_step_timing in s: 0.2382 | consumed_samples: 184
    Training epoch 0, iteration 46/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 46 | reduced_train_loss: 1.193 | train_step_timing in s: 0.2287 | consumed_samples: 188
    Training epoch 0, iteration 47/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 47 | reduced_train_loss: 1.25 | train_step_timing in s: 0.2333 | consumed_samples: 192
    Training epoch 0, iteration 48/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 48 | reduced_train_loss: 1.289 | train_step_timing in s: 0.2463 | consumed_samples: 196
    Training epoch 0, iteration 49/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 49 | reduced_train_loss: 1.309 | train_step_timing in s: 0.2482 | consumed_samples: 200
    Epoch 0, global step 49: 'val_loss' was not in top 5
    [NeMo I 2025-06-18 16:52:20 nemo_logging:393] Using FullyParallelSaveStrategyWrapper(torch_dist, 1) dist-ckpt save strategy.
    [NeMo I 2025-06-18 16:52:43 nemo_logging:393] Global Checkpoint Save : Rank: 0 : Iteration: 49 : Start time: 1750290740.432s : Save duration: 23.107s
    [NeMo I 2025-06-18 16:52:54 nemo_logging:393] Scheduled async checkpoint save for /workspace/bionemo2/data/results/evo2/checkpoints/evo2--val_loss=0.0000-epoch=0-consumed_samples=200.0-last.ckpt
    [NeMo I 2025-06-18 16:52:54 nemo_logging:393] Async finalization time took 0.001 s
    Validation: iteration 1/20
    Validation: iteration 2/20
    Validation: iteration 3/20
    Validation: iteration 4/20
    Validation: iteration 5/20
    Validation: iteration 6/20
    Validation: iteration 7/20
    Validation: iteration 8/20
    Validation: iteration 9/20
    Validation: iteration 10/20
    Validation: iteration 11/20
    Validation: iteration 12/20
    Validation: iteration 13/20
    Validation: iteration 14/20
    Validation: iteration 15/20
    Validation: iteration 16/20
    Validation: iteration 17/20
    Validation: iteration 18/20
    Validation: iteration 19/20
    Validation: iteration 20/20
    [NeMo W 2025-06-18 16:52:59 nemo_logging:405] /usr/local/lib/python3.12/dist-packages/lightning/pytorch/trainer/connectors/logger_connector/result.py:431: It is recommended to use `self.log('global_batch_size', ..., sync_dist=True)` when logging on epoch level in distributed setting to accumulate the metric across devices.
        
    [NeMo W 2025-06-18 16:53:00 nemo_logging:405] /usr/local/lib/python3.12/dist-packages/lightning/pytorch/trainer/connectors/logger_connector/result.py:431: It is recommended to use `self.log('val_loss', ..., sync_dist=True)` when logging on epoch level in distributed setting to accumulate the metric across devices.
        
    Training epoch 0, iteration 50/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 50 | reduced_train_loss: 1.257 | train_step_timing in s: 0.685 | consumed_samples: 204 | val_loss: 1.619
    [NeMo I 2025-06-18 16:53:00 nemo_logging:393] Async finalization time took 0.000 s
    Training epoch 0, iteration 51/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 51 | reduced_train_loss: 1.257 | train_step_timing in s: 0.6542 | consumed_samples: 208 | val_loss: 1.619
    [NeMo I 2025-06-18 16:53:01 nemo_logging:393] Async finalization time took 0.043 s
    Training epoch 0, iteration 52/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 52 | reduced_train_loss: 1.29 | train_step_timing in s: 0.6231 | consumed_samples: 212 | val_loss: 1.619
    [NeMo I 2025-06-18 16:53:02 nemo_logging:393] Async finalization time took 0.000 s
    Training epoch 0, iteration 53/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 53 | reduced_train_loss: 1.285 | train_step_timing in s: 0.6661 | consumed_samples: 216 | val_loss: 1.619
    [NeMo I 2025-06-18 16:53:02 nemo_logging:393] Async finalization time took 0.018 s
    Training epoch 0, iteration 54/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 54 | reduced_train_loss: 1.231 | train_step_timing in s: 0.6769 | consumed_samples: 220 | val_loss: 1.619
    [NeMo I 2025-06-18 16:53:03 nemo_logging:393] Async finalization time took 0.000 s
    Training epoch 0, iteration 55/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 55 | reduced_train_loss: 1.291 | train_step_timing in s: 0.6632 | consumed_samples: 224 | val_loss: 1.619
    [NeMo I 2025-06-18 16:53:04 nemo_logging:393] Async finalization time took 0.000 s
    Training epoch 0, iteration 56/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 56 | reduced_train_loss: 0.7497 | train_step_timing in s: 0.6125 | consumed_samples: 228 | val_loss: 1.619
    [NeMo I 2025-06-18 16:53:04 nemo_logging:393] Async finalization time took 0.000 s
    Training epoch 0, iteration 57/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 57 | reduced_train_loss: 1.1 | train_step_timing in s: 0.5248 | consumed_samples: 232 | val_loss: 1.619
    [NeMo I 2025-06-18 16:53:05 nemo_logging:393] Async finalization time took 0.000 s
    Training epoch 0, iteration 58/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 58 | reduced_train_loss: 1.289 | train_step_timing in s: 0.4839 | consumed_samples: 236 | val_loss: 1.619
    [NeMo I 2025-06-18 16:53:05 nemo_logging:393] Async finalization time took 0.019 s
    Training epoch 0, iteration 59/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 59 | reduced_train_loss: 1.237 | train_step_timing in s: 0.4548 | consumed_samples: 240 | val_loss: 1.619
    [NeMo I 2025-06-18 16:53:06 nemo_logging:393] Async finalization time took 0.004 s
    Training epoch 0, iteration 60/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 60 | reduced_train_loss: 1.315 | train_step_timing in s: 0.4651 | consumed_samples: 244 | val_loss: 1.619
    [NeMo I 2025-06-18 16:53:06 nemo_logging:393] Async finalization time took 0.039 s
    Training epoch 0, iteration 61/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 61 | reduced_train_loss: 1.299 | train_step_timing in s: 0.479 | consumed_samples: 248 | val_loss: 1.619
    [NeMo I 2025-06-18 16:53:07 nemo_logging:393] Async finalization time took 0.000 s
    Training epoch 0, iteration 62/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 62 | reduced_train_loss: 1.268 | train_step_timing in s: 0.4226 | consumed_samples: 252 | val_loss: 1.619
    [NeMo I 2025-06-18 16:53:07 nemo_logging:393] Async finalization time took 0.013 s
    Training epoch 0, iteration 63/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 63 | reduced_train_loss: 1.274 | train_step_timing in s: 0.2902 | consumed_samples: 256 | val_loss: 1.619
    [NeMo I 2025-06-18 16:53:08 nemo_logging:393] Successfully saved checkpoint from iteration      49 to /workspace/bionemo2/data/results/evo2/checkpoints/evo2--val_loss=0.0000-epoch=0-consumed_samples=200.0-last.ckpt
    [NeMo I 2025-06-18 16:53:08 nemo_logging:393] Async checkpoint save for step 50 (/workspace/bionemo2/data/results/evo2/checkpoints/evo2--val_loss=0.0000-epoch=0-consumed_samples=200.0-last.ckpt) finalized successfully.
    [NeMo I 2025-06-18 16:53:08 nemo_logging:393] Async finalization time took 0.109 s
    Training epoch 0, iteration 64/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 64 | reduced_train_loss: 1.302 | train_step_timing in s: 0.2374 | consumed_samples: 260 | val_loss: 1.619
    Training epoch 0, iteration 65/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 65 | reduced_train_loss: 1.293 | train_step_timing in s: 0.2455 | consumed_samples: 264 | val_loss: 1.619
    Training epoch 0, iteration 66/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 66 | reduced_train_loss: 1.265 | train_step_timing in s: 0.2503 | consumed_samples: 268 | val_loss: 1.619
    Training epoch 0, iteration 67/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 67 | reduced_train_loss: 1.654 | train_step_timing in s: 0.2569 | consumed_samples: 272 | val_loss: 1.619
    Training epoch 0, iteration 68/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 68 | reduced_train_loss: 1.272 | train_step_timing in s: 0.2526 | consumed_samples: 276 | val_loss: 1.619
    Training epoch 0, iteration 69/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 69 | reduced_train_loss: 1.236 | train_step_timing in s: 0.2423 | consumed_samples: 280 | val_loss: 1.619
    Training epoch 0, iteration 70/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 70 | reduced_train_loss: 1.259 | train_step_timing in s: 0.2459 | consumed_samples: 284 | val_loss: 1.619
    Training epoch 0, iteration 71/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 71 | reduced_train_loss: 1.067 | train_step_timing in s: 0.2419 | consumed_samples: 288 | val_loss: 1.619
    Training epoch 0, iteration 72/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 72 | reduced_train_loss: 1.233 | train_step_timing in s: 0.2557 | consumed_samples: 292 | val_loss: 1.619
    Training epoch 0, iteration 73/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 73 | reduced_train_loss: 1.295 | train_step_timing in s: 0.2385 | consumed_samples: 296 | val_loss: 1.619
    Training epoch 0, iteration 74/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 74 | reduced_train_loss: 1.303 | train_step_timing in s: 0.245 | consumed_samples: 300 | val_loss: 1.619
    Training epoch 0, iteration 75/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 75 | reduced_train_loss: 1.172 | train_step_timing in s: 0.2687 | consumed_samples: 304 | val_loss: 1.619
    Training epoch 0, iteration 76/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 76 | reduced_train_loss: 1.262 | train_step_timing in s: 0.2488 | consumed_samples: 308 | val_loss: 1.619
    Training epoch 0, iteration 77/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 77 | reduced_train_loss: 1.313 | train_step_timing in s: 0.2331 | consumed_samples: 312 | val_loss: 1.619
    Training epoch 0, iteration 78/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 78 | reduced_train_loss: 1.217 | train_step_timing in s: 0.2372 | consumed_samples: 316 | val_loss: 1.619
    Training epoch 0, iteration 79/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 79 | reduced_train_loss: 1.265 | train_step_timing in s: 0.2702 | consumed_samples: 320 | val_loss: 1.619
    Training epoch 0, iteration 80/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 80 | reduced_train_loss: 1.299 | train_step_timing in s: 0.2481 | consumed_samples: 324 | val_loss: 1.619
    Training epoch 0, iteration 81/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 81 | reduced_train_loss: 1.257 | train_step_timing in s: 0.2361 | consumed_samples: 328 | val_loss: 1.619
    Training epoch 0, iteration 82/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 82 | reduced_train_loss: 1.274 | train_step_timing in s: 0.2447 | consumed_samples: 332 | val_loss: 1.619
    Training epoch 0, iteration 83/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 83 | reduced_train_loss: 1.292 | train_step_timing in s: 0.2405 | consumed_samples: 336 | val_loss: 1.619
    Training epoch 0, iteration 84/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 84 | reduced_train_loss: 1.286 | train_step_timing in s: 0.2574 | consumed_samples: 340 | val_loss: 1.619
    Training epoch 0, iteration 85/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 85 | reduced_train_loss: 1.224 | train_step_timing in s: 0.2498 | consumed_samples: 344 | val_loss: 1.619
    Training epoch 0, iteration 86/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 86 | reduced_train_loss: 0.9719 | train_step_timing in s: 0.2515 | consumed_samples: 348 | val_loss: 1.619
    Training epoch 0, iteration 87/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 87 | reduced_train_loss: 1.301 | train_step_timing in s: 0.2456 | consumed_samples: 352 | val_loss: 1.619
    Training epoch 0, iteration 88/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 88 | reduced_train_loss: 1.172 | train_step_timing in s: 0.2315 | consumed_samples: 356 | val_loss: 1.619
    Training epoch 0, iteration 89/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 89 | reduced_train_loss: 1.286 | train_step_timing in s: 0.2256 | consumed_samples: 360 | val_loss: 1.619
    Training epoch 0, iteration 90/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 90 | reduced_train_loss: 1.235 | train_step_timing in s: 0.2389 | consumed_samples: 364 | val_loss: 1.619
    Training epoch 0, iteration 91/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 91 | reduced_train_loss: 1.294 | train_step_timing in s: 0.2343 | consumed_samples: 368 | val_loss: 1.619
    Training epoch 0, iteration 92/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 92 | reduced_train_loss: 1.292 | train_step_timing in s: 0.2349 | consumed_samples: 372 | val_loss: 1.619
    Training epoch 0, iteration 93/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 93 | reduced_train_loss: 1.305 | train_step_timing in s: 0.2382 | consumed_samples: 376 | val_loss: 1.619
    Training epoch 0, iteration 94/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 94 | reduced_train_loss: 1.3 | train_step_timing in s: 0.2584 | consumed_samples: 380 | val_loss: 1.619
    Training epoch 0, iteration 95/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 95 | reduced_train_loss: 1.289 | train_step_timing in s: 0.2356 | consumed_samples: 384 | val_loss: 1.619
    Training epoch 0, iteration 96/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 96 | reduced_train_loss: 1.325 | train_step_timing in s: 0.2629 | consumed_samples: 388 | val_loss: 1.619
    Training epoch 0, iteration 97/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 97 | reduced_train_loss: 1.265 | train_step_timing in s: 0.2319 | consumed_samples: 392 | val_loss: 1.619
    Training epoch 0, iteration 98/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 98 | reduced_train_loss: 1.211 | train_step_timing in s: 0.24 | consumed_samples: 396 | val_loss: 1.619
    Training epoch 0, iteration 99/99 | lr: 3e-05 | global_batch_size: 4 | global_step: 99 | reduced_train_loss: 1.294 | train_step_timing in s: 0.261 | consumed_samples: 400 | val_loss: 1.619
    Epoch 0, global step 99: 'val_loss' reached 1.61913 (best 1.61913), saving model to '/workspace/bionemo2/data/results/evo2/checkpoints/evo2--val_loss=1.6191-epoch=0-consumed_samples=400.0.ckpt' as top 5
    [NeMo I 2025-06-18 16:53:19 nemo_logging:393] Global Checkpoint Save : Rank: 0 : Iteration: 99 : Start time: 1750290797.320s : Save duration: 1.718s
    [NeMo I 2025-06-18 16:53:19 nemo_logging:393] Scheduled async checkpoint save for /workspace/bionemo2/data/results/evo2/checkpoints/evo2--val_loss=1.6191-epoch=0-consumed_samples=400.0.ckpt
    [NeMo I 2025-06-18 16:53:20 nemo_logging:393] Global Checkpoint Save : Rank: 0 : Iteration: 99 : Start time: 1750290799.546s : Save duration: 1.136s
    [NeMo I 2025-06-18 16:53:21 nemo_logging:393] Scheduled async checkpoint save for /workspace/bionemo2/data/results/evo2/checkpoints/evo2--val_loss=1.6191-epoch=0-consumed_samples=400.0-last.ckpt
    [NeMo I 2025-06-18 16:53:21 nemo_logging:393] Async finalization time took 0.001 s
    Validation: iteration 1/20
    Validation: iteration 2/20
    Validation: iteration 3/20
    Validation: iteration 4/20
    Validation: iteration 5/20
    Validation: iteration 6/20
    Validation: iteration 7/20
    Validation: iteration 8/20
    Validation: iteration 9/20
    Validation: iteration 10/20
    Validation: iteration 11/20
    Validation: iteration 12/20
    Validation: iteration 13/20
    Validation: iteration 14/20
    Validation: iteration 15/20
    Validation: iteration 16/20
    Validation: iteration 17/20
    Validation: iteration 18/20
    Validation: iteration 19/20
    Validation: iteration 20/20
    [NeMo I 2025-06-18 16:53:31 nemo_logging:393] Async finalization time took 0.059 s
    `Trainer.fit` stopped: `max_steps=100` reached.
    [NeMo I 2025-06-18 16:53:31 nemo_logging:393] Pending async checkpoint saves. Finalizing them synchronously now
    [NeMo I 2025-06-18 16:53:41 nemo_logging:393] Successfully saved checkpoint from iteration      99 to /workspace/bionemo2/data/results/evo2/checkpoints/evo2--val_loss=1.6191-epoch=0-consumed_samples=400.0.ckpt
    [NeMo I 2025-06-18 16:53:41 nemo_logging:393] Async checkpoint save for step 100 (/workspace/bionemo2/data/results/evo2/checkpoints/evo2--val_loss=1.6191-epoch=0-consumed_samples=400.0.ckpt) finalized successfully.
    [NeMo I 2025-06-18 16:53:42 nemo_logging:393] Successfully saved checkpoint from iteration      99 to /workspace/bionemo2/data/results/evo2/checkpoints/evo2--val_loss=1.6191-epoch=0-consumed_samples=400.0-last.ckpt
    [NeMo I 2025-06-18 16:53:42 nemo_logging:393] Async checkpoint save for step 100 (/workspace/bionemo2/data/results/evo2/checkpoints/evo2--val_loss=1.6191-epoch=0-consumed_samples=400.0-last.ckpt) finalized successfully.
    [NeMo I 2025-06-18 16:53:42 nemo_logging:393] Async finalization time took 10.692 s



```python
!echo $MASTER_PORT
```

    29530



```python

```
