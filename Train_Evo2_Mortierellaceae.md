## Evo2 Finetuning with 96 Genomes from the Mortierellaceae


```python
import os

concat_path = '/home/johnsh29/Evo2_Mortierellaceae/AllGenomes.fna'
```


```python
full_fasta_path = os.path.abspath(concat_path)
output_dir = os.path.abspath("preprocessed_data")
output_yaml = f"""
- datapaths: ["{full_fasta_path}"]
  output_dir: "{output_dir}"
  output_prefix: Mortierellaceae_uint8_distinct
  train_split: 0.9
  valid_split: 0.05
  test_split: 0.05
  overwrite: True
  embed_reverse_complement: true
  random_reverse_complement: 0.0
  random_lineage_dropout: 0.0
  include_sequence_id: false
  transcribe: "back_transcribe"
  force_uppercase: True
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

    [NeMo I 2025-06-18 21:41:53 nemo_logging:393] Using byte-level tokenization
    [NeMo I 2025-06-18 21:41:53 nemo_logging:393] Created temporary binary datasets: /home/johnsh29/Evo2_Mortierellaceae/preprocessed_data/Mortierellaceae_uint8_distinct_byte-level_train.bin.tmp /home/johnsh29/Evo2_Mortierellaceae/preprocessed_data/Mortierellaceae_uint8_distinct_byte-level_val.bin.tmp /home/johnsh29/Evo2_Mortierellaceae/preprocessed_data/Mortierellaceae_uint8_distinct_byte-level_test.bin.tmp
    [NeMo I 2025-06-18 21:55:43 nemo_logging:393] Average preprocessing time per sequence: 0.0002956886035248247
    [NeMo I 2025-06-18 21:55:43 nemo_logging:393] Average indexing time per sequence: 0.0008919303745573302
    [NeMo I 2025-06-18 21:55:43 nemo_logging:393] Number of sequences processed: 538586
    [NeMo I 2025-06-18 21:55:46 nemo_logging:393] Finished preprocessing Mortierellaceae_uint8_distinct ([PosixPath('/home/johnsh29/Evo2_Mortierellaceae/AllGenomes.fna')]) in 832.294 seconds with 1 workers.



```python
!ls -lh preprocessed_data/
```

    total 3.8G
    -rw-r--r-- 1 johnsh29 domain users 405M Jun 18 21:55 Mortierellaceae_uint8_distinct_byte-level_test.bin
    -rw-r--r-- 1 johnsh29 domain users 528K Jun 18 21:55 Mortierellaceae_uint8_distinct_byte-level_test.idx
    -rw-r--r-- 1 johnsh29 domain users 6.6G Jun 18 21:55 Mortierellaceae_uint8_distinct_byte-level_train.bin
    -rw-r--r-- 1 johnsh29 domain users 9.3M Jun 18 21:55 Mortierellaceae_uint8_distinct_byte-level_train.idx
    -rw-r--r-- 1 johnsh29 domain users 397M Jun 18 21:55 Mortierellaceae_uint8_distinct_byte-level_val.bin
    -rw-r--r-- 1 johnsh29 domain users 520K Jun 18 21:55 Mortierellaceae_uint8_distinct_byte-level_val.idx



```python
!evo2_convert_to_nemo2 \
  --model-path hf://arcinstitute/savanna_evo2_1b_base \
  --model-size 1b --output-dir nemo2_evo2_1b_8k
```

    Could not find the bitsandbytes CUDA binary at PosixPath('/usr/local/lib/python3.12/dist-packages/bitsandbytes/libbitsandbytes_cuda129.so')
    The installed version of bitsandbytes was compiled without GPU support. 8-bit optimizers, 8-bit multiplication, and GPU quantization are unavailable.
    [NeMo I 2025-06-18 21:56:00 nemo_logging:393] Using byte-level tokenization
    GPU available: True (cuda), used: False
    TPU available: False, using: 0 TPU cores
    HPU available: False, using: 0 HPUs
    [NeMo W 2025-06-18 21:56:00 nemo_logging:405] /usr/local/lib/python3.12/dist-packages/lightning/pytorch/trainer/setup.py:177: GPU available but not used. You can set it by doing `Trainer(accelerator='gpu')`.
        
    [NeMo I 2025-06-18 21:56:00 nemo_logging:393] Fixing mis-match between ddp-config & mcore-optimizer config
    [NeMo I 2025-06-18 21:56:00 nemo_logging:393] Rank 0 has data parallel group : [0]
    [NeMo I 2025-06-18 21:56:00 nemo_logging:393] Rank 0 has combined group of data parallel and context parallel : [0]
    [NeMo I 2025-06-18 21:56:00 nemo_logging:393] All data parallel group ranks with context parallel combined: [[0]]
    [NeMo I 2025-06-18 21:56:00 nemo_logging:393] Ranks 0 has data parallel rank: 0
    [NeMo I 2025-06-18 21:56:00 nemo_logging:393] Rank 0 has context parallel group: [0]
    [NeMo I 2025-06-18 21:56:00 nemo_logging:393] All context parallel group ranks: [[0]]
    [NeMo I 2025-06-18 21:56:00 nemo_logging:393] Ranks 0 has context parallel rank: 0
    [NeMo I 2025-06-18 21:56:00 nemo_logging:393] Rank 0 has model parallel group: [0]
    [NeMo I 2025-06-18 21:56:00 nemo_logging:393] All model parallel group ranks: [[0]]
    [NeMo I 2025-06-18 21:56:00 nemo_logging:393] Rank 0 has tensor model parallel group: [0]
    [NeMo I 2025-06-18 21:56:00 nemo_logging:393] All tensor model parallel group ranks: [[0]]
    [NeMo I 2025-06-18 21:56:00 nemo_logging:393] Rank 0 has tensor model parallel rank: 0
    [NeMo I 2025-06-18 21:56:00 nemo_logging:393] Rank 0 has pipeline model parallel group: [0]
    [NeMo I 2025-06-18 21:56:00 nemo_logging:393] Rank 0 has embedding group: [0]
    [NeMo I 2025-06-18 21:56:00 nemo_logging:393] All pipeline model parallel group ranks: [[0]]
    [NeMo I 2025-06-18 21:56:00 nemo_logging:393] Rank 0 has pipeline model parallel rank 0
    [NeMo I 2025-06-18 21:56:00 nemo_logging:393] All embedding group ranks: [[0]]
    [NeMo I 2025-06-18 21:56:00 nemo_logging:393] Rank 0 has embedding rank: 0
    ----------------------------------------------------------------------------------------------------
    distributed_backend=gloo
    All distributed processes registered. Starting with 1 processes
    ----------------------------------------------------------------------------------------------------
    
    [NeMo W 2025-06-18 21:56:00 nemo_logging:405] /usr/local/lib/python3.12/dist-packages/lightning/pytorch/trainer/trainer.py:1090: `trainer.init_module` cannot fully support proper instantiation of your model with the `MegatronStrategy` strategy. Please instantiate your model inside the`LightningModule.configure_model` hook instead
        
    [NeMo I 2025-06-18 21:56:00 nemo_logging:393] Padded vocab_size: 512, original vocab_size: 512, dummy tokens: 0.
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
    [NeMo W 2025-06-18 21:56:02 nemo_logging:405] Could not copy Trainer's 'max_steps' to LR scheduler's 'max_steps'. If you are not using an LR scheduler, this warning can safely be ignored.
    [NeMo I 2025-06-18 21:56:02 nemo_logging:393] Using FullyParallelSaveStrategyWrapper(torch_dist, 1) dist-ckpt save strategy.
    [NeMo I 2025-06-18 21:56:22 nemo_logging:393] Global Checkpoint Save : Rank: 0 : Iteration: 0 : Start time: 1750308962.206s : Save duration: 20.774s
    [NeMo I 2025-06-18 21:56:23 nemo_logging:393] Converted Hyena model to Nemo, model saved to nemo2_evo2_1b_8k



```python
import os
from pathlib import Path
output_pfx = str(Path(os.path.abspath("preprocessed_data"))/"Mortierellaceae_uint8_distinct_byte-level")
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
## Estimate number of steps

TotalBasePairs = 3875446222
TotalTrainingBasePairs = TotalBasePairs * 0.9  # 90% of total base pairs
GlobalBatchSize = 16 # microbatch of 2 times 8 GPUs
ContextLength = 8192

BasePairsPerBatch = ContextLength * GlobalBatchSize

StepsPerEpoch = TotalTrainingBasePairs / BasePairsPerBatch

print("Number of Steps for One Full Pass of the Data: ", int(StepsPerEpoch))
```

    Number of Steps for One Full Pass of the Data:  26610



```python
import time

# Start timer
start_time = time.time()



# For evo2 training and fine-tuning follow the same set of steps, so we use the same train_evo2 command.
#  the big difference is the --ckpt-dir argument which points to a pre-existing checkpoint from some other training run.
!train_evo2 \
    -d training_data_config.yaml \
    --dataset-dir {preprocessed_data} \
    --result-dir pretraining_mortierellaceae \
    --model-size 1b \
    --devices 1 \
    --num-nodes 1 \
    --seq-length 8192 \
    --micro-batch-size 1 \
    --lr 0.0001 \
    --warmup-steps 5 \
    --max-steps 10 \
    --tensor-parallel-size 1 \
    --ckpt-dir nemo2_evo2_1b_8k \
    --clip-grad 1 \
    --wd 0.01 \
    --activation-checkpoint-recompute-num-layers 1 \
    --val-check-interval 5 \
    --ckpt-async-save \
    --wandb-offline



# End timer and calculate duration
end_time = time.time()
training_duration = end_time - start_time

# Print timing results
print("\nTraining completed!")
print(f"Total training time: {training_duration:.2f} seconds")
print(f"({training_duration/60:.2f} minutes)")
print(f"({training_duration/3600:.2f} hours)")
```

    Could not find the bitsandbytes CUDA binary at PosixPath('/usr/local/lib/python3.12/dist-packages/bitsandbytes/libbitsandbytes_cuda129.so')
    The installed version of bitsandbytes was compiled without GPU support. 8-bit optimizers, 8-bit multiplication, and GPU quantization are unavailable.
    [NeMo I 2025-06-18 21:56:36 nemo_logging:393] Using byte-level tokenization
    [NeMo W 2025-06-18 21:56:36 nemo_logging:405] WandB is currently turned off.
    [NeMo W 2025-06-18 21:56:36 nemo_logging:405] User-set tensorboard is currently turned off. Internally one may still be set by NeMo2.
    Trainer already configured with model summary callbacks: [<class 'lightning.pytorch.callbacks.rich_model_summary.RichModelSummary'>]. Skipping setting a default `ModelSummary` callback.
    GPU available: True (cuda), used: True
    TPU available: False, using: 0 TPU cores
    HPU available: False, using: 0 HPUs
    [NeMo I 2025-06-18 21:56:36 nemo_logging:393] Experiments will be logged at pretraining_mortierellaceae/evo2/dev
    [NeMo W 2025-06-18 21:56:36 nemo_logging:405] There were no checkpoints found in checkpoint_dir or no checkpoint folder at checkpoint_dir :pretraining_mortierellaceae/evo2/checkpoints. Training from scratch.
    [NeMo I 2025-06-18 21:56:36 nemo_logging:393] Fixing mis-match between ddp-config & mcore-optimizer config
    [NeMo I 2025-06-18 21:56:36 nemo_logging:393] Rank 0 has data parallel group : [0]
    [NeMo I 2025-06-18 21:56:36 nemo_logging:393] Rank 0 has combined group of data parallel and context parallel : [0]
    [NeMo I 2025-06-18 21:56:36 nemo_logging:393] All data parallel group ranks with context parallel combined: [[0]]
    [NeMo I 2025-06-18 21:56:36 nemo_logging:393] Ranks 0 has data parallel rank: 0
    [NeMo I 2025-06-18 21:56:36 nemo_logging:393] Rank 0 has context parallel group: [0]
    [NeMo I 2025-06-18 21:56:36 nemo_logging:393] All context parallel group ranks: [[0]]
    [NeMo I 2025-06-18 21:56:36 nemo_logging:393] Ranks 0 has context parallel rank: 0
    [NeMo I 2025-06-18 21:56:36 nemo_logging:393] Rank 0 has model parallel group: [0]
    [NeMo I 2025-06-18 21:56:36 nemo_logging:393] All model parallel group ranks: [[0]]
    [NeMo I 2025-06-18 21:56:36 nemo_logging:393] Rank 0 has tensor model parallel group: [0]
    [NeMo I 2025-06-18 21:56:36 nemo_logging:393] All tensor model parallel group ranks: [[0]]
    [NeMo I 2025-06-18 21:56:36 nemo_logging:393] Rank 0 has tensor model parallel rank: 0
    [NeMo I 2025-06-18 21:56:36 nemo_logging:393] Rank 0 has pipeline model parallel group: [0]
    [NeMo I 2025-06-18 21:56:36 nemo_logging:393] Rank 0 has embedding group: [0]
    [NeMo I 2025-06-18 21:56:36 nemo_logging:393] All pipeline model parallel group ranks: [[0]]
    [NeMo I 2025-06-18 21:56:36 nemo_logging:393] Rank 0 has pipeline model parallel rank 0
    [NeMo I 2025-06-18 21:56:36 nemo_logging:393] All embedding group ranks: [[0]]
    [NeMo I 2025-06-18 21:56:36 nemo_logging:393] Rank 0 has embedding rank: 0
    ----------------------------------------------------------------------------------------------------
    distributed_backend=nccl
    All distributed processes registered. Starting with 1 processes
    ----------------------------------------------------------------------------------------------------
    
    [NeMo I 2025-06-18 21:56:37 utils:554] Building Evo2Dataset splits with sizes=[10, 60, 1] and config=GPTDatasetConfig(random_seed=1234, sequence_length=8192, blend=None, blend_per_split=[(['/home/johnsh29/Evo2_Mortierellaceae/preprocessed_data/Mortierellaceae_uint8_distinct_byte-level_train'], [1.0]), (['/home/johnsh29/Evo2_Mortierellaceae/preprocessed_data/Mortierellaceae_uint8_distinct_byte-level_val'], [1.0]), (['/home/johnsh29/Evo2_Mortierellaceae/preprocessed_data/Mortierellaceae_uint8_distinct_byte-level_test'], [1.0])], split=None, split_matrix=None, num_dataset_builder_threads=1, path_to_cache=None, mmap_bin_files=True, mock=False, tokenizer=<nemo.collections.common.tokenizers.bytelevel_tokenizers.ByteLevelTokenizer object at 0x7f1efa5c0830>, mid_level_dataset_surplus=0.005, reset_position_ids=False, reset_attention_mask=False, eod_mask_loss=False, create_attention_mask=False, drop_last_partial_validation_sequence=True, add_extra_token_to_sequence=True, object_storage_cache_path=None)
    [NeMo I 2025-06-18 21:56:37 utils:554] Load the _IndexReader from /home/johnsh29/Evo2_Mortierellaceae/preprocessed_data/Mortierellaceae_uint8_distinct_byte-level_train.idx
    [NeMo I 2025-06-18 21:56:38 utils:554] 	Extract the sequence lengths
    [NeMo I 2025-06-18 21:56:38 utils:554] 	Extract the sequence pointers
    [NeMo I 2025-06-18 21:56:38 utils:554] 	Extract the document indices
    [NeMo I 2025-06-18 21:56:38 utils:554] > total number of sequences: 484998
    [NeMo I 2025-06-18 21:56:38 utils:554] > total number of documents: 484998
    [NeMo I 2025-06-18 21:56:38 utils:554] Build and save the Evo2Dataset train indices
    [NeMo I 2025-06-18 21:56:38 utils:554] > total number of samples: 863395
    [NeMo I 2025-06-18 21:56:38 utils:554] > total number of epochs: 1
    [NeMo I 2025-06-18 21:56:38 utils:554] Load the _IndexReader from /home/johnsh29/Evo2_Mortierellaceae/preprocessed_data/Mortierellaceae_uint8_distinct_byte-level_val.idx
    [NeMo I 2025-06-18 21:56:38 utils:554] 	Extract the sequence lengths
    [NeMo I 2025-06-18 21:56:38 utils:554] 	Extract the sequence pointers
    [NeMo I 2025-06-18 21:56:38 utils:554] 	Extract the document indices
    [NeMo I 2025-06-18 21:56:38 utils:554] > total number of sequences: 26594
    [NeMo I 2025-06-18 21:56:38 utils:554] > total number of documents: 26594
    [NeMo I 2025-06-18 21:56:38 utils:554] Build and save the Evo2Dataset valid indices
    [NeMo I 2025-06-18 21:56:38 utils:554] > total number of samples: 50710
    [NeMo I 2025-06-18 21:56:38 utils:554] > total number of epochs: 1
    [NeMo I 2025-06-18 21:56:38 utils:554] Load the _IndexReader from /home/johnsh29/Evo2_Mortierellaceae/preprocessed_data/Mortierellaceae_uint8_distinct_byte-level_test.idx
    [NeMo I 2025-06-18 21:56:38 utils:554] 	Extract the sequence lengths
    [NeMo I 2025-06-18 21:56:38 utils:554] 	Extract the sequence pointers
    [NeMo I 2025-06-18 21:56:38 utils:554] 	Extract the document indices
    [NeMo I 2025-06-18 21:56:38 utils:554] > total number of sequences: 26994
    [NeMo I 2025-06-18 21:56:38 utils:554] > total number of documents: 26994
    [NeMo I 2025-06-18 21:56:38 utils:554] Build and save the Evo2Dataset test indices
    [NeMo I 2025-06-18 21:56:38 utils:554] > total number of samples: 51815
    [NeMo I 2025-06-18 21:56:38 utils:554] > total number of epochs: 1
    [NeMo I 2025-06-18 21:56:38 nemo_logging:393] Padded vocab_size: 512, original vocab_size: 512, dummy tokens: 0.
    [NeMo W 2025-06-18 21:56:38 random:222] CPU RNG state changed within GPU RNG context
    [NeMo W 2025-06-18 21:56:38 random:222] CPU RNG state changed within GPU RNG context
    [NeMo W 2025-06-18 21:56:38 random:222] CPU RNG state changed within GPU RNG context
    [NeMo W 2025-06-18 21:56:38 random:222] CPU RNG state changed within GPU RNG context
    [NeMo W 2025-06-18 21:56:38 random:222] CPU RNG state changed within GPU RNG context
    [NeMo W 2025-06-18 21:56:38 random:222] CPU RNG state changed within GPU RNG context
    [NeMo W 2025-06-18 21:56:38 random:222] CPU RNG state changed within GPU RNG context
    [NeMo W 2025-06-18 21:56:38 random:222] CPU RNG state changed within GPU RNG context
    [NeMo W 2025-06-18 21:56:38 random:222] CPU RNG state changed within GPU RNG context
    [NeMo W 2025-06-18 21:56:38 random:222] CPU RNG state changed within GPU RNG context
    [NeMo W 2025-06-18 21:56:38 random:222] CPU RNG state changed within GPU RNG context
    [NeMo W 2025-06-18 21:56:38 random:222] CPU RNG state changed within GPU RNG context
    [NeMo W 2025-06-18 21:56:38 random:222] CPU RNG state changed within GPU RNG context
    [NeMo W 2025-06-18 21:56:38 random:222] CPU RNG state changed within GPU RNG context
    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1,2,3]
    [NeMo I 2025-06-18 21:56:38 nemo_logging:393] Copying Trainer's 'max_steps' (10) to LR scheduler's 'max_steps'.
    [NeMo I 2025-06-18 21:56:38 num_microbatches_calculator:228] setting number of microbatches to constant 1
    [NeMo I 2025-06-18 21:56:38 nemo_logging:393]  > number of parameters on (tensor, pipeline) model parallel rank (0 ,0): 1108204800
    [NeMo I 2025-06-18 21:56:38 utils:554] Setting up DistributedDataParallel with config DistributedDataParallelConfig(grad_reduce_in_fp32=False, overlap_grad_reduce=False, overlap_param_gather=False, align_param_gather=False, use_distributed_optimizer=True, num_distributed_optimizer_instances=1, check_for_nan_in_grad=True, check_for_large_grads=False, bucket_size=None, pad_buckets_for_high_nccl_busbw=False, average_in_collective=True, fp8_param_gather=False, use_custom_fsdp=False, data_parallel_sharding_strategy='no_shard', gradient_reduce_div_fusion=True, suggested_communication_unit_size=None, preserve_fp32_weights=True, keep_fp8_transpose_cache_when_using_custom_fsdp=False)
    [NeMo I 2025-06-18 21:56:38 utils:575] Number of buckets for gradient all-reduce / reduce-scatter: 1
        Params for bucket 1 (1108204800 elements, 1108204800 padded size):
        	module.decoder.layers.21.mixer.dense.weight
        	module.decoder.layers.15.mlp.linear_fc2.weight
        	module.decoder.layers.11.mlp.linear_fc1.layer_norm_weight
        	module.decoder.layers.8.mlp.linear_fc2.weight
        	module.decoder.layers.4.mlp.linear_fc1.layer_norm_weight
        	module.decoder.layers.1.mixer.dense_projection.weight
        	module.decoder.layers.20.mixer.dense_projection.weight
        	module.decoder.layers.9.mixer.dense.weight
        	module.decoder.layers.2.mixer.dense.weight
        	module.decoder.layers.23.mixer.mixer.filter.R
        	module.decoder.layers.20.mlp.linear_fc2.weight
        	module.decoder.layers.18.mlp.linear_fc1.weight
        	module.decoder.layers.10.mlp.linear_fc1.weight
        	module.decoder.layers.3.mlp.linear_fc1.weight
        	module.decoder.layers.1.mixer.mixer.filter.h
        	module.embedding.word_embeddings.weight
        	module.decoder.layers.22.mixer.dense_projection.layer_norm_weight
        	module.decoder.layers.21.mixer.dense.bias
        	module.decoder.layers.14.mixer.dense_projection.layer_norm_weight
        	module.decoder.layers.12.mixer.dense.bias
        	module.decoder.layers.11.mlp.linear_fc2.weight
        	module.decoder.layers.7.mixer.dense_projection.layer_norm_weight
        	module.decoder.layers.4.mlp.linear_fc2.weight
        	module.decoder.layers.1.mixer.dense.weight
        	module.decoder.layers.22.mlp.linear_fc2.weight
        	module.decoder.layers.18.mlp.linear_fc1.layer_norm_weight
        	module.decoder.layers.12.mlp.linear_fc1.weight
        	module.decoder.layers.5.mlp.linear_fc1.weight
        	module.decoder.layers.1.mlp.linear_fc1.weight
        	module.decoder.layers.0.mixer.dense.bias
        	module.decoder.layers.0.mixer.dense_projection.weight
        	module.decoder.layers.23.mixer.dense.weight
        	module.decoder.layers.16.mlp.linear_fc1.layer_norm_weight
        	module.decoder.layers.16.mixer.dense_projection.layer_norm_weight
        	module.decoder.layers.15.mixer.mixer.conv_bias
        	module.decoder.layers.11.mixer.dense.weight
        	module.decoder.layers.9.mixer.dense_projection.layer_norm_weight
        	module.decoder.layers.8.mixer.mixer.conv_bias
        	module.decoder.layers.4.mixer.dense.weight
        	module.decoder.layers.2.mixer.mixer.conv_bias
        	module.decoder.layers.24.mlp.linear_fc1.weight
        	module.decoder.layers.17.mlp.linear_fc1.weight
        	module.decoder.layers.9.mlp.linear_fc1.layer_norm_weight
        	module.decoder.layers.2.mlp.linear_fc1.layer_norm_weight
        	module.decoder.layers.21.mixer.dense_projection.layer_norm_weight
        	module.decoder.layers.18.mlp.linear_fc2.weight
        	module.decoder.layers.13.mixer.mixer.filter.R
        	module.decoder.layers.11.mixer.dense.bias
        	module.decoder.layers.10.mlp.linear_fc2.weight
        	module.decoder.layers.6.mixer.mixer.filter.R
        	module.decoder.layers.3.mlp.linear_fc2.weight
        	module.decoder.layers.0.mlp.linear_fc1.weight
        	module.decoder.layers.19.mlp.linear_fc1.weight
        	module.decoder.layers.14.mixer.mixer.short_conv.short_conv_weight
        	module.decoder.layers.12.mixer.dense_projection.layer_norm_weight
        	module.decoder.layers.7.mixer.mixer.short_conv.short_conv_weight
        	module.decoder.layers.5.mixer.dense_projection.layer_norm_weight
        	module.decoder.layers.23.mixer.dense_projection.layer_norm_weight
        	module.decoder.layers.22.mixer.mixer.conv_bias
        	module.decoder.layers.18.mixer.dense.weight
        	module.decoder.layers.15.mixer.hyena_proj_conv.short_conv_weight
        	module.decoder.layers.12.mlp.linear_fc2.weight
        	module.decoder.layers.12.mixer.mixer.conv_bias
        	module.decoder.layers.8.mixer.hyena_proj_conv.short_conv_weight
        	module.decoder.layers.5.mlp.linear_fc2.weight
        	module.decoder.layers.4.mixer.dense.bias
        	module.decoder.layers.0.mixer.mixer.short_conv.short_conv_weight
        	module.decoder.layers.23.mlp.linear_fc1.layer_norm_weight
        	module.decoder.layers.16.mixer.mixer.filter.gamma
        	module.decoder.layers.13.mixer.dense.weight
        	module.decoder.layers.10.self_attention.linear_proj.bias
        	module.decoder.layers.9.mixer.mixer.filter.gamma
        	module.decoder.layers.6.mixer.dense.weight
        	module.decoder.layers.2.mixer.dense_projection.layer_norm_weight
        	module.decoder.final_norm.weight
        	module.decoder.layers.24.mlp.linear_fc2.weight
        	module.decoder.layers.20.mixer.mixer.filter.R
        	module.decoder.layers.17.mlp.linear_fc2.weight
        	module.decoder.layers.15.mixer.mixer.filter.h
        	module.decoder.layers.10.self_attention.linear_proj.weight
        	module.decoder.layers.8.mixer.mixer.filter.h
        	module.decoder.layers.5.mixer.dense.bias
        	module.decoder.layers.3.self_attention.linear_proj.weight
        	module.decoder.layers.21.mixer.mixer.short_conv.short_conv_weight
        	module.decoder.layers.19.mixer.dense_projection.layer_norm_weight
        	module.decoder.layers.15.mixer.dense.bias
        	module.decoder.layers.15.mixer.dense.weight
        	module.decoder.layers.11.mixer.dense_projection.layer_norm_weight
        	module.decoder.layers.8.mixer.dense.bias
        	module.decoder.layers.8.mixer.dense.weight
        	module.decoder.layers.4.mixer.dense_projection.layer_norm_weight
        	module.decoder.layers.0.mlp.linear_fc2.weight
        	module.decoder.layers.22.mixer.hyena_proj_conv.short_conv_weight
        	module.decoder.layers.19.mlp.linear_fc2.weight
        	module.decoder.layers.16.mixer.mixer.filter.p
        	module.decoder.layers.14.mixer.hyena_proj_conv.short_conv_weight
        	module.decoder.layers.9.mixer.mixer.filter.p
        	module.decoder.layers.7.mixer.hyena_proj_conv.short_conv_weight
        	module.decoder.layers.5.mixer.mixer.conv_bias
        	module.decoder.layers.2.mixer.mixer.filter.p
        	module.decoder.layers.2.mixer.dense_projection.weight
        	module.decoder.layers.23.mixer.mixer.filter.gamma
        	module.decoder.layers.20.mixer.dense.weight
        	module.decoder.layers.15.mixer.dense_projection.weight
        	module.decoder.layers.13.mixer.dense_projection.layer_norm_weight
        	module.decoder.layers.8.mixer.dense_projection.weight
        	module.decoder.layers.6.mixer.dense_projection.layer_norm_weight
        	module.decoder.layers.0.mixer.dense.weight
        	module.decoder.layers.17.self_attention.linear_proj.weight
        	module.decoder.layers.24.self_attention.linear_proj.weight
        	module.decoder.layers.22.mixer.mixer.filter.h
        	module.decoder.layers.19.mixer.dense.bias
        	module.decoder.layers.16.mixer.hyena_proj_conv.short_conv_weight
        	module.decoder.layers.13.mlp.linear_fc1.layer_norm_weight
        	module.decoder.layers.9.mixer.hyena_proj_conv.short_conv_weight
        	module.decoder.layers.6.mlp.linear_fc1.layer_norm_weight
        	module.decoder.layers.3.self_attention.linear_proj.bias
        	module.decoder.layers.22.mixer.dense.bias
        	module.decoder.layers.22.mixer.dense.weight
        	module.decoder.layers.18.mixer.dense_projection.layer_norm_weight
        	module.decoder.layers.14.mixer.dense.bias
        	module.decoder.layers.9.mlp.linear_fc1.weight
        	module.decoder.layers.6.mixer.mixer.conv_bias
        	module.decoder.layers.2.mlp.linear_fc1.weight
        	module.decoder.layers.0.mixer.dense_projection.layer_norm_weight
        	module.decoder.layers.23.mixer.mixer.filter.p
        	module.decoder.layers.21.mixer.hyena_proj_conv.short_conv_weight
        	module.decoder.layers.19.mixer.mixer.conv_bias
        	module.decoder.layers.15.mlp.linear_fc1.layer_norm_weight
        	module.decoder.layers.11.mixer.mixer.short_conv.short_conv_weight
        	module.decoder.layers.8.mlp.linear_fc1.layer_norm_weight
        	module.decoder.layers.4.mixer.mixer.short_conv.short_conv_weight
        	module.decoder.layers.1.mixer.dense_projection.layer_norm_weight
        	module.decoder.layers.22.mixer.dense_projection.weight
        	module.decoder.layers.20.mixer.dense_projection.layer_norm_weight
        	module.decoder.layers.18.mixer.dense.bias
        	module.decoder.layers.16.mixer.dense.weight
        	module.decoder.layers.14.mixer.dense_projection.weight
        	module.decoder.layers.12.mixer.hyena_proj_conv.short_conv_weight
        	module.decoder.layers.9.mixer.dense.bias
        	module.decoder.layers.7.mixer.dense_projection.weight
        	module.decoder.layers.5.mixer.hyena_proj_conv.short_conv_weight
        	module.decoder.layers.2.mixer.dense.bias
        	module.decoder.layers.23.mixer.hyena_proj_conv.short_conv_weight
        	module.decoder.layers.20.mlp.linear_fc1.layer_norm_weight
        	module.decoder.layers.13.mixer.mixer.filter.gamma
        	module.decoder.layers.10.self_attention.linear_qkv.weight
        	module.decoder.layers.6.mixer.mixer.filter.gamma
        	module.decoder.layers.3.self_attention.linear_qkv.weight
        	module.decoder.layers.1.mixer.dense.bias
        	module.decoder.layers.23.mlp.linear_fc1.weight
        	module.decoder.layers.20.mixer.mixer.conv_bias
        	module.decoder.layers.16.mlp.linear_fc1.weight
        	module.decoder.layers.16.mixer.dense_projection.weight
        	module.decoder.layers.12.mixer.mixer.filter.h
        	module.decoder.layers.9.mixer.dense_projection.weight
        	module.decoder.layers.2.mixer.hyena_proj_conv.short_conv_weight
        	module.decoder.layers.1.mixer.mixer.conv_bias
        	module.decoder.layers.22.mlp.linear_fc1.layer_norm_weight
        	module.decoder.layers.18.mixer.mixer.short_conv.short_conv_weight
        	module.decoder.layers.14.mlp.linear_fc1.weight
        	module.decoder.layers.12.mixer.dense.weight
        	module.decoder.layers.9.mlp.linear_fc2.weight
        	module.decoder.layers.7.mlp.linear_fc1.weight
        	module.decoder.layers.5.mixer.dense.weight
        	module.decoder.layers.2.mlp.linear_fc2.weight
        	module.decoder.layers.23.mixer.dense.bias
        	module.decoder.layers.21.mixer.dense_projection.weight
        	module.decoder.layers.19.mixer.hyena_proj_conv.short_conv_weight
        	module.decoder.layers.16.mixer.mixer.conv_bias
        	module.decoder.layers.13.mixer.mixer.filter.p
        	module.decoder.layers.11.mixer.hyena_proj_conv.short_conv_weight
        	module.decoder.layers.9.mixer.mixer.conv_bias
        	module.decoder.layers.6.mixer.mixer.filter.p
        	module.decoder.layers.4.mixer.hyena_proj_conv.short_conv_weight
        	module.decoder.layers.17.self_attention.linear_qkv.weight
        	module.decoder.layers.24.self_attention.linear_qkv.weight
        	module.decoder.layers.20.mixer.mixer.filter.gamma
        	module.decoder.layers.14.mlp.linear_fc1.layer_norm_weight
        	module.decoder.layers.12.mixer.dense_projection.weight
        	module.decoder.layers.7.mlp.linear_fc1.layer_norm_weight
        	module.decoder.layers.5.mixer.dense_projection.weight
        	module.decoder.layers.2.mixer.mixer.filter.R
        	module.decoder.layers.1.mlp.linear_fc2.weight
        	module.decoder.layers.23.mixer.dense_projection.weight
        	module.decoder.layers.13.mixer.hyena_proj_conv.short_conv_weight
        	module.decoder.layers.10.mlp.linear_fc1.layer_norm_weight
        	module.decoder.layers.6.mixer.hyena_proj_conv.short_conv_weight
        	module.decoder.layers.3.mlp.linear_fc1.layer_norm_weight
        	module.decoder.layers.23.mlp.linear_fc2.weight
        	module.decoder.layers.21.mlp.linear_fc1.weight
        	module.decoder.layers.19.mixer.dense.weight
        	module.decoder.layers.16.mlp.linear_fc2.weight
        	module.decoder.layers.16.mixer.dense.bias
        	module.decoder.layers.13.mlp.linear_fc1.weight
        	module.decoder.layers.6.mlp.linear_fc1.weight
        	module.decoder.layers.5.mixer.mixer.filter.h
        	module.decoder.layers.2.mixer.mixer.filter.gamma
        	module.decoder.layers.1.mlp.linear_fc1.layer_norm_weight
        	module.decoder.layers.23.mixer.mixer.conv_bias
        	module.decoder.layers.20.mixer.mixer.filter.p
        	module.decoder.layers.18.mixer.hyena_proj_conv.short_conv_weight
        	module.decoder.layers.14.mlp.linear_fc2.weight
        	module.decoder.layers.12.mlp.linear_fc1.layer_norm_weight
        	module.decoder.layers.10.self_attention.linear_qkv.layer_norm_weight
        	module.decoder.layers.7.mlp.linear_fc2.weight
        	module.decoder.layers.5.mlp.linear_fc1.layer_norm_weight
        	module.decoder.layers.3.self_attention.linear_qkv.layer_norm_weight
        	module.decoder.layers.0.mixer.hyena_proj_conv.short_conv_weight
        	module.decoder.layers.21.mlp.linear_fc1.layer_norm_weight
        	module.decoder.layers.19.mixer.dense_projection.weight
        	module.decoder.layers.15.mlp.linear_fc1.weight
        	module.decoder.layers.13.mixer.dense.bias
        	module.decoder.layers.11.mixer.dense_projection.weight
        	module.decoder.layers.8.mlp.linear_fc1.weight
        	module.decoder.layers.6.mixer.dense.bias
        	module.decoder.layers.4.mixer.dense_projection.weight
        	module.decoder.layers.1.mixer.hyena_proj_conv.short_conv_weight
        	module.decoder.layers.24.mlp.linear_fc1.layer_norm_weight
        	module.decoder.layers.24.self_attention.linear_proj.bias
        	module.decoder.layers.20.mixer.hyena_proj_conv.short_conv_weight
        	module.decoder.layers.17.mlp.linear_fc1.layer_norm_weight
        	module.decoder.layers.17.self_attention.linear_proj.bias
        	module.decoder.layers.14.mixer.dense.weight
        	module.decoder.layers.7.mixer.dense.weight
        	module.decoder.layers.20.mlp.linear_fc1.weight
        	module.decoder.layers.19.mixer.mixer.filter.h
        	module.decoder.layers.13.mixer.dense_projection.weight
        	module.decoder.layers.6.mixer.dense_projection.weight
        	module.decoder.layers.17.self_attention.linear_qkv.layer_norm_weight
        	module.decoder.layers.24.self_attention.linear_qkv.layer_norm_weight
        	module.decoder.layers.21.mlp.linear_fc2.weight
        	module.decoder.layers.19.mlp.linear_fc1.layer_norm_weight
        	module.decoder.layers.16.mixer.mixer.filter.R
        	module.decoder.layers.13.mlp.linear_fc2.weight
        	module.decoder.layers.11.mlp.linear_fc1.weight
        	module.decoder.layers.9.mixer.mixer.filter.R
        	module.decoder.layers.6.mlp.linear_fc2.weight
        	module.decoder.layers.4.mlp.linear_fc1.weight
        	module.decoder.layers.22.mlp.linear_fc1.weight
        	module.decoder.layers.20.mixer.dense.bias
        	module.decoder.layers.18.mixer.dense_projection.weight
        	module.decoder.layers.15.mixer.dense_projection.layer_norm_weight
        	module.decoder.layers.13.mixer.mixer.conv_bias
        	module.decoder.layers.8.mixer.dense_projection.layer_norm_weight
        	module.decoder.layers.7.mixer.dense.bias
        	module.decoder.layers.0.mlp.linear_fc1.layer_norm_weight
    [NeMo I 2025-06-18 21:56:38 utils:554] Setting up optimizer with config OptimizerConfig(optimizer='adam', lr=0.0001, min_lr=None, decoupled_lr=None, decoupled_min_lr=None, weight_decay=0.01, fp16=False, bf16=True, params_dtype=torch.bfloat16, use_precision_aware_optimizer=False, store_param_remainders=True, main_grads_dtype=torch.float32, main_params_dtype=torch.float32, exp_avg_dtype=torch.float32, exp_avg_sq_dtype=torch.float32, loss_scale=None, initial_loss_scale=4294967296, min_loss_scale=1.0, loss_scale_window=1000, hysteresis=2, adam_beta1=0.9, adam_beta2=0.95, adam_eps=1e-08, sgd_momentum=0.9, use_distributed_optimizer=True, overlap_param_gather_with_optimizer_step=False, optimizer_cpu_offload=False, optimizer_offload_fraction=0.0, use_torch_optimizer_for_cpu_offload=False, overlap_cpu_optimizer_d2h_h2d=False, pin_cpu_grads=True, pin_cpu_params=True, clip_grad=1.0, log_num_zeros_in_grad=False, barrier_with_L1_time=False, timers=None, config_logger_dir='')
    [NeMo I 2025-06-18 21:56:38 nemo_logging:393] Doing selective restore from RestoreConfig(path='nemo2_evo2_1b_8k', adapter_path=None, load_model_state=True, load_optim_state=False, load_artifacts=True)
    [NeMo I 2025-06-18 21:56:38 nemo_logging:393] Using <megatron.core.dist_checkpointing.strategies.fully_parallel.FullyParallelLoadStrategyWrapper object at 0x7f1ef0331f40> dist-ckpt load strategy.
    [NeMo I 2025-06-18 21:57:00 nemo_logging:393] Global Checkpoint Load : Rank : 0 : Start time : 1750308998.535s : Time spent in load_checkpoint: 22.456s
    [NeMo I 2025-06-18 21:57:00 nemo_logging:393] Restoring model weights from RestoreConfig(path='nemo2_evo2_1b_8k', adapter_path=None, load_model_state=True, load_optim_state=False, load_artifacts=True)
    [NeMo I 2025-06-18 21:57:00 nemo_logging:393] Finished restoring from RestoreConfig(path='nemo2_evo2_1b_8k', adapter_path=None, load_model_state=True, load_optim_state=False, load_artifacts=True), cleaning up.
    ┏━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┓
    ┃[1;35m [0m[1;35m [0m[1;35m [0m┃[1;35m [0m[1;35mName                               [0m[1;35m [0m┃[1;35m [0m[1;35mType                  [0m[1;35m [0m┃[1;35m [0m[1;35mParams[0m[1;35m [0m┃[1;35m [0m[1;35mMode [0m[1;35m [0m┃
    ┡━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━┩
    │[2m [0m[2m0[0m[2m [0m│ module                              │ DDP                    │  1.1 B │ train │
    │[2m [0m[2m1[0m[2m [0m│ module.module                       │ Float16Module          │  1.1 B │ train │
    │[2m [0m[2m2[0m[2m [0m│ module.module.module                │ HyenaModel             │  1.1 B │ train │
    │[2m [0m[2m3[0m[2m [0m│ module.module.module.embedding      │ LanguageModelEmbedding │  983 K │ train │
    │[2m [0m[2m4[0m[2m [0m│ module.module.module.rotary_pos_emb │ RotaryEmbedding        │      0 │ train │
    │[2m [0m[2m5[0m[2m [0m│ module.module.module.decoder        │ HyenaStack             │  1.1 B │ train │
    │[2m [0m[2m6[0m[2m [0m│ module.module.module.output_layer   │ ColumnParallelLinear   │      0 │ train │
    └───┴─────────────────────────────────────┴────────────────────────┴────────┴───────┘
    [1mTrainable params[0m: 1.1 B                                                                                                                                        
    [1mNon-trainable params[0m: 0                                                                                                                                        
    [1mTotal params[0m: 1.1 B                                                                                                                                            
    [1mTotal estimated model params size (MB)[0m: 4.4 K                                                                                                                  
    [1mModules in train mode[0m: 356                                                                                                                                     
    [1mModules in eval mode[0m: 0                                                                                                                                        
    In file included from [01m[K/usr/include/python3.12/Python.h:12[m[K,
                     from [01m[K/tmp/tmpccltraq9/main.c:4[m[K:
    [01m[K/usr/include/python3.12/pyconfig.h:3:12:[m[K [01;31m[Kfatal error: [m[Kx86_64-linux-gnu/python3.12/pyconfig.h: No such file or directory
        3 | #  include [01;31m[K<x86_64-linux-gnu/python3.12/pyconfig.h>[m[K
          |            [01;31m[K^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~[m[K
    compilation terminated.
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277] WON'T CONVERT filter /usr/local/lib/python3.12/dist-packages/nemo/collections/llm/gpt/model/megatron/hyena/hyena_utils.py line 616 
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277] due to: 
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277] Traceback (most recent call last):
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_dynamo/convert_frame.py", line 1210, in __call__
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     result = self._inner_convert(
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]              ^^^^^^^^^^^^^^^^^^^^
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_dynamo/convert_frame.py", line 597, in __call__
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     return _compile(
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]            ^^^^^^^^^
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_dynamo/convert_frame.py", line 1056, in _compile
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     guarded_code = compile_inner(code, one_graph, hooks, transform)
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_utils_internal.py", line 97, in wrapper_function
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     return function(*args, **kwargs)
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]            ^^^^^^^^^^^^^^^^^^^^^^^^^
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_dynamo/convert_frame.py", line 758, in compile_inner
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     return _compile_inner(code, one_graph, hooks, transform)
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_dynamo/convert_frame.py", line 794, in _compile_inner
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     out_code = transform_code_object(code, transform)
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_dynamo/bytecode_transformation.py", line 1418, in transform_code_object
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     transformations(instructions, code_options)
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_dynamo/convert_frame.py", line 256, in _fn
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     return fn(*args, **kwargs)
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]            ^^^^^^^^^^^^^^^^^^^
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_dynamo/convert_frame.py", line 712, in transform
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     tracer.run()
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_dynamo/symbolic_convert.py", line 3315, in run
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     super().run()
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_dynamo/symbolic_convert.py", line 1216, in run
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     while self.step():
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]           ^^^^^^^^^^^
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_dynamo/symbolic_convert.py", line 1126, in step
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     self.dispatch_table[inst.opcode](self, inst)
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_dynamo/symbolic_convert.py", line 3511, in RETURN_VALUE
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     self._return(inst)
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_dynamo/symbolic_convert.py", line 3496, in _return
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     self.output.compile_subgraph(
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_dynamo/output_graph.py", line 1141, in compile_subgraph
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     self.compile_and_call_fx_graph(
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_dynamo/output_graph.py", line 1434, in compile_and_call_fx_graph
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     compiled_fn = self.call_user_compiler(gm)
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_dynamo/output_graph.py", line 1484, in call_user_compiler
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     return self._call_user_compiler(gm)
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_dynamo/output_graph.py", line 1516, in _call_user_compiler
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     compiled_fn = compiler_fn(gm, self.example_inputs())
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_dynamo/repro/after_dynamo.py", line 150, in __call__
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     compiled_gm = compiler_fn(gm, example_inputs)
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/__init__.py", line 2349, in __call__
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     return compile_fx(model_, inputs_, config_patches=self.config)
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/compile_fx.py", line 1890, in compile_fx
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     return compile_fx(
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]            ^^^^^^^^^^^
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/compile_fx.py", line 2260, in compile_fx
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     raise e.remove_dynamo_frames() from None  # see TORCHDYNAMO_VERBOSE=1
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/compile_fx.py", line 763, in _compile_fx_inner
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     raise InductorError(e, currentframe()).with_traceback(
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/compile_fx.py", line 748, in _compile_fx_inner
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     mb_compiled_graph = fx_codegen_and_compile(
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]                         ^^^^^^^^^^^^^^^^^^^^^^^
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/compile_fx.py", line 1454, in fx_codegen_and_compile
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     return scheme.codegen_and_compile(gm, example_inputs, inputs_to_check, graph_kwargs)
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/compile_fx.py", line 1174, in codegen_and_compile
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     compiled_fn = graph.compile_to_module().call
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]                   ^^^^^^^^^^^^^^^^^^^^^^^^^
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/graph.py", line 2088, in compile_to_module
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     return self._compile_to_module()
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]            ^^^^^^^^^^^^^^^^^^^^^^^^^
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/graph.py", line 2135, in _compile_to_module
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     mod = PyCodeCache.load_by_key_path(
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/codecache.py", line 2712, in load_by_key_path
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     mod = _reload_python_module(key, path)
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/runtime/compile_tasks.py", line 36, in _reload_python_module
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     exec(code, mod.__dict__, mod.__dict__)
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/tmp/torchinductor_johnsh29/v5/cv5d3thjjjuaajwx3grjifwfqlnnv24ifefi37aymjgb7i3wt2vl.py", line 80, in <module>
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     async_compile.wait(globals())
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/async_compile.py", line 424, in wait
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     self._wait_futures(scope)
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/async_compile.py", line 445, in _wait_futures
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     scope[key] = result.result()
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]                  ^^^^^^^^^^^^^^^
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/codecache.py", line 3189, in result
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     return self.result_fn()
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]            ^^^^^^^^^^^^^^^^
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/async_compile.py", line 325, in get_result
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     kernel.precompile(
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/runtime/triton_heuristics.py", line 277, in precompile
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     self._make_launchers()
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/runtime/triton_heuristics.py", line 434, in _make_launchers
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     launchers.append(result.make_launcher())
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]                      ^^^^^^^^^^^^^^^^^^^^^^
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/runtime/triton_heuristics.py", line 1079, in make_launcher
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     binary._init_handles()
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/triton/compiler/compiler.py", line 384, in _init_handles
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     self.run = driver.active.launcher_cls(self.src, self.metadata)
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/driver.py", line 440, in __init__
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     mod = compile_module_from_src(src, "__triton_launcher")
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/driver.py", line 57, in compile_module_from_src
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     so = _build(name, src_path, tmpdir, library_dirs(), include_dir, libraries)
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/triton/runtime/build.py", line 50, in _build
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     ret = subprocess.check_call(cc_cmd)
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/lib/python3.12/subprocess.py", line 413, in check_call
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     raise CalledProcessError(retcode, cmd)
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277] torch._inductor.exc.InductorError: CalledProcessError: Command '['/home/johnsh29/miniconda3/bin/x86_64-conda-linux-gnu-cc', '/tmp/tmpccltraq9/main.c', '-O3', '-shared', '-fPIC', '-Wno-psabi', '-o', '/tmp/tmpccltraq9/__triton_launcher.cpython-312-x86_64-linux-gnu.so', '-lcuda', '-L/usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/lib', '-L/usr/lib/x86_64-linux-gnu', '-L/usr/local/cuda/compat/lib', '-I/usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/include', '-I/tmp/tmpccltraq9', '-I/usr/include/python3.12']' returned non-zero exit status 1.
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277] 
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277] Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277] 
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277] Traceback (most recent call last):
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_dynamo/convert_frame.py", line 1210, in __call__
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     result = self._inner_convert(
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]              ^^^^^^^^^^^^^^^^^^^^
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_dynamo/convert_frame.py", line 597, in __call__
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     return _compile(
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]            ^^^^^^^^^
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_dynamo/convert_frame.py", line 1056, in _compile
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     guarded_code = compile_inner(code, one_graph, hooks, transform)
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_utils_internal.py", line 97, in wrapper_function
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     return function(*args, **kwargs)
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]            ^^^^^^^^^^^^^^^^^^^^^^^^^
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_dynamo/convert_frame.py", line 758, in compile_inner
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     return _compile_inner(code, one_graph, hooks, transform)
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_dynamo/convert_frame.py", line 794, in _compile_inner
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     out_code = transform_code_object(code, transform)
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_dynamo/bytecode_transformation.py", line 1418, in transform_code_object
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     transformations(instructions, code_options)
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_dynamo/convert_frame.py", line 256, in _fn
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     return fn(*args, **kwargs)
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]            ^^^^^^^^^^^^^^^^^^^
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_dynamo/convert_frame.py", line 712, in transform
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     tracer.run()
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_dynamo/symbolic_convert.py", line 3315, in run
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     super().run()
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_dynamo/symbolic_convert.py", line 1216, in run
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     while self.step():
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]           ^^^^^^^^^^^
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_dynamo/symbolic_convert.py", line 1126, in step
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     self.dispatch_table[inst.opcode](self, inst)
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_dynamo/symbolic_convert.py", line 3511, in RETURN_VALUE
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     self._return(inst)
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_dynamo/symbolic_convert.py", line 3496, in _return
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     self.output.compile_subgraph(
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_dynamo/output_graph.py", line 1141, in compile_subgraph
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     self.compile_and_call_fx_graph(
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_dynamo/output_graph.py", line 1434, in compile_and_call_fx_graph
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     compiled_fn = self.call_user_compiler(gm)
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_dynamo/output_graph.py", line 1484, in call_user_compiler
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     return self._call_user_compiler(gm)
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_dynamo/output_graph.py", line 1516, in _call_user_compiler
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     compiled_fn = compiler_fn(gm, self.example_inputs())
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_dynamo/repro/after_dynamo.py", line 150, in __call__
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     compiled_gm = compiler_fn(gm, example_inputs)
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/__init__.py", line 2349, in __call__
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     return compile_fx(model_, inputs_, config_patches=self.config)
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/compile_fx.py", line 1890, in compile_fx
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     return compile_fx(
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]            ^^^^^^^^^^^
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/compile_fx.py", line 2260, in compile_fx
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     raise e.remove_dynamo_frames() from None  # see TORCHDYNAMO_VERBOSE=1
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/compile_fx.py", line 763, in _compile_fx_inner
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     raise InductorError(e, currentframe()).with_traceback(
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/compile_fx.py", line 748, in _compile_fx_inner
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     mb_compiled_graph = fx_codegen_and_compile(
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]                         ^^^^^^^^^^^^^^^^^^^^^^^
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/compile_fx.py", line 1454, in fx_codegen_and_compile
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     return scheme.codegen_and_compile(gm, example_inputs, inputs_to_check, graph_kwargs)
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/compile_fx.py", line 1174, in codegen_and_compile
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     compiled_fn = graph.compile_to_module().call
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]                   ^^^^^^^^^^^^^^^^^^^^^^^^^
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/graph.py", line 2088, in compile_to_module
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     return self._compile_to_module()
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]            ^^^^^^^^^^^^^^^^^^^^^^^^^
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/graph.py", line 2135, in _compile_to_module
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     mod = PyCodeCache.load_by_key_path(
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/codecache.py", line 2712, in load_by_key_path
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     mod = _reload_python_module(key, path)
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/runtime/compile_tasks.py", line 36, in _reload_python_module
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     exec(code, mod.__dict__, mod.__dict__)
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/tmp/torchinductor_johnsh29/v5/cv5d3thjjjuaajwx3grjifwfqlnnv24ifefi37aymjgb7i3wt2vl.py", line 80, in <module>
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     async_compile.wait(globals())
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/async_compile.py", line 424, in wait
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     self._wait_futures(scope)
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/async_compile.py", line 445, in _wait_futures
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     scope[key] = result.result()
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]                  ^^^^^^^^^^^^^^^
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/codecache.py", line 3189, in result
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     return self.result_fn()
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]            ^^^^^^^^^^^^^^^^
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/async_compile.py", line 325, in get_result
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     kernel.precompile(
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/runtime/triton_heuristics.py", line 277, in precompile
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     self._make_launchers()
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/runtime/triton_heuristics.py", line 434, in _make_launchers
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     launchers.append(result.make_launcher())
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]                      ^^^^^^^^^^^^^^^^^^^^^^
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/runtime/triton_heuristics.py", line 1079, in make_launcher
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     binary._init_handles()
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/triton/compiler/compiler.py", line 384, in _init_handles
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     self.run = driver.active.launcher_cls(self.src, self.metadata)
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/driver.py", line 440, in __init__
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     mod = compile_module_from_src(src, "__triton_launcher")
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/driver.py", line 57, in compile_module_from_src
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     so = _build(name, src_path, tmpdir, library_dirs(), include_dir, libraries)
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/local/lib/python3.12/dist-packages/triton/runtime/build.py", line 50, in _build
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     ret = subprocess.check_call(cc_cmd)
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]   File "/usr/lib/python3.12/subprocess.py", line 413, in check_call
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277]     raise CalledProcessError(retcode, cmd)
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277] torch._inductor.exc.InductorError: CalledProcessError: Command '['/home/johnsh29/miniconda3/bin/x86_64-conda-linux-gnu-cc', '/tmp/tmpccltraq9/main.c', '-O3', '-shared', '-fPIC', '-Wno-psabi', '-o', '/tmp/tmpccltraq9/__triton_launcher.cpython-312-x86_64-linux-gnu.so', '-lcuda', '-L/usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/lib', '-L/usr/lib/x86_64-linux-gnu', '-L/usr/local/cuda/compat/lib', '-I/usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/include', '-I/tmp/tmpccltraq9', '-I/usr/include/python3.12']' returned non-zero exit status 1.
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277] 
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277] Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"
    [rank0]:W0618 21:57:02.561000 1489278 torch/_dynamo/convert_frame.py:1277] 
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
    [NeMo W 2025-06-18 21:57:03 rerun_state_machine:1264] Implicit initialization of Rerun State Machine!
    [NeMo W 2025-06-18 21:57:03 rerun_state_machine:239] RerunStateMachine initialized in mode RerunMode.DISABLED
    Training epoch 0, iteration 0/9 | lr: 0 | global_batch_size: 1 | global_step: 0 | reduced_train_loss: 1.371 | train_step_timing in s: 2.193
    Training epoch 0, iteration 1/9 | lr: 2e-05 | global_batch_size: 1 | global_step: 1 | reduced_train_loss: 1.37 | train_step_timing in s: 0.3513 | consumed_samples: 2
    Training epoch 0, iteration 2/9 | lr: 4e-05 | global_batch_size: 1 | global_step: 2 | reduced_train_loss: 1.259 | train_step_timing in s: 0.3506 | consumed_samples: 3
    Training epoch 0, iteration 3/9 | lr: 6e-05 | global_batch_size: 1 | global_step: 3 | reduced_train_loss: 1.277 | train_step_timing in s: 0.3502 | consumed_samples: 4
    Training epoch 0, iteration 4/9 | lr: 8e-05 | global_batch_size: 1 | global_step: 4 | reduced_train_loss: 1.257 | train_step_timing in s: 0.3505 | consumed_samples: 5
    Epoch 0, global step 4: 'val_loss' was not in top 5
    [NeMo I 2025-06-18 21:57:04 nemo_logging:393] Using FullyParallelSaveStrategyWrapper(torch_dist, 1) dist-ckpt save strategy.
    [NeMo I 2025-06-18 21:57:05 nemo_logging:393] Global Checkpoint Save : Rank: 0 : Iteration: 4 : Start time: 1750309024.990s : Save duration: 0.217s
    [NeMo I 2025-06-18 21:57:12 nemo_logging:393] Scheduled async checkpoint save for /home/johnsh29/Evo2_Mortierellaceae/pretraining_mortierellaceae/evo2/checkpoints/evo2--val_loss=0.0000-epoch=0-consumed_samples=5.0-last.ckpt
    [NeMo I 2025-06-18 21:57:12 nemo_logging:393] Async finalization time took 0.002 s
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
    [NeMo W 2025-06-18 21:57:14 nemo_logging:405] /usr/local/lib/python3.12/dist-packages/lightning/pytorch/trainer/connectors/logger_connector/result.py:431: It is recommended to use `self.log('global_batch_size', ..., sync_dist=True)` when logging on epoch level in distributed setting to accumulate the metric across devices.
        
    [NeMo W 2025-06-18 21:57:14 nemo_logging:405] /usr/local/lib/python3.12/dist-packages/lightning/pytorch/trainer/connectors/logger_connector/result.py:431: It is recommended to use `self.log('val_loss', ..., sync_dist=True)` when logging on epoch level in distributed setting to accumulate the metric across devices.
        
    Training epoch 0, iteration 5/9 | lr: 0.0001 | global_batch_size: 1 | global_step: 5 | reduced_train_loss: 1.26 | train_step_timing in s: 0.3964 | consumed_samples: 6 | val_loss: 1.264
    [NeMo I 2025-06-18 21:57:15 nemo_logging:393] Async finalization time took 0.000 s
    Training epoch 0, iteration 6/9 | lr: 3e-05 | global_batch_size: 1 | global_step: 6 | reduced_train_loss: 1.323 | train_step_timing in s: 0.4149 | consumed_samples: 7 | val_loss: 1.264
    [NeMo I 2025-06-18 21:57:15 nemo_logging:393] Async finalization time took 0.000 s
    Training epoch 0, iteration 7/9 | lr: 3e-05 | global_batch_size: 1 | global_step: 7 | reduced_train_loss: 1.338 | train_step_timing in s: 0.399 | consumed_samples: 8 | val_loss: 1.264
    [NeMo I 2025-06-18 21:57:16 nemo_logging:393] Async finalization time took 0.000 s
    Training epoch 0, iteration 8/9 | lr: 3e-05 | global_batch_size: 1 | global_step: 8 | reduced_train_loss: 1.257 | train_step_timing in s: 0.4168 | consumed_samples: 9 | val_loss: 1.264
    [NeMo I 2025-06-18 21:57:16 nemo_logging:393] Async finalization time took 0.000 s
    Training epoch 0, iteration 9/9 | lr: 3e-05 | global_batch_size: 1 | global_step: 9 | reduced_train_loss: 1.292 | train_step_timing in s: 0.4083 | consumed_samples: 10 | val_loss: 1.264
    Epoch 0, global step 9: 'val_loss' reached 1.26396 (best 1.26396), saving model to '/home/johnsh29/Evo2_Mortierellaceae/pretraining_mortierellaceae/evo2/checkpoints/evo2--val_loss=1.2640-epoch=0-consumed_samples=10.0.ckpt' as top 5
    [NeMo I 2025-06-18 21:57:20 nemo_logging:393] Global Checkpoint Save : Rank: 0 : Iteration: 9 : Start time: 1750309039.748s : Save duration: 0.476s
    [NeMo I 2025-06-18 21:57:20 nemo_logging:393] Scheduled async checkpoint save for /home/johnsh29/Evo2_Mortierellaceae/pretraining_mortierellaceae/evo2/checkpoints/evo2--val_loss=1.2640-epoch=0-consumed_samples=10.0.ckpt
    [NeMo I 2025-06-18 21:57:23 nemo_logging:393] Global Checkpoint Save : Rank: 0 : Iteration: 9 : Start time: 1750309040.832s : Save duration: 2.746s
    [NeMo I 2025-06-18 21:57:35 nemo_logging:393] Scheduled async checkpoint save for /home/johnsh29/Evo2_Mortierellaceae/pretraining_mortierellaceae/evo2/checkpoints/evo2--val_loss=1.2640-epoch=0-consumed_samples=10.0-last.ckpt
    [NeMo I 2025-06-18 21:57:35 nemo_logging:393] Async finalization time took 0.001 s
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
    [NeMo I 2025-06-18 21:58:10 nemo_logging:393] Async finalization time took 0.000 s
    `Trainer.fit` stopped: `max_steps=10` reached.
    [NeMo I 2025-06-18 21:58:10 nemo_logging:393] Pending async checkpoint saves. Finalizing them synchronously now
    [NeMo I 2025-06-18 22:03:04 nemo_logging:393] Successfully saved checkpoint from iteration       4 to /home/johnsh29/Evo2_Mortierellaceae/pretraining_mortierellaceae/evo2/checkpoints/evo2--val_loss=0.0000-epoch=0-consumed_samples=5.0-last.ckpt
    [NeMo I 2025-06-18 22:03:05 nemo_logging:393] Async checkpoint save for step 5 (/home/johnsh29/Evo2_Mortierellaceae/pretraining_mortierellaceae/evo2/checkpoints/evo2--val_loss=0.0000-epoch=0-consumed_samples=5.0-last.ckpt) finalized successfully.
    [NeMo I 2025-06-18 22:03:27 nemo_logging:393] Successfully saved checkpoint from iteration       9 to /home/johnsh29/Evo2_Mortierellaceae/pretraining_mortierellaceae/evo2/checkpoints/evo2--val_loss=1.2640-epoch=0-consumed_samples=10.0.ckpt
    [NeMo I 2025-06-18 22:03:28 nemo_logging:393] Async checkpoint save for step 10 (/home/johnsh29/Evo2_Mortierellaceae/pretraining_mortierellaceae/evo2/checkpoints/evo2--val_loss=1.2640-epoch=0-consumed_samples=10.0.ckpt) finalized successfully.
    [NeMo I 2025-06-18 22:03:55 nemo_logging:393] Successfully saved checkpoint from iteration       9 to /home/johnsh29/Evo2_Mortierellaceae/pretraining_mortierellaceae/evo2/checkpoints/evo2--val_loss=1.2640-epoch=0-consumed_samples=10.0-last.ckpt
    [NeMo I 2025-06-18 22:03:55 nemo_logging:393] Async checkpoint save for step 10 (/home/johnsh29/Evo2_Mortierellaceae/pretraining_mortierellaceae/evo2/checkpoints/evo2--val_loss=1.2640-epoch=0-consumed_samples=10.0-last.ckpt) finalized successfully.
    [NeMo I 2025-06-18 22:03:55 nemo_logging:393] Async finalization time took 344.980 s
    
    Training completed!
    Total training time: 461.33 seconds
    (7.69 minutes)
    (0.13 hours)



```python

```


```python

```


```python

```
