# ReaLchords

PyTorch implementation of ReaLchords, ReaLJam and GAPT: real-time music accompaniment systems with generative models trained via reinforcement learning.

## Table of Contents

- [Setup and Play ReaLJam](#setup-and-play-realjam)
  - [ONNX Speedup for ReaLJam Server](#onnx-speedup-for-realjam-server)
  - [Start ReaLJam Server via this Codebase](#start-realjam-server-via-this-codebase)
- [Setup Model Training and Development](#setup-model-training-and-development)
- [Dataset](#dataset)
  - [Quick Start](#quick-start)
  - [Dataset Statistics](#dataset-statistics)
  - [Data Augmentation](#data-augmentation)
  - [Using Datasets](#using-datasets)
- [Model Training](#model-training)
  - [ReaLchords Model](#realchords-model-only-trained-on-hooktheory-dataset)
  - [GAPT Model](#gapt-model-trained-on-3-datasets-hooktheory-pop909-nottingham)
  - [RL Training](#rl-training)
  - [RL Training for Melody Model](#rl-training-for-melody-model)
- [Model Checkpoints](#model-checkpoints)
- [Model Checkpoint Conversion for ReaLJam](#model-checkpoint-conversion-for-realjam)
- [Citation](#citation)
- [A Note on argbind](#a-note-on-argbind)

## Setup and Play ReaLJam

To just install and play with ReaLJam, simply run: `pip install realjam` (python>=3.10).

To start the ReaLJam backend server:

1. Make sure you are at the project root directory.
2. Run `realjam-start-server` to start the ReaLJam server on 8080. The pretrained checkpoints will be downloaded automatically (the checkpoints will be downloaded to `$HOME/.realjam/checkpoints/`).
3. When the server starts, it will output several URLs. Open the URL showing `http://127.0.0.1:8080` (localhost) in your browser.

You can set the port (default: 8080) and whether to use SSL by:

`realjam-start-server --port 1234 --ssl`

You should include `--ssl` if you want to run the backend on a server. In that case, you can access ReaLJam by visiting `http://server_ip:8080` (replace `server_ip` with your server's IP address).

The interface of ReaLJam is as follows. We support both MIDI keyboard as input and computer keyboard (see the letters labeled on piano keys). Select your MIDI input source from the upper right. Begin jamming by clicking "Start Live Session", and the model will start to accompany you after `Initial Beats of Silence` beats (default: 8). You can also adjust `Lookahead Beats` and `Commit Beats` to control how far into the future the model will generate and how much of the future buffer is kept fixed. You can select different models from the `Model` drop-down menu.

For more details on how ReaLJam works, please see our [paper](https://arxiv.org/abs/2502.21267).

### ONNX Speedup for ReaLJam server

We also support ONNX speedup that allows you to run models faster on both CPU and GPU. You can replace the PyTorch model with an ONNX model via `realjam-start-server --port 1234 --ssl --onnx`. When a CUDA device is detected, the ONNX model (as well as the PyTorch model) will run on the CUDA device by default. You can adjust this using `onnx_provider`. See `realjam-start-server -h` for details.

### Start ReaLJam server via this codebase

You can start the ReaLJam server via this codebase rather than using the `realjam-start-server` command in the `realjam` package by running `python -m realchords.realjam.server --port 1234 --ssl`. This will start the server on the port 1234 with SSL enabled.


## Setup Model Training and Development

Simply install this codebase locally via:

`pip install -e .`

For logging the generated results as audio, we use [FluidSynth](https://www.fluidsynth.org/). See function [`play_midi_with_soundfont`](realchords/utils/log_utils.py) in `realchords/utils/log_utils.py` for details. To do that, you need to:

1. Install [FluidSynth CLI](https://github.com/FluidSynth/fluidsynth/wiki/Download)
2. Download the SF2 soundfont from [https://sites.google.com/site/soundfonts4u/](https://sites.google.com/site/soundfonts4u/) to `soundfonts/Yamaha C5 Grand-v2.4.sf2` (or use other soundfonts you prefer)

## Dataset

ReaLchords supports four music datasets (Hooktheory, POP909, Nottingham, Wikifonia). All datasets are converted to a unified cache format (`.jsonl`) for training.

### Quick Start

Download and convert all four datasets using these commands:

```bash
# 1. Hooktheory (19K songs)
python scripts/download_hooktheory_dataset.py --verify
python scripts/convert_hooktheory_to_cache.py
python scripts/convert_hooktheory_to_cache.py --augmentation

# 2. POP909 (909 popular songs)
python scripts/download_pop909_dataset.py --verify
python scripts/convert_pop909_to_cache.py
python scripts/convert_pop909_to_cache.py --augmentation

# 3. Nottingham (1,019 folk tunes)
python scripts/download_nottingham_dataset.py --verify
python scripts/convert_nottingham_to_cache.py
python scripts/convert_nottingham_to_cache.py --augmentation

# 4. Wikifonia (6K+ lead sheets)
python scripts/download_wikifonia_dataset.py
python scripts/convert_wikifonia_to_cache.py
python scripts/convert_wikifonia_to_cache.py --augmentation
```

**Output**: Converted datasets are stored in `data/cache/<dataset_name>/*.jsonl` with train/valid/test splits (80/10/10). Each line in the `.jsonl` file is a JSON object containing melody notes, chord annotations, and timing information. The above commands will create both regular and augmented versions of the datasets.

**Chord Vocabularies**: Global chord names are maintained in:
- `data/cache/chord_names.json` - Without augmentation
- `data/cache/chord_names_augmented.json` - With transposition augmentation

### Dataset Statistics

| Dataset | Songs | Unique Chords | Source | Format |
|---------|-------|---------------|--------|--------|
| **Hooktheory** | 19,086 | ~2,821 | [SheetSage](https://sheetsage.s3.amazonaws.com/hooktheory/Hooktheory.json.gz) | JSON |
| **POP909** | 909 | ~175 | [GitHub](https://github.com/music-x-lab/POP909-Dataset) | MIDI + TXT |
| **Nottingham** | 1,019 | ~32 | [GitHub](https://github.com/jukedeck/nottingham-dataset) | ABC |
| **Wikifonia** | ~6,000 | varies | [Wikifonia Archive](https://github.com/andreamust/WikifoniaDataset) | MusicXML |

### Data Augmentation

The `--augmentation` flag transposes all songs by -6 to +6 semitones (13 versions each), creating 13x more training data and significantly improving model generalization across different keys.

### Using Datasets

All datasets use the same `HooktheoryDataset` interface:

```python
from realchords.dataset.hooktheory_dataloader import HooktheoryDataset

# Load any dataset
dataset = HooktheoryDataset(
    cache_dir="data/cache/hooktheory",  # or pop909, nottingham, wikifonia
    split="train",
    model_type="decoder_only",
    model_part="chord",
    max_len=512,
    data_augmentation=True  # Enable augmentation
)
```

## Model Training

Follow the following process to train the final RL-finetuned models. This codebase uses [Weights & Biases (wandb)](https://wandb.ai/) to log training data. If you want to view your training curves on wandb, please configure and login to wandb before training. The same applies to reward model training and RL training.

### ReaLchords Model (only trained on Hooktheory dataset)

**Supervised Pre-training (Maximum Likelihood Estimation, MLE):**

```bash
# Online accompaniment models
python scripts/train_decoder_only.py --args.load configs/gen_models/decoder_only_online_chord.yml --save_dir logs/decoder_only_online_chord

# Encoder-decoder models
python scripts/train_enc_dec.py --args.load configs/gen_models/enc_dec_base_chord.yml --save_dir logs/enc_dec_base_chord
```

**Reward Model Training**

```bash
# Contrastive reward models
python scripts/train_contrastive_reward.py --args.load configs/reward_models/contrastive_reward.yml --save_dir logs/contrastive_reward
python scripts/train_contrastive_reward.py --args.load configs/reward_models/contrastive_reward_2.yml --save_dir logs/contrastive_reward_2

# Discriminative reward models (with 128 batch size)
python scripts/train_discriminative_reward.py --args.load configs/reward_models/discriminative_reward.yml --save_dir logs/discriminative_reward_128_bs
python scripts/train_discriminative_reward.py --args.load configs/reward_models/discriminative_reward_2.yml --save_dir logs/discriminative_reward_128_bs_2

# Rhythm-specific reward models, which do not use augmentation
python scripts/train_contrastive_reward_rhythm.py --args.load configs/reward_models/contrastive_reward_no_augmentation_rhythm.yml --save_dir logs/contrastive_reward_no_augmentation_rhythm
python scripts/train_contrastive_reward_rhythm.py --args.load configs/reward_models/contrastive_reward_no_augmentation_rhythm_2.yml --save_dir logs/contrastive_reward_no_augmentation_rhythm_2
python scripts/train_discriminative_reward_rhythm.py --args.load configs/reward_models/discriminative_reward_no_augmentation_rhythm.yml --save_dir logs/discriminative_reward_no_augmentation_rhythm
python scripts/train_discriminative_reward_rhythm.py --args.load configs/reward_models/discriminative_reward_no_augmentation_rhythm_2.yml --save_dir logs/discriminative_reward_no_augmentation_rhythm_2
```

### GAPT Model (trained on 3 datasets: Hooktheory, POP909, Nottingham):

**Supervised Pre-training (Maximum Likelihood Estimation, MLE):**

```bash
# Online accompaniment model (3 datasets)
python scripts/train_decoder_only.py --args.load configs/gen_models/decoder_only_online_chord_3_datasets.yml --save_dir logs/decoder_only_online_chord_3_datasets

# Encoder-decoder model (3 datasets)
python scripts/train_enc_dec.py --args.load configs/gen_models/enc_dec_base_chord_3_datasets.yml --save_dir logs/enc_dec_base_chord_3_datasets
```

**Reward Model Training**

```bash
# Contrastive reward models (3 datasets)
python scripts/train_contrastive_reward.py --args.load configs/reward_models/contrastive_reward_3_datasets.yml --save_dir logs/contrastive_reward_3_datasets
python scripts/train_contrastive_reward.py --args.load configs/reward_models/contrastive_reward_2_3_datasets.yml --save_dir logs/contrastive_reward_2_3_datasets

# Discriminative reward models (3 datasets, 128 batch size)
python scripts/train_discriminative_reward.py --args.load configs/reward_models/discriminative_reward_128_bs_3_datasets.yml --save_dir logs/discriminative_reward_128_bs_3_datasets
python scripts/train_discriminative_reward.py --args.load configs/reward_models/discriminative_reward_128_bs_2_3_datasets.yml --save_dir logs/discriminative_reward_128_bs_2_3_datasets

# Rhythm-specific reward models (3 datasets, no augmentation)
python scripts/train_contrastive_reward_rhythm.py --args.load configs/reward_models/contrastive_reward_no_augmentation_rhythm_3_datasets.yml --save_dir logs/contrastive_reward_no_augmentation_rhythm_3_datasets
python scripts/train_contrastive_reward_rhythm.py --args.load configs/reward_models/contrastive_reward_no_augmentation_rhythm_2_3_datasets.yml --save_dir logs/contrastive_reward_no_augmentation_rhythm_2_3_datasets
python scripts/train_discriminative_reward_rhythm.py --args.load configs/reward_models/discriminative_reward_no_augmentation_rhythm_3_datasets.yml --save_dir logs/discriminative_reward_no_augmentation_rhythm_3_datasets
python scripts/train_discriminative_reward_rhythm.py --args.load configs/reward_models/discriminative_reward_no_augmentation_rhythm_2_3_datasets.yml --save_dir logs/discriminative_reward_no_augmentation_rhythm_2_3_datasets
```


### RL Training

Post-train MLE models with reinforcement learning. First, install `mpi4py` for distributed RL training, e.g. via `conda`:

```bash
conda install mpi4py
```

**Train ReaLchords model (reproduction of the original paper):**

This requires the following pre-trained models:
- Base model: `decoder_only_online_chord`
- Anchor model: `enc_dec_base_chord`
- Reward models: `contrastive_reward`, `contrastive_reward_2`, `discriminative_reward_128_bs`, `discriminative_reward_128_bs_2`
- Rhythm reward models: `contrastive_reward_no_augmentation_rhythm`, `contrastive_reward_no_augmentation_rhythm_2`, `discriminative_reward_no_augmentation_rhythm`, `discriminative_reward_no_augmentation_rhythm_2`

```bash
python scripts/train_rl_ensemble_rhythm_reward_offline_anchor.py \
  --args.load configs/single_agent_rl/realchords.yml \
  --save_dir logs/realchords
```

**Train GAPT model (Generative Adversarial Preference Training):**

This requires the following pre-trained models (3 datasets):
- Base model: `decoder_only_online_chord_3_datasets`
- Anchor model: `enc_dec_base_chord_3_datasets`
- Reward models: `contrastive_reward_3_datasets`, `contrastive_reward_2_3_datasets`, `discriminative_reward_128_bs_3_datasets`, `discriminative_reward_128_bs_2_3_datasets`
- Rhythm reward models: `contrastive_reward_no_augmentation_rhythm_3_datasets`, `contrastive_reward_no_augmentation_rhythm_2_3_datasets`, `discriminative_reward_no_augmentation_rhythm_3_datasets`, `discriminative_reward_no_augmentation_rhythm_2_3_datasets`


```bash
python scripts/train_rl_ensemble_rhythm_reward_offline_anchor_gail.py \
  --args.load configs/single_agent_rl/gapt.yml \
  --save_dir logs/gapt
```

**Note on OpenRLHF:** The RL training framework is adapted from [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) v0.6.1.post1. To eliminate dependency on the outdated OpenRLHF version, the relevant code has been copied to `realchords/rl/openrlhf_local`. Those sources remain under the Apache License 2.0; see `third_party/openrlhf/LICENSE` and the notices in `realchords/rl/openrlhf_local` for attribution details.

**Hardware Requirements:**

All training (MLE, reward models, and RL) can run on GPUs with 48GB VRAM (e.g., L40s, A6000). The RL training currently supports single GPU only.

**Note on OpenMPI:**
If you encounter MPI errors, you can try the following to set the environment variables and run the training via `mpirun`:

```bash
export RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500  # Any available port between 10000-65535

mpirun -np 1 \
  -x MASTER_PORT \
  -x MASTER_ADDR \
  -x RANK \
  python scripts/train_rl_ensemble_rhythm_reward_offline_anchor_gail.py \
    --args.load configs/single_agent_rl/gapt.yml \
    --save_dir logs/gapt
```

**Memory Optimization (Optional):**

If you encounter out-of-memory (OOM) errors, you can:

1. Reduce `micro_rollout_batch_size` and `train_batch_size` in your config
2. Enable VRAM swapping by setting these flags to `true` (see `configs/single_agent_rl/rl_realchords_base.yml` for details):

```yaml
reward_vram_swap: true          # Swap reward model to CPU when not in use
counterpart_vram_swap: true     # Swap reference model to CPU when not in use
trainer_empty_cache: true       # Clear CUDA cache after each training step
logits_vram_swap: true          # Swap logits to CPU to reduce memory usage
```

> **Note:** These options reduce memory usage but may increase training time.

### RL Training for Melody Model

If you want to train an online model that generates melody given chords, you can train the models with the following commands:

**1. MLE Pre-training:**

```bash
python scripts/train_decoder_only.py --args.load configs/gen_models/decoder_only_online_melody_4_datasets.yml --save_dir logs/decoder_only_online_melody_4_datasets

python scripts/train_enc_dec.py --args.load configs/gen_models/enc_dec_base_melody_4_datasets.yml --save_dir logs/enc_dec_base_melody_4_datasets
```

**2. Reward Model Training:**

```bash
# Contrastive reward models
python scripts/train_contrastive_reward.py --args.load configs/reward_models/contrastive_reward_4_datasets.yml --save_dir logs/contrastive_reward_4_datasets
python scripts/train_contrastive_reward.py --args.load configs/reward_models/contrastive_reward_2_4_datasets.yml --save_dir logs/contrastive_reward_2_4_datasets

# Discriminative reward models
python scripts/train_discriminative_reward.py --args.load configs/reward_models/discriminative_reward_128_bs_4_datasets.yml --save_dir logs/discriminative_reward_128_bs_4_datasets
python scripts/train_discriminative_reward.py --args.load configs/reward_models/discriminative_reward_128_bs_2_4_datasets.yml --save_dir logs/discriminative_reward_128_bs_2_4_datasets

# Rhythm-specific reward models
python scripts/train_contrastive_reward_rhythm.py --args.load configs/reward_models/contrastive_reward_no_augmentation_rhythm_4_datasets.yml --save_dir logs/contrastive_reward_no_augmentation_rhythm_4_datasets
python scripts/train_contrastive_reward_rhythm.py --args.load configs/reward_models/contrastive_reward_no_augmentation_rhythm_2_4_datasets.yml --save_dir logs/contrastive_reward_no_augmentation_rhythm_2_4_datasets
python scripts/train_discriminative_reward_rhythm.py --args.load configs/reward_models/discriminative_reward_no_augmentation_rhythm_4_datasets.yml --save_dir logs/discriminative_reward_no_augmentation_rhythm_4_datasets
python scripts/train_discriminative_reward_rhythm.py --args.load configs/reward_models/discriminative_reward_no_augmentation_rhythm_2_4_datasets.yml --save_dir logs/discriminative_reward_no_augmentation_rhythm_2_4_datasets
```

**3. RL Training with GAIL:**

```bash
python scripts/train_rl_ensemble_rhythm_reward_offline_anchor_gail.py \
  --args.load configs/single_agent_rl/rl_melody_gail_4_datasets.yml \
  --save_dir logs/rl_melody_gail_4_datasets
```

## Model Checkpoints

All trained model checkpoints (supervised models, reward models, and RL-trained models) can be downloaded from the [huggingface page](https://huggingface.co/lukewys/realchords-pytorch/tree/main).

## Model Checkpoint conversion for ReaLJam

To convert the MLE model checkpoints to the format compatible with ReaLJam, you need to use the following command:

```bash
python scripts/generate_checkpoint_for_realjam.py \
  --checkpoint_path logs/decoder_only_online_chord/step=11000.ckpt \
  --save_dir checkpoints
```

You can directly use the RL checkpoints for ReaLJam.

## Citation

If you find this work useful, consider citing the following papers:

```bibtex
@inproceedings{wu2025generative,
  title={Generative Adversarial Post-Training Mitigates Reward Hacking in Live Human-{AI} Music Interaction},
  author={Wu, Yusong and Brade, Stephen and Ma, Teng and Fowler, Tia-Jane and Yang, Enning and Banar, Berker and Courville, Aaron and Jaques, Natasha and Huang, Cheng-Zhi Anna},
  booktitle={arXiv preprint},
  year={2025},
}

@inproceedings{wu2024adaptive,
  title={Adaptive accompaniment with ReaLchords},
  author={Wu, Yusong and Cooijmans, Tim and Kastner, Kyle and Roberts, Adam and Simon, Ian and Scarlatos, Alexander and Donahue, Chris and Tarakajian, Cassie and Omidshafiei, Shayegan and Courville, Aaron and others},
  booktitle={Forty-first International Conference on Machine Learning},
  year={2024},
}

@inproceedings{scarlatos2025realjam,
  title={ReaLJam: Real-Time Human-AI Music Jamming with Reinforcement Learning-Tuned Transformers},
  author={Scarlatos, Alexander and Wu, Yusong and Simon, Ian and Roberts, Adam and Cooijmans, Tim and Jaques, Natasha and Tarakajian, Cassie and Huang, Anna},
  booktitle={Proceedings of the Extended Abstracts of the CHI Conference on Human Factors in Computing Systems},
  pages={1--9},
  year={2025}
}
```

## A Note on argbind

We use [argbind](https://github.com/pseeth/argbind) to manage configuration and override function/class arguments. 

**Important limitations:**
- Argbind only supports keyword arguments (kwargs)
- May not properly support `bool` type arguments or kwargs with `None` as the default value

**Known bug:** If a `bool` keyword argument is not explicitly loaded from the config file, its default value will always be `False`, regardless of the default value specified in the function declaration. However, if the argument is explicitly specified in the YAML config, it will correctly use the value set in the config.
