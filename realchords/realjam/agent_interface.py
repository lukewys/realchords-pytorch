"""Interface for interacting with generative models.

@Author Tia-Jane Fowler, Alex Scarlatos, Yusong Wu
"""

from typing import List, Optional, Tuple, TypedDict
import note_seq
import numpy as np
import torch
import logging
import os
import time
import json
import copy
import wget
from pathlib import Path
from realchords.utils.inference_utils import (
    load_gen_model_from_state_dict,
    load_rl_model,
)
from realchords.model.gen_model import (
    DecoderTransformer,
    EncoderDecoderTransformer,
)
from realchords.constants import (
    CHORD_OCTAVE,
    ZERO_OCTAVE,
    FRAME_PER_BEAT,
    BASS_OCTAVE,
    REALJAM_CHECKPOINT_DIR,
)
from realchords.dataset.hooktheory_tokenizer import HooktheoryTokenizer
from realchords.model.sampling import (
    filter_invalid_tokens_generate_unconditional,
    filter_invalid_tokens_generate_online,
    filter_invalid_tokens_generate_single_part,
)
from functools import partial
import onnxruntime as ort
from realchords.realjam.onnx_kv import generate_tokens, generate_tokens_online
from realchords.realjam.prepare_onnx import prepare_online

logging.basicConfig(
    level=logging.INFO,  # show INFO, WARNING, ERROR, CRITICAL
    format="%(asctime)s [%(levelname)s] %(message)s",
)

ChordInfo = Tuple[str, List[int], bool]


class NoteInfo(TypedDict):
    pitch: int
    frame: int
    on: bool


ONLINE_MODEL_PATH = os.path.join(
    REALJAM_CHECKPOINT_DIR,
    "realjam_decoder_only_online_chord_3_datasets",
    "step=11000.ckpt",
)

OFFLINE_MODEL_PATH = os.path.join(
    REALJAM_CHECKPOINT_DIR,
    "realjam_enc_dec_base_chord_random_truncate_3_datasets",
    "step=20000.ckpt",
)

MODEL_PATHS = {
    "GAPT": os.path.join(
        REALJAM_CHECKPOINT_DIR,
        "gapt",
        "actor.pth",
    ),
    "ReaLchords": os.path.join(
        REALJAM_CHECKPOINT_DIR,
        "realchords",
        "actor.pth",
    ),
    "Online MLE": os.path.join(
        REALJAM_CHECKPOINT_DIR,
        "realjam_decoder_only_online_chord_3_datasets",
        "step=11000.ckpt",
    ),
}


def download_checkpoints(
    checkpoint_dir: Optional[str] = None, force: bool = False
) -> bool:
    """Download checkpoint files from Hugging Face using wget.

    This function downloads the required checkpoint files and models from the
    lukewys/realchords-pytorch repository on Hugging Face. It uses wget for
    downloading, which should be available on most Linux systems.

    Args:
        checkpoint_dir: Directory to save checkpoints. If None, uses REALJAM_CHECKPOINT_DIR.
        force: If True, re-download files even if they already exist.

    Returns:
        True if download was successful, False otherwise.

    Example:
        >>> download_checkpoints()  # Downloads to default checkpoint directory
        >>> download_checkpoints("/path/to/checkpoints", force=True)  # Force re-download
    """
    if checkpoint_dir is None:
        checkpoint_dir = REALJAM_CHECKPOINT_DIR

    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Define files and directories to download
    base_url = "https://huggingface.co/lukewys/realchords-pytorch/resolve/main"

    downloads = [
        # Single file
        {
            "url": f"{base_url}/chord_names_augmented.json",
            "output": checkpoint_path / "chord_names_augmented.json",
            "is_file": True,
        },
        # Directories
        {
            "name": "gapt",
            "url": f"{base_url}/gapt",
            "output": checkpoint_path / "gapt",
            "files": ["actor.pth", "args.yml"],
        },
        {
            "name": "realchords",
            "url": f"{base_url}/realchords",
            "output": checkpoint_path / "realchords",
            "files": ["actor.pth", "args.yml"],
        },
        {
            "name": "realjam_decoder_only_online_chord_3_datasets",
            "url": f"{base_url}/realjam_decoder_only_online_chord_3_datasets",
            "output": checkpoint_path
            / "realjam_decoder_only_online_chord_3_datasets",
            "files": ["step=11000.ckpt", "args.yml"],
        },
        {
            "name": "realjam_enc_dec_base_chord_random_truncate_3_datasets",
            "url": f"{base_url}/realjam_enc_dec_base_chord_random_truncate_3_datasets",
            "output": checkpoint_path
            / "realjam_enc_dec_base_chord_random_truncate_3_datasets",
            "files": ["step=20000.ckpt", "args.yml"],
        },
    ]

    logging.info(f"Starting checkpoint download to {checkpoint_path}")

    # Download chord_names_augmented.json
    chord_names_file = downloads[0]
    output_file = chord_names_file["output"]

    if not output_file.exists() or force:
        logging.info(f"Downloading {chord_names_file['url']}...")
        try:
            wget.download(chord_names_file["url"], str(output_file))
            logging.info(
                f"\nSuccessfully downloaded chord_names_augmented.json"
            )
        except Exception as e:
            logging.error(f"Failed to download chord_names_augmented.json: {e}")
            return False
    else:
        logging.info(
            f"chord_names_augmented.json already exists, skipping download"
        )

    # Download model directories
    for download_item in downloads[1:]:
        output_dir = download_item["output"]
        output_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"Downloading {download_item['name']}...")

        for file_name in download_item["files"]:
            output_file = output_dir / file_name
            file_url = f"{download_item['url']}/{file_name}"

            if not output_file.exists() or force:
                logging.info(f"  Downloading {file_name}...")
                try:
                    wget.download(file_url, str(output_file))
                    logging.info(f"\n  Successfully downloaded {file_name}")
                except Exception as e:
                    logging.error(f"  Failed to download {file_name}: {e}")
                    return False
            else:
                logging.info(f"  {file_name} already exists, skipping download")

    logging.info("All checkpoints downloaded successfully!")
    return True


class Agent:
    """Interface for interacting with generative chord model."""

    def __init__(
        self,
        onnx: bool = False,
        provider=None,
        compile: bool = True,
        auto_download: bool = True,
    ) -> None:
        """Initialize the agent.

        Args:
            onnx: Whether to use ONNX runtime for inference.
            provider: ONNX execution provider (e.g., 'CUDAExecutionProvider', 'CPUExecutionProvider').
            compile: Whether to compile the model with torch.compile.
            auto_download: Whether to automatically download checkpoints if they don't exist.
        """
        logging.info("Creating model...")

        # Check if checkpoints exist, download if needed
        chord_names_path = os.path.join(
            REALJAM_CHECKPOINT_DIR, "chord_names_augmented.json"
        )
        if auto_download and not os.path.exists(chord_names_path):
            logging.info(
                "Checkpoints not found. Downloading from Hugging Face..."
            )
            if not download_checkpoints():
                raise RuntimeError(
                    "Failed to download checkpoints. Please download manually or "
                    "set auto_download=False and ensure checkpoints are available."
                )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # decide the onnx provider based on the device
        if provider is None:
            if self.device.type == "cuda":
                provider = "CUDAExecutionProvider"
            else:
                provider = "CPUExecutionProvider"

        self.compile = compile
        self.onnx = onnx
        self.max_frames = 256

        # Create tokenizer
        chord_names_path = os.path.join(
            REALJAM_CHECKPOINT_DIR, "chord_names_augmented.json"
        )
        with open(chord_names_path, "r") as f:
            chord_names = json.load(f)
        tokenizer = HooktheoryTokenizer(chord_names=chord_names)
        self.tokenizer = tokenizer

        # Load non-causal (offline) model
        self.enc_dec_model_path = OFFLINE_MODEL_PATH
        self.non_causal_model = load_gen_model_from_state_dict(
            self.enc_dec_model_path,
            EncoderDecoderTransformer,
            compile=False,
            override_args={
                "EncoderDecoderTransformer.enc_num_tokens": tokenizer.num_tokens,
                "EncoderDecoderTransformer.dec_num_tokens": tokenizer.num_tokens,
                "EncoderDecoderTransformer.pad_value": tokenizer.pad_token,
            },
        )
        self.non_causal_model = self._prepare_model(self.non_causal_model)

        self.mle_model = load_gen_model_from_state_dict(
            ONLINE_MODEL_PATH,
            DecoderTransformer,
            compile=False,
            override_args={
                "DecoderTransformer.num_tokens": tokenizer.num_tokens,
                "DecoderTransformer.pad_value": tokenizer.pad_token,
            },
        )

        self.models = {}
        for model_name, model_path in MODEL_PATHS.items():

            if ".pth" not in os.path.basename(model_path):
                model = load_gen_model_from_state_dict(
                    model_path,
                    DecoderTransformer,
                    compile=False,
                    override_args={
                        "DecoderTransformer.num_tokens": tokenizer.num_tokens,
                        "DecoderTransformer.pad_value": tokenizer.pad_token,
                    },
                )

            else:
                model = load_rl_model(
                    model_path,
                    copy.deepcopy(self.mle_model),
                    compile=False,
                )

            model = self._prepare_model(model)

            if self.onnx:
                model = self._prepare_onnx_model(provider, model_path)
            self.models[model_name] = model

        # Prepare MLE model after all models are loaded because
        # loading RL model needs un-compiled MLE model
        self.mle_model = self._prepare_model(self.mle_model)

        # Define filter functions to avoid invalid tokens during generation
        self.online_model_filter_fn = partial(
            filter_invalid_tokens_generate_unconditional,
            model_part="chord",
            tokenizer=self.tokenizer,
            filter_opposite_part=True,
        )
        self.online_model_filter_kwargs = dict()

        self.online_model_commit_filter_fn = partial(
            filter_invalid_tokens_generate_online,
            model_part="melody",
            tokenizer=self.tokenizer,
        )
        self.online_model_commit_filter_kwargs = dict()

        self.non_causal_model_filter_fn = partial(
            filter_invalid_tokens_generate_single_part,
            model_part="chord",
            tokenizer=self.tokenizer,
        )
        self.non_causal_model_filter_kwargs = dict()

    def _prepare_onnx_model(self, provider, model_path):
        logging.info(f"===========Preparing model for ONNX...===========")
        save_dir = os.path.dirname(model_path)
        save_dir = os.path.join(save_dir, "onnx")
        onnx_init, onnx_step = prepare_online(
            self.mle_model,
            self.tokenizer,
            save_dir=save_dir,
            model_name="onnx_online",
            max_gen_seq_len=self.max_frames * 2,
            provider=provider,
        )
        logging.info(f"===========Created ONNX model===========")
        return onnx_init, onnx_step

    def get_models(self) -> List[str]:
        return list(self.models.keys())

    def melody_to_frame_tokens(
        self, melody_data: List[NoteInfo], end_frame: int = 0
    ) -> List[int]:
        """Given melody, convert to homophony and return frame-based tokens.

        Args:
          melody_data: list of note on/off events from frontend
          end_frame: frame to pad till or cut off at (if > 0)

        Returns:
          list of note tokens (rest/onset/hold), one for each frame
        """

        in_progress = {}
        notes = []
        for event in melody_data:
            if event["on"]:
                # Add note to in-progress notes
                in_progress[event["pitch"]] = event["frame"]
            elif event["pitch"] in in_progress:
                # Remove note from in-progress notes
                note_start_frame = in_progress.pop(event["pitch"])
                # Make sure this note's end frame doesn't exceed the overall end frame
                if end_frame > 0:
                    note_end_frame = min(end_frame, event["frame"])
                else:
                    note_end_frame = event["frame"]

                # Calculate other fields, then add note to notes list
                notes.append(
                    {
                        "onset": note_start_frame / FRAME_PER_BEAT,
                        "offset": note_end_frame / FRAME_PER_BEAT,
                        "octave": ((event["pitch"] - ZERO_OCTAVE) // 12),
                        "pitch_class": (event["pitch"] % 12),
                    }
                )

        # Truncate any notes extending past end_frame
        for pitch, note_start_frame in in_progress.items():
            if end_frame > 0:
                notes.append(
                    {
                        "onset": note_start_frame / FRAME_PER_BEAT,
                        "offset": end_frame / FRAME_PER_BEAT,
                        "octave": (pitch - ZERO_OCTAVE) // 12,
                        "pitch_class": pitch % 12,
                    }
                )

        # Determine the total sequence duration
        if end_frame > 0:
            duration = end_frame
        else:
            # If no end frame passed in, calculate it from the notes
            duration = notes[-1]["offset"] * FRAME_PER_BEAT if notes else 0

        tokens = self.tokenizer.melody_to_frames(notes, duration)
        return tokens

    def decode_chord_token(self, chord_token: int) -> ChordInfo:
        """Extract the underlying information from a given chord token.

        Args:
          chord_token: the chord token to decode

        Returns:
          chord_symbol: string symbolic representation of chord
          chord_pitches: list of pitches for chord
          is_onset: if the chord is a hit or a hold
        """
        if self.tokenizer.is_silence(chord_token):
            return "", [], True
        is_onset = self.tokenizer.is_chord_on(chord_token)
        token_name = self.tokenizer.id_to_name[chord_token]
        if is_onset:
            chord_symbol = token_name.replace("CHORD_ON_", "")
        else:
            chord_symbol = token_name.replace("CHORD_", "")
        try:
            chord_pitches = note_seq.chord_symbol_pitches(chord_symbol)
            bass_pitch = note_seq.chord_symbol_bass(chord_symbol)
            pitches = [
                *[CHORD_OCTAVE * 12 + pitch for pitch in chord_pitches],
                BASS_OCTAVE * 12 + bass_pitch,
            ]
        except (note_seq.ChordSymbolError, TypeError) as e:
            logging.warning(
                "Failed to convert chord symbol %s to pitches: %s",
                chord_symbol,
                e,
            )
            chord_symbol = ""
            pitches = []
        return chord_symbol, pitches, is_onset

    def _prepare_model(self, model):
        """Prepare model for inference.
        Args:
          model: the model to prepare
        """
        model.eval()
        model.to(self.device)
        if self.compile:
            model = torch.compile(model)
        return model

    def gen_online_model(
        self,
        model: torch.nn.Module,
        prompt: torch.Tensor,
        seq_len: int,
        temperature: float,
    ) -> torch.Tensor:
        """Generate tokens using the online model.

        Args:
            model: The model to use for generation
            prompt: The input prompt tensor
            seq_len: Number of new tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated tokens tensor
        """
        if self.onnx:
            # Use ONNX implementation
            onnx_init, onnx_step = model
            gen_tokens_np, _ = generate_tokens(
                onnx_init,
                onnx_step,
                prompt,
                temperature=temperature,
                n_new=seq_len,
                filter_logits_fn=self.online_model_filter_fn,
                filter_kwargs=self.online_model_filter_kwargs,
            )
            return torch.tensor(gen_tokens_np, device=prompt.device)
        else:
            return model.generate(
                prompts=prompt,
                seq_len=seq_len,
                temperature=temperature,
                cache_kv=True,
                filter_logits_fn=self.online_model_filter_fn,
                filter_kwargs=self.online_model_filter_kwargs,
            )

    def gen_commit_online_model_postfill(
        self,
        model: torch.nn.Module,
        prompt: torch.Tensor,
        chord_tokens: List[int],
        note_token_hist: List[int],
        chord_token_hist: List[int],
        commit_frames: int,
        frame: int,
        temperature: float,
    ) -> torch.Tensor:
        """Generate melody frames for committed period using the online model.
        We implement this by *post-filling* the committed chord tokens.

        Here we are generating for the period that already got commited,
            so the chord tokens cannot change.

        We are:
        - Condition on melody and previously generated chord tokens.
        - Generate lookahead chord & melody frames into the future.
        - Force chords to not change by filling in the committed chord tokens.
        - Effectively generating the imaginated melody frames that will be played.
        - But we are not conditioned on the correct chord tokens,
            so this is just an approximation.
        """
        if self.onnx:
            prefix = "[ONNX Online model commit]"
        else:
            prefix = "[Online model commit]"

        logging.info(
            f"{prefix} len(chord_tokens): %s",
            len(chord_tokens),
        )
        logging.info(
            f"{prefix} len(note_token_hist): %s",
            len(note_token_hist),
        )
        logging.info(
            f"{prefix} len(chord_token_hist): %s",
            len(chord_token_hist),
        )
        logging.info(
            f"{prefix} Generating commit frames: %s",
            commit_frames,
        )
        logging.info(f"{prefix} Prompt: %s", prompt)

        start_time = time.time()
        generated_commit = self.gen_online_model(
            model=model,
            prompt=prompt,
            seq_len=commit_frames * 2,
            temperature=temperature,
        )
        end_time = time.time()

        logging.info(
            f"{prefix} Time taken: %s",
            end_time - start_time,
        )
        logging.info(f"{prefix} Generated: %s", generated_commit)
        generated_commit = generated_commit.cpu().numpy()
        generated_commit[0, 0::2] = chord_tokens[frame : frame + commit_frames]
        generated_commit_tensor = torch.tensor(
            generated_commit, device=prompt.device
        )
        logging.info(
            f"{prefix} Generated after chord fill in: %s",
            generated_commit_tensor,
        )
        prompt = torch.cat([prompt, generated_commit_tensor], dim=1)
        return prompt

    def gen_commit_online_model_teacher_forcing(
        self,
        model: torch.nn.Module,
        prompt: torch.Tensor,
        chord_tokens: List[int],
        note_token_hist: List[int],
        chord_token_hist: List[int],
        commit_frames: int,
        frame: int,
        temperature: float,
    ) -> torch.Tensor:
        """Generate melody frames for committed period using the online model.
        We implement this by *teacher-forcing* the committed chord tokens.

        Here we are generating for the period that already got commited,
            so the chord tokens cannot change.

        We are:
        - Condition on melody and previously generated chord tokens.
        - Generate lookahead chord & melody frames into the future.
        - Force chords to not change by filling in the committed chord tokens *at each step*.
        - Effectively generating the imaginated melody frames that will be played *at each step*.

        in this case it actually becomes a "delay of 1" model:
            It always knows what the other part (chord) will play in present time
            and predict itself (melody) at the next time step.
        """
        if self.onnx:
            prefix = "[ONNX Online model commit]"
        else:
            prefix = "[Online model commit]"
        logging.info(
            f"{prefix} len(chord_tokens): %s",
            len(chord_tokens),
        )
        logging.info(
            f"{prefix} len(note_token_hist): %s",
            len(note_token_hist),
        )
        logging.info(
            f"{prefix} len(chord_token_hist): %s",
            len(chord_token_hist),
        )
        logging.info(
            f"{prefix} Generating commit frames: %s",
            commit_frames,
        )
        logging.info(f"{prefix} Prompt: %s", prompt)
        conditions = torch.tensor(
            [chord_tokens[frame : frame + commit_frames]],
            device=prompt.device,
        )

        logging.info(f"{prefix} Conditions: %s", conditions)

        start_time = time.time()
        if self.onnx:
            # Use ONNX implementation
            # Note: The ONNX branch uses the function `generate_tokens_online` with the parameter `prompt`.
            # This differs from the PyTorch branch, which uses `generate_online` with the parameter `prompts`.
            onnx_init, onnx_step = model
            prompt_np = prompt.cpu().numpy()
            gen_commit_np, _ = generate_tokens_online(
                onnx_init,
                onnx_step,
                prompt,
                conditions,
                n_new=commit_frames * 2,
                temperature=temperature,
                filter_logits_fn=self.online_model_commit_filter_fn,
                filter_kwargs=self.online_model_commit_filter_kwargs,
                reverse_condition_order=True,
            )
            generated_commit = torch.tensor(gen_commit_np, device=prompt.device)
        else:
            # prompt = [bos, chord, note, chord, note, ..., chord, note]
            # conditions = [chord1, chord2, ..., chordn]
            # generated_commit = [chord1, note1, chord2, note2, ..., chordn, noten]
            generated_commit = model.generate_online(
                prompts=prompt,
                conditions=conditions,
                seq_len=commit_frames * 2,
                temperature=temperature,
                cache_kv=True,
                filter_logits_fn=self.online_model_commit_filter_fn,
                filter_kwargs=self.online_model_commit_filter_kwargs,
                reverse_condition_order=True,
            )

        end_time = time.time()

        logging.info(
            f"{prefix} Time taken: %s",
            end_time - start_time,
        )
        logging.info(f"{prefix} Generated: %s", generated_commit)

        if not self.onnx:
            generated_commit = generated_commit.cpu().numpy()
            generated_commit_tensor = torch.tensor(
                generated_commit, device=prompt.device
            )
        else:
            generated_commit_tensor = generated_commit

        prompt = torch.cat([prompt, generated_commit_tensor], dim=1)
        return prompt

    def generate_live(
        self,
        model_name: str,
        notes: List[NoteInfo],
        chord_tokens: List[int],
        frame: int,
        lookahead: int,
        commitahead: int,
        temperature: float,
        silence_till: int,
        intro_set: bool,
    ) -> Tuple[List[ChordInfo], List[int], Optional[List[int]]]:
        """Generate chords for a given frame.

        - Condition on melody and previously generated chord tokens.
        - Generate lookahead frames into the future.
        - Force chords to not change during commitahead from previous generation.
        - Generate predicted intro chords before silence cutoff.

        Args:
          model_name: model to use, from MODELS keys
          notes: list of all note events in session
          chord_tokens: list of all chord tokens in session
          frame: frame to start generating at
          lookahead: how many frames into the future to generate
          commitahead: how many frames to leave chords unchanged since last call
          temperature: model sampling temperature
          silence_till: frames before generating with online model
          intro_set: if intro chords have been filled in by offline model

        Returns:
          new_chords: list of ChordInfo, one for each frame in lookahead
          new_chord_tokens: list of chord tokens, one for each frame in lookahead
          intro_chord_tokens: list of chord tokens to fill in session beginning,
            only returned when first generated right before silence cutoff frame
        """
        # Wait until at most 4 frames before start to have enough context
        gen_start_frame = silence_till - min(lookahead, 4)
        if frame < gen_start_frame or frame == 0:
            logging.info(
                "Waiting for more user input before generating; frame: %s, lookahead:"
                " %s, silence_till: %s",
                frame,
                lookahead,
                silence_till,
            )
            return [], [], None

        logging.info("[Request] Generate chords for frame %s", frame)
        logging.info("[Request] Model name: %s", model_name)
        logging.info("[Request] Notes: %s", notes)
        logging.info("[Request] Chord tokens: %s", chord_tokens)
        logging.info("[Request] Lookahead: %s", lookahead)
        logging.info("[Request] Commitahead: %s", commitahead)
        logging.info("[Request] Temperature: %s", temperature)
        logging.info("[Request] Silence till: %s", silence_till)

        model = self.models[model_name]

        # Convert notes to frame token format
        note_token_hist = self.melody_to_frame_tokens(notes, frame)
        note_token_hist = note_token_hist.tolist()

        # Use non-causal model to generate likely chords for introduction section
        intro_chord_tokens = None
        if not intro_set:
            note_tokens_bos_eos = (
                [self.tokenizer.bos_token]
                + note_token_hist
                + [self.tokenizer.eos_token]
            )
            enc_inputs = (
                torch.tensor(note_tokens_bos_eos, dtype=torch.long)
                .unsqueeze(0)
                .to(self.device)
            )
            gen_inputs = torch.full(
                (enc_inputs.shape[0], 1),
                self.tokenizer.bos_token,
                dtype=torch.long,
                device=self.device,
            )

            start_time = time.time()
            intro_chord_tokens = self.non_causal_model.generate(
                seq_in=enc_inputs,
                seq_out_start=gen_inputs,
                seq_len=frame,
                cache_kv=True,
                filter_logits_fn=self.non_causal_model_filter_fn,
                filter_kwargs=self.non_causal_model_filter_kwargs,
            )
            end_time = time.time()

            intro_chord_tokens = (
                intro_chord_tokens.squeeze(0).detach().cpu().tolist()
            )

            chord_tokens[:frame] = intro_chord_tokens
            logging.info("[Offline model] Input: %s", enc_inputs)
            logging.info("[Offline model] Output: %s", intro_chord_tokens)
            logging.info(
                "[Offline model] Time taken: %s", end_time - start_time
            )
        # Fill any gaps in chord tokens with rests
        # If there are -1 values in chord_tokens, fill them with SILENCE
        # If chord_tokens length is less than frame, pad with SILENCE
        for i in range(len(chord_tokens)):
            if chord_tokens[i] == -1:
                chord_tokens[i] = self.tokenizer.name_to_id["SILENCE"]
        if len(chord_tokens) < frame:
            chord_tokens.extend(
                [self.tokenizer.name_to_id["SILENCE"]]
                * (frame - len(chord_tokens))
            )

        # Create initial prompt (up to target frame)
        # If chord_tokens length is greater than frame, take the first frame tokens
        # If chord_tokens length is less than frame, take all chord_tokens
        chord_token_hist = chord_tokens[:frame]
        # Trim beginning of context to avoid surpassing max length
        max_context_len = self.max_frames - lookahead
        note_token_hist = note_token_hist[-max_context_len:]
        chord_token_hist = chord_token_hist[-max_context_len:]

        if self.tokenizer.is_note_hold(note_token_hist[0]):
            note_token_hist[0] = self.tokenizer.note_hold_id_to_note_id(
                note_token_hist[0]
            )

        if self.tokenizer.is_chord_hold(chord_token_hist[0]):
            chord_token_hist[0] = self.tokenizer.chord_hold_id_to_chord_id(
                chord_token_hist[0]
            )

        # Create interleaved sequence with BOS at start, EOS at end
        interleaved_tokens = [self.tokenizer.bos_token]
        for i in range(min(len(chord_token_hist), len(note_token_hist))):
            interleaved_tokens.extend([chord_token_hist[i], note_token_hist[i]])

        prompt = torch.tensor(
            [interleaved_tokens], dtype=torch.long, device=self.device
        )

        start_idx = prompt.shape[1]
        new_chord_tokens = None
        # First predict up to commit point to get predicted user notes
        # chord_tokens represents the history of generated chords + future chords from the frontend
        # If frame > length of chord_tokens, it means the current frame (time of this frontend message)
        # has exceeded previously generated chords, indicating a missed generation period
        # Therefore, if we reduce commitahead or commit_frames
        commit_frames = len(chord_tokens) - lookahead + commitahead - frame
        commit_frames = max(commit_frames, 0)  # If frame is past commit point
        logging.debug("commit_frames: %s", commit_frames)
        logging.debug("lookahead: %s", lookahead)
        if commit_frames >= lookahead:
            # Can just look up new tokens in cache if lookahead was decreased by user
            # Otherwise will try to generate <= 0 tokens and will break model api
            new_chord_tokens = np.array(
                chord_tokens[frame : frame + min(commit_frames, lookahead)]
            )
        else:
            if commit_frames > 0:
                prompt = self.gen_commit_online_model_teacher_forcing(
                    model,
                    prompt,
                    chord_tokens,
                    note_token_hist,
                    chord_token_hist,
                    commit_frames,
                    frame,
                    temperature,
                )
            # -1 because we don't need to generate the last melody token
            # So here we are saving some time by not generating the last token
            seq_len = lookahead * 2 - commit_frames * 2 - 1

            if self.onnx:
                prefix = "[ONNX Online model]"
            else:
                prefix = "[Online model]"

            logging.info(
                f"{prefix} Generating seq_len: %s, prompt: %s",
                seq_len,
                prompt,
            )
            start_time = time.time()
            generated_lookahead = self.gen_online_model(
                model=model,
                prompt=prompt,
                seq_len=seq_len,
                temperature=temperature,
            )
            end_time = time.time()
            logging.info(f"{prefix} Time taken: %s", end_time - start_time)
            logging.info(f"{prefix} Generated: %s", generated_lookahead)

            if self.onnx:
                prompt_np = prompt.cpu().numpy()
                all_chord_tokens_np = np.concatenate(
                    [prompt_np, generated_lookahead.cpu().numpy()], axis=1
                )
                new_chord_tokens = all_chord_tokens_np[0, start_idx::2]
            else:
                all_chord_tokens = torch.cat(
                    [prompt, generated_lookahead], dim=1
                )
                new_chord_tokens = (
                    all_chord_tokens[0, start_idx::2].cpu().numpy()
                )

        # Decode and return new chord tokens
        new_chords = [
            self.decode_chord_token(chord_tok) for chord_tok in new_chord_tokens
        ]

        logging.info("[Response] Chords: %s", new_chords)
        logging.info("[Response] Chord tokens: %s", new_chord_tokens.tolist())
        logging.info("[Response] Intro chord tokens: %s", intro_chord_tokens)
        logging.info("[Response] Frame: %s", frame)
        return new_chords, new_chord_tokens.tolist(), intro_chord_tokens
