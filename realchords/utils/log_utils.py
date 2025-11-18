"""Utility functions for logging and visualization."""

import os
import subprocess
import tempfile

import note_seq
import librosa
import numpy as np
from bokeh.io.export import export_png
from PIL import Image

from realchords.constants import SF2_PATH, MIDI_SYNTH_SR


def play_midi_with_soundfont(pretty_midi_obj, sf2_path=SF2_PATH):
    # Save PrettyMIDI object to a temporary MIDI file
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as temp_file:
        midi_path = temp_file.name
    pretty_midi_obj.write(midi_path)

    # Convert the MIDI to audio using the FluidSynth CLI
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        audio_path = temp_file.name

    # Command to invoke fluidsynth and generate audio from MIDI with the specified SoundFont
    fluidsynth_cmd = [
        "fluidsynth",
        "-ni",  # Non-interactive mode
        sf2_path,  # Path to the SoundFont
        midi_path,  # Path to the MIDI file
        "-F",
        audio_path,  # Output audio file path
        "-r",
        f"{MIDI_SYNTH_SR}",  # Sample rate (can adjust if needed)
    ]

    # Run the FluidSynth command
    subprocess.run(fluidsynth_cmd, check=True)

    # Load and play the audio in Jupyter Notebook
    audio, _ = librosa.load(audio_path, sr=None)
    return audio


def bokeh_fig_to_image(fig):
    """Convert a Bokeh figure to an RGB array."""
    # Export the figure to a PNG file
    # Use tempfile to create a temporary file for the PNG
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        temp_filename = temp_file.name  # Get the name of the temporary file
    export_png(fig, filename=temp_filename)

    # Convert the PNG to an RGB array
    image = Image.open(temp_filename)
    rgb_array = np.array(image.convert("RGB"))
    os.remove(temp_filename)  # Remove the temporary file
    return rgb_array


def midi_to_image(midi):
    sq = note_seq.midi_io.midi_to_note_sequence(midi)
    fig = note_seq.plot_sequence(sq, show_figure=False)
    return bokeh_fig_to_image(fig)


def midi_to_audio_image(midi, sf2_path=SF2_PATH):
    """Render a pretty MIDI object as audio and pianoroll for logging."""
    # Play the MIDI file
    audio = play_midi_with_soundfont(midi, sf2_path=sf2_path)
    image = midi_to_image(midi)
    return audio, image
