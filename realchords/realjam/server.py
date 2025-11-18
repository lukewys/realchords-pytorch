"""Serves frontend and model endpoints for genjam interface.

@Author Alex Scarlatos, Tia-Jane Fowler, Yusong Wu
"""

import json
import os
import flask
import argparse

from realchords.realjam import agent_interface

DEFAULT_PORT = 8080

base_dir = os.path.dirname(os.path.abspath(__file__))
frontend_dir = os.path.join(base_dir, "frontend")

app = flask.Flask(__name__, static_url_path="", static_folder=frontend_dir)

agent: agent_interface.Agent = None


@app.get("/")
def index() -> str:
    """Get index page."""
    return flask.send_file(os.path.join(frontend_dir, "index.html"))


@app.get("/models")
def get_models() -> str:
    """Get available model names."""
    assert agent is not None
    return json.dumps(agent.get_models())


@app.post("/play")
def play() -> str:
    """Generate new chords given context."""
    assert agent is not None
    payload = flask.request.get_json()
    new_chords, new_chord_tokens, intro_chord_tokens = agent.generate_live(
        payload["model"],
        payload["notes"],
        payload["chordTokens"],
        payload["frame"],
        payload["lookahead"],
        payload["commitahead"],
        float(payload["temperature"]),
        payload["silenceTill"],
        payload["introSet"],
    )
    return json.dumps(
        {
            "newChords": new_chords,
            "newChordTokens": new_chord_tokens,
            "introChordTokens": intro_chord_tokens,
            "frame": payload["frame"],
        }
    )


def main() -> None:
    global agent

    parser = argparse.ArgumentParser(description="Run the RealJam server")
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port to run the server on (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--ssl", action="store_true", help="Enable SSL with adhoc certificate"
    )
    parser.add_argument(
        "--onnx", action="store_true", help="Use ONNX model instead of PyTorch"
    )
    parser.add_argument(
        "--onnx_provider",
        type=str,
        default=None,
        help=(
            "Execution provider for ONNX Runtime. "
            "If omitted, the script selects `CUDAExecutionProvider` when CUDA is "
            "available, otherwise `CPUExecutionProvider`. "
            "Use this flag to override the default—for example: "
            "`--onnx_provider CUDAExecutionProvider` or "
            "`--onnx_provider CPUExecutionProvider`."
        ),
    )
    args = parser.parse_args()
    agent = agent_interface.Agent(onnx=args.onnx, provider=args.onnx_provider)

    ssl_context = "adhoc" if args.ssl else None
    app.run(
        host="0.0.0.0", port=args.port, ssl_context=ssl_context, threaded=False
    )


if __name__ == "__main__":
    main()
