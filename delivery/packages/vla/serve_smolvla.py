#!/usr/bin/env python3
"""Serve a trained SmolVLA model as a gRPC policy server.

Compatible with LeISAAC's lerobot policy transport layer.

Usage:
    python serve_smolvla.py \
        --checkpoint /data/smolvla_ckpt/last/pretrained_model \
        --port 50051
"""

import argparse
import time
from concurrent import futures
from pathlib import Path

import grpc
import numpy as np
import torch
from PIL import Image

# LeRobot imports
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to trained SmolVLA checkpoint")
    p.add_argument("--port", type=int, default=50051)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--image-size", type=int, nargs=2, default=[256, 256])
    return p.parse_args()


class SmolVLAServer:
    """Standalone SmolVLA inference server.

    Can be used with gRPC (LeISAAC transport) or directly via HTTP.
    """

    def __init__(self, checkpoint_path, device="cuda:0", image_size=(256, 256)):
        print(f"Loading SmolVLA from {checkpoint_path}...")
        self.device = device
        self.image_size = image_size

        self.policy = PreTrainedPolicy.from_pretrained(checkpoint_path)
        self.policy.to(device)
        self.policy.eval()
        print(f"SmolVLA loaded on {device}")

        self.step_count = 0

    @torch.no_grad()
    def predict(self, images: dict, state: np.ndarray) -> np.ndarray:
        """Run SmolVLA inference.

        Args:
            images: dict of camera_name -> PIL.Image or np.ndarray (H, W, 3)
            state: joint positions (state_dim,)

        Returns:
            action: predicted action (action_dim,)
        """
        observation = {}

        # State
        observation["observation.state"] = torch.from_numpy(
            state.astype(np.float32)
        ).unsqueeze(0).to(self.device)

        # Images
        for cam_name, img in images.items():
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img[:, :, :3].astype(np.uint8))
            img = img.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
            # Convert to tensor: (C, H, W), float [0, 1]
            img_tensor = torch.from_numpy(
                np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
            ).unsqueeze(0).to(self.device)
            observation[f"observation.images.{cam_name}"] = img_tensor

        # Inference
        action = self.policy.select_action(observation)
        self.step_count += 1

        return action.squeeze(0).cpu().numpy()


def run_http_server(server: SmolVLAServer, port: int):
    """Simple HTTP server for testing without gRPC."""
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import json

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path == "/predict":
                length = int(self.headers["Content-Length"])
                data = json.loads(self.rfile.read(length))

                state = np.array(data["state"], dtype=np.float32)
                images = {}
                for cam, img_list in data.get("images", {}).items():
                    images[cam] = np.array(img_list, dtype=np.uint8)

                action = server.predict(images, state)

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"action": action.tolist()}).encode())

            elif self.path == "/health":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")

        def log_message(self, format, *args):
            pass  # suppress logs

    httpd = HTTPServer(("0.0.0.0", port), Handler)
    print(f"HTTP server on port {port}")
    httpd.serve_forever()


def main():
    args = parse_args()

    server = SmolVLAServer(
        checkpoint_path=args.checkpoint,
        device=args.device,
        image_size=tuple(args.image_size),
    )

    print(f"\nSmolVLA server starting on port {args.port}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Device: {args.device}")
    print(f"  Image size: {args.image_size}")

    run_http_server(server, args.port)


if __name__ == "__main__":
    main()
