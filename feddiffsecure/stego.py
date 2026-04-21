from __future__ import annotations

import hashlib
import io
import math
import zlib
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from PIL import Image


def _keystream(key: str, length: int) -> bytes:
    output = bytearray()
    counter = 0
    while len(output) < length:
        block = hashlib.sha256(f"{key}:{counter}".encode("utf-8")).digest()
        output.extend(block)
        counter += 1
    return bytes(output[:length])


def xor_bytes(data: bytes, key: str) -> bytes:
    stream = _keystream(key, len(data))
    return bytes(a ^ b for a, b in zip(data, stream))


def serialize_state_dict(state_dict: Dict[str, torch.Tensor]) -> bytes:
    cpu_state = {k: v.detach().cpu() for k, v in state_dict.items()}
    buffer = io.BytesIO()
    torch.save(cpu_state, buffer)
    return buffer.getvalue()


def deserialize_state_dict(blob: bytes) -> Dict[str, torch.Tensor]:
    buffer = io.BytesIO(blob)
    return torch.load(buffer, map_location="cpu", weights_only=False)


def _bytes_to_bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def _bits_to_bytes(bits: np.ndarray) -> bytes:
    return np.packbits(bits.astype(np.uint8)).tobytes()


def embed_payload_in_png(payload: bytes, output_path: str | Path, seed: int = 0) -> tuple[Path, dict]:
    compressed = zlib.compress(payload)
    header = len(compressed).to_bytes(4, "big")
    packet = header + compressed
    bits = _bytes_to_bits(packet)

    num_channels = 3
    total_values = int(math.ceil(len(bits) / 1.0))
    side = int(math.ceil(math.sqrt(total_values / num_channels)))
    side = max(side, 32)

    rng = np.random.default_rng(seed)
    cover = rng.integers(0, 256, size=(side, side, num_channels), dtype=np.uint8)
    stego = cover.copy().reshape(-1)
    stego[: len(bits)] = (stego[: len(bits)] & 0xFE) | bits
    stego = stego.reshape(cover.shape)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(stego, mode="RGB").save(output_path)

    mse = float(np.mean((cover.astype(np.float32) - stego.astype(np.float32)) ** 2))
    psnr = 99.0 if mse == 0.0 else float(20 * np.log10(255.0 / np.sqrt(mse)))
    return output_path, {
        "payload_bytes_raw": len(payload),
        "payload_bytes_compressed": len(compressed),
        "carrier_shape": list(cover.shape),
        "mse": mse,
        "psnr": psnr,
    }


def extract_payload_from_png(path: str | Path) -> bytes:
    img = np.array(Image.open(path).convert("RGB"), dtype=np.uint8).reshape(-1)
    length_bits = img[:32] & 1
    length = int.from_bytes(_bits_to_bytes(length_bits), "big")
    payload_bits = img[32 : 32 + length * 8] & 1
    compressed = _bits_to_bytes(payload_bits)
    return zlib.decompress(compressed)


def package_state_as_stego_png(state_dict: Dict[str, torch.Tensor], output_path: str | Path, key: str, seed: int) -> tuple[Path, dict]:
    raw = serialize_state_dict(state_dict)
    encrypted = xor_bytes(raw, key)
    sha256 = hashlib.sha256(raw).hexdigest()
    stego_path, info = embed_payload_in_png(encrypted, output_path=output_path, seed=seed)
    info["raw_sha256"] = sha256
    return stego_path, info


def recover_state_from_stego_png(path: str | Path, key: str) -> tuple[Dict[str, torch.Tensor], dict]:
    encrypted = extract_payload_from_png(path)
    raw = xor_bytes(encrypted, key)
    sha256 = hashlib.sha256(raw).hexdigest()
    state = deserialize_state_dict(raw)
    return state, {"raw_sha256": sha256, "payload_bytes_raw": len(raw)}
