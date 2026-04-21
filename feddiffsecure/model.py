from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora import LoRAConv2d


def _group_count(num_channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, num_channels), 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1


def _expand_stage_depths(
    value: int | Sequence[int],
    expected_length: int,
    name: str,
) -> list[int]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        depths = [int(x) for x in value]
    else:
        depths = [int(value)] * expected_length
    if len(depths) != expected_length:
        raise ValueError(f"{name} must provide exactly {expected_length} entries")
    if any(depth < 1 for depth in depths):
        raise ValueError(f"{name} entries must all be >= 1")
    return depths


def _normalize_level_set(
    value: Sequence[int] | None,
    upper_bound: int,
    name: str,
    default_all: bool = False,
) -> set[int]:
    if value is None:
        return set(range(upper_bound)) if default_all else set()
    levels = {int(level) for level in value}
    invalid_levels = sorted(level for level in levels if level < 0 or level >= upper_bound)
    if invalid_levels:
        raise ValueError(f"{name} contains invalid levels: {invalid_levels}")
    return levels


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        emb = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None].float() * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, rank: int, alpha: float) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_ch), in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = LoRAConv2d(in_ch, out_ch, kernel_size=3, padding=1, rank=rank, alpha=alpha)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(_group_count(out_ch), out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = LoRAConv2d(out_ch, out_ch, kernel_size=3, padding=1, rank=rank, alpha=alpha)
        self.skip = (
            LoRAConv2d(in_ch, out_ch, kernel_size=1, rank=rank, alpha=alpha)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.skip(x)


class SpatialSelfAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        head_dim: int | None = None,
    ) -> None:
        super().__init__()
        heads = max(1, min(int(num_heads), int(channels)))
        if head_dim is None:
            while heads > 1 and channels % heads != 0:
                heads -= 1
            inner_dim = channels
            head_dim = inner_dim // heads
        else:
            head_dim = int(head_dim)
            inner_dim = heads * head_dim

        self.channels = int(channels)
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        self.inner_dim = int(inner_dim)
        self.scale = self.head_dim ** -0.5

        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.to_qkv = nn.Conv2d(channels, self.inner_dim * 3, kernel_size=1, bias=False)
        self.proj_out = nn.Conv2d(self.inner_dim, channels, kernel_size=1)

    def _reshape_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = tensor.shape
        return tensor.view(batch, self.heads, self.head_dim, height * width).permute(0, 1, 3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm(x)
        q, k, v = self.to_qkv(h).chunk(3, dim=1)
        q = self._reshape_heads(q)
        k = self._reshape_heads(k)
        v = self._reshape_heads(v)

        if hasattr(F, "scaled_dot_product_attention"):
            attn_out = F.scaled_dot_product_attention(q, k, v)
        else:
            attention = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attention = attention.softmax(dim=-1)
            attn_out = torch.matmul(attention, v)

        batch, _, height, width = x.shape
        attn_out = attn_out.permute(0, 1, 3, 2).reshape(batch, self.inner_dim, height, width)
        attn_out = self.proj_out(attn_out)
        return residual + attn_out


class LevelTransition(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        rank: int,
        alpha: float,
        downsample: bool = True,
    ) -> None:
        super().__init__()
        if not downsample and in_ch == out_ch:
            self.conv = nn.Identity()
        else:
            self.conv = LoRAConv2d(
                in_ch,
                out_ch,
                kernel_size=3,
                stride=2 if downsample else 1,
                padding=1,
                rank=rank,
                alpha=alpha,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, rank: int, alpha: float) -> None:
        super().__init__()
        self.conv = LoRAConv2d(in_ch, out_ch, kernel_size=3, padding=1, rank=rank, alpha=alpha)

    def forward(self, x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        x = F.interpolate(x, size=size, mode="nearest")
        return self.conv(x)


class TinyUNet(nn.Module):
    def __init__(
        self,
        channels: int = 1,
        base_channels: int = 32,
        lora_rank: int = 8,
        lora_alpha: float = 8.0,
    ) -> None:
        super().__init__()
        time_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.init_conv = LoRAConv2d(channels, base_channels, kernel_size=3, padding=1, rank=lora_rank, alpha=lora_alpha)
        self.down1 = ResBlock(base_channels, base_channels, time_dim, lora_rank, lora_alpha)
        self.down2 = ResBlock(base_channels, base_channels * 2, time_dim, lora_rank, lora_alpha)
        self.mid = ResBlock(base_channels * 2, base_channels * 2, time_dim, lora_rank, lora_alpha)
        self.up1 = ResBlock(base_channels * 3, base_channels, time_dim, lora_rank, lora_alpha)
        self.up2 = ResBlock(base_channels * 2, base_channels, time_dim, lora_rank, lora_alpha)
        self.final_norm = nn.GroupNorm(_group_count(base_channels), base_channels)
        self.final_act = nn.SiLU()
        self.final_conv = LoRAConv2d(base_channels, channels, kernel_size=1, rank=lora_rank, alpha=lora_alpha)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(time)

        x0 = self.init_conv(x)
        x1 = self.down1(x0, t_emb)
        x2 = self.down2(self.pool(x1), t_emb)
        xm = self.mid(x2, t_emb)

        xu = F.interpolate(xm, scale_factor=2.0, mode="nearest")
        xu = self.up1(torch.cat([xu, x1], dim=1), t_emb)
        xu = self.up2(torch.cat([xu, x0], dim=1), t_emb)
        out = self.final_conv(self.final_act(self.final_norm(xu)))
        return out


class UNet(nn.Module):
    def __init__(
        self,
        channels: int = 1,
        base_channels: int = 16,
        channel_multipliers: Sequence[int] = (1, 2, 4),
        blocks_per_level: int | Sequence[int] = 2,
        lora_rank: int = 8,
        lora_alpha: float = 8.0,
        encoder_blocks_per_level: Sequence[int] | None = None,
        decoder_blocks_per_level: Sequence[int] | None = None,
        mid_blocks: int = 2,
        downsample_levels: Sequence[int] | None = None,
        attention_levels: Sequence[int] = (),
        attention_heads: int = 4,
        attention_head_dim: int | None = None,
        mid_attention: bool = False,
    ) -> None:
        super().__init__()
        if len(channel_multipliers) < 2:
            raise ValueError("channel_multipliers must contain at least two levels")
        if isinstance(blocks_per_level, int) and blocks_per_level < 1:
            raise ValueError("blocks_per_level must be >= 1")
        if mid_blocks < 1:
            raise ValueError("mid_blocks must be >= 1")

        self.level_channels = [base_channels * int(mult) for mult in channel_multipliers]
        num_levels = len(self.level_channels)
        num_transitions = num_levels - 1
        self.attention_levels = _normalize_level_set(attention_levels, num_levels, "attention_levels")
        self.downsample_levels = _normalize_level_set(
            downsample_levels,
            num_transitions,
            "downsample_levels",
            default_all=True,
        )
        encoder_depths = _expand_stage_depths(
            encoder_blocks_per_level if encoder_blocks_per_level is not None else blocks_per_level,
            num_levels,
            "encoder_blocks_per_level",
        )
        decoder_depths = _expand_stage_depths(
            decoder_blocks_per_level if decoder_blocks_per_level is not None else blocks_per_level,
            num_transitions,
            "decoder_blocks_per_level",
        )
        time_dim = base_channels * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.init_conv = LoRAConv2d(
            channels,
            self.level_channels[0],
            kernel_size=3,
            padding=1,
            rank=lora_rank,
            alpha=lora_alpha,
        )

        self.encoder_blocks = nn.ModuleList()
        self.encoder_attentions = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        current_ch = self.level_channels[0]
        for level_idx, out_ch in enumerate(self.level_channels):
            blocks = nn.ModuleList()
            for block_idx in range(encoder_depths[level_idx]):
                block_in = current_ch if block_idx == 0 else out_ch
                blocks.append(ResBlock(block_in, out_ch, time_dim, lora_rank, lora_alpha))
            self.encoder_blocks.append(blocks)
            self.encoder_attentions.append(
                SpatialSelfAttention(out_ch, num_heads=attention_heads, head_dim=attention_head_dim)
                if level_idx in self.attention_levels
                else nn.Identity()
            )
            current_ch = out_ch

            if level_idx < len(self.level_channels) - 1:
                next_ch = self.level_channels[level_idx + 1]
                self.downsamples.append(
                    LevelTransition(
                        current_ch,
                        next_ch,
                        lora_rank,
                        lora_alpha,
                        downsample=level_idx in self.downsample_levels,
                    )
                )
                current_ch = next_ch

        self.mid_blocks = nn.ModuleList(
            [ResBlock(current_ch, current_ch, time_dim, lora_rank, lora_alpha) for _ in range(mid_blocks)]
        )
        self.mid_attention = (
            SpatialSelfAttention(current_ch, num_heads=attention_heads, head_dim=attention_head_dim)
            if mid_attention
            else nn.Identity()
        )

        self.upsamples = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.decoder_attentions = nn.ModuleList()
        for level_idx, skip_ch in zip(reversed(range(len(self.level_channels) - 1)), reversed(self.level_channels[:-1])):
            self.upsamples.append(Upsample(current_ch, skip_ch, lora_rank, lora_alpha))
            blocks = nn.ModuleList([ResBlock(skip_ch * 2, skip_ch, time_dim, lora_rank, lora_alpha)])
            for _ in range(decoder_depths[level_idx] - 1):
                blocks.append(ResBlock(skip_ch, skip_ch, time_dim, lora_rank, lora_alpha))
            self.decoder_blocks.append(blocks)
            self.decoder_attentions.append(
                SpatialSelfAttention(skip_ch, num_heads=attention_heads, head_dim=attention_head_dim)
                if level_idx in self.attention_levels
                else nn.Identity()
            )
            current_ch = skip_ch

        self.final_norm = nn.GroupNorm(_group_count(current_ch), current_ch)
        self.final_act = nn.SiLU()
        self.final_conv = LoRAConv2d(current_ch, channels, kernel_size=1, rank=lora_rank, alpha=lora_alpha)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(time)
        x = self.init_conv(x)

        skips: list[torch.Tensor] = []
        for level_idx, (blocks, attention) in enumerate(zip(self.encoder_blocks, self.encoder_attentions)):
            for block in blocks:
                x = block(x, t_emb)
            x = attention(x)
            skips.append(x)
            if level_idx < len(self.downsamples):
                x = self.downsamples[level_idx](x)

        for block in self.mid_blocks:
            x = block(x, t_emb)
        x = self.mid_attention(x)

        for upsample, blocks, attention, skip in zip(
            self.upsamples,
            self.decoder_blocks,
            self.decoder_attentions,
            reversed(skips[:-1]),
        ):
            x = upsample(x, size=skip.shape[-2:])
            x = torch.cat([x, skip], dim=1)
            for block in blocks:
                x = block(x, t_emb)
            x = attention(x)

        return self.final_conv(self.final_act(self.final_norm(x)))


def build_model(dataset_cfg: dict, model_cfg: dict) -> nn.Module:
    architecture = str(model_cfg.get("architecture", "tiny_unet")).lower()
    common_kwargs = {
        "channels": int(dataset_cfg["channels"]),
        "base_channels": int(model_cfg["base_channels"]),
        "lora_rank": int(model_cfg["lora_rank"]),
        "lora_alpha": float(model_cfg["lora_alpha"]),
    }

    if architecture in {"tiny", "tiny_unet"}:
        return TinyUNet(**common_kwargs)

    if architecture in {"unet", "large_unet", "larger_unet"}:
        return UNet(
            **common_kwargs,
            channel_multipliers=tuple(int(x) for x in model_cfg.get("channel_multipliers", [1, 2, 4])),
            blocks_per_level=model_cfg.get("blocks_per_level", 2),
            encoder_blocks_per_level=(
                tuple(int(x) for x in model_cfg["encoder_blocks_per_level"])
                if model_cfg.get("encoder_blocks_per_level") is not None
                else None
            ),
            decoder_blocks_per_level=(
                tuple(int(x) for x in model_cfg["decoder_blocks_per_level"])
                if model_cfg.get("decoder_blocks_per_level") is not None
                else None
            ),
            mid_blocks=int(model_cfg.get("mid_blocks", 2)),
            downsample_levels=(
                tuple(int(x) for x in model_cfg["downsample_levels"])
                if model_cfg.get("downsample_levels") is not None
                else None
            ),
            attention_levels=tuple(int(x) for x in model_cfg.get("attention_levels", [])),
            attention_heads=int(model_cfg.get("attention_heads", 4)),
            attention_head_dim=(
                int(model_cfg["attention_head_dim"])
                if model_cfg.get("attention_head_dim") is not None
                else None
            ),
            mid_attention=bool(model_cfg.get("mid_attention", False)),
        )

    raise ValueError(f"Unsupported model architecture: {architecture}")
