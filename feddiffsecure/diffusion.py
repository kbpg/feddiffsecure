from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class DiffusionConfig:
    timesteps: int = 100
    sample_steps: int = 30
    sampler: str = "ddpm"
    ddim_eta: float = 0.0
    clip_denoised: bool = True
    x0_loss_weight: float = 0.0
    beta_schedule: str = "linear"


def resolve_effective_sampler(sampler: str, sample_steps: int, timesteps: int) -> str:
    sampler = str(sampler).lower()
    if sampler not in {"ddpm", "ddim"}:
        raise ValueError(f"Unsupported sampler: {sampler}")
    if sampler == "ddpm" and int(sample_steps) < int(timesteps):
        return "ddim"
    return sampler


class GaussianDiffusion:
    def __init__(self, timesteps: int = 100, beta_schedule: str = "linear") -> None:
        self.timesteps = timesteps
        self.beta_schedule = str(beta_schedule).lower()
        betas = self._build_beta_schedule(timesteps=self.timesteps, schedule=self.beta_schedule)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]], dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_mean_coef2 = torch.sqrt(alphas) * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    def _build_beta_schedule(self, timesteps: int, schedule: str) -> torch.Tensor:
        schedule = str(schedule).lower()
        if schedule == "linear":
            return torch.linspace(1e-4, 0.02, timesteps, dtype=torch.float32)
        if schedule == "cosine":
            steps = torch.arange(timesteps + 1, dtype=torch.float64)
            s = 0.008
            x = (steps / timesteps + s) / (1.0 + s)
            alphas_cumprod = torch.cos(x * torch.pi * 0.5).pow(2)
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return betas.clamp(1e-6, 0.999).to(dtype=torch.float32)
        raise ValueError(f"Unsupported beta schedule: {schedule}")

    def to(self, device: str) -> "GaussianDiffusion":
        for name, value in vars(self).items():
            if torch.is_tensor(value):
                setattr(self, name, value.to(device))
        return self

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_alpha * x_start + sqrt_one_minus * noise

    def p_losses(
        self,
        model: torch.nn.Module,
        x_start: torch.Tensor,
        t: torch.Tensor,
        x0_loss_weight: float = 0.0,
    ) -> torch.Tensor:
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = model(x_noisy, t)
        loss = F.mse_loss(predicted_noise, noise)
        if float(x0_loss_weight) > 0.0:
            predicted_x0 = self._predict_x0_from_noise(x_noisy, t, predicted_noise)
            loss = loss + F.mse_loss(predicted_x0, x_start) * float(x0_loss_weight)
        return loss

    def _predict_x0_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, pred_noise: torch.Tensor) -> torch.Tensor:
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return (x_t - sqrt_one_minus * pred_noise) / torch.clamp(sqrt_alpha_bar, min=1e-8)

    def _posterior_mean_variance(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coef1 = self.posterior_mean_coef1[t][:, None, None, None]
        coef2 = self.posterior_mean_coef2[t][:, None, None, None]
        mean = coef1 * x_start + coef2 * x_t
        var = self.posterior_variance[t][:, None, None, None]
        return mean, var

    def _make_schedule(self, sample_steps: int) -> list[int]:
        sample_steps = max(1, min(int(sample_steps), self.timesteps))
        if sample_steps >= self.timesteps:
            return list(range(self.timesteps - 1, -1, -1))

        raw = torch.linspace(self.timesteps - 1, 0, sample_steps).round().long().tolist()
        schedule: list[int] = []
        for value in raw:
            current = int(value)
            if not schedule or schedule[-1] != current:
                schedule.append(current)
        if schedule[-1] != 0:
            schedule.append(0)
        return schedule

    def _should_capture(
        self,
        timestep: int,
        capture_every: int,
        capture_steps: set[int] | None,
    ) -> bool:
        if capture_steps is not None and int(timestep) in capture_steps:
            return True
        if int(capture_every) > 0 and (int(timestep) == 0 or int(timestep) % int(capture_every) == 0):
            return True
        return False

    def _record_capture(
        self,
        captures: list[dict[str, torch.Tensor]] | None,
        timestep: int,
        x: torch.Tensor,
    ) -> None:
        if captures is None:
            return
        captures.append({"timestep": int(timestep), "samples": x.detach().cpu().clone()})

    @torch.no_grad()
    def _sample_ddpm(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        clip_denoised: bool,
        captures: list[dict[str, torch.Tensor]] | None = None,
        capture_every: int = 0,
        capture_steps: set[int] | None = None,
    ) -> torch.Tensor:
        batch_size = x.size(0)
        device = x.device

        for timestep in range(self.timesteps - 1, -1, -1):
            t = torch.full((batch_size,), timestep, device=device, dtype=torch.long)
            pred_noise = model(x, t)
            x_start = self._predict_x0_from_noise(x, t, pred_noise)
            if clip_denoised:
                x_start = x_start.clamp(-1.0, 1.0)

            model_mean, posterior_var = self._posterior_mean_variance(x_start, x, t)
            if timestep == 0:
                x = x_start if clip_denoised else model_mean
            else:
                noise = torch.randn_like(x)
                x = model_mean + torch.sqrt(torch.clamp(posterior_var, min=1e-20)) * noise
            if self._should_capture(timestep=timestep, capture_every=capture_every, capture_steps=capture_steps):
                self._record_capture(captures, timestep=timestep, x=x)

        return x

    @torch.no_grad()
    def _sample_ddim(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        sample_steps: int,
        eta: float,
        clip_denoised: bool,
        captures: list[dict[str, torch.Tensor]] | None = None,
        capture_every: int = 0,
        capture_steps: set[int] | None = None,
    ) -> torch.Tensor:
        batch_size = x.size(0)
        device = x.device
        schedule = self._make_schedule(sample_steps)

        for idx, timestep in enumerate(schedule):
            next_timestep = schedule[idx + 1] if idx + 1 < len(schedule) else -1
            t = torch.full((batch_size,), timestep, device=device, dtype=torch.long)

            pred_noise = model(x, t)
            x_start = self._predict_x0_from_noise(x, t, pred_noise)
            if clip_denoised:
                x_start = x_start.clamp(-1.0, 1.0)

            if next_timestep < 0:
                x = x_start
                if self._should_capture(timestep=timestep, capture_every=capture_every, capture_steps=capture_steps):
                    self._record_capture(captures, timestep=timestep, x=x)
                continue

            alpha_bar_t = self.alphas_cumprod[t][:, None, None, None]
            next_t = torch.full((batch_size,), next_timestep, device=device, dtype=torch.long)
            alpha_bar_next = self.alphas_cumprod[next_t][:, None, None, None]

            sigma = eta * torch.sqrt((1.0 - alpha_bar_next) / (1.0 - alpha_bar_t))
            sigma = sigma * torch.sqrt(torch.clamp(1.0 - alpha_bar_t / alpha_bar_next, min=0.0))
            direction = torch.sqrt(torch.clamp(1.0 - alpha_bar_next - sigma * sigma, min=0.0)) * pred_noise
            x = torch.sqrt(alpha_bar_next) * x_start + direction
            if eta > 0.0:
                x = x + sigma * torch.randn_like(x)
            if self._should_capture(timestep=timestep, capture_every=capture_every, capture_steps=capture_steps):
                self._record_capture(captures, timestep=timestep, x=x)

        return x

    @torch.no_grad()
    def sample(
        self,
        model: torch.nn.Module,
        shape: tuple[int, int, int, int],
        device: str,
        sample_steps: int | None = None,
        sampler: str = "ddpm",
        ddim_eta: float = 0.0,
        clip_denoised: bool = True,
        initial_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        model.eval()
        sample_steps = sample_steps or self.timesteps
        sampler = resolve_effective_sampler(
            sampler=sampler,
            sample_steps=int(sample_steps),
            timesteps=int(self.timesteps),
        )
        x = torch.randn(shape, device=device) if initial_noise is None else initial_noise.to(device).clone()

        if sampler == "ddpm":
            x = self._sample_ddpm(model=model, x=x, clip_denoised=clip_denoised)
        else:
            x = self._sample_ddim(
                model=model,
                x=x,
                sample_steps=int(sample_steps),
                eta=float(ddim_eta),
                clip_denoised=clip_denoised,
            )

        model.train()
        return torch.clamp(x, -1.0, 1.0)

    @torch.no_grad()
    def sample_with_trajectory(
        self,
        model: torch.nn.Module,
        shape: tuple[int, int, int, int],
        device: str,
        sample_steps: int | None = None,
        sampler: str = "ddpm",
        ddim_eta: float = 0.0,
        clip_denoised: bool = True,
        initial_noise: torch.Tensor | None = None,
        capture_every: int = 0,
        capture_steps: list[int] | None = None,
        include_initial_noise: bool = True,
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        model.eval()
        sample_steps = sample_steps or self.timesteps
        sampler = resolve_effective_sampler(
            sampler=sampler,
            sample_steps=int(sample_steps),
            timesteps=int(self.timesteps),
        )
        x = torch.randn(shape, device=device) if initial_noise is None else initial_noise.to(device).clone()
        captures: list[dict[str, torch.Tensor]] = []
        capture_set = {int(step) for step in capture_steps} if capture_steps else None
        if include_initial_noise:
            self._record_capture(captures, timestep=self.timesteps, x=x)

        if sampler == "ddpm":
            x = self._sample_ddpm(
                model=model,
                x=x,
                clip_denoised=clip_denoised,
                captures=captures,
                capture_every=capture_every,
                capture_steps=capture_set,
            )
        else:
            x = self._sample_ddim(
                model=model,
                x=x,
                sample_steps=int(sample_steps),
                eta=float(ddim_eta),
                clip_denoised=clip_denoised,
                captures=captures,
                capture_every=capture_every,
                capture_steps=capture_set,
            )

        model.train()
        return torch.clamp(x, -1.0, 1.0), captures
