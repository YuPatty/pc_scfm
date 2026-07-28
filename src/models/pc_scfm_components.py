import torch
import torch.nn as nn
import torch.nn.functional as F

class FlowBaselineHead(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.coeff_count = h.get("flow_coeff_count", 32)
        self.basis_type = h.get("flow_basis", "cosine")
        hidden = h.get("policy_hidden", 128)
        self.condition = nn.Sequential(
            nn.Linear(h.dense_channel, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.velocity = nn.Sequential(
            nn.Linear(hidden + self.coeff_count + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.coeff_count),
        )
        self.logvar = nn.Linear(hidden, self.coeff_count)

    def _basis(self, signal_len, device, dtype):
        if self.basis_type != "cosine":
            return None
        t = torch.linspace(0.0, 1.0, signal_len, device=device, dtype=dtype)
        basis = torch.stack([torch.cos(torch.pi * k * t) for k in range(self.coeff_count)], dim=0)
        return basis / basis.pow(2).sum(dim=-1, keepdim=True).clamp_min(1e-6).sqrt()

    def coeff_to_waveform(self, coeff, signal_len):
        basis = self._basis(signal_len, coeff.device, coeff.dtype)
        if basis is None:
            return F.interpolate(coeff.unsqueeze(1), size=signal_len, mode="linear", align_corners=False)
        return torch.matmul(coeff, basis).unsqueeze(1)

    def predict_velocity(self, z_tau, tau, condition):
        if tau.dim() == 1:
            tau = tau.unsqueeze(-1)
        return self.velocity(torch.cat([z_tau, tau, condition], dim=-1))

    def flow_matching_loss(self, encoded_feature, target_coeff):
        pooled = encoded_feature.mean(dim=(2, 3))
        condition = self.condition(pooled)
        z0 = torch.randn_like(target_coeff)
        tau = torch.rand((target_coeff.shape[0], 1), device=target_coeff.device)
        z_tau = (1.0 - tau) * z0 + tau * target_coeff
        target_velocity = target_coeff - z0
        pred_velocity = self.predict_velocity(z_tau, tau, condition)
        return F.mse_loss(pred_velocity, target_velocity)

    def forward(self, encoded_feature, signal_len, nfe=4, samples=4):
        pooled = encoded_feature.mean(dim=(2, 3))
        condition = self.condition(pooled)
        coeff_samples = []
        nfe = max(int(nfe), 1)
        samples = max(int(samples), 1)
        dt = 1.0 / nfe
        for _ in range(samples):
            z = torch.randn((pooled.shape[0], self.coeff_count), device=pooled.device, dtype=pooled.dtype)
            for i in range(nfe):
                tau = torch.full((pooled.shape[0], 1), (i + 0.5) * dt, device=pooled.device, dtype=pooled.dtype)
                z = z + dt * self.predict_velocity(z, tau, condition)
            coeff_samples.append(z)
        coeff_stack = torch.stack(coeff_samples, dim=0)
        mu_coeff = coeff_stack.mean(dim=0)
        if samples > 1:
            var_coeff = coeff_stack.var(dim=0, unbiased=True)
        else:
            var_coeff = self.logvar(condition).exp()
        mu = self.coeff_to_waveform(mu_coeff, signal_len)
        logvar = self.coeff_to_waveform((var_coeff + 1e-6).log(), signal_len)
        return mu, logvar.clamp(min=-8.0, max=6.0), pooled, mu_coeff


class RiskPolicyHead(nn.Module):
    def __init__(self, h, state_dim):
        super().__init__()
        hidden = h.get("policy_hidden", 128)
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.alpha = nn.Linear(hidden, 1)
        self.alpha_concentration = nn.Linear(hidden, 2)
        self.weights = nn.Linear(hidden, 3)
        self.stop = nn.Linear(hidden, 1)
        self.reject = nn.Linear(hidden, 1)
        self.value = nn.Linear(hidden, 1)
        self.cost_value = nn.Linear(hidden, 1)

    def _params(self, state):
        h = self.net(state)
        alpha_beta = F.softplus(self.alpha_concentration(h)) + 1e-4
        weight_logits = self.weights(h)
        stop_logits = self.stop(h)
        reject_logits = self.reject(h)
        return h, alpha_beta, weight_logits, stop_logits, reject_logits

    def forward(self, state):
        h, alpha_beta, weight_logits, stop_logits, reject_logits = self._params(state)
        return {
            "alpha": torch.sigmoid(self.alpha(h)).view(-1, 1, 1),
            "alpha_beta": alpha_beta.view(-1, 2),
            "weight_logits": weight_logits,
            "weights": torch.softmax(weight_logits, dim=-1).view(-1, 3, 1),
            "stop_logit": stop_logits.view(-1, 1, 1),
            "reject_logit": reject_logits.view(-1, 1, 1),
            "stop_prob": torch.sigmoid(stop_logits).view(-1, 1, 1),
            "reject_prob": torch.sigmoid(reject_logits).view(-1, 1, 1),
            "value": self.value(h).view(-1, 1, 1),
            "cost_value": self.cost_value(h).view(-1, 1, 1),
        }

    def sample_action(self, state, deterministic=False):
        out = self.forward(state)
        if deterministic:
            out["log_prob"] = torch.zeros_like(out["alpha"])
            out["entropy"] = torch.zeros_like(out["alpha"])
            out["weight_index"] = out["weights"].squeeze(-1).argmax(dim=-1, keepdim=True)
            out["stop_sample"] = (out["stop_prob"] >= 0.5).float()
            out["reject_sample"] = (out["reject_prob"] >= 0.5).float()
            return out

        beta_dist = torch.distributions.Beta(out["alpha_beta"][:, 0], out["alpha_beta"][:, 1])
        alpha = beta_dist.rsample().clamp(1e-4, 1.0 - 1e-4).view(-1, 1, 1)
        weight_dist = torch.distributions.Categorical(logits=out["weight_logits"])
        weight_index = weight_dist.sample().view(-1, 1)
        weights = F.one_hot(weight_index.squeeze(-1), num_classes=3).float().unsqueeze(-1)
        stop_dist = torch.distributions.Bernoulli(logits=out["stop_logit"].view(-1))
        reject_dist = torch.distributions.Bernoulli(logits=out["reject_logit"].view(-1))
        stop_sample = stop_dist.sample().view(-1, 1, 1)
        reject_sample = reject_dist.sample().view(-1, 1, 1)
        log_prob = (
            beta_dist.log_prob(alpha.view(-1))
            + weight_dist.log_prob(weight_index.view(-1))
            + stop_dist.log_prob(stop_sample.view(-1))
            + reject_dist.log_prob(reject_sample.view(-1))
        ).view(-1, 1, 1)
        entropy = (
            beta_dist.entropy()
            + weight_dist.entropy()
            + stop_dist.entropy()
            + reject_dist.entropy()
        ).view(-1, 1, 1)
        out.update(
            {
                "alpha": alpha,
                "weights": weights,
                "weight_index": weight_index,
                "stop_sample": stop_sample,
                "reject_sample": reject_sample,
                "log_prob": log_prob,
                "entropy": entropy,
            }
        )
        return out


class MorphologySafetyLayer(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.kappa = h.get("safety_kappa", 0.15)
        self.eps = h.get("safety_eps", 1e-6)

    def morphology_cost(self, delta):
        derivative = delta[..., 1:] - delta[..., :-1]
        second_derivative = derivative[..., 1:] - derivative[..., :-1]
        qrs_proxy = derivative.abs().amax(dim=-1, keepdim=True)
        return (
            derivative.abs().mean(dim=-1, keepdim=True)
            + 0.2 * second_derivative.abs().mean(dim=-1, keepdim=True)
            + 0.05 * qrs_proxy
        )

    def forward(self, alpha, delta, cumulative_cost):
        cost = self.morphology_cost(delta)
        remaining = (self.kappa - cumulative_cost).clamp_min(0.0)
        safe_limit = remaining / (cost + self.eps)
        alpha_safe = torch.minimum(alpha, safe_limit)
        cumulative_cost = cumulative_cost + alpha_safe * cost
        return alpha_safe, cumulative_cost, cost
