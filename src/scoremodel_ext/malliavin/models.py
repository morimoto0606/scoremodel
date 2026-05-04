import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeScoreMLP(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, t, y):
        if t.ndim == 1:
            t = t[:, None]
        if y.ndim == 1:
            y = y[:, None]
        return self.net(torch.cat([t, y], dim=1))


class ScoreMLP2D(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


class TimeScoreMLP2D(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),   # t, x1, x2
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, t, x):
        if t.ndim == 1:
            t = t[:, None]
        inp = torch.cat([t, x], dim=1)
        return self.net(inp)


class FourierFeatures(nn.Module):
    def __init__(self, in_dim: int, num_frequencies: int = 16, scale: float = 10.0):
        super().__init__()
        B = torch.randn(in_dim, num_frequencies) * scale
        self.register_buffer("B", B)

    def forward(self, x):
        proj = 2.0 * math.pi * x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.net(x))


class MirafzaliSkorokhodNet(nn.Module):
    """
    N_theta(t, x) ≈ E[delta_t(u_t) | X_t=x]
    For practical memory, default hidden=512. For faithful large run, use hidden=4096.
    """
    def __init__(
        self,
        x_dim: int = 2,
        hidden: int = 512,
        n_blocks: int = 6,
        num_frequencies: int = 16,
        fourier_scale: float = 10.0,
    ):
        super().__init__()
        in_dim = x_dim + 1
        self.ff = FourierFeatures(in_dim, num_frequencies, fourier_scale)
        ff_dim = 2 * num_frequencies
        self.in_layer = nn.Sequential(
            nn.Linear(in_dim + ff_dim, hidden),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden) for _ in range(n_blocks)])
        self.out_layer = nn.Linear(hidden, x_dim)

    def forward(self, t, x):
        if t.ndim == 1:
            t = t[:, None]
        z = torch.cat([t, x], dim=1)
        zff = self.ff(z)
        h = torch.cat([z, zff], dim=1)
        h = self.in_layer(h)
        h = self.blocks(h)
        return self.out_layer(h)


class NormalizedSkorokhodModel(nn.Module):
    def __init__(self, net, x_mean, x_std, t_mean, t_std, y_mean, y_std):
        super().__init__()
        self.net = net
        self.register_buffer("x_mean", x_mean)
        self.register_buffer("x_std", x_std)
        self.register_buffer("t_mean", t_mean)
        self.register_buffer("t_std", t_std)
        self.register_buffer("y_mean", y_mean)
        self.register_buffer("y_std", y_std)

    def forward(self, t, x):
        if t.ndim == 1:
            t_col = t[:, None]
        else:
            t_col = t
        tn = ((t_col - self.t_mean) / self.t_std.clamp_min(1e-6)).squeeze(-1)
        xn = (x - self.x_mean) / self.x_std.clamp_min(1e-6)
        yn = self.net(tn, xn)
        return yn * self.y_std + self.y_mean


def train_mirafzali_skorokhod_net(
    t,
    x,
    delta,
    n_epochs=1000,
    batch_size=2048,
    lr=2e-4,
    weight_decay=1e-5,
    hidden=512,
    n_blocks=6,
    num_frequencies=16,
    device="cuda",
):
    """
    Algorithm 6 style:
        input  : (X_t, t)
        target : delta_t(u_t)
        output : E[delta_t(u_t) | X_t]
    """
    t = t.to(device)
    x = x.to(device)
    delta = delta.to(device)

    x_mean = x.mean(dim=0, keepdim=True)
    x_std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
    t_col = t[:, None]
    t_mean = t_col.mean(dim=0, keepdim=True)
    t_std = t_col.std(dim=0, keepdim=True).clamp_min(1e-6)
    y_mean = delta.mean(dim=0, keepdim=True)
    y_std = delta.std(dim=0, keepdim=True).clamp_min(1e-6)

    x_n = (x - x_mean) / x_std
    t_n = ((t_col - t_mean) / t_std).squeeze(-1)
    y_n = (delta - y_mean) / y_std

    net = MirafzaliSkorokhodNet(
        x_dim=x.shape[1],
        hidden=hidden,
        n_blocks=n_blocks,
        num_frequencies=num_frequencies,
    ).to(device)

    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    n = x.shape[0]
    best_loss = float("inf")
    best_state = None

    for ep in range(1, n_epochs + 1):
        idx = torch.randint(0, n, (batch_size,), device=device)
        pred = net(t_n[idx], x_n[idx])
        loss = F.mse_loss(pred, y_n[idx])

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        if ep % 500 == 0:
            with torch.no_grad():
                # validation proxy on random subset to avoid huge full pass
                vidx = torch.randint(0, n, (min(20000, n),), device=device)
                vpred = net(t_n[vidx], x_n[vidx])
                vloss = F.mse_loss(vpred, y_n[vidx]).item()

            if vloss < best_loss:
                best_loss = vloss
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                print(f"  *** mirafzali best updated: {best_loss:.6e}")

            print(f"  epoch={ep:5d}  loss={vloss:.6e}  best={best_loss:.6e}")

            sched.step()
        else:
            sched.step()

    if best_state is not None:
        net.load_state_dict(best_state)

    wrapped = NormalizedSkorokhodModel(
        net,
        x_mean.detach(),
        x_std.detach(),
        t_mean.detach(),
        t_std.detach(),
        y_mean.detach(),
        y_std.detach(),
    ).to(device)

    return wrapped