import torch

from ode_nn.data import C19Dataset
from ode_nn.history import HistoryWithSoftmax
from ode_nn.seiturd import History, SeiturdModel

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ds = C19Dataset()
history = HistoryWithSoftmax.from_dataset(ds, requires_grad=True)
A = ds.adjacency


model = SeiturdModel(history.num_days, A).to(device)
print(
    f"init:  E->: {model.decay_I:.3}   I->: {model.decay_E:.3}   T->: {model.decay_T:.3}"
)

opt = torch.optim.Adam(list(model.parameters()) + list(history.parameters()), lr=1e-2)

for step in range(100_000):
    opt.zero_grad()
    regs = {}
    for n in ["contagion_I", "detection_rate", "recovery_rate"]:
        regs[n] = 100 * getattr(model, n).diff(dim=0).abs().sum()
    loss = -model.log_prob(history) + sum(regs.values())
    loss.backward()
    opt.step()
    print(
        f"step {step:5,}  loss: {loss.detach():>15.3f}   "
        f"E->: {model.decay_I:<5.3}   I->: {model.decay_E:<5.3}   T->: {model.decay_T:<5.3}   "
        "regs: " + " ".join(f"{n}: {v:>10.0f}" for n, v in regs.items())
    )
    if step % 2_000 == 0:
        d = {"model": model, "history": history, "opt": opt}
        torch.save(d, f"checkpt_{step}.pt")
