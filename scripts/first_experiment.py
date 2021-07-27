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

opt = torch.optim.SGD(list(model.parameters()) + list(history.parameters()), lr=1e-12)

for step in range(100_000):
    opt.zero_grad()
    log_prob = model.log_prob(history)
    loss = -log_prob.sum()
    loss.backward()
    opt.step()
    print(
        f"step {step:5,}  loss: {loss.detach():15.3f}   "
        f"E->: {model.decay_I:<5.3}   I->: {model.decay_E:<5.3}   T->: {model.decay_T:<5.3}"
    )
