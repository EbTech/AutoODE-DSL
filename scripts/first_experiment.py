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

opt = torch.optim.SGD(list(model.parameters()) + list(history.parameters()), lr=1e-10)

for step in range(10):
    print(f"starting step {step}")
    opt.zero_grad()

    log_prob = model.log_prob(history)
    loss = -log_prob.sum()

    loss.backward()

    opt.step()
