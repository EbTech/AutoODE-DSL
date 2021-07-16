from ode_nn.data import C19Dataset
from ode_nn.history import HistoryWithSoftmax
from ode_nn.seiturd import History, SeiturdModel

ds = C19Dataset()
history = HistoryWithSoftmax.from_dataset(ds)
A = ds.adjacency

model = SeiturdModel(history.num_days, A)

log_prob = model.log_prob(history)

print(log_prob)
