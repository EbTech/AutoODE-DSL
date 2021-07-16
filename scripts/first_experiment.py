import torch

from ode_nn.data import C19Dataset
from ode_nn.history import HistoryWithSoftmax
from ode_nn.seiturd import History, SeiturdModel

ds = C19Dataset()
history = HistoryWithSoftmax.from_dataset(ds, requires_grad=True)
A = ds.adjacency

model = SeiturdModel(history.num_days, A)

log_prob = model.log_prob(history)
total_log_prob = log_prob.sum()

print(total_log_prob)

total_log_prob.backward()
lr = 1e-5

print("Logits SEIU before update:\n", history.logits_SEIU[-1, 0])
print("Population of bama on day 1 before update:\n", history[-1, 0])
# print("logic_decay_E before update: ", model.logit_decay_E)
with torch.no_grad():
    for name, param in model.named_parameters():
        print(name)
        param -= lr * param.grad

    for name, param in history.named_parameters():
        print(name, param.grad.norm())
        param -= lr * param.grad

print("Logits SEIU after update:\n", history.logits_SEIU[-1, 0])
print("Population of bama on day 1 after update:\n", history[-1, 0])
# print("logic_decay_E after update: ", model.logit_decay_E)
