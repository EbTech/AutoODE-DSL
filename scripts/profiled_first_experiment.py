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

# Non-default profiler schedule allows user to turn profiler on and off
# on different iterations of the training loop;
# trace_handler is called every time a new trace becomes available
def trace_handler(prof):
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=-1))
    # prof.export_chrome_trace("/tmp/test_trace_" + str(prof.step_num) + ".json")

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],

    # In this example with wait=1, warmup=1, active=2,
    # profiler will skip the first step/iteration,
    # start warming up on the second, record
    # the third and the forth iterations,
    # after which the trace will become available
    # and on_trace_ready (when set) is called;
    # the cycle repeats starting with the next step
    schedule=torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("logs"),
    with_stack=True
) as profiler:
    for step in range(100_000):
        opt.zero_grad()
        regs = {}
        for n in ["contagion_I", "detection_rate", "recovery_rate"]:
            regs[n] = 100 * getattr(model, n).diff(dim=0).abs().sum()
        loss = -model.log_prob(history) + sum(regs.values())
        loss.backward()
        opt.step()
        profiler.step()
        print(
            f"step {step:5,}  loss: {loss.detach():>15.3f}   "
            f"E->: {model.decay_I:<5.3}   I->: {model.decay_E:<5.3}   T->: {model.decay_T:<5.3}   "
            "regs: " + " ".join(f"{n}: {v:>10.0f}" for n, v in regs.items())
        )
        if step % 2_000 == 0:
            d = {"model": model, "history": history, "opt": opt}
            torch.save(d, f"checkpt_{step}.pt")
