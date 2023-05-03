import torch
import pyro
import pyro.distributions as dist
from pyro.infer.tracegraph_elbo import TrackNonReparam
from pyro.ops.provenance import get_provenance
from pyro.poutine import trace

def model():
    probs_a = torch.tensor([0.3, 0.7])
    probs_b = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    probs_c = torch.tensor([[0.5, 0.5], [0.6, 0.4]])
    a = pyro.sample("a", dist.Categorical(probs_a))
    b = pyro.sample("b", dist.Categorical(probs_b[a]))
    pyro.sample("c", dist.Categorical(probs_c[b]), obs=torch.tensor(0))

with TrackNonReparam():
    model_tr = trace(model).get_trace()
model_tr.compute_log_prob()

print(get_provenance(model_tr.nodes["a"]["log_prob"]))  
frozenset({'a'})
print(get_provenance(model_tr.nodes["b"]["log_prob"]))  
frozenset({'b', 'a'})
print(get_provenance(model_tr.nodes["c"]["log_prob"]))  
frozenset({'b', 'a'})
