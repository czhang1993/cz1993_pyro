# https://pyro.ai/examples/intro_long.html
# Introduction to Pyro

import numpy as np
import pandas as pd
import torch
import pyro

df = pd.DataFrame(data)

# prior = torch.distribution.Normal(loc=0, scale=1)




class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()


model = Model()
guide = pyro.infer.autoguide.AutoNormal(model)
elbo_ = pyro.infer.Trace_ELBO(num_particles=10)
elbo = elbo_(model, guide)

optim = torch.optim.Adam(
  elbo.parameters(),
  lr=0.001
)

for _ in range(100):
    optim.zero_grad()
    loss = elbo(data)
    loss.backward()
    optim.step()

