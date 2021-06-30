import  torch
from    torch import nn
from    torch.nn import functional as F


class gumble_softmax(nn.Module):

    def __init__(self, N, tau_max=100):
        super().__init__()
        self.tau_max = tau_max
        self.N = N
        self.n = 0
        
    def forward(self, prob):
        epsilon = torch.rand_like(prob).detach_()
        G = torch.log(-torch.log(epsilon))
        sigma = min(self.tau_max, (self.tau_max ** (self.n / self.N)))
        # sigma = self.tau_max ** (self.n / self.N)
        return F.softmax((torch.log(prob) - G) * sigma, dim=-1)
        # return F.softmax((prob - G) * 10000, dim=-1)
    
    def step_n(self):
        self.n += 1
    
    def init_n(self):
        self.n = 0

