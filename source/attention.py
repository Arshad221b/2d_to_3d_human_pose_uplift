import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class Attention(nn.Module): 
    def __init__(self, embed_size, heads):
        super(Attention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads 
        self.head_dim = embed_size // heads 


        self.q_w = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.k_w = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.v_w = nn.Linear(self.embed_size, self.embed_size, bias=False)

    
    def forward(self, x, mode = "vanilla"):
        B, N, C, T = x.shape

        if mode == "vanilla":
            print(self.q_w.weight.shape)
            print(x.shape)
            q = self.q_w(x)
            k = self.k_w(x)
            v = self.v_w(x)

            score = (q @ k.transpose(-2, -1)) / (self.embed_size ** 0.5)

            output = F.softmax(score @ v, dim=-1)
            return output

        # if mode == "temporal": 
        #     q

    

a = Attention(embed_size=10*17*2, heads=2)

x = torch.randn(2, 17, 2, 10)

print(a(x))

