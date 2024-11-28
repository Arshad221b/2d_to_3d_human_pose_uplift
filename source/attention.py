import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class MLPlayer(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super(MLPlayer, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(self.embed_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.embed_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x): 
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

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
        if mode == "vanilla":
            return self.vanilla_attention(x)
        elif mode == "temporal":
            return self.temporal_attention(x, seqlen=8)
        elif mode == "spatial":
            return self.vanilla_attention(x)

    def vanilla_attention(self, x):
        B, T, N = x.shape
        q = self.q_w(x)
        k = self.k_w(x)
        v = self.v_w(x)
        
        score = (q @ k.transpose(-2, -1)) / (self.embed_size ** 0.5)
        
        score = F.softmax(score, dim=-1)
        output = score @ v
        return output

    def temporal_attention(self, x, seqlen=8):
        
        B, T, N = x.shape
        qt = self.q_w(x).view(B, T, self.heads, self.head_dim)
        kt = self.k_w(x).view(B, T, self.heads, self.head_dim)
        vt = self.v_w(x).view(B, T, self.heads, self.head_dim)  
        
        score = (qt @ kt.transpose(-2, -1)) / (self.embed_size ** 0.5)
        score = F.softmax(score, dim=-1)
        output = score @ vt
        output = output.view(B, T, N)
        
        return output
        
    
    
class Block(nn.Module): 
    def __init__(self, embed_size, heads):
        super(Block, self).__init__()
        self.embed_size = embed_size
        self.heads = heads

        self.attention = Attention(embed_size=embed_size, heads=heads)
        self.mlp = MLPlayer(embed_size=embed_size, hidden_size=embed_size*4)
        
    def forward(self, x): 
        print('before attention', x.shape)
        x = x + self.attention(x).forward(mode="temporal")
        print('after attention', x.shape)
        x = x + self.mlp(x)
        print('after mlp', x.shape)
        return x

d_model = 64
embed_size = 10 *d_model
# a = Attention(embed_size=embed_size, heads=2)

x = torch.randn(2, 10, embed_size)
b = Block(embed_size=embed_size, heads=2)

print(b(x).shape)