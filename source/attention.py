import torch 
import torch.nn as nn 
import torch.nn.functional as F 

device = 'cuda' if torch.cuda.is_available() == True else 'cpu'

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


class DSTFormer(nn.Module):
    def __init__(self, embed_size, heads):
        super(DSTFormer, self).__init__()
        self.embed_size = embed_size
        self.heads = heads

        self.attention = Attention(embed_size=embed_size, heads=heads).to(device)
        self.mlp = MLPlayer(embed_size=embed_size, hidden_size=embed_size).to(device)

    def forward(self, x):
        # DST former is a dual stream attention, (S + T) + (T + S)

        s1 = self.attention(x, "spatial")
        t1 = self.attention(s1, "temporal")

        t2 = self.attention(x, "temporal")
        s2 = self.attention(t2, "spatial")

        fused = t1 + s2

        output = self.mlp(fused)

        return output 



        


d_model = 64
embed_size = 10 *d_model
num_heads = 8
# a = Attention(embed_size=embed_size, heads=2)

x = torch.randn(8, 10, embed_size).to(device)
dst = DSTFormer(embed_size=embed_size, heads=8)




print(dst(x).shape)