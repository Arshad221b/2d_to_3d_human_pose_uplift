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
    def __init__(self, dim_in=3, dim_out=3, embed_size=256, heads=2, max_len=10, num_joints=17, fusion_depth = 5, attn_depth =2, fusion = True):
        super(DSTFormer, self).__init__()

        self.joint_embed = nn.Linear(dim_in, embed_size)
        self.fusion = fusion

        self.pos_embedding = nn.Parameter(torch.zeros(1, num_joints, embed_size))
        self.temp_embedding = nn.Parameter(torch.zeros(1, max_len, 1, embed_size))

        self.embed_size = embed_size
        self.heads = heads

        self.attention = Attention(embed_size=embed_size, heads=heads).to(device)
        self.mlp = MLPlayer(embed_size=embed_size, hidden_size=embed_size).to(device)
        # self.fusion_model = nn.ModuleList([nn.Linear(2*embed_size, 2) for i in range(fusion_depth)])
        self.fusion_model = nn.Linear(2*embed_size, 2)
        self.head = nn.Linear(embed_size, dim_out)

    def forward(self, x):
        # DST former is a dual stream attention, (S + T) + (T + S)
        B, F, J, C = x.shape # batch, frames, joints, channels 

        x = x.reshape(-1, J, C) # merging the batch and the frames

        x = self.joint_embed(x) # (C, J) x (X, J, C) = (X, J, J)
        x = x + self.pos_embedding # (17, J) x (X, J, J) = (X, J, J)
        _, J, C = x.shape # J changing from 3 to 64
        x = x.reshape(-1, F, J, C) + self.temp_embedding[:, :F, :, :]
        x = x.reshape(B*F, J, C)

        s1 = self.attention(x, "spatial")
        t1 = self.attention(s1, "temporal")

        t2 = self.attention(x, "temporal")
        s2 = self.attention(t2, "spatial")

        if self.fusion:
            # fusion_model = self.fusion_model()
            alpha = torch.cat([t1, s2], dim = -1)
            alpha = self.fusion_model(alpha)
            alpha = alpha.softmax(-1)
            x = t1 * alpha[:,:, 0:1] + s2 * alpha[:,:, 1:2]
            # output = 
            

        else: 
            fused = (t1 + s2) /2
            x = self.mlp(fused)
        output = self.head(x)
        return output  



        


num_joints = 17
embed_size = 64
num_heads = 8
# a = Attention(embed_size=embed_size, heads=2)

x = torch.randn(8, 10, num_joints, 3).to(device)
dst = DSTFormer(embed_size=embed_size, heads=8)

print(dst(x).shape)