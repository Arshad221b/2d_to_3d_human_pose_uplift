import torch
from DataLoaders import Dataset
from DSTFormer import DSTFormer
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from loss_function import loss_mpjpe
scaler = GradScaler()


class train: 
    def __init__(self, train_loader, num_epochs): 
        self.model = DSTFormer(dim_in=2, dim_out=1, embed_size=64, heads=8, max_len=5, num_joints=17, fusion_depth = 3, attn_depth =2, fusion = True)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        # self.criterion = torch.nn.MSELoss()
        self.train_loader = train_loader
        self.num_epochs = num_epochs

    def train(self): 
        for epoch in tqdm(range(self.num_epochs)): 
            for i, (video, target) in tqdm(enumerate(self.train_loader)): 
                self.optimizer.zero_grad()
                outputs = self.model(video)
                # loss = self.criterion(outputs.to('cuda'), target.to('cuda'))
                loss = loss_mpjpe(outputs.to('cuda'), target.to('cuda'))
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item()}")

            if epoch % 10 == 0: 
                torch.save(self.model.state_dict(), f"model_{epoch}.pth")



batch_size = 256
num_epochs = 10

path = '/teamspace/studios/this_studio/data/AMASS/AMASS/amass_joints_h36m_60.pkl'
dataset = Dataset(path, 1000, max_frames=5)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=dataset.collate_fn)

train = train(train_loader, num_epochs)
train.train()