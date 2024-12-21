import torch
from DSTFormer import DSTFormer


class train: 
    def __init__(): 
        self.model = DSTFormer()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = torch.nn.MSELoss()
        self.train_loader = 
        self.val_loader = 
        self.test_loader = 

    def train(self): 
        for epoch in range(self.num_epochs): 
            for i, (images, labels) in enumerate(self.train_loader): 
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                print(f"Epoch [{epoch+1}/{self.num_epochs}], Step [{i+1}/{len(self.train_loader)}], Loss: {loss.item()}")



def train_loader(data)