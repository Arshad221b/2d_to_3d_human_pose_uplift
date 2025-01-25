import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader as TorchDataLoader
from tqdm import tqdm

from DataLoaders import Dataset, collate_fn
from DSTFormer import DSTFormer
from loss_function import loss_mpjpe

scaler = GradScaler()


class train:
    def __init__(self, train_loader, num_epochs):
        """Initialize the training class.
        
        Args:
            train_loader: DataLoader for training data
            num_epochs: Number of epochs to train
        """
        self.model = DSTFormer(
            dim_in=2,
            dim_out=2,
            embed_size=64,
            heads=8,
            max_len=5,
            num_joints=17,
            fusion_depth=2,
            attn_depth=2,
            fusion=True
        )
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        self.train_loader = train_loader
        self.num_epochs = num_epochs

    def train(self):
        # self.model.load_state_dict(torch.load('/teamspace/studios/this_studio/model_100.pth'))

        for epoch in tqdm(range(101, self.num_epochs), desc='Epochs'):
            for video, target in tqdm(self.train_loader, desc=f'Epoch {epoch + 1}'):
                self.optimizer.zero_grad()
                outputs = self.model(video)
                
                loss = loss_mpjpe(outputs.to('cuda'), target.to('cuda'))
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

            print(f"Epoch [{epoch + 1}/{self.num_epochs}], Loss: {loss.item()}")

            if epoch % 5 == 0:
                torch.save(
                    self.model.state_dict(),
                    f"model_{epoch}.pth"
                )


def main():
    batch_size = 32
    num_epochs = 200
    data_path = 'path/to/data.pkl'

    dataset = Dataset(data_path, 100, max_frames=5, if_train=True)
    dataloader = TorchDataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=lambda batch: collate_fn(batch, dataset)
    )

    trainer = train(dataloader, num_epochs)
    trainer.train()


if __name__ == "__main__":
    main()