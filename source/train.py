import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader

from tqdm import tqdm

from DataLoaders import Dataset, collate_fn
from DSTFormer import DSTFormer
from loss_function import loss_mpjpe


class Trainer:
    """Trainer class for DSTFormer model with transfer learning capabilities."""
    
    def __init__(self, train_loader: TorchDataLoader, num_epochs: int):
        """
        Initialize the trainer.
        
        Args:
            train_loader: DataLoader for training data
            num_epochs: Number of epochs to train
        """
        # Initialize model
        self.model = DSTFormer(
            dim_in=2,
            dim_out=1,
            embed_size=64,
            heads=8,
            max_len=5,
            num_joints=17,
            fusion_depth=2,
            attn_depth=2,
            fusion=True
        ).to('cuda')

        # Freeze all parameters except the head
        for param in self.model.parameters():
            param.requires_grad = False

        # Define new head for transfer learning
        self.model.head = nn.Sequential(
            nn.Linear(self.model.embed_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Unfreeze head parameters
        for param in self.model.head.parameters():
            param.requires_grad = True

        # Load pre-trained weights
        pre_trained_weights = torch.load('/teamspace/studios/this_studio/model_105.pth')
        self.model.load_state_dict(pre_trained_weights, strict=False)
        self.model.to('cuda')

        # Initialize optimizer and loss
        self.optimizer = torch.optim.AdamW(
            self.model.head.parameters(),
            lr=0.001,
            weight_decay=1e-4
        )
        self.criterion = torch.nn.MSELoss()
        self.train_loader = train_loader
        self.num_epochs = num_epochs

    def train(self):
        """Train the model for the specified number of epochs."""
        # Print model parameters info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%")

        # Training loop
        for epoch in tqdm(range(self.num_epochs)):
            for i, (video, target) in tqdm(enumerate(self.train_loader)):
                self.optimizer.zero_grad()
                outputs = self.model(video.to('cuda'))
                loss = loss_mpjpe(outputs.to('cuda'), target.to('cuda'))
                loss.backward()
                self.optimizer.step()
            
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item()}")

            # Save model checkpoint every 5 epochs
            if epoch % 5 == 0:
                torch.save(self.model.state_dict(), f"final_model_{epoch}.pth")


def main():
    """Main function to set up and start training."""
    # Training parameters
    batch_size = 128
    num_epochs = 50

    # Dataset setup
    path = '/teamspace/studios/this_studio/data/AMASS/AMASS/amass_joints_h36m_60.pkl'
    dataset = Dataset(path, 1000, max_frames=5, if_train=False)
    
    dataloader = TorchDataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=lambda batch: collate_fn(batch, dataset)
    )

    # Initialize and start training
    trainer = Trainer(dataloader, num_epochs)
    trainer.train()


if __name__ == "__main__":
    main()

# model = DSTFormer(dim_in=2, dim_out=2, embed_size=64, heads=8, max_len=5, num_joints=17, fusion_depth = 2, attn_depth =2, fusion = True).to('cuda')
# for name, module in model.named_modules():
#     print(f"Variable Name: {name}")
#     print(f"Module: {module}")
#     print("-" * 50)
# print(model.head.weight.shape)