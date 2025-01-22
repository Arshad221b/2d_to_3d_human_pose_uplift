import torch 
import numpy as np
import pickle
from torch.utils.data import DataLoader as TorchDataLoader

class DataLoader:
    def __init__(self, data_path, frame_threshold): 
        self.data_path = data_path
        self.frame_threshold = frame_threshold

    def load_data(self):
        with open(self.data_path, 'rb') as f: 
            self.data = pickle.load(f)
        return self.data

    def prefiltering(self): 
        new_data = []
        for i in self.data: 
            if i.shape[1] >= self.frame_threshold: 
                new_data.append(i)

        self.data = new_data
        return self.data

    def main(self): 
        data = self.load_data()
        data = self.prefiltering()
        return data



class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, frame_threshold, max_frames =10, if_train = True): 
        self.data = DataLoader(data_path, frame_threshold).main()
        self.max_frames = max_frames
        self.if_train = if_train

    def split_video(self, video): 
        max_splits = video.shape[1] // self.max_frames
        return torch.tensor(video[:, :max_splits*self.max_frames,:].reshape(max_splits, self.max_frames, video.shape[0], video.shape[2]))

    def __len__(self): 
        return len(self.data)
    
    def __getitem__(self, idx): 
        return self.data[idx]

    def create_masked_data(self,data, mask_prob = 0.15):
        masked_data = data
        mask = np.random.rand(*masked_data.shape) < mask_prob
        gaussian_noise = np.random.normal(0, 1, masked_data.shape)
        masked_data[mask] = gaussian_noise[mask]

        return masked_data

def collate_fn(batch, dataset, mask_prob=0.15):
    videos = []
    targets = []

    for i in batch:
        if dataset.if_train:
            # Create masked data and split it
            masked_video = dataset.split_video(dataset.create_masked_data(i, mask_prob))
            target_video = dataset.split_video(i)
            videos.append(masked_video[:, :, :, :2])  # Take first two channels
            targets.append(target_video[:, :, :, :2])  # Same for target

        else:
            # No masking for inference
            split_video = dataset.split_video(i)
            videos.append(split_video[:, :, :, :2])  # First two channels
            targets.append(split_video[:, :, :, 2:])  # Remaining channels

    # Combine into tensors
    return (
        torch.cat(videos, dim=0).to(torch.float32),
        torch.cat(targets, dim=0).to(torch.float32),
    )



# path = '/teamspace/studios/this_studio/data/AMASS/AMASS/amass_joints_h36m_60.pkl'
# # data = DataLoader.main(path, 1000)

# dataset = Dataset(path, 500)


# # train_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=dataset.collate_fn)

# # for vidoes, target in train_loader: 
# #     print(vidoes.shape, target.shape)
# #     break

# # dataloader = DataLoader(dataset, collate_fn=lambda batch: collate_fn(batch, dataset))
# dataloader = TorchDataLoader(
#     dataset,
#     batch_size=4,
#     collate_fn=lambda batch: collate_fn(batch, dataset)
# )


# # Iterate through the DataLoader
# for videos, targets in dataloader:
#     print("Videos shape:", videos.shape)
#     print("Targets shape:", targets.shape)
#     break