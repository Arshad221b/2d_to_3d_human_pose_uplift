import torch 
import numpy as np
import pickle

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
    def __init__(self, data_path, frame_threshold, max_frames =10): 
        self.data = DataLoader(data_path, frame_threshold).main()
        self.max_frames = max_frames

    def split_video(self, video): 
        max_splits = video.shape[1] // self.max_frames
        return torch.tensor(video[:, :max_splits*self.max_frames,:].reshape(max_splits, self.max_frames, video.shape[0], video.shape[2]))

    def __len__(self): 
        return len(self.data)
    
    def __getitem__(self, idx): 
        return self.data[idx]

    def collate_fn(self, batch):
        videos = []
        targets = []
        for i in batch: 
            videos.append(self.split_video(i)[:,:,:,:2])
            targets.append(self.split_video(i)[:,:,:,2:])
        return torch.cat(videos, dim=0).to(torch.float32), torch.cat(targets, dim=0).to(torch.float32)
        # return torch.cat(videos, dim=0).to(torch.float32), torch.cat(targets, dim=0).to(torch.float32)



# path = '/teamspace/studios/this_studio/data/AMASS/AMASS/amass_joints_h36m_60.pkl'
# # data = DataLoader.main(path, 1000)

# dataset = Dataset(path, 500)


# train_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=dataset.collate_fn)

# for i in train_loader: 
#     print(i)
#     break