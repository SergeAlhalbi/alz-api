from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

class MRIClassifierDataset(Dataset):
    def __init__(self, split="train", transform=None):
        self.dataset = load_dataset("Falah/Alzheimer_MRI", split=split)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Convert 1-channel to 3-channel
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        label = item["label"]

        if isinstance(image, Image.Image):
            # print(f"[DEBUG] Original image size at index {idx}: {image.size}")  # (width, height)
            image = self.transform(image)
            # print(f"[DEBUG] Transformed image shape: {image.shape}")  # torch.Size([3, 224, 224])

        return image, label