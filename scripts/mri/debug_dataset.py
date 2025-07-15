import os
import matplotlib.pyplot as plt
from domains.mri.dataset import MRIClassifierDataset
import torchvision.transforms.functional as F
import torch

if __name__ == "__main__":
    # Load both splits
    train_dataset = MRIClassifierDataset(split="train")
    test_dataset = MRIClassifierDataset(split="test")

    # Dataset sizes
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Show image shape and label
    img, label = train_dataset[0]
    print(f"Image shape: {img.shape}")
    print(f"Sample label: {label}")

    # Count unique classes in train set
    all_train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
    unique_labels = torch.unique(torch.tensor(all_train_labels))
    print(f"Unique class labels: {unique_labels.tolist()}")
    print(f"Total classes: {len(unique_labels)}")

    # Visualize a sample image
    plt.imshow(F.to_pil_image(img))
    plt.title(f"Label: {label}")
    plt.axis("off")
    plt.show()

    # Save the image for inference
    save_path = os.path.join("C:/Users/serge/OneDrive/Desktop/Projects/alz-api/tests/mri", "inference_sample.png")
    F.to_pil_image(img).save(save_path)
    print(f"Saved image to: {save_path}")