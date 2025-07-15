from domains.mri.dataset import MRIClassifierDataset

def test_dataset_length():
    dataset = MRIClassifierDataset(split="train")
    assert len(dataset) > 0

def test_dataset_sample_shape():
    dataset = MRIClassifierDataset(split="train")
    image, label = dataset[0]
    assert image.shape[1:] == (224, 224)
    assert image.shape[0] in [1, 3]  # allow grayscale or RGB
    assert isinstance(label, int)