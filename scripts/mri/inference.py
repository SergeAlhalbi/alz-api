import io
import os
from PIL import Image
import torchvision.transforms as transforms
from domains.mri.model import AlzheimerMRIModel
from common.utils.config import load_config

def load_image(image_input, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])

    if isinstance(image_input, str):  # File path
        image = Image.open(image_input).convert("RGB")
    else:  # BytesIO or UploadFile.file
        image = Image.open(io.BytesIO(image_input)).convert("RGB")

    image = transform(image).unsqueeze(0)
    return image.to(device)

def predict(image_input, config_path="configs/mri/config.yaml"):
    config = load_config(config_path)
    device = config["device"]
    
    model_wrapper = AlzheimerMRIModel(
        num_classes=config["model"]["num_classes"],
        device=device
    )
    model_path = os.path.join(config["paths"]["model_dir"], config["paths"]["model_file"])
    model_wrapper.load_weights(model_path)

    image_tensor = load_image(image_input, device)
    probs = model_wrapper.predict(image_tensor, return_probs=True).cpu().numpy()[0]
    predicted_class = int(probs.argmax())
    return predicted_class, probs.tolist()

# Example
if __name__ == "__main__":
    result = predict(os.path.join("C:/Users/serge/OneDrive/Desktop/Projects/alz-api/tests/mri", "inference_sample.png"))
    print(result)