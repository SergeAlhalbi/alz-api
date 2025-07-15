from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, model, dataloader, criterion, optimizer, device, log_dir=None):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.writer = SummaryWriter(log_dir=log_dir) if log_dir else None

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0

        for i, (images, labels) in enumerate(tqdm(self.dataloader, desc=f"Epoch {epoch+1}")):
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.dataloader)

        # Log to TensorBoard
        if self.writer:
            self.writer.add_scalar("Loss/train", avg_loss, epoch)

        return avg_loss

    def train(self, epochs):
        for epoch in range(epochs):
            loss = self.train_epoch(epoch)
            print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss:.4f}")

        if self.writer:
            self.writer.close()