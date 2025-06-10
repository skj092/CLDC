import timm
from torch import nn


class ImageClassifier(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)

        # Replace the final classification layer
        if hasattr(self.model, 'fc'):
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        elif hasattr(self.model, 'classifier') and isinstance(self.model.classifier, nn.Linear):
            self.model.classifier = nn.Linear(
                self.model.classifier.in_features, num_classes)
        elif hasattr(self.model, 'head') and isinstance(self.model.head, nn.Linear):
            self.model.head = nn.Linear(
                self.model.head.in_features, num_classes)
        else:
            raise NotImplementedError(
                f"Unknown model head structure for {model_name}")

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        return loss, acc

    def validation_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(1)
        acc = (preds == y).float().mean()
        return loss, acc, preds, y
