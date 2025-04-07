import torch.nn as nn
import torch

class MaxPoolInstanceMIL(nn.Module):
    def __init__(self, instance_encoder, classifier):
        super().__init__()
        self.instance_encoder = instance_encoder
        self.classifier = classifier

    def forward(self, x):
        bag_scores = []
        for bag in x:
            out = self.instance_encoder(bag)
            out = self.classifier(out)
            out, _ = torch.max(out, dim=0)
            bag_scores.append(out)
        return torch.stack(bag_scores)

class AttentionMIL(nn.Module):
    def __init__(self, instance_encoder, classifier, attention_dim):
        super().__init__()
        self.instance_encoder = instance_encoder
        self.classifier = classifier
        self.attention_layer = nn.Linear(128, attention_dim)
        self.attention_weights = nn.Linear(attention_dim, 1)

    def forward(self, x):
        bag_scores = []
        for bag in x:
            instance_embeddings = self.instance_encoder(bag)
            attention_scores = torch.tanh(self.attention_layer(instance_embeddings))
            attention_weights = torch.softmax(self.attention_weights(attention_scores), dim=0)
            bag_embedding = torch.sum(attention_weights * instance_embeddings, dim=0)
            bag_score = self.classifier(bag_embedding.unsqueeze(0))
            bag_scores.append(bag_score)
        return torch.cat(bag_scores, dim=0)
