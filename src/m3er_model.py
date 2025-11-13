import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cross_decomposition import CCA


class ModalityCheckStep(nn.Module):
    """
    Modality Check Step menggunakan CCA untuk detect ineffectual modalities
    """

    def __init__(self, feature_dims, n_components=100, threshold=0.3):
        super(ModalityCheckStep, self).__init__()
        self.feature_dims = feature_dims
        self.n_components = n_components
        self.threshold = threshold
        self.cca_models = {}

    def fit_cca(self, modality_pairs, train_data):
        """
        Fit CCA models untuk setiap pair of modalities

        Args:
            modality_pairs: List of tuples [(mod1, mod2), ...]
            train_data: Dictionary dengan keys 'speech', 'text', 'visual'
        """
        for mod1, mod2 in modality_pairs:
            # Flatten temporal dimension
            X = train_data[mod1].reshape(train_data[mod1].shape[0], -1)
            Y = train_data[mod2].reshape(train_data[mod2].shape[0], -1)

            # Fit CCA
            cca = CCA(n_components=self.n_components)
            cca.fit(X, Y)

            self.cca_models[(mod1, mod2)] = cca
            print(f"✓ CCA fitted for {mod1}-{mod2}")

    def check_modality(self, features_dict):
        """
        Check if modalities are effectual based on correlation

        Returns:
            indicators: Dictionary dengan indicator untuk setiap modality
        """
        indicators = {mod: 1.0 for mod in features_dict.keys()}

        # Check correlation untuk setiap modality
        modalities = list(features_dict.keys())

        for i, mod_i in enumerate(modalities):
            correlations = []

            for j, mod_j in enumerate(modalities):
                if i != j:
                    # Get CCA model
                    if (mod_i, mod_j) in self.cca_models:
                        cca = self.cca_models[(mod_i, mod_j)]

                        # Transform features
                        feat_i = features_dict[mod_i].reshape(1, -1)
                        feat_j = features_dict[mod_j].reshape(1, -1)

                        X_c, Y_c = cca.transform(feat_i, feat_j)

                        # Calculate correlation
                        corr = np.corrcoef(X_c.flatten(), Y_c.flatten())[0, 1]
                        correlations.append(abs(corr))

            # Check if modality is ineffectual
            if correlations and max(correlations) < self.threshold:
                indicators[mod_i] = 0.0

        return indicators


class ProxyFeatureGenerator(nn.Module):
    """
    Generate proxy features untuk ineffectual modalities
    """

    def __init__(self, feature_dims):
        super(ProxyFeatureGenerator, self).__init__()
        self.feature_dims = feature_dims

        # Linear transformations untuk generate proxy features
        # Speech -> Text, Speech -> Visual
        self.speech_to_text = nn.Linear(feature_dims["speech"], feature_dims["text"])
        self.speech_to_visual = nn.Linear(
            feature_dims["speech"], feature_dims["visual"]
        )

        # Text -> Speech, Text -> Visual
        self.text_to_speech = nn.Linear(feature_dims["text"], feature_dims["speech"])
        self.text_to_visual = nn.Linear(feature_dims["text"], feature_dims["visual"])

        # Visual -> Speech, Visual -> Text
        self.visual_to_speech = nn.Linear(
            feature_dims["visual"], feature_dims["speech"]
        )
        self.visual_to_text = nn.Linear(feature_dims["visual"], feature_dims["text"])

    def forward(self, features_dict, indicators):
        """
        Generate proxy features untuk ineffectual modalities

        Args:
            features_dict: Dictionary dengan features untuk setiap modality
            indicators: Dictionary dengan indicators (0 or 1) untuk setiap modality

        Returns:
            Updated features_dict dengan proxy features
        """
        updated_features = features_dict.copy()

        # Generate proxy features based on indicators
        # Speech ineffectual
        if indicators["speech"] == 0.0:
            proxy_from_text = self.text_to_speech(features_dict["text"])
            proxy_from_visual = self.visual_to_speech(features_dict["visual"])
            updated_features["speech"] = (proxy_from_text + proxy_from_visual) / 2

        # Text ineffectual
        if indicators["text"] == 0.0:
            proxy_from_speech = self.speech_to_text(features_dict["speech"])
            proxy_from_visual = self.visual_to_text(features_dict["visual"])
            updated_features["text"] = (proxy_from_speech + proxy_from_visual) / 2

        # Visual ineffectual
        if indicators["visual"] == 0.0:
            proxy_from_speech = self.speech_to_visual(features_dict["speech"])
            proxy_from_text = self.text_to_visual(features_dict["text"])
            updated_features["visual"] = (proxy_from_speech + proxy_from_text) / 2

        return updated_features


class MultiplicativeFusion(nn.Module):
    """
    M3ER: Multiplicative Multimodal Emotion Recognition
    """

    def __init__(self, config):
        super(M3ER, self).__init__()

        self.config = config
        self.timestep = config["timestep"]
        self.feature_dims = config["feature_dims"]
        self.n_classes = config["n_classes"]

        # LSTM untuk setiap modality (seperti di paper)
        self.lstm_speech = nn.LSTM(
            input_size=self.feature_dims["speech"],
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        self.lstm_text = nn.LSTM(
            input_size=self.feature_dims["text"],
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        self.lstm_visual = nn.LSTM(
            input_size=self.feature_dims["visual"],
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        # Memory variable
        self.memory_dim = 128

        # Attention module
        self.attention_dim = 32
        self.attention_speech = nn.Linear(32 + self.memory_dim, self.attention_dim)
        self.attention_text = nn.Linear(32 + self.memory_dim, self.attention_dim)
        self.attention_visual = nn.Linear(32 + self.memory_dim, self.attention_dim)

        # Memory update
        self.memory_update = nn.Linear(96 + self.memory_dim, self.memory_dim)

        # Classification heads untuk setiap modality
        self.classifier_speech = nn.Sequential(
            nn.Linear(32, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, self.n_classes)
        )

        self.classifier_text = nn.Sequential(
            nn.Linear(32, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, self.n_classes)
        )

        self.classifier_visual = nn.Sequential(
            nn.Linear(32, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, self.n_classes)
        )

        # Final fusion layer
        self.fusion = MultiplicativeFusion(
            n_modalities=3, n_classes=self.n_classes, beta=config.get("beta", 1.0)
        )

        # Final classifier - INPUT DARI FUSION OUTPUT (n_classes dimension)
        # Fusion output shape: (batch_size, n_classes)
        # Concat dengan memory: n_classes + memory_dim
        self.final_classifier = nn.Sequential(
            nn.Linear(self.n_classes + self.memory_dim, 64),  # 6 + 128 = 134
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, self.n_classes),
        )

        # Proxy feature generator
        self.proxy_generator = ProxyFeatureGenerator(self.feature_dims)

    def forward(self, speech, text, visual, memory=None):
        """
        Forward pass sesuai paper architecture
        """
        batch_size = speech.size(0)

        # Initialize memory if not provided
        if memory is None:
            memory = torch.zeros(batch_size, self.memory_dim).to(speech.device)

        # LSTM processing
        lstm_out_speech, _ = self.lstm_speech(speech)
        lstm_out_text, _ = self.lstm_text(text)
        lstm_out_visual, _ = self.lstm_visual(visual)

        # Take last timestep output
        h_speech = lstm_out_speech[:, -1, :]  # (batch, 32)
        h_text = lstm_out_text[:, -1, :]
        h_visual = lstm_out_visual[:, -1, :]

        # Attention mechanism
        speech_mem = torch.cat([h_speech, memory], dim=1)
        text_mem = torch.cat([h_text, memory], dim=1)
        visual_mem = torch.cat([h_visual, memory], dim=1)

        attn_speech = torch.sigmoid(self.attention_speech(speech_mem))
        attn_text = torch.sigmoid(self.attention_text(text_mem))
        attn_visual = torch.sigmoid(self.attention_visual(visual_mem))

        # Apply attention
        h_speech = h_speech * attn_speech
        h_text = h_text * attn_text
        h_visual = h_visual * attn_visual

        # Update memory
        combined_features = torch.cat([h_speech, h_text, h_visual, memory], dim=1)
        memory = torch.tanh(self.memory_update(combined_features))

        # Individual predictions untuk multiplicative fusion
        pred_speech = self.classifier_speech(h_speech)  # (batch, n_classes)
        pred_text = self.classifier_text(h_text)
        pred_visual = self.classifier_visual(h_visual)

        # Multiplicative fusion
        fused_pred = self.fusion(
            [pred_speech, pred_text, pred_visual]
        )  # (batch, n_classes)

        # Final prediction - GUNAKAN FUSION OUTPUT + MEMORY
        final_input = torch.cat(
            [fused_pred, memory], dim=1
        )  # (batch, n_classes + memory_dim)
        final_pred = self.final_classifier(final_input)  # (batch, n_classes)

        return {
            "final_pred": final_pred,
            "fused_pred": fused_pred,
            "pred_speech": pred_speech,
            "pred_text": pred_text,
            "pred_visual": pred_visual,
            "memory": memory,
        }

    def compute_loss(self, predictions, targets):
        """
        Compute modified multiplicative loss (Equation 3)

        Args:
            predictions: List of prediction tensors [pred_speech, pred_text, pred_visual]
            targets: Ground truth labels (batch_size,)
        """
        batch_size = targets.size(0)
        loss = 0.0

        for pred in predictions:
            # Apply softmax
            prob = F.softmax(pred, dim=1)

            # Get probability untuk true class
            true_class_prob = prob[range(batch_size), targets]

            # Modified multiplicative loss: -sum(p_i^(beta/(M-1)) * log(p_i))
            weighted_log_prob = (
                true_class_prob ** (self.beta / (self.n_modalities - 1))
            ) * torch.log(true_class_prob + 1e-10)
            loss += -weighted_log_prob.mean()

        return loss


class M3ER(nn.Module):
    """
    M3ER: Multiplicative Multimodal Emotion Recognition
    """

    def __init__(self, config):
        super(M3ER, self).__init__()

        self.config = config
        self.timestep = config["timestep"]
        self.feature_dims = config["feature_dims"]
        self.n_classes = config["n_classes"]

        # LSTM untuk setiap modality (seperti di paper)
        self.lstm_speech = nn.LSTM(
            input_size=self.feature_dims["speech"],
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        self.lstm_text = nn.LSTM(
            input_size=self.feature_dims["text"],
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        self.lstm_visual = nn.LSTM(
            input_size=self.feature_dims["visual"],
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        # Memory variable
        self.memory_dim = 128

        # Attention module
        self.attention_dim = 32
        self.attention_speech = nn.Linear(32 + self.memory_dim, self.attention_dim)
        self.attention_text = nn.Linear(32 + self.memory_dim, self.attention_dim)
        self.attention_visual = nn.Linear(32 + self.memory_dim, self.attention_dim)

        # Memory update
        self.memory_update = nn.Linear(96 + self.memory_dim, self.memory_dim)

        # Classification heads untuk setiap modality
        self.classifier_speech = nn.Sequential(
            nn.Linear(32, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, self.n_classes)
        )

        self.classifier_text = nn.Sequential(
            nn.Linear(32, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, self.n_classes)
        )

        self.classifier_visual = nn.Sequential(
            nn.Linear(32, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, self.n_classes)
        )

        # Final fusion layer
        self.fusion = MultiplicativeFusion(
            n_modalities=3, n_classes=self.n_classes, beta=config.get("beta", 1.0)
        )

        # Final classifier (after fusion)
        self.final_classifier = nn.Sequential(
            nn.Linear(160, 64),  # 32*3 (LSTM outputs) + 128 (memory) = 160
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, self.n_classes),
        )

        # Proxy feature generator
        self.proxy_generator = ProxyFeatureGenerator(self.feature_dims)

    def forward(self, speech, text, visual, memory=None):
        """
        Forward pass

        Args:
            speech: (batch_size, timestep, speech_dim)
            text: (batch_size, timestep, text_dim)
            visual: (batch_size, timestep, visual_dim)
            memory: (batch_size, memory_dim) - optional
        """
        batch_size = speech.size(0)

        # Initialize memory if not provided
        if memory is None:
            memory = torch.zeros(batch_size, self.memory_dim).to(speech.device)

        # LSTM processing
        lstm_out_speech, _ = self.lstm_speech(speech)  # (batch, timestep, 32)
        lstm_out_text, _ = self.lstm_text(text)
        lstm_out_visual, _ = self.lstm_visual(visual)

        # Take last timestep output
        h_speech = lstm_out_speech[:, -1, :]  # (batch, 32)
        h_text = lstm_out_text[:, -1, :]
        h_visual = lstm_out_visual[:, -1, :]

        # Attention mechanism (simplified)
        # Concat dengan memory
        speech_mem = torch.cat([h_speech, memory], dim=1)
        text_mem = torch.cat([h_text, memory], dim=1)
        visual_mem = torch.cat([h_visual, memory], dim=1)

        # Compute attention
        attn_speech = torch.sigmoid(self.attention_speech(speech_mem))
        attn_text = torch.sigmoid(self.attention_text(text_mem))
        attn_visual = torch.sigmoid(self.attention_visual(visual_mem))

        # Apply attention
        h_speech = h_speech * attn_speech
        h_text = h_text * attn_text
        h_visual = h_visual * attn_visual

        # Update memory
        combined_features = torch.cat([h_speech, h_text, h_visual, memory], dim=1)
        memory = torch.tanh(self.memory_update(combined_features))

        # Individual predictions untuk multiplicative fusion
        pred_speech = self.classifier_speech(h_speech)
        pred_text = self.classifier_text(h_text)
        pred_visual = self.classifier_visual(h_visual)

        # Multiplicative fusion
        fused_pred = self.fusion([pred_speech, pred_text, pred_visual])

        # Final prediction
        final_features = torch.cat([h_speech, h_text, h_visual, memory], dim=1)
        final_pred = self.final_classifier(final_features)

        return {
            "final_pred": final_pred,
            "fused_pred": fused_pred,
            "pred_speech": pred_speech,
            "pred_text": pred_text,
            "pred_visual": pred_visual,
            "memory": memory,
        }

    def compute_loss(self, outputs, targets):
        """
        Compute total loss
        """
        # Multiplicative fusion loss
        fusion_loss = self.fusion.compute_loss(
            [outputs["pred_speech"], outputs["pred_text"], outputs["pred_visual"]],
            targets,
        )

        # Final prediction loss
        final_loss = F.cross_entropy(outputs["final_pred"], targets)

        # Combined loss
        total_loss = 0.7 * fusion_loss + 0.3 * final_loss

        return total_loss, {
            "fusion_loss": fusion_loss.item(),
            "final_loss": final_loss.item(),
        }


# Training utilities
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in dataloader:
        speech = batch["speech"].to(device)
        text = batch["text"].to(device)
        visual = batch["visual"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(speech, text, visual)
        loss, loss_dict = model.compute_loss(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs["final_pred"], 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return total_loss / len(dataloader), 100 * correct / total


def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            speech = batch["speech"].to(device)
            text = batch["text"].to(device)
            visual = batch["visual"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(speech, text, visual)
            loss, _ = model.compute_loss(outputs, labels)

            total_loss += loss.item()

            _, predicted = torch.max(outputs["final_pred"], 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return total_loss / len(dataloader), 100 * correct / total
