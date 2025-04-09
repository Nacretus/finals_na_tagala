import warnings
warnings.filterwarnings("ignore")
import torch.nn.functional as F
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib import style
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_recall_curve, roc_auc_score, precision_score, recall_score
import logging
import os
import random
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

class TextProcessor:
    def __init__(self, max_length=512):
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        
    def preprocess(self, text):
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        return text
    
    def tokenize(self, text, return_tensors=None):
        return self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors=return_tensors
        )
    
    def transform_text_batch(self, texts):
        preprocessed_texts = [self.preprocess(text) for text in texts]
        encoding = self.tokenize(
            preprocessed_texts, 
            return_tensors='pt'
        )
        return encoding


class TextDataset(Dataset):
    def __init__(self, texts, labels, processor):
        self.texts = texts
        self.labels = torch.tensor(labels.values, dtype=torch.float32)
        self.processor = processor
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        preprocessed = self.processor.preprocess(text)
        encoding = self.processor.tokenize(preprocessed, return_tensors=None)
        
        input_ids = torch.tensor(encoding['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(encoding['attention_mask'], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': self.labels[idx]
        }


class BiLSTMCNNAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, dropout_rate=0.5):
        super(BiLSTMCNNAttention, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # CNN layers with standardized sequence length
        self.conv1 = nn.Conv1d(embedding_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embedding_dim, hidden_dim, kernel_size=4, padding=1)
        self.conv3 = nn.Conv1d(embedding_dim, hidden_dim, kernel_size=5, padding=2)
        
        # Adaptive pooling to ensure 512 sequence length (BERT's max length)
        self.adaptive_pool = nn.AdaptiveMaxPool1d(512)
        
        # Bi-LSTM layer
        self.bilstm = nn.LSTM(
            hidden_dim * 3,  # Concatenated CNN features
            hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_rate if dropout_rate > 0 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Classification layer
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes),
            nn.Sigmoid()
        )
    
    def forward(self, input_ids, attention_mask=None):
        # Embedding
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        
        # CNN feature extraction
        embedded_permuted = embedded.permute(0, 2, 1)  # [batch_size, embedding_dim, seq_len]
        
        conv1_out = F.relu(self.conv1(embedded_permuted))  # [batch_size, hidden_dim, seq_len]
        conv2_out = F.relu(self.conv2(embedded_permuted))
        conv3_out = F.relu(self.conv3(embedded_permuted))
        
        # Apply adaptive pooling to ensure 512 sequence length
        conv1_out = self.adaptive_pool(conv1_out)
        conv2_out = self.adaptive_pool(conv2_out)
        conv3_out = self.adaptive_pool(conv3_out)
        
        # Concatenate CNN features
        cnn_features = torch.cat([conv1_out, conv2_out, conv3_out], dim=1)  # [batch_size, hidden_dim*3, 512]
        cnn_features = cnn_features.permute(0, 2, 1)  # [batch_size, 512, hidden_dim*3]
        
        # Apply Bi-LSTM
        if attention_mask is not None:
            lstm_out, _ = self.bilstm(cnn_features)
            lstm_out = lstm_out * attention_mask.unsqueeze(-1)
        else:
            lstm_out, _ = self.bilstm(cnn_features)  # [batch_size, seq_len, hidden_dim*2]
        
        # Apply attention mechanism
        attention_weights = self.attention(lstm_out)  # [batch_size, seq_len, 1]
        context = torch.sum(attention_weights * lstm_out, dim=1)  # [batch_size, hidden_dim*2]
        
        # Classification
        output = self.classifier(context)  # [batch_size, num_classes]
        
        return output


def calculate_class_weights(df, label_columns):
    class_counts = df[label_columns].sum().to_dict()
    total_samples = len(df)
    
    class_weights = {}
    for label, count in class_counts.items():
        weight = total_samples / (count + 1e-5)
        class_weights[label] = weight
    
    weight_sum = sum(class_weights.values())
    for label in class_weights:
        class_weights[label] = class_weights[label] / weight_sum * len(class_weights)
        
    return class_weights


class PrecisionFocusedLoss(nn.Module):
    def __init__(self, class_weights, alpha=1, gamma=2, fp_weight=4.0, fn_weight=1.0):
        super(PrecisionFocusedLoss, self).__init__()
        self.class_weights = class_weights
        self.weight_keys = list(class_weights.keys())
        self.alpha = alpha
        self.gamma = gamma
        self.fp_weight = fp_weight
        self.fn_weight = fn_weight
        
    def forward(self, outputs, targets):
        weights = torch.ones(len(self.weight_keys), device=outputs.device)
        for i, label in enumerate(self.weight_keys):
            weights[i] = self.class_weights[label]
            
        bce_loss = F.binary_cross_entropy(outputs, targets, reduction='none')
        
        fps = (outputs > 0.5).float() * (1 - targets)
        fns = (outputs <= 0.5).float() * targets
        
        pt = torch.exp(-bce_loss)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        
        asymmetric_weight = 1.0 + fps * (self.fp_weight - 1.0) + fns * (self.fn_weight - 1.0)
        combined_weight = focal_weight * asymmetric_weight
        
        weighted_loss = bce_loss * weights.unsqueeze(0) * combined_weight
        
        return weighted_loss.mean()


class TextClassificationTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler=None,
        device=torch.device("cpu"),
        model_path="toxic_classifier_model.pt",
        label_columns=None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.model_path = model_path
        self.best_val_loss = float('inf')
        self.label_columns = label_columns
        
    def train_epoch(self):
        self.model.train()
        train_loss = 0
        
        for batch in tqdm(self.train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask)
            
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            train_loss += loss.item()
            
        return train_loss / len(self.train_loader)
    
    def evaluate(self):
        self.model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        all_outputs = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                
                all_outputs.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        all_outputs = np.vstack(all_outputs)
        all_labels = np.vstack(all_labels)
        
        all_preds = (all_outputs > 0.5).astype(float)
        accuracy = accuracy_score(all_labels.flatten(), all_preds.flatten())
        precision = precision_score(all_labels, all_preds, average='micro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='micro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
        
        auc_scores = {}
        precision_scores = {}
        recall_scores = {}
        
        for i, label in enumerate(self.label_columns):
            try:
                auc = roc_auc_score(all_labels[:, i], all_outputs[:, i])
                auc_scores[label] = auc
                
                precision_scores[label] = precision_score(all_labels[:, i], all_preds[:, i], zero_division=0)
                recall_scores[label] = recall_score(all_labels[:, i], all_preds[:, i], zero_division=0)
                
            except ValueError:
                auc_scores[label] = float('nan')
                precision_scores[label] = float('nan')
                recall_scores[label] = float('nan')
                
        valid_aucs = [auc for auc in auc_scores.values() if not np.isnan(auc)]
        avg_auc = sum(valid_aucs) / len(valid_aucs) if valid_aucs else 0
        
        valid_precisions = [p for p in precision_scores.values() if not np.isnan(p)]
        avg_precision = sum(valid_precisions) / len(valid_precisions) if valid_precisions else 0
        
        logger.info(f"Validation - Precision: {precision:.4f}, Recall: {recall:.4f}, AUC-ROC: {avg_auc:.4f}")
        
        return val_loss / len(self.val_loader), accuracy, f1, all_outputs, all_labels, auc_scores, precision_scores, recall_scores
    
    def calibrate_thresholds(self, min_precision=0.85, min_recall=0.3):
        _, _, _, all_outputs, all_labels, _ = self.evaluate()
        optimal_thresholds = {}
        
        for i, label in enumerate(self.label_columns):
            precision, recall, thresholds = precision_recall_curve(all_labels[:, i], all_outputs[:, i])
            
            valid_indices = [j for j, r in enumerate(recall) if r >= min_recall]
            
            if valid_indices:
                valid_precision = [precision[j] for j in valid_indices]
                valid_thresholds = [thresholds[min(j, len(thresholds)-1)] for j in valid_indices]
                
                best_idx = np.argmax(valid_precision)
                optimal_thresholds[label] = valid_thresholds[best_idx]
                
                logger.info(f"Label {label}: Selected threshold {valid_thresholds[best_idx]:.4f} "
                            f"with precision {valid_precision[best_idx]:.4f} at "
                            f"recall {recall[valid_indices[best_idx]]:.4f}")
            else:
                optimal_thresholds[label] = 0.5
                logger.warning(f"Label {label}: Could not find threshold with min_recall={min_recall}. "
                               f"Using default threshold=0.5")
                
        return optimal_thresholds
        
    def train(self, num_epochs, patience=3):
        logger.info(f"Starting training for {num_epochs} epochs")
        
        best_val_loss = float('inf')
        best_precision = 0.0
        no_improvement = 0
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss, accuracy, f1, _, _, auc_scores, precision_scores, recall_scores = self.evaluate()
            
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            valid_aucs = [auc for auc in auc_scores.values() if not np.isnan(auc)]
            avg_auc = sum(valid_aucs) / len(valid_aucs) if valid_aucs else 0
            
            valid_precisions = [p for p in precision_scores.values() if not np.isnan(p)]
            avg_precision = sum(valid_precisions) / len(valid_precisions) if valid_precisions else 0
            
            valid_recalls = [r for r in recall_scores.values() if not np.isnan(r)]
            avg_recall = sum(valid_recalls) / len(valid_recalls) if valid_recalls else 0
            
            logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                        f"Train Loss: {train_loss:.4f}, "
                        f"Val Loss: {val_loss:.4f}, "
                        f"Accuracy: {accuracy:.4f}, "
                        f"F1: {f1:.4f}, "
                        f"Precision: {avg_precision:.4f}, "
                        f"Recall: {avg_recall:.4f}, "
                        f"AUC-ROC: {avg_auc:.4f}")
            
            for label, precision in precision_scores.items():
                if not np.isnan(precision):
                    logger.info(f"  - {label} precision: {precision:.4f}")
            
            if avg_precision > best_precision:
                best_precision = avg_precision
                no_improvement = 0
                
                self.save_model()
                logger.info(f"New best model saved with precision: {avg_precision:.4f}")
            else:
                no_improvement += 1
                
            if no_improvement >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
                
        logger.info("Training completed")
        
        logger.info("Calibrating thresholds for high precision...")
        optimal_thresholds = self.calibrate_thresholds(min_precision=0.85, min_recall=0.3)
        logger.info(f"Optimal thresholds: {optimal_thresholds}")
        
        self.save_model(thresholds=optimal_thresholds)
        
        return optimal_thresholds
        
    def save_model(self, thresholds=None):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        
        if thresholds is not None:
            checkpoint['thresholds'] = thresholds
            
        torch.save(checkpoint, self.model_path)
    
    def load_model(self):
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            thresholds = checkpoint.get('thresholds', None)
            return thresholds
        else:
            logger.error(f"No model found at {self.model_path}")
            return None


class Predictor:
    def __init__(self, model, processor, device, thresholds=None):
        self.model = model
        self.processor = processor
        self.device = device
        self.thresholds = thresholds if thresholds else 0.5
        self.model.to(device)
        self.model.eval()
        
    def predict(self, text):
        preprocessed = self.processor.preprocess(text)
        encoding = self.processor.tokenize(preprocessed, return_tensors='pt')
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            outputs_np = outputs.cpu().numpy()
            
            if isinstance(self.thresholds, dict):
                predictions = np.zeros_like(outputs_np)
                for i, label in enumerate(self.thresholds.keys()):
                    predictions[0, i] = outputs_np[0, i] > self.thresholds[label]
            else:
                predictions = (outputs_np > self.thresholds).astype(float)
            
        return predictions
    
    def predict_batch(self, texts):
        encodings = self.processor.transform_text_batch(texts)
        
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            outputs_np = outputs.cpu().numpy()
            
            if isinstance(self.thresholds, dict):
                predictions = np.zeros_like(outputs_np)
                for i, label in enumerate(self.thresholds.keys()):
                    predictions[:, i] = outputs_np[:, i] > self.thresholds[label]
            else:
                predictions = (outputs_np > self.thresholds).astype(float)
            
        return predictions


def visualize_data(df):
    style.use('ggplot')
    
    plt.figure(figsize=(10, 6))
    x = df.iloc[:, 1:].sum()
    plt.bar(x.index, x.values, alpha=0.8, color='tab:blue')
    plt.title("Label Distribution", fontsize=14)
    plt.ylabel("Count", fontsize=12)
    plt.xlabel("Labels", fontsize=12)
    plt.savefig("label_distribution.png", dpi=300, bbox_inches='tight')
    
    plt.figure(figsize=(10, 6))
    rowsum = df.iloc[:, 1:].sum(axis=1)
    plt.hist(rowsum.values, bins=range(0, max(rowsum) + 2), alpha=0.8, color='tab:orange')
    plt.title("Labels per Comment", fontsize=14)
    plt.ylabel("Number of Comments", fontsize=12)
    plt.xlabel("Number of Labels", fontsize=12)
    plt.xticks(range(0, max(rowsum) + 1))
    plt.savefig("labels_per_comment.png", dpi=300, bbox_inches='tight')


def main():
    DATA_PATH = "FF.csv"
    MODEL_PATH = "toxic_classifier_model.pt"
    MAX_LENGTH = 512
    BATCH_SIZE = 24
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-5
    EMBEDDING_DIM = 256
    HIDDEN_DIM = 256
    VALIDATION_SPLIT = 0.1
    
    logger.info(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    
    visualize_data(df)
    
    processor = TextProcessor(max_length=MAX_LENGTH)
    
    X = df['comment']
    y = df.iloc[:, 1:]  # Assuming labels start from column 1
    
    class_weights = calculate_class_weights(df, y.columns)
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=df.iloc[:, 1:].sum(axis=1)
    )
    
    val_ratio = VALIDATION_SPLIT / 0.3
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED, 
        stratify=y_temp.sum(axis=1)
    )
    
    logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    train_dataset = TextDataset(X_train, y_train, processor)
    val_dataset = TextDataset(X_val, y_val, processor)
    test_dataset = TextDataset(X_test, y_test, processor)
    
    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    vocab_size = len(processor.tokenizer.vocab)
    num_classes = y.shape[1]
    
    # Initialize the BiLSTM+CNN+Attention model
    model = BiLSTMCNNAttention(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=num_classes
    ).to(device)
    
    # Initialize optimizer with weight decay for regularization
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    # Use learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2, patience=3, verbose=True, min_lr=1e-6
    )
    
    # Use precision-focused loss
    criterion = PrecisionFocusedLoss(
        class_weights, 
        alpha=1, 
        gamma=2, 
        fp_weight=4.0,  # Higher penalty for false positives
        fn_weight=1.0
    )
    
    trainer = TextClassificationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        model_path=MODEL_PATH,
        label_columns=y.columns
    )
    
    # Train and get optimal thresholds
    optimal_thresholds = trainer.train(NUM_EPOCHS, patience=5)
    
    # Load the best model for evaluation
    trainer.load_model()
    
    # Evaluate on test set
    model.eval()
    all_preds = []
    all_labels = []
    all_raw_outputs = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            outputs_np = outputs.cpu().numpy()
            all_raw_outputs.append(outputs_np)
            
            # Apply calibrated thresholds
            predictions = np.zeros_like(outputs_np)
            for i, label in enumerate(optimal_thresholds.keys()):
                predictions[:, i] = outputs_np[:, i] > optimal_thresholds[label]
            
            all_preds.append(predictions)
            all_labels.append(labels.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    all_raw_outputs = np.vstack(all_raw_outputs)
    
    # Calculate and log metrics
    accuracy = accuracy_score(all_labels.flatten(), all_preds.flatten())
    precision = precision_score(all_labels, all_preds, average='micro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='micro', zero_division=0)
    f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # Calculate AUC-ROC for each label
    auc_scores = {}
    for i, label in enumerate(y.columns):
        try:
            auc = roc_auc_score(all_labels[:, i], all_raw_outputs[:, i])
            auc_scores[label] = auc
        except ValueError:
            auc_scores[label] = float('nan')
    
    # Calculate average AUC-ROC
    valid_aucs = [auc for auc in auc_scores.values() if not np.isnan(auc)]
    avg_auc = sum(valid_aucs) / len(valid_aucs) if valid_aucs else 0
    
    logger.info(f"Test Results - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
                f"Recall: {recall:.4f}, F1 (micro): {f1_micro:.4f}, "
                f"F1 (macro): {f1_macro:.4f}, Avg AUC-ROC: {avg_auc:.4f}")
    
    # Per-label precision and recall
    for i, label in enumerate(y.columns):
        label_precision = precision_score(all_labels[:, i], all_preds[:, i], zero_division=0)
        label_recall = recall_score(all_labels[:, i], all_preds[:, i], zero_division=0)
        logger.info(f"{label}: Precision={label_precision:.4f}, Recall={label_recall:.4f}, AUC-ROC={auc_scores[label]:.4f}")
    
    logger.info("\nClassification Report:")
    logger.info(classification_report(all_labels, all_preds, target_names=y.columns))
    
    # Initialize predictor with calibrated thresholds
    predictor = Predictor(model, processor, device, thresholds=optimal_thresholds)
    
    logger.info("\nDemo Prediction:")
    sample_text = "This is a sample text for prediction"
    prediction = predictor.predict(sample_text)
    logger.info(f"Text: {sample_text}")
    logger.info(f"Prediction: {prediction}")
    
    print("\nInteractive prediction mode (type 'exit' to quit):")
    while True:
        user_input = input("Enter a comment for prediction: ")
        if user_input.lower() == 'exit':
            break
            
        prediction = predictor.predict(user_input)
        class_predictions = {y.columns[i]: pred for i, pred in enumerate(prediction[0])}
        print("Predicted labels:")
        for label, value in class_predictions.items():
            if value > 0:
                print(f"- {label}")
        
        if all(value == 0 for value in class_predictions.values()):
            print("- No labels predicted")


if __name__ == "__main__":
    main()
