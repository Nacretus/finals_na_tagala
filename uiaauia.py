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
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score, precision_recall_curve
import logging
import os
import random
import contextlib
import seaborn as sns
from tqdm import tqdm
from stopwordsiso import stopwords as stopwords_iso

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
        self.stop_words = set(stopwords_iso("en")).union(stopwords_iso("tl"))
        
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


class EfficientTextClassifier(nn.Module):
    def __init__(self, hidden_dim, num_classes, dropout_rate=0.3):
        super(EfficientTextClassifier, self).__init__()
        
        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.bert_embedding_dim = self.bert.config.hidden_size
        
        for param in self.bert.parameters():
            param.requires_grad = False
            
        self.projection = nn.Linear(self.bert_embedding_dim, hidden_dim)
        
        self.lstm = nn.LSTM(
            hidden_dim, 
            hidden_dim, 
            batch_first=True,
            bidirectional=True
        )
        
        self.conv = nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1)
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
            # Removed Sigmoid since we'll use BCEWithLogitsLoss
        )
    
    def forward(self, input_ids, attention_mask=None):
        with torch.no_grad():
            bert_outputs = self.bert(input_ids, attention_mask=attention_mask)
            embedded = bert_outputs.last_hidden_state
        
        projected = self.projection(embedded)
        
        lstm_out, _ = self.lstm(projected)
        if attention_mask is not None:
            lstm_out = lstm_out * attention_mask.unsqueeze(-1)
        
        lstm_out = lstm_out.permute(0, 2, 1)
        cnn_out = self.conv(lstm_out)
        cnn_out = cnn_out.permute(0, 2, 1)
        
        attn_weights = self.attention(cnn_out)
        context = torch.sum(attn_weights * cnn_out, dim=1)
        
        output = self.classifier(context)
        return output


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
        model_path="text_classifier_model.pt"
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
        
    def train_epoch(self):
        self.model.train()
        train_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        for batch in progress_bar:
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
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        return train_loss / len(self.train_loader)
    
    def evaluate(self):
        self.model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                
                # Apply sigmoid for predictions since we're using BCEWithLogitsLoss
                preds = (torch.sigmoid(outputs) > 0.5).float()
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        
        accuracy = accuracy_score(all_labels.flatten(), all_preds.flatten())
        f1 = f1_score(all_labels, all_preds, average='micro')
        
        return val_loss / len(self.val_loader), accuracy, f1
    
    def train(self, num_epochs, patience=3):
        logger.info(f"Starting training for {num_epochs} epochs")
        
        best_val_loss = float('inf')
        no_improvement = 0
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            
            val_loss, accuracy, f1 = self.evaluate()
            
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                        f"Train Loss: {train_loss:.4f}, "
                        f"Val Loss: {val_loss:.4f}, "
                        f"Accuracy: {accuracy:.4f}, "
                        f"F1: {f1:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improvement = 0
                
                self.save_model()
                logger.info(f"New best model saved with validation loss: {val_loss:.4f}")
            else:
                no_improvement += 1
                
            if no_improvement >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
                
        logger.info("Training completed")
        
    def save_model(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, self.model_path)
    
    def load_model(self):
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"Model loaded from {self.model_path}")
        else:
            logger.error(f"No model found at {self.model_path}")


class Predictor:
    def __init__(self, model, processor, device, threshold=0.5):
        self.model = model
        self.processor = processor
        self.device = device
        self.threshold = threshold
        self.model.to(device)
        self.model.eval()
        
    def predict(self, text):
        preprocessed = self.processor.preprocess(text)
        encoding = self.processor.tokenize(preprocessed, return_tensors='pt')
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            # Apply sigmoid to raw logits since model no longer has sigmoid
            outputs = torch.sigmoid(outputs)
            predictions = (outputs > self.threshold).float().cpu().numpy()
            
        return predictions
    
    def predict_batch(self, texts):
        encodings = self.processor.transform_text_batch(texts)
        
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            # Apply sigmoid to raw logits since model no longer has sigmoid
            outputs = torch.sigmoid(outputs)
            predictions = (outputs > self.threshold).float().cpu().numpy()
            
        return predictions


def visualize_data(df):
    style.use('ggplot')
    
    plt.figure(figsize=(12, 7))
    x = df.iloc[:, 1:].sum()
    colors = ['#FF6B6B', '#FF9E7A', '#FFBF86', '#FFDF8C', '#FFF47D', '#C4F7A1']
    plt.bar(x.index, x.values, alpha=0.8, color=colors[:len(x)])
    plt.title("Toxicity Category Distribution", fontsize=16)
    plt.ylabel("Number of Instances", fontsize=14)
    plt.xlabel("Toxicity Categories", fontsize=14)
    plt.xticks(rotation=30, ha='right')
    
    for i, v in enumerate(x):
        plt.text(i, v + 0.5, str(v), ha='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig("toxicity_distribution.png", dpi=300, bbox_inches='tight')
    
    plt.figure(figsize=(12, 7))
    rowsum = df.iloc[:, 1:].sum(axis=1)
    plt.hist(rowsum.values, bins=range(0, max(rowsum) + 2), alpha=0.8, color='#FF7E79')
    plt.title("Toxicity Categories per Comment", fontsize=16)
    plt.ylabel("Number of Comments", fontsize=14)
    plt.xlabel("Number of Toxicity Categories", fontsize=14)
    plt.xticks(range(0, max(rowsum) + 1))
    
    total = len(df)
    for i in range(0, max(rowsum) + 1):
        count = (rowsum == i).sum()
        percentage = count / total * 100
        plt.text(i, count + 5, f"{percentage:.1f}%", ha='center')
    
    plt.tight_layout()
    plt.savefig("toxicity_per_comment.png", dpi=300, bbox_inches='tight')
    
    plt.figure(figsize=(14, 12))
    corr = df.iloc[:, 1:].corr()
    cmap = plt.cm.RdYlGn_r
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, 
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-0.6,
        center=0,
        square=True,
        linewidths=.5,
        annot=True,
        fmt=".2f"
    )
    plt.title("Co-occurrence of Toxicity Categories", fontsize=16)
    plt.tight_layout()
    plt.savefig("toxicity_correlation.png", dpi=300, bbox_inches='tight')
    
    non_toxic = (rowsum == 0).sum()
    toxic = (rowsum > 0).sum()
    multi_category = (rowsum > 1).sum()
    
    logger.info(f"Total comments: {len(df)}")
    logger.info(f"Non-toxic comments: {non_toxic} ({non_toxic/len(df)*100:.2f}%)")
    logger.info(f"Toxic comments: {toxic} ({toxic/len(df)*100:.2f}%)")
    logger.info(f"Comments with multiple toxicity categories: {multi_category} ({multi_category/len(df)*100:.2f}%)")
    logger.info(f"Total toxicity labels: {x.sum()}")

def calculate_class_weights(df, label_columns):
    logger.info("Calculating class weights...")
    
    class_counts = df[label_columns].sum().to_dict()
    total_samples = len(df)
    
    class_weights = {}
    for label, count in class_counts.items():
        # Calculate positive weight for each class
        # This represents how much to scale the positive examples vs negative ones
        pos_weight = (total_samples - count) / (count + 1e-5)
        class_weights[label] = pos_weight
        
    for label, weight in class_weights.items():
        logger.info(f"Positive weight for {label}: {weight:.4f}")
        
    return class_weights

class WeightedBCELoss(nn.Module):
    def __init__(self, class_weights):
        super(WeightedBCELoss, self).__init__()
        self.class_weights = class_weights
        self.weight_keys = list(class_weights.keys())
        
    def forward(self, outputs, targets):
        weights = torch.ones(len(self.weight_keys), device=outputs.device)
        for i, label in enumerate(self.weight_keys):
            weights[i] = self.class_weights[label]
            
        loss = F.binary_cross_entropy(outputs, targets, reduction='none')
        weighted_loss = loss * weights.unsqueeze(0)
        return weighted_loss.mean()

def main():
    DATA_PATH = "FF.csv"
    MODEL_PATH = "toxic_aiaoiaia_model.pt"
    MAX_LENGTH = 512
    BATCH_SIZE = 8
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    HIDDEN_DIM = 128
    VALIDATION_SPLIT = 0.1
    
    TOXICITY_LABELS = ['toxic', 'very toxic', 'profanity', 'threat', 'insult', 'identity_hate']
    
    logger.info(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    
    logger.info("Toxicity label distribution in dataset:")
    toxicity_counts = df.iloc[:, 1:].sum()
    for label, count in toxicity_counts.items():
        logger.info(f"{label}: {count} instances ({count/len(df)*100:.2f}%)")
    
    visualize_data(df)
    
    processor = TextProcessor(max_length=MAX_LENGTH)
    
    X = df['comment']
    y = df.iloc[:, 1:]
    
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
    
    num_classes = y.shape[1]
    model = EfficientTextClassifier(
        hidden_dim=HIDDEN_DIM,
        num_classes=num_classes,
        dropout_rate=0.3
    ).to(device)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} of total)")
    
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], 
        lr=LEARNING_RATE
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # Always calculate class weights for imbalanced multi-label classification
    class_weights = calculate_class_weights(df, y.columns)
    
    # Convert class weights to tensor format for BCEWithLogitsLoss
    pos_weight = torch.FloatTensor([class_weights[label] for label in y.columns]).to(device)
    
    # Use BCEWithLogitsLoss with class weights for multi-label classification
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    logger.info("Using BCEWithLogitsLoss with positive class weights")
    
    trainer = TextClassificationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        model_path=MODEL_PATH
    )
    
    trainer.train(NUM_EPOCHS, patience=3)
    
    trainer.load_model()
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            outputs = torch.sigmoid(outputs)  # Apply sigmoid to raw logits
            preds = (outputs > 0.5).float()
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    accuracy = accuracy_score(all_labels.flatten(), all_preds.flatten())
    f1_micro = f1_score(all_labels, all_preds, average='micro')
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    
    logger.info(f"Test Results - Accuracy: {accuracy:.4f}, F1 (micro): {f1_micro:.4f}, F1 (macro): {f1_macro:.4f}")
    logger.info("\nClassification Report:")
    logger.info(classification_report(all_labels, all_preds, target_names=y.columns))
    
    predictor = Predictor(model, processor, device)
    
    logger.info("\nDemo Predictions:")
    
    sample_text1 = "This is a normal comment about the weather today."
    prediction1 = predictor.predict(sample_text1)
    logger.info(f"Text: {sample_text1}")
    logger.info(f"Prediction: {prediction1}")
    
    sample_text2 = "I strongly disagree with your opinion and think you should reconsider."
    prediction2 = predictor.predict(sample_text2)
    logger.info(f"Text: {sample_text2}")
    logger.info(f"Prediction: {prediction2}")
    
    print("\nToxicity Detection Mode (type 'exit' to quit):")
    print(f"This model detects the following types of toxic content: {', '.join(y.columns)}")
    while True:
        user_input = input("Enter a comment to analyze for toxicity: ")
        if user_input.lower() == 'exit':
            break
            
        prediction = predictor.predict(user_input)
        class_predictions = {y.columns[i]: pred for i, pred in enumerate(prediction[0])}
        
        print("Toxicity analysis:")
        has_toxicity = False
        for label, value in class_predictions.items():
            if value > 0:
                has_toxicity = True
                print(f"- {label.replace('_', ' ').title()}: Detected")
        
        if not has_toxicity:
            print("- No toxic content detected")


if __name__ == "__main__":
    main()