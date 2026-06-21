import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

# Configurer le device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Périphérique d'exécution détecté : {device}")

# Créer le répertoire des résultats
os.makedirs("results", exist_ok=True)

def load_and_preprocess_data():
    # 1. Chargement du dataset Breast Cancer Wisconsin
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    # Séparation en Train (70%), Val (15%), Test (15%)
    # D'abord séparer le Test (15%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    # Ensuite séparer le Val (15% du total, soit 15/85 de la partie restante)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.15/0.85, random_state=42, stratify=y_train_val
    )
    
    # Standardisation des données
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Conversion en tenseurs PyTorch
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)
    
    print(f"Forme des données d'entraînement : {X_train_t.shape}")
    print(f"Forme des données de validation   : {X_val_t.shape}")
    print(f"Forme des données de test         : {X_test_t.shape}")
    
    return X_train_t, y_train_t, X_val_t, y_val_t, X_test_t, y_test_t, X_train.shape[1]

# 2. Implémentations du MLP

# Version A : nn.Sequential
def get_sequential_mlp(input_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 16),
        nn.ReLU(),
        nn.Linear(16, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
        nn.Sigmoid()
    ).to(device)

# Version B : Classe personnalisée nn.Module
class CustomMLP(nn.Module):
    def __init__(self, input_dim):
        super(CustomMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(16, 8)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

# Fonction d'initialisation des poids
def init_weights(model, method):
    for name, param in model.named_parameters():
        if 'weight' in name:
            if method == 'gaussian':
                nn.init.normal_(param, mean=0.0, std=0.1)
            elif method == 'constant':
                nn.init.constant_(param, 0.5)
            elif method == 'xavier':
                nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.constant_(param, 0.0)

# 3. Fonction d'entraînement
def train_model(model, X_train, y_train, X_val, y_val, epochs=150, lr=0.01):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        # Évaluation sur la validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        
        # Sauvegarde du meilleur état
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_model_state = model.state_dict().copy()
            
    return train_losses, val_losses, best_model_state

def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        predictions = (outputs >= 0.5).float()
        
    y_test_cpu = y_test.cpu().numpy()
    pred_cpu = predictions.cpu().numpy()
    
    accuracy = accuracy_score(y_test_cpu, pred_cpu)
    precision = precision_score(y_test_cpu, pred_cpu)
    recall = recall_score(y_test_cpu, pred_cpu)
    f1 = f1_score(y_test_cpu, pred_cpu)
    cm = confusion_matrix(y_test_cpu, pred_cpu)
    
    return accuracy, precision, recall, f1, cm

def run_part1():
    print("\n" + "="*50)
    print("PARTIE I : MLP ET INGÉNIERIE PYTORCH")
    print("="*50)
    
    X_train_t, y_train_t, X_val_t, y_val_t, X_test_t, y_test_t, input_dim = load_and_preprocess_data()
    
    # A. Inspection des paramètres
    print("\n--- A. Inspection des paramètres d'un MLP Custom ---")
    temp_model = CustomMLP(input_dim).to(device)
    print(f"Modèle : {temp_model}")
    
    print("\nInspection via named_parameters() :")
    for name, param in temp_model.named_parameters():
        print(f"  Nom: {name:<12} | Taille: {str(list(param.shape)):<15} | Requiert Gradient: {param.requires_grad}")
        
    print("\nInspection des clés du state_dict :")
    print(list(temp_model.state_dict().keys()))
    
    # B. Comparaison des stratégies d'initialisation
    print("\n--- B. Comparaison des stratégies d'initialisation ---")
    initializations = ['gaussian', 'constant', 'xavier']
    history = {}
    best_states = {}
    
    plt.figure(figsize=(10, 6))
    for init_method in initializations:
        model = CustomMLP(input_dim).to(device)
        init_weights(model, init_method)
        
        train_l, val_l, best_state = train_model(model, X_train_t, y_train_t, X_val_t, y_val_t, epochs=150)
        history[init_method] = (train_l, val_l)
        best_states[init_method] = best_state
        
        plt.plot(train_l, label=f"Train - {init_method.capitalize()}")
        plt.plot(val_l, linestyle='--', label=f"Val - {init_method.capitalize()}")
        print(f"Initialisation {init_method.capitalize()} : Perte Val finale minimale = {min(val_l):.4f}")
        
    plt.title("Impact des stratégies d'initialisation sur la perte de l'MLP")
    plt.xlabel("Époque")
    plt.ylabel("Perte (BCE)")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/part1_loss.png", dpi=300)
    plt.close()
    print("Graphique d'initialisation sauvegardé dans results/part1_loss.png")
    
    # C. Sauvegarde et chargement du meilleur modèle (sur la base de Xavier)
    print("\n--- C. Sauvegarde et rechargement du meilleur modèle ---")
    best_xavier_model = CustomMLP(input_dim).to(device)
    best_xavier_model.load_state_dict(best_states['xavier'])
    
    model_path = "results/best_mlp_model.pth"
    torch.save(best_xavier_model.state_dict(), model_path)
    print(f"Meilleur modèle (Xavier) sauvegardé à {model_path}")
    
    # Rechargement et vérification de la cohérence du device
    loaded_model = CustomMLP(input_dim).to(device)
    loaded_model.load_state_dict(torch.load(model_path, map_location=device))
    loaded_model.eval()
    print("Modèle rechargé avec succès et affecté au device.")
    
    # Évaluation sur le jeu de test
    acc, prec, rec, f1, cm = evaluate_model(loaded_model, X_test_t, y_test_t)
    
    print("\n--- D. Métriques d'évaluation sur l'ensemble de Test ---")
    print(f"Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Précision : {prec:.4f}")
    print(f"Rappel    : {rec:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print("Matrice de Confusion :")
    print(cm)
    
    # Sauvegarder la matrice de confusion sous forme de figure
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Sain', 'Tumeur'], yticklabels=['Sain', 'Tumeur'])
    plt.xlabel('Prédiction')
    plt.ylabel('Réel')
    plt.title('Matrice de Confusion - MLP (Breast Cancer)')
    plt.tight_layout()
    plt.savefig("results/part1_confusion.png", dpi=300)
    plt.close()
    print("Matrice de confusion sauvegardée dans results/part1_confusion.png")
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "history": history
    }

if __name__ == "__main__":
    run_part1()
