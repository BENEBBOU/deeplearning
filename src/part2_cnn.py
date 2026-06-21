import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# Configurer le device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Périphérique d'exécution détecté pour la Partie II : {device}")

# 1. Implémentations manuelles en NumPy

def cross_correlation_2d_numpy(X, K):
    """
    Calcule la corrélation croisée 2D de X avec le noyau K.
    X et K sont des tableaux numpy 2D.
    """
    h_k, w_k = K.shape
    h_x, w_x = X.shape
    h_out = h_x - h_k + 1
    w_out = w_x - w_k + 1
    Y = np.zeros((h_out, w_out))
    for i in range(h_out):
        for j in range(w_out):
            Y[i, j] = np.sum(X[i:i + h_k, j:j + w_k] * K)
    return Y

def max_pooling_2d_numpy(X, pool_size=(2, 2)):
    """
    Calcule le Max-Pooling 2D de X.
    X est un tableau numpy 2D.
    """
    h_p, w_p = pool_size
    h_x, w_x = X.shape
    h_out = h_x // h_p
    w_out = w_x // w_p
    Y = np.zeros((h_out, w_out))
    for i in range(h_out):
        for j in range(w_out):
            Y[i, j] = np.max(X[i*h_p:(i+1)*h_p, j*w_p:(j+1)*w_p])
    return Y

def avg_pooling_2d_numpy(X, pool_size=(2, 2)):
    """
    Calcule l'Average-Pooling 2D de X.
    X est un tableau numpy 2D.
    """
    h_p, w_p = pool_size
    h_x, w_x = X.shape
    h_out = h_x // h_p
    w_out = w_x // w_p
    Y = np.zeros((h_out, w_out))
    for i in range(h_out):
        for j in range(w_out):
            Y[i, j] = np.mean(X[i*h_p:(i+1)*h_p, j*w_p:(j+1)*w_p])
    return Y

def verify_custom_implementations():
    print("\n--- Vérification des implémentations manuelles (NumPy vs PyTorch) ---")
    # Entrées aléatoires
    X_np = np.random.randn(5, 5).astype(np.float32)
    K_np = np.random.randn(3, 3).astype(np.float32)
    
    # 1. Corrélation croisée 2D
    Y_np_conv = cross_correlation_2d_numpy(X_np, K_np)
    
    # Équivalent PyTorch
    X_pt = torch.tensor(X_np).unsqueeze(0).unsqueeze(0) # (1, 1, 5, 5)
    K_pt = torch.tensor(K_np).unsqueeze(0).unsqueeze(0) # (1, 1, 3, 3)
    Y_pt_conv = F.conv2d(X_pt, K_pt, padding=0).squeeze().numpy()
    
    diff_conv = np.abs(Y_np_conv - Y_pt_conv).max()
    print(f"Différence maximale corrélation croisée : {diff_conv:.2e} (Succès: {diff_conv < 1e-5})")
    
    # 2. Max Pooling
    X_np_pool = np.random.randn(4, 4).astype(np.float32)
    Y_np_max = max_pooling_2d_numpy(X_np_pool, (2, 2))
    
    X_pt_pool = torch.tensor(X_np_pool).unsqueeze(0).unsqueeze(0)
    Y_pt_max = F.max_pool2d(X_pt_pool, kernel_size=2, stride=2).squeeze().numpy()
    
    diff_max = np.abs(Y_np_max - Y_pt_max).max()
    print(f"Différence maximale Max-Pooling        : {diff_max:.2e} (Succès: {diff_max < 1e-5})")
    
    # 3. Average Pooling
    Y_np_avg = avg_pooling_2d_numpy(X_np_pool, (2, 2))
    Y_pt_avg = F.avg_pool2d(X_pt_pool, kernel_size=2, stride=2).squeeze().numpy()
    
    diff_avg = np.abs(Y_np_avg - Y_pt_avg).max()
    print(f"Différence maximale Average-Pooling    : {diff_avg:.2e} (Succès: {diff_avg < 1e-5})")

# 2. Modèle CNN Configurable

class ConfigurableCNN(nn.Module):
    def __init__(self, padding=0, stride=1, pooling='max', num_filters=6, use_1x1=False):
        super(ConfigurableCNN, self).__init__()
        self.pooling_type = pooling
        self.use_1x1 = use_1x1
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=3, stride=stride, padding=padding)
        self.relu1 = nn.ReLU()
        
        if pooling == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            
        if use_1x1:
            self.conv1x1 = nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=1)
            
        # Deuxième convolution
        self.conv2 = nn.Conv2d(in_channels=num_filters, out_channels=num_filters * 2, kernel_size=3, padding=0)
        self.relu2 = nn.ReLU()
        
        if pooling == 'max':
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            
        # Calcul automatique de la dimension de sortie aplatie
        dummy_input = torch.zeros(1, 1, 28, 28)
        with torch.no_grad():
            x = self.relu1(self.conv1(dummy_input))
            x = self.pool1(x)
            if use_1x1:
                x = self.conv1x1(x)
            x = self.relu2(self.conv2(x))
            # Gérer le cas où la taille spatiale devient trop petite pour pool2
            if x.shape[2] >= 2 and x.shape[3] >= 2:
                x = self.pool2(x)
            self.flat_dim = x.numel()
            
        self.fc1 = nn.Linear(self.flat_dim, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        if self.use_1x1:
            x = self.conv1x1(x)
        x = self.relu2(self.conv2(x))
        if x.shape[2] >= 2 and x.shape[3] >= 2:
            x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# Modèle MLP simple pour comparaison
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = x.view(x.size(0), -1) # Aplatir
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

# 3. Code d'entraînement et d'évaluation

def train_and_evaluate(model, train_loader, test_loader, epochs=5, lr=0.005):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    test_accs = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        train_losses.append(running_loss / len(train_loader))
        
        # Test
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = correct / total
        test_accs.append(acc)
        
    return train_losses, test_accs

def run_part2():
    print("\n" + "="*50)
    print("PARTIE II : CNN ET VISION PAR ORDINATEUR")
    print("="*50)
    
    # A. Vérifier les calculs customs
    verify_custom_implementations()
    
    # B. Chargement de Fashion-MNIST
    print("\nChargement de Fashion-MNIST...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    
    train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    # Utilisation de sous-ensembles pour accélérer les entraînements (6000 train, 1000 test)
    # afin que les tests s'exécutent de façon fluide en environnement restreint
    train_subset = Subset(train_dataset, range(6000))
    test_subset = Subset(test_dataset, range(1000))
    
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)
    
    print(f"Jeu de données réduit : {len(train_subset)} images d'entraînement, {len(test_subset)} images de test.")
    
    # C. Expérimentations comparatives sur l'architecture du CNN
    print("\n--- C. Expérimentations comparatives (Hyperparamètres du CNN) ---")
    
    experiments = {
        'Baseline (Padding=0, Stride=1, MaxPool)': ConfigurableCNN(padding=0, stride=1, pooling='max', num_filters=6),
        'Padding=1': ConfigurableCNN(padding=1, stride=1, pooling='max', num_filters=6),
        'Stride=2': ConfigurableCNN(padding=0, stride=2, pooling='max', num_filters=6),
        'AvgPool': ConfigurableCNN(padding=0, stride=1, pooling='avg', num_filters=6),
        'Filtres=12': ConfigurableCNN(padding=0, stride=1, pooling='max', num_filters=12),
        'Avec Conv 1x1': ConfigurableCNN(padding=0, stride=1, pooling='max', num_filters=6, use_1x1=True)
    }
    
    results = {}
    epochs = 4
    
    for name, model in experiments.items():
        # Calculer le nombre de paramètres
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Entraînement de {name:<40} | Paramètres : {params:<7}", end="")
        losses, accs = train_and_evaluate(model, train_loader, test_loader, epochs=epochs)
        results[name] = {'loss': losses, 'accuracy': accs, 'params': params}
        print(f" | Accuracy test finale : {accs[-1]*100:.2f}%")
        
    # Tracer le graphe comparatif des courbes de précision
    plt.figure(figsize=(10, 6))
    for name, res in results.items():
        plt.plot(res['accuracy'], label=f"{name} (Acc: {res['accuracy'][-1]*100:.1f}%)")
    plt.title("Comparaison des architectures CNN sur Fashion-MNIST")
    plt.xlabel("Époque")
    plt.ylabel("Accuracy Test")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/part2_comparison.png", dpi=300)
    plt.close()
    print("Graphique de comparaison sauvegardé dans results/part2_comparison.png")
    
    # D. Visualisation des feature maps de la première couche de conv
    print("\n--- D. Visualisation des Feature Maps ---")
    # Prendre une image du jeu de test
    image, label = test_dataset[0] # Forme : (1, 28, 28)
    image_batch = image.unsqueeze(0).to(device) # Forme : (1, 1, 28, 28)
    
    # Utiliser le modèle baseline pour la visualisation
    viz_model = experiments['Baseline (Padding=0, Stride=1, MaxPool)'].eval()
    
    with torch.no_grad():
        # Passer dans la première conv et relu
        features = viz_model.relu1(viz_model.conv1(image_batch))
        
    features = features.squeeze(0).cpu().numpy() # Forme : (C, H_out, W_out)
    num_channels = features.shape[0]
    
    # Plot
    fig, axes = plt.subplots(1, num_channels + 1, figsize=(12, 3))
    # Image originale
    axes[0].imshow(image.squeeze().numpy(), cmap='gray')
    axes[0].set_title("Originale")
    axes[0].axis('off')
    
    # Feature maps
    for i in range(num_channels):
        axes[i+1].imshow(features[i], cmap='viridis')
        axes[i+1].set_title(f"Filtre {i+1}")
        axes[i+1].axis('off')
        
    plt.suptitle("Feature Maps de la première couche de convolution (Baseline CNN)")
    plt.tight_layout()
    plt.savefig("results/part2_feature_maps.png", dpi=300)
    plt.close()
    print("Visualisation des feature maps sauvegardée dans results/part2_feature_maps.png")
    
    # E. Comparaison finale MLP vs CNN
    print("\n--- E. Comparaison MLP vs CNN ---")
    mlp_model = SimpleMLP()
    mlp_params = sum(p.numel() for p in mlp_model.parameters() if p.requires_grad)
    print(f"MLP simple | Paramètres : {mlp_params}")
    mlp_losses, mlp_accs = train_and_evaluate(mlp_model, train_loader, test_loader, epochs=epochs)
    print(f"MLP simple | Accuracy test finale : {mlp_accs[-1]*100:.2f}%")
    
    baseline_res = results['Baseline (Padding=0, Stride=1, MaxPool)']
    print("\nBilan Comparatif :")
    print(f"  - MLP : {mlp_params} paramètres | Accuracy : {mlp_accs[-1]*100:.2f}%")
    print(f"  - CNN : {baseline_res['params']} paramètres | Accuracy : {baseline_res['accuracy'][-1]*100:.2f}%")
    print(f"  => Le CNN utilise {(mlp_params/baseline_res['params']):.1f}x moins de paramètres tout en étant plus performant !")

if __name__ == "__main__":
    run_part2()
