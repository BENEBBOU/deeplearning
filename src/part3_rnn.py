import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import string
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Configurer le device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Périphérique d'exécution détecté pour la Partie III : {device}")

# 1. Démonstration expérimentale du Gradient Clipping
def demo_gradient_clipping():
    print("\n--- Démonstration expérimentale du Gradient Clipping ---")
    seq_len = 80
    input_dim = 5
    hidden_dim = 10
    num_layers = 2
    
    # Créer deux modèles RNN identiques
    rnn_no_clip = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True).to(device)
    rnn_clip = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True).to(device)
    
    # Initialiser les poids avec de grandes valeurs pour provoquer une explosion de gradient
    for model in [rnn_no_clip, rnn_clip]:
        for name, param in model.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=2.5) # std élevé pour provoquer l'explosion
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
                
    # Données synthétiques
    X = torch.randn(16, seq_len, input_dim).to(device)
    y_target = torch.randn(16, seq_len, hidden_dim).to(device)
    
    criterion = nn.MSELoss()
    
    norms_no_clip = []
    norms_clip = []
    
    # Simulation d'apprentissage sur 50 étapes
    for step in range(50):
        # Sans clipping
        rnn_no_clip.zero_grad()
        out, _ = rnn_no_clip(X)
        loss = criterion(out, y_target)
        loss.backward()
        
        # Calcul de la norme L2 totale du gradient
        total_norm_no_clip = 0.0
        for p in rnn_no_clip.parameters():
            if p.grad is not None:
                total_norm_no_clip += p.grad.data.norm(2).item() ** 2
        total_norm_no_clip = total_norm_no_clip ** 0.5
        norms_no_clip.append(total_norm_no_clip)
        
        # Avec clipping (max_norm = 1.0)
        rnn_clip.zero_grad()
        out_c, _ = rnn_clip(X)
        loss_c = criterion(out_c, y_target)
        loss_c.backward()
        
        # Calcul de la norme avant clipping
        total_norm_clip_before = 0.0
        for p in rnn_clip.parameters():
            if p.grad is not None:
                total_norm_clip_before += p.grad.data.norm(2).item() ** 2
        total_norm_clip_before = total_norm_clip_before ** 0.5
        norms_clip.append(total_norm_clip_before)
        
        # Appliquer le clipping de gradient
        nn.utils.clip_grad_norm_(rnn_clip.parameters(), max_norm=1.0)
        
    plt.figure(figsize=(10, 5))
    plt.plot(norms_no_clip, label="Sans Gradient Clipping (Norme Réelle)")
    plt.plot(norms_clip, linestyle='--', label="Avec Gradient Clipping (Norme avant clip)")
    plt.axhline(y=1.0, color='r', linestyle=':', label="Seuil de Clipping (max_norm=1.0)")
    plt.title("Évolution de la norme du gradient (Explosion vs Stabilité)")
    plt.xlabel("Étape d'apprentissage")
    plt.ylabel("Norme L2 du Gradient")
    plt.yscale("log") # Échelle logarithmique pour mieux voir l'explosion
    plt.legend()
    plt.grid(True)
    plt.savefig("results/part3_grad_clipping.png", dpi=300)
    plt.close()
    print("Graphique du gradient clipping sauvegardé dans results/part3_grad_clipping.png")

# 2. Préparation du jeu de données de Traduction Anglais-Français

def create_and_save_dataset():
    # Génération d'un corpus parallèle de traduction simplifié
    os.makedirs("data", exist_ok=True)
    
    pronouns_en = ["i am", "you are", "he is", "she is", "we are", "they are"]
    pronouns_fr = ["je suis", "tu es", "il est", "elle est", "nous sommes", "ils sont"]
    
    adjectives_en = ["cold", "warm", "tired", "hungry", "happy", "sad", "strong", "ready", "late", "small", "tall"]
    adjectives_fr = ["froid", "chaud", "fatigue", "affame", "heureux", "triste", "fort", "pret", "en retard", "petit", "grand"]
    
    extra_pairs = [
        ("i like dogs", "j'aime les chiens"),
        ("she likes cats", "elle aime les chats"),
        ("we love peace", "nous aimons la paix"),
        ("they want food", "ils veulent manger"),
        ("he works here", "il travaille ici"),
        ("she speaks french", "elle parle francais"),
        ("they speak english", "ils parlent anglais"),
        ("it is cold today", "il fait froid aujourd'hui"),
        ("it is warm outside", "il fait chaud dehors"),
        ("i am very happy", "je suis tres heureux"),
        ("we are very strong", "nous sommes tres forts"),
        ("they are very late", "ils sont tres en retard"),
        ("he is a doctor", "il est medecin"),
        ("she is a teacher", "elle est enseignante")
    ]
    
    pairs = []
    # Générer des combinaisons
    for p_en, p_fr in zip(pronouns_en, pronouns_fr):
        for adj_en, adj_fr in zip(adjectives_en, adjectives_fr):
            pairs.append((f"{p_en} {adj_en}", f"{p_fr} {adj_fr}"))
            # Ajouter une variante avec "very / tres"
            pairs.append((f"{p_en} very {adj_en}", f"{p_fr} tres {adj_fr}"))
            
    pairs.extend(extra_pairs)
    # Multiplier un peu le jeu de données pour l'entraînement
    pairs = pairs * 3
    
    file_path = "data/english_french.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        for en, fr in pairs:
            f.write(f"{en}\t{fr}\n")
            
    print(f"Jeu de données de traduction généré avec {len(pairs)} paires de phrases dans {file_path}")
    return file_path

class TranslationVocabulary:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        self.idx2word = {0: "<PAD>", 1: "<UNK>", 2: "<SOS>", 3: "<EOS>"}
        self.word_counts = {}
        self.num_words = 4
        
    def add_sentence(self, sentence):
        for word in sentence.lower().split():
            # Supprimer la ponctuation simple
            word = word.strip(string.punctuation)
            if word:
                if word not in self.word2idx:
                    self.word2idx[word] = self.num_words
                    self.idx2word[self.num_words] = word
                    self.num_words += 1
                    self.word_counts[word] = 1
                else:
                    self.word_counts[word] += 1

    def numericalize(self, sentence):
        tokens = []
        for word in sentence.lower().split():
            word = word.strip(string.punctuation)
            if word:
                tokens.append(self.word2idx.get(word, self.word2idx["<UNK>"]))
        return tokens

class TranslationDataset(Dataset):
    def __init__(self, file_path, vocab_en=None, vocab_fr=None):
        self.pairs = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    self.pairs.append((parts[0], parts[1]))
                    
        # Initialiser ou réutiliser les vocabulaires
        if vocab_en is None:
            self.vocab_en = TranslationVocabulary()
            self.vocab_fr = TranslationVocabulary()
            for en, fr in self.pairs:
                self.vocab_en.add_sentence(en)
                self.vocab_fr.add_sentence(fr)
        else:
            self.vocab_en = vocab_en
            self.vocab_fr = vocab_fr
            
        self.numerical_pairs = []
        for en, fr in self.pairs:
            en_num = self.vocab_en.numericalize(en)
            fr_num = [self.vocab_fr.word2idx["<SOS>"]] + self.vocab_fr.numericalize(fr) + [self.vocab_fr.word2idx["<EOS>"]]
            self.numerical_pairs.append((en_num, fr_num))
            
    def __len__(self):
        return len(self.pairs)
        
    def __getitem__(self, idx):
        return self.numerical_pairs[idx]

def collate_fn(batch):
    # Padding dynamique des lots
    src_list, trg_list = zip(*batch)
    
    max_src_len = max(len(s) for s in src_list)
    max_trg_len = max(len(t) for t in trg_list)
    
    padded_src = []
    padded_trg = []
    
    for src in src_list:
        padded_src.append(src + [0] * (max_src_len - len(src))) # 0 = <PAD>
    for trg in trg_list:
        padded_trg.append(trg + [0] * (max_trg_len - len(trg)))
        
    return torch.tensor(padded_src, dtype=torch.long), torch.tensor(padded_trg, dtype=torch.long)

# 3. Modèle Seq2Seq (Encoder-Decoder avec GRU)

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=0)
        self.rnn = nn.GRU(emb_dim, hid_dim, batch_first=True)
        
    def forward(self, src):
        # src: [batch_size, src_len]
        embedded = self.embedding(src) # [batch_size, src_len, emb_dim]
        outputs, hidden = self.rnn(embedded)
        # hidden: [1, batch_size, hid_dim]
        return hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=0)
        self.rnn = nn.GRU(emb_dim, hid_dim, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
    def forward(self, input, hidden):
        # input: [batch_size, 1]
        # hidden: [1, batch_size, hid_dim]
        embedded = self.embedding(input) # [batch_size, 1, emb_dim]
        output, hidden = self.rnn(embedded, hidden)
        # output: [batch_size, 1, hid_dim]
        prediction = self.fc_out(output.squeeze(1)) # [batch_size, output_dim]
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [batch_size, src_len]
        # trg: [batch_size, trg_len]
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # Encoder cache l'état
        hidden = self.encoder(src)
        
        # Premier token du décodeur : <SOS>
        input = trg[:, 0].unsqueeze(1)
        
        for t in range(1, trg_len):
            prediction, hidden = self.decoder(input, hidden)
            outputs[:, t] = prediction
            
            teacher_force = np.random.random() < teacher_forcing_ratio
            top1 = prediction.argmax(1).unsqueeze(1)
            input = trg[:, t].unsqueeze(1) if teacher_force else top1
            
        return outputs

# 4. Stratégies de Décodage

def translate_greedy(model, src_tokens, vocab_fr, max_len=15):
    model.eval()
    with torch.no_grad():
        src_tensor = torch.tensor(src_tokens, dtype=torch.long).unsqueeze(0).to(device)
        hidden = model.encoder(src_tensor)
        
        # SOS
        input_token = torch.tensor([[vocab_fr.word2idx["<SOS>"]]], dtype=torch.long).to(device)
        
        translated_tokens = []
        for _ in range(max_len):
            prediction, hidden = model.decoder(input_token, hidden)
            top1 = prediction.argmax(1).item()
            
            if top1 == vocab_fr.word2idx["<EOS>"]:
                break
                
            translated_tokens.append(top1)
            input_token = torch.tensor([[top1]], dtype=torch.long).to(device)
            
    return translated_tokens

def translate_beam_search(model, src_tokens, vocab_fr, beam_width=3, max_len=15):
    model.eval()
    with torch.no_grad():
        src_tensor = torch.tensor(src_tokens, dtype=torch.long).unsqueeze(0).to(device)
        hidden = model.encoder(src_tensor)
        
        # Chaque élément du faisceau : (seq_indices, hidden_state, log_probability)
        beams = [([vocab_fr.word2idx["<SOS>"]], hidden, 0.0)]
        
        for _ in range(max_len):
            candidates = []
            all_done = True
            for seq, hid, log_prob in beams:
                if seq[-1] == vocab_fr.word2idx["<EOS>"]:
                    candidates.append((seq, hid, log_prob))
                    continue
                
                all_done = False
                last_token = torch.tensor([[seq[-1]]], dtype=torch.long).to(device)
                
                prediction, next_hidden = model.decoder(last_token, hid)
                log_probs = F.log_softmax(prediction, dim=1).squeeze(0) # [vocab_size]
                
                # Récupérer les top k meilleures probabilités
                topk_probs, topk_idx = torch.topk(log_probs, beam_width)
                for val, idx in zip(topk_probs, topk_idx):
                    candidates.append((seq + [idx.item()], next_hidden, log_prob + val.item()))
                    
            if all_done:
                break
                
            # Classer les candidats et garder les beam_width meilleurs
            beams = sorted(candidates, key=lambda x: x[2], reverse=True)[:beam_width]
            
        # Renvoyer la meilleure séquence sans <SOS> et <EOS>
        best_seq = beams[0][0]
        cleaned_seq = [idx for idx in best_seq if idx not in [vocab_fr.word2idx["<SOS>"], vocab_fr.word2idx["<EOS>"]]]
        return cleaned_seq

# 5. Entraînement et exécution générale de la Partie III

def run_part3():
    print("\n" + "="*50)
    print("PARTIE III : RNN, LSTM, GRU ET SEQ2SEQ")
    print("="*50)
    
    # A. Démo du Gradient Clipping
    demo_gradient_clipping()
    
    # B. Dataset et Vocabulaire
    file_path = create_and_save_dataset()
    dataset = TranslationDataset(file_path)
    
    # Séparation train/test (90% / 10%)
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate_fn)
    
    print(f"Taille du vocabulaire Anglais : {dataset.vocab_en.num_words}")
    print(f"Taille du vocabulaire Français : {dataset.vocab_fr.num_words}")
    
    # C. Construction du modèle Seq2Seq
    INPUT_DIM = dataset.vocab_en.num_words
    OUTPUT_DIM = dataset.vocab_fr.num_words
    ENC_EMB_DIM = 32
    DEC_EMB_DIM = 32
    HID_DIM = 64
    
    encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM)
    decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM)
    model = Seq2Seq(encoder, decoder, device).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=0) # ignorer <PAD>
    
    # D. Entraînement
    epochs = 15
    train_losses = []
    
    print("\nEntraînement du modèle Seq2Seq...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for src, trg in train_loader:
            src, trg = src.to(device), trg.to(device)
            optimizer.zero_grad()
            
            # trg: [batch_size, trg_len]
            output = model(src, trg, teacher_forcing_ratio=0.5)
            # output: [batch_size, trg_len, vocab_size]
            
            # Réarranger les dimensions pour CrossEntropyLoss
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            
            loss = criterion(output, trg)
            loss.backward()
            
            # Clip gradient
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        perplexity = np.exp(avg_loss)
        print(f"Époque {epoch+1:02d} | Perte : {avg_loss:.4f} | Perplexité : {perplexity:.2f}")
        
    # Sauvegarde de la courbe d'apprentissage
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Perte d'entraînement")
    plt.title("Évolution de la perte - Traducteur Seq2Seq")
    plt.xlabel("Époque")
    plt.ylabel("Perte")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/part3_loss.png", dpi=300)
    plt.close()
    print("Graphique de perte de traduction sauvegardé dans results/part3_loss.png")
    
    # E. Évaluation (BLEU Score et Décodage)
    print("\n--- Évaluation des décodages (Greedy vs Beam Search) ---")
    
    # Sélectionner quelques phrases de test
    smooth_fn = SmoothingFunction().method1
    bleu_greedy_scores = []
    bleu_beam_scores = []
    
    # Afficher quelques exemples de traductions
    test_indices = np.random.choice(len(test_set), size=min(5, len(test_set)), replace=False)
    
    for i, idx in enumerate(test_indices):
        src_tokens, trg_tokens = test_set[idx]
        
        # Convertir en mots
        src_words = [dataset.vocab_en.idx2word[t] for t in src_tokens]
        ref_words = [dataset.vocab_fr.idx2word[t] for t in trg_tokens if t not in [0, 2, 3]] # Exclure PAD, SOS, EOS
        
        # Décodages
        pred_greedy_idx = translate_greedy(model, src_tokens, dataset.vocab_fr)
        pred_beam_idx = translate_beam_search(model, src_tokens, dataset.vocab_fr, beam_width=3)
        
        # Convertir les prédictions en mots
        pred_greedy_words = [dataset.vocab_fr.idx2word[t] for t in pred_greedy_idx]
        pred_beam_words = [dataset.vocab_fr.idx2word[t] for t in pred_beam_idx]
        
        # Calcul BLEU
        bleu_greedy = sentence_bleu([ref_words], pred_greedy_words, smoothing_function=smooth_fn)
        bleu_beam = sentence_bleu([ref_words], pred_beam_words, smoothing_function=smooth_fn)
        
        bleu_greedy_scores.append(bleu_greedy)
        bleu_beam_scores.append(bleu_beam)
        
        print(f"\nExemple {i+1} :")
        print(f"  - Source (EN)   : {' '.join(src_words)}")
        print(f"  - Référence (FR): {' '.join(ref_words)}")
        print(f"  - Greedy (FR)   : {' '.join(pred_greedy_words)} | BLEU: {bleu_greedy:.4f}")
        print(f"  - Beam (FR)     : {' '.join(pred_beam_words)} | BLEU: {bleu_beam:.4f}")
        
    # Score BLEU moyen sur tout l'ensemble de test
    all_bleu_greedy = []
    all_bleu_beam = []
    for idx in range(len(test_set)):
        src_tokens, trg_tokens = test_set[idx]
        ref_words = [dataset.vocab_fr.idx2word[t] for t in trg_tokens if t not in [0, 2, 3]]
        
        pred_greedy = translate_greedy(model, src_tokens, dataset.vocab_fr)
        pred_beam = translate_beam_search(model, src_tokens, dataset.vocab_fr, beam_width=3)
        
        words_greedy = [dataset.vocab_fr.idx2word[t] for t in pred_greedy]
        words_beam = [dataset.vocab_fr.idx2word[t] for t in pred_beam]
        
        all_bleu_greedy.append(sentence_bleu([ref_words], words_greedy, smoothing_function=smooth_fn))
        all_bleu_beam.append(sentence_bleu([ref_words], words_beam, smoothing_function=smooth_fn))
        
    print("\n" + "-"*40)
    print(f"Score BLEU moyen - Greedy Search : {np.mean(all_bleu_greedy):.4f}")
    print(f"Score BLEU moyen - Beam Search   : {np.mean(all_bleu_beam):.4f}")
    print("-"*40)

if __name__ == "__main__":
    run_part3()
