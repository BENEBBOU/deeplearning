import os
import sys

# Ajouter le répertoire de travail au PATH pour pouvoir importer src
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.part1_mlp import run_part1
from src.part2_cnn import run_part2
from src.part3_rnn import run_part3

def main():
    print("="*60)
    # S'assurer que le dossier results existe
    os.makedirs("results", exist_ok=True)
    
    # Exécuter la Partie I
    print("\n>>> DÉMARRAGE DE LA PARTIE I (MLP)...")
    part1_res = run_part1()
    
    # Exécuter la Partie II
    print("\n>>> DÉMARRAGE DE LA PARTIE II (CNN)...")
    run_part2()
    
    # Exécuter la Partie III
    print("\n>>> DÉMARRAGE DE LA PARTIE III (RNN & Seq2Seq)...")
    run_part3()
    
    print("\n" + "="*60)
    print("TOUTES LES SIMULATIONS ONT ÉTÉ EXÉCUTÉES AVEC SUCCÈS !")
    print("Tous les graphiques et modèles ont été enregistrés dans le dossier 'results/'.")
    print("="*60)

if __name__ == "__main__":
    main()
