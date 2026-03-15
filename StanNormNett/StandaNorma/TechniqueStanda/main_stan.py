"""
Programme complet de standardisation avec menu interactif
Gère 3 approches : manuelle, pandas, scikit-learn pour chaque méthode
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
import sys

class StandardisationManager:
    """
    Classe principale pour gérer toutes les méthodes de standardisation
    """
    
    def __init__(self):
        self.data = None
        self.data_original = None
        self.methodes_info = self._init_methodes_info()
        
    def _init_methodes_info(self):
        """Initialise les informations sur chaque méthode"""
        return {
            '1': {
                'nom': 'Z-Score Standardization',
                'formule': "X' = (X - μ) / σ",
                'description': 'Standardisation classique qui centre et réduit les données',
                'resultat': 'moyenne = 0, écart-type = 1',
                'usecase': [
                    '✅ Cas général de Machine Learning',
                    '✅ Données normalement distribuées',
                    '✅ Algorithmes sensibles aux échelles (SVM, PCA, régression)',
                    '✅ Méthode par défaut recommandée'
                ],
                'avantages': [
                    '✓ Conserve les relations entre les données',
                    '✓ Interprétation facile',
                    '✓ Bien adapté aux données gaussiennes'
                ],
                'inconvenients': [
                    '✗ Sensible aux outliers',
                    '✗ Suppose une distribution symétrique'
                ]
            },
            '2': {
                'nom': 'Robust Standardization',
                'formule': "X' = (X - médiane) / IQR\nIQR = Q3 - Q1",
                'description': 'Utilise la médiane et l\'IQR pour être robuste aux outliers',
                'resultat': 'centré autour de 0, échelle basée sur l\'IQR',
                'usecase': [
                    '✅ Données avec outliers',
                    '✅ Distributions asymétriques',
                    '✅ Données financières',
                    '✅ Capteurs avec valeurs aberrantes'
                ],
                'avantages': [
                    '✓ Robuste aux valeurs extrêmes',
                    '✓ Pas d\'hypothèse de distribution',
                    '✓ Médiane moins sensible que la moyenne'
                ],
                'inconvenients': [
                    '✗ Moins standard que Z-Score',
                    '✗ Perte d\'information sur l\'écart-type'
                ]
            },
            '3': {
                'nom': 'Mean Normalization',
                'formule': "X' = (X - μ) / (max - min)",
                'description': 'Centre autour de 0 en utilisant l\'étendue au lieu de l\'écart-type',
                'resultat': 'centré autour de 0, borné entre -0.5 et 0.5 environ',
                'usecase': [
                    '✅ Recherche académique',
                    '✅ Modèles spécifiques',
                    '✅ Quand l\'étendue est plus importante que la variance'
                ],
                'avantages': [
                    '✓ Simple à comprendre',
                    '✓ Borné dans un intervalle fixe'
                ],
                'inconvenients': [
                    '✗ Sensible aux outliers',
                    '✗ Rarement utilisé en pratique',
                    '✗ Perte de la notion d\'écart-type'
                ]
            },
            '4': {
                'nom': 'Unit Vector Standardization',
                'formule': "X' = X / ||X|| (norme L1 ou L2)",
                'description': 'Normalise la longueur du vecteur à 1',
                'resultat': 'norme = 1',
                'usecase': [
                    '✅ NLP (Traitement du langage naturel)',
                    '✅ Similarité cosinus',
                    '✅ Recherche par similarité',
                    '✅ Algorithmes basés sur les distances angulaires'
                ],
                'avantages': [
                    '✓ Préserve les directions',
                    '✓ Idéal pour similarité cosinus',
                    '✓ Indépendant de l\'échelle'
                ],
                'inconvenients': [
                    '✗ Perte de l\'information de magnitude',
                    '✗ Ne préserve pas les distances euclidiennes'
                ]
            }
        }
    
    def saisir_donnees(self):
        """Permet à l'utilisateur de saisir ses données"""
        print("\n" + "="*60)
        print("SAISIE DES DONNÉES")
        print("="*60)
        
        print("\nChoisissez le mode de saisie:")
        print("1. Données exemple (par défaut)")
        print("2. Saisie manuelle")
        print("3. Charger depuis un fichier CSV")
        print("4. Générer des données aléatoires")
        
        choix = input("\nVotre choix (1-4) [1]: ").strip() or "1"
        
        if choix == "1":
            # Données exemple
            self.data = pd.DataFrame({
                'A': [10, 20, 30, 40, 50],
                'B': [2, 4, 6, 8, 100],  # Avec outlier
                'C': [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]  # Suite de Fibonacci
            })
            print("\n✅ Données exemple chargées:")
            print(self.data)
            
        elif choix == "2":
            # Saisie manuelle
            print("\nSaisissez vos données (une ligne par valeur, 'fin' pour terminer):")
            valeurs = []
            while True:
                val = input("> ").strip()
                if val.lower() == 'fin':
                    break
                try:
                    valeurs.append(float(val))
                except ValueError:
                    print("❌ Valeur invalide, essayez encore")
            
            if valeurs:
                self.data = pd.DataFrame({'valeur': valeurs})
                print(f"\n✅ {len(valeurs)} valeurs saisies")
            else:
                print("❌ Aucune valeur saisie, utilisation des données exemple")
                self.data = pd.DataFrame({'A': [10, 20, 30, 40, 50]})
                
        elif choix == "3":
            # Charger depuis CSV
            fichier = input("Nom du fichier CSV: ").strip()
            try:
                self.data = pd.read_csv(fichier)
                print(f"\n✅ Fichier chargé: {self.data.shape[0]} lignes, {self.data.shape[1]} colonnes")
            except Exception as e:
                print(f"❌ Erreur: {e}")
                print("Utilisation des données exemple")
                self.data = pd.DataFrame({'A': [10, 20, 30, 40, 50]})
                
        elif choix == "4":
            # Générer données aléatoires
            n_lignes = int(input("Nombre de lignes [20]: ") or "20")
            n_colonnes = int(input("Nombre de colonnes [3]: ") or "3")
            
            np.random.seed(42)
            data_dict = {}
            for i in range(n_colonnes):
                if i == 1:  # Une colonne avec outlier
                    data_dict[f'col_{i+1}'] = np.random.normal(50, 15, n_lignes)
                    data_dict[f'col_{i+1}'][0] = 1000  # Outlier
                else:
                    data_dict[f'col_{i+1}'] = np.random.normal(50, 15, n_lignes)
            
            self.data = pd.DataFrame(data_dict)
            print(f"\n✅ Données aléatoires générées: {self.data.shape}")
        
        self.data_original = self.data.copy()
        return self.data
    
    def afficher_infos_methode(self, methode_id):
        """Affiche les informations détaillées sur une méthode"""
        info = self.methodes_info[methode_id]
        
        print("\n" + "="*60)
        print(f"📊 {info['nom']}")
        print("="*60)
        print(f"\n📐 Formule: {info['formule']}")
        print(f"\n📝 Description: {info['description']}")
        print(f"\n🎯 Résultat: {info['resultat']}")
        
        print("\n💡 Cas d'utilisation:")
        for cas in info['usecase']:
            print(f"   {cas}")
        
        print("\n✅ Avantages:")
        for avant in info['avantages']:
            print(f"   {avant}")
        
        print("\n❌ Inconvénients:")
        for inconv in info['inconvenients']:
            print(f"   {inconv}")
    
    def afficher_statistiques(self, df_avant, df_apres, colonnes):
        """Affiche les statistiques avant/après de manière détaillée"""
        print("\n" + "-"*50)
        print("📊 STATISTIQUES COMPARATIVES")
        print("-"*50)
        
        # Création d'un DataFrame de comparaison
        stats_avant = df_avant[colonnes].describe()
        stats_apres = df_apres[colonnes].describe()
        
        # Ajouter des métriques supplémentaires
        for col in colonnes:
            print(f"\n▶ Colonne: {col}")
            print(f"   Avant - μ={df_avant[col].mean():.3f}, σ={df_avant[col].std():.3f}, "
                  f"min={df_avant[col].min():.3f}, max={df_avant[col].max():.3f}")
            print(f"   Après - μ={df_apres[col].mean():.3f}, σ={df_apres[col].std():.3f}, "
                  f"min={df_apres[col].min():.3f}, max={df_apres[col].max():.3f}")
            
            # Vérification des propriétés
            if abs(df_apres[col].mean()) < 1e-10:
                print(f"   ✅ Moyenne ≈ 0 (vérifié)")
            if abs(df_apres[col].std() - 1) < 1e-10:
                print(f"   ✅ Écart-type ≈ 1 (vérifié)")
    
    def methode_manuelle_zscore(self, data):
        """Implémentation manuelle du Z-Score"""
        print("\n" + "="*60)
        print("🔧 MÉTHODE MANUELLE - Z-SCORE")
        print("="*60)
        
        result = data.copy()
        colonnes = data.select_dtypes(include=[np.number]).columns
        
        for col in colonnes:
            moy = data[col].mean()
            std = data[col].std()
            result[col] = (data[col] - moy) / std
            print(f"\n{col}: μ={moy:.3f}, σ={std:.3f}")
        
        return result
    
    def methode_pandas_zscore(self, data):
        """Implémentation pandas du Z-Score"""
        print("\n" + "="*60)
        print("🐼 MÉTHODE PANDAS - Z-SCORE")
        print("="*60)
        
        result = data.copy()
        colonnes = data.select_dtypes(include=[np.number]).columns
        
        for col in colonnes:
            result[col] = (data[col] - data[col].mean()) / data[col].std()
        
        return result
    
    def methode_sklearn_zscore(self, data):
        """Implémentation scikit-learn du Z-Score"""
        print("\n" + "="*60)
        print("🤖 MÉTHODE SCIKIT-LEARN - Z-SCORE")
        print("="*60)
        
        result = data.copy()
        colonnes = data.select_dtypes(include=[np.number]).columns
        
        scaler = StandardScaler()
        result[colonnes] = scaler.fit_transform(data[colonnes])
        
        print("\nParamètres du scaler:")
        for i, col in enumerate(colonnes):
            print(f"{col}: mean={scaler.mean_[i]:.3f}, scale={scaler.scale_[i]:.3f}")
        
        return result
    
    def methode_manuelle_robust(self, data):
        """Implémentation manuelle du Robust"""
        print("\n" + "="*60)
        print("🔧 MÉTHODE MANUELLE - ROBUST")
        print("="*60)
        
        result = data.copy()
        colonnes = data.select_dtypes(include=[np.number]).columns
        
        for col in colonnes:
            mediane = data[col].median()
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            result[col] = (data[col] - mediane) / iqr
            print(f"\n{col}: médiane={mediane:.3f}, Q1={q1:.3f}, Q3={q3:.3f}, IQR={iqr:.3f}")
        
        return result
    
    def methode_pandas_robust(self, data):
        """Implémentation pandas du Robust"""
        print("\n" + "="*60)
        print("🐼 MÉTHODE PANDAS - ROBUST")
        print("="*60)
        
        result = data.copy()
        colonnes = data.select_dtypes(include=[np.number]).columns
        
        for col in colonnes:
            mediane = data[col].median()
            iqr = data[col].quantile(0.75) - data[col].quantile(0.25)
            result[col] = (data[col] - mediane) / iqr
        
        return result
    
    def methode_sklearn_robust(self, data):
        """Implémentation scikit-learn du Robust"""
        print("\n" + "="*60)
        print("🤖 MÉTHODE SCIKIT-LEARN - ROBUST")
        print("="*60)
        
        result = data.copy()
        colonnes = data.select_dtypes(include=[np.number]).columns
        
        scaler = RobustScaler()
        result[colonnes] = scaler.fit_transform(data[colonnes])
        
        print("\nParamètres du scaler:")
        for i, col in enumerate(colonnes):
            print(f"{col}: center={scaler.center_[i]:.3f}, scale={scaler.scale_[i]:.3f}")
        
        return result
    
    def methode_manuelle_mean_norm(self, data):
        """Implémentation manuelle de Mean Normalization"""
        print("\n" + "="*60)
        print("🔧 MÉTHODE MANUELLE - MEAN NORMALIZATION")
        print("="*60)
        
        result = data.copy()
        colonnes = data.select_dtypes(include=[np.number]).columns
        
        for col in colonnes:
            moy = data[col].mean()
            min_val = data[col].min()
            max_val = data[col].max()
            range_val = max_val - min_val
            
            if range_val != 0:
                result[col] = (data[col] - moy) / range_val
            else:
                result[col] = 0
            
            print(f"\n{col}: μ={moy:.3f}, min={min_val:.3f}, max={max_val:.3f}, range={range_val:.3f}")
        
        return result
    
    def methode_pandas_mean_norm(self, data):
        """Implémentation pandas de Mean Normalization"""
        print("\n" + "="*60)
        print("🐼 MÉTHODE PANDAS - MEAN NORMALIZATION")
        print("="*60)
        
        result = data.copy()
        colonnes = data.select_dtypes(include=[np.number]).columns
        
        for col in colonnes:
            moy = data[col].mean()
            range_val = data[col].max() - data[col].min()
            
            if range_val != 0:
                result[col] = (data[col] - moy) / range_val
            else:
                result[col] = 0
        
        return result
    
    def methode_sklearn_mean_norm(self, data):
        """Approche scikit-learn pour Mean Normalization (pas de classe directe)"""
        print("\n" + "="*60)
        print("🤖 MÉTHODE SCIKIT-LEARN - MEAN NORMALIZATION")
        print("="*60)
        print("(Pas de classe directe dans sklearn, implémentation personnalisée)")
        
        result = data.copy()
        colonnes = data.select_dtypes(include=[np.number]).columns
        
        for col in colonnes:
            moy = data[col].mean()
            min_val = data[col].min()
            max_val = data[col].max()
            range_val = max_val - min_val
            
            if range_val != 0:
                result[col] = (data[col] - moy) / range_val
            else:
                result[col] = 0
            
            print(f"\n{col}: μ={moy:.3f}, range={range_val:.3f}")
        
        return result
    
    def methode_manuelle_unit_vector(self, data, norm_type='l2'):
        """Implémentation manuelle de Unit Vector"""
        print("\n" + "="*60)
        print(f"🔧 MÉTHODE MANUELLE - UNIT VECTOR (norme {norm_type.upper()})")
        print("="*60)
        
        result = data.copy()
        colonnes = data.select_dtypes(include=[np.number]).columns
        
        for col in colonnes:
            if norm_type == 'l2':
                norm = np.linalg.norm(data[col])
            else:  # l1
                norm = np.sum(np.abs(data[col]))
            
            if norm != 0:
                result[col] = data[col] / norm
            print(f"\n{col}: norme avant={norm:.3f}")
        
        return result
    
    def methode_pandas_unit_vector(self, data, norm_type='l2'):
        """Implémentation pandas de Unit Vector"""
        print("\n" + "="*60)
        print(f"🐼 MÉTHODE PANDAS - UNIT VECTOR (norme {norm_type.upper()})")
        print("="*60)
        
        from sklearn.preprocessing import normalize
        
        result = data.copy()
        colonnes = data.select_dtypes(include=[np.number]).columns
        
        # Normalisation par colonne
        data_normalized = normalize(data[colonnes], norm=norm_type, axis=0)
        result[colonnes] = data_normalized
        
        return result
    
    def methode_sklearn_unit_vector(self, data, norm_type='l2'):
        """Implémentation scikit-learn de Unit Vector"""
        print("\n" + "="*60)
        print(f"🤖 MÉTHODE SCIKIT-LEARN - UNIT VECTOR (norme {norm_type.upper()})")
        print("="*60)
        
        from sklearn.preprocessing import normalize
        
        result = data.copy()
        colonnes = data.select_dtypes(include=[np.number]).columns
        
        result[colonnes] = normalize(data[colonnes], norm=norm_type, axis=0)
        
        return result
    
    def executer_methode(self, methode_id, approche_id):
        """Exécute une méthode spécifique avec l'approche choisie"""
        
        # Mapping des méthodes
        methodes = {
            '1': {  # Z-Score
                '1': self.methode_manuelle_zscore,
                '2': self.methode_pandas_zscore,
                '3': self.methode_sklearn_zscore
            },
            '2': {  # Robust
                '1': self.methode_manuelle_robust,
                '2': self.methode_pandas_robust,
                '3': self.methode_sklearn_robust
            },
            '3': {  # Mean Normalization
                '1': self.methode_manuelle_mean_norm,
                '2': self.methode_pandas_mean_norm,
                '3': self.methode_sklearn_mean_norm
            },
            '4': {  # Unit Vector
                '1': self.methode_manuelle_unit_vector,
                '2': self.methode_pandas_unit_vector,
                '3': self.methode_sklearn_unit_vector
            }
        }
        
        # Afficher les informations de la méthode
        self.afficher_infos_methode(methode_id)
        
        # Demander le type de norme pour Unit Vector si nécessaire
        if methode_id == '4':
            print("\nType de norme:")
            print("1. L2 (Euclidienne) [défaut]")
            print("2. L1 (Manhattan)")
            norm_choix = input("Choix (1-2) [1]: ").strip() or "1"
            norm_type = 'l2' if norm_choix == '1' else 'l1'
        else:
            norm_type = None
        
        # Exécuter la méthode
        print(f"\n📊 AVANT STANDARDISATION:")
        colonnes = self.data.select_dtypes(include=[np.number]).columns
        print(self.data[colonnes].describe().round(3))
        
        # Appliquer la transformation
        if methode_id == '4' and norm_type:
            resultat = methodes[methode_id][approche_id](self.data, norm_type)
        else:
            resultat = methodes[methode_id][approche_id](self.data)
        
        print(f"\n📊 APRÈS STANDARDISATION:")
        print(resultat[colonnes].describe().round(3))
        
        # Afficher les statistiques comparatives détaillées
        self.afficher_statistiques(self.data, resultat, colonnes)
        
        # Proposer de visualiser
        self.visualiser_resultats(self.data, resultat, colonnes)
        
        return resultat
    
    def visualiser_resultats(self, avant, apres, colonnes):
        """Visualise les résultats avant/après"""
        choix = input("\n📈 Voulez-vous visualiser les résultats? (o/n) [n]: ").strip().lower()
        
        if choix == 'o':
            n_cols = len(colonnes)
            fig, axes = plt.subplots(2, n_cols, figsize=(5*n_cols, 10))
            
            if n_cols == 1:
                axes = axes.reshape(2, 1)
            
            for i, col in enumerate(colonnes):
                # Avant
                axes[0, i].hist(avant[col], bins=20, alpha=0.7, color='blue')
                axes[0, i].axvline(avant[col].mean(), color='red', linestyle='--', label=f'Moyenne: {avant[col].mean():.2f}')
                axes[0, i].axvline(avant[col].median(), color='green', linestyle='--', label=f'Médiane: {avant[col].median():.2f}')
                axes[0, i].set_title(f'{col} - AVANT')
                axes[0, i].legend()
                
                # Après
                axes[1, i].hist(apres[col], bins=20, alpha=0.7, color='orange')
                axes[1, i].axvline(apres[col].mean(), color='red', linestyle='--', label=f'Moyenne: {apres[col].mean():.2f}')
                axes[1, i].axvline(apres[col].median(), color='green', linestyle='--', label=f'Médiane: {apres[col].median():.2f}')
                axes[1, i].set_title(f'{col} - APRÈS')
                axes[1, i].legend()
            
            plt.tight_layout()
            plt.show()
    
    def menu_principal(self):
        """Affiche le menu principal"""
        while True:
            print("\n" + "="*60)
            print("🔷 SYSTÈME DE STANDARDISATION - MENU PRINCIPAL")
            print("="*60)
            
            print("\n📋 DONNÉES ACTUELLES:")
            if self.data is not None:
                print(f"   Shape: {self.data.shape}")
                print(f"   Colonnes: {list(self.data.columns)}")
                print("\n   Aperçu:")
                print(self.data.head())
            else:
                print("   ❌ Aucune donnée chargée")
            
            print("\n" + "-"*60)
            print("1. 📂 Charger/Saisir des données")
            print("2. 🔍 Choisir une méthode de standardisation")
            print("3. ℹ️  Informations sur les méthodes")
            print("4. 🔄 Réinitialiser les données")
            print("5. 🚪 Quitter")
            
            choix = input("\n👉 Votre choix: ").strip()
            
            if choix == '1':
                self.saisir_donnees()
            elif choix == '2':
                if self.data is None:
                    print("❌ Veuillez d'abord charger des données!")
                    continue
                self.menu_methodes()
            elif choix == '3':
                self.menu_informations()
            elif choix == '4':
                if self.data_original is not None:
                    self.data = self.data_original.copy()
                    print("✅ Données réinitialisées")
                else:
                    print("❌ Aucune donnée originale à restaurer")
            elif choix == '5':
                print("\n👋 Au revoir!")
                sys.exit(0)
            else:
                print("❌ Choix invalide")
    
    def menu_methodes(self):
        """Menu pour choisir la méthode de standardisation"""
        print("\n" + "="*60)
        print("🔍 CHOIX DE LA MÉTHODE DE STANDARDISATION")
        print("="*60)
        
        print("\nMéthodes disponibles:")
        print("1. 📊 Z-Score Standardization")
        print("2. 🛡️ Robust Standardization")
        print("3. 📐 Mean Normalization")
        print("4. 🔮 Unit Vector Standardization")
        print("5. 🔙 Retour")
        
        choix_methode = input("\n👉 Choisissez une méthode (1-5): ").strip()
        
        if choix_methode == '5':
            return
        
        if choix_methode not in ['1', '2', '3', '4']:
            print("❌ Méthode invalide")
            return
        
        print("\n" + "-"*40)
        print("Approche d'implémentation:")
        print("1. 🔧 Manuel (pur Python)")
        print("2. 🐼 Pandas")
        print("3. 🤖 Scikit-learn")
        print("4. 🔙 Retour")
        
        choix_approche = input("\n👉 Choisissez une approche (1-4): ").strip()
        
        if choix_approche == '4':
            return
        
        if choix_approche not in ['1', '2', '3']:
            print("❌ Approche invalide")
            return
        
        # Exécuter la méthode choisie
        self.executer_methode(choix_methode, choix_approche)
    
    def menu_informations(self):
        """Menu pour afficher les informations sur les méthodes"""
        print("\n" + "="*60)
        print("ℹ️ INFORMATIONS SUR LES MÉTHODES")
        print("="*60)
        
        for i in range(1, 5):
            self.afficher_infos_methode(str(i))
            if i < 4:
                input("\nAppuyez sur Entrée pour continuer...")

def main():
    """Fonction principale"""
    print("\n" + "="*70)
    print("🌟 BIENVENUE DANS LE SYSTÈME COMPLET DE STANDARDISATION 🌟")
    print("="*70)
    print("\nCe programme vous permet de:")
    print("• Comparer 4 méthodes de standardisation")
    print("• Utiliser 3 approches différentes (manuelle, pandas, sklearn)")
    print("• Visualiser les résultats avant/après")
    print("• Comprendre les cas d'utilisation de chaque méthode")
    
    # Créer le gestionnaire
    manager = StandardisationManager()
    
    # Démarrer le menu
    manager.menu_principal()

if __name__ == "__main__":
    main()