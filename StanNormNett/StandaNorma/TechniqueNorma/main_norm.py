"""
Programme complet de normalisation avec menu interactif
Gère 3 approches : manuelle, pandas, scikit-learn pour chaque méthode
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import sys

class NormalisationManager:
    """
    Classe principale pour gérer toutes les méthodes de normalisation
    """
    
    def __init__(self):
        self.data = None
        self.data_original = None
        self.methodes_info = self._init_methodes_info()
        
    def _init_methodes_info(self):
        """Initialise les informations sur chaque méthode de normalisation"""
        return {
            '1': {
                'nom': 'Min-Max Normalization',
                'formule': "X' = (X - min) / (max - min)",
                'description': 'Met à l\'échelle les données dans un intervalle fixe [0, 1]',
                'resultat': 'valeurs comprises entre 0 et 1',
                'usecase': [
                    '✅ Algorithmes basés sur les distances (KNN, K-means)',
                    '✅ Réseaux de neurones (sortie sigmoïde)',
                    '✅ Quand on a besoin de bornes fixes',
                    '✅ Traitement d\'images (pixels entre 0 et 1)'
                ],
                'avantages': [
                    '✓ Borné dans un intervalle connu',
                    '✓ Préserve la forme de la distribution',
                    '✓ Simple à interpréter',
                    '✓ Idéal pour les données bornées'
                ],
                'inconvenients': [
                    '✗ Très sensible aux outliers',
                    '✗ Nouvelle donnée hors intervalle pose problème',
                    '✗ Ne gère pas bien les distributions asymétriques'
                ]
            },
            '2': {
                'nom': 'MaxAbs Normalization',
                'formule': "X' = X / max(|X|)",
                'description': 'Normalise par la valeur absolue maximale, préserve le signe',
                'resultat': 'valeurs comprises entre -1 et 1',
                'usecase': [
                    '✅ Données centrées autour de 0',
                    '✅ Données avec valeurs négatives',
                    '✅ Données parcimonieuses (sparse data)',
                    '✅ Algorithmes sensibles au signe'
                ],
                'avantages': [
                    '✓ Préserve le signe des données',
                    '✓ Borné entre -1 et 1',
                    '✓ Idéal pour données centrées',
                    '✓ Bon pour les matrices creuses'
                ],
                'inconvenients': [
                    '✗ Sensible aux outliers',
                    '✗ Suppose que 0 est le centre',
                    '✗ Moins connu que Min-Max'
                ]
            },
            '3': {
                'nom': 'Robust Normalization',
                'formule': "X' = (X - médiane) / (Q3 - Q1)",
                'description': 'Utilise des statistiques robustes pour normaliser',
                'resultat': 'centré autour de 0, échelle basée sur l\'IQR',
                'usecase': [
                    '✅ Données avec outliers',
                    '✅ Distributions asymétriques',
                    '✅ Données financières',
                    '✅ Capteurs avec valeurs aberrantes'
                ],
                'avantages': [
                    '✓ Robuste aux outliers',
                    '✓ Pas d\'hypothèse de distribution',
                    '✓ Médiane moins sensible que la moyenne'
                ],
                'inconvenients': [
                    '✗ Pas borné dans un intervalle fixe',
                    '✗ Moins standard que Min-Max'
                ]
            },
            '4': {
                'nom': 'Quantile Normalization',
                'formule': "X' = G⁻¹(F(X))",
                'description': 'Transforme pour suivre une distribution uniforme ou normale',
                'resultat': 'distribution uniforme ou normale',
                'usecase': [
                    '✅ Données non-normales',
                    '✅ Quand on veut une distribution spécifique',
                    '✅ Pour éliminer les effets d\'échelle',
                    '✅ Analyse de données biologiques (RNA-seq)'
                ],
                'avantages': [
                    '✓ Rend la distribution uniforme/normale',
                    '✓ Très robuste',
                    '✓ Idéal pour comparer des distributions différentes'
                ],
                'inconvenients': [
                    '✗ Perd la forme originale',
                    '✗ Complexe à interpréter',
                    '✗ Peut créer des artefacts'
                ]
            },
            '5': {
                'nom': 'Decimal Scaling Normalization',
                'formule': "X' = X / 10^k, où k = log10(max|X|)",
                'description': 'Normalise en déplaçant la virgule décimale',
                'resultat': 'valeurs entre -1 et 1',
                'usecase': [
                    '✅ Données avec grands nombres',
                    '✅ Cas simples sans outliers extrêmes',
                    '✅ Pré-traitement rapide'
                ],
                'avantages': [
                    '✓ Très simple à calculer',
                    '✓ Préserve les relations',
                    '✓ Intuitif'
                ],
                'inconvenients': [
                    '✗ Dépend de la base 10',
                    '✗ Pas optimal pour tous les cas',
                    '✗ Rarement utilisé en pratique'
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
            # Données exemple avec différents cas
            np.random.seed(42)
            self.data = pd.DataFrame({
                'A': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],  # Linéaire
                'B': [-5, -3, 0, 2, 5, 7, 10, 12, 15, 18],  # Avec négatifs
                'C': np.random.normal(50, 15, 20).round(1),  # Normale
                'D': [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],  # Exponentielle
                'E': [10, 20, 30, 40, 50, 60, 70, 80, 90, 1000]  # Avec outlier
            })
            print("\n✅ Données exemple chargées:")
            print(self.data.head())
            
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
            type_dist = input("Type de distribution (normale/uniforme/exponentielle) [normale]: ").strip() or "normale"
            
            np.random.seed(42)
            data_dict = {}
            for i in range(n_colonnes):
                if type_dist.lower() == "normale":
                    data_dict[f'col_{i+1}'] = np.random.normal(50, 15, n_lignes)
                elif type_dist.lower() == "uniforme":
                    data_dict[f'col_{i+1}'] = np.random.uniform(0, 100, n_lignes)
                elif type_dist.lower() == "exponentielle":
                    data_dict[f'col_{i+1}'] = np.random.exponential(20, n_lignes)
            
            self.data = pd.DataFrame(data_dict)
            print(f"\n✅ Données {type_dist} générées: {self.data.shape}")
        
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
        
        for col in colonnes:
            print(f"\n▶ Colonne: {col}")
            print(f"   Avant - min={df_avant[col].min():.3f}, max={df_avant[col].max():.3f}, "
                  f"μ={df_avant[col].mean():.3f}, σ={df_avant[col].std():.3f}")
            print(f"   Après - min={df_apres[col].min():.3f}, max={df_apres[col].max():.3f}, "
                  f"μ={df_apres[col].mean():.3f}, σ={df_apres[col].std():.3f}")
    
    # ==================== MIN-MAX NORMALIZATION ====================
    
    def methode_manuelle_minmax(self, data, feature_range=(0, 1)):
        """Implémentation manuelle du Min-Max"""
        print("\n" + "="*60)
        print(f"🔧 MÉTHODE MANUELLE - MIN-MAX [{feature_range[0]}, {feature_range[1]}]")
        print("="*60)
        
        result = data.copy()
        colonnes = data.select_dtypes(include=[np.number]).columns
        min_target, max_target = feature_range
        
        for col in colonnes:
            min_val = data[col].min()
            max_val = data[col].max()
            range_val = max_val - min_val
            
            if range_val != 0:
                # Min-Max standard
                result[col] = (data[col] - min_val) / range_val
                # Ajustement à l'intervalle cible
                result[col] = result[col] * (max_target - min_target) + min_target
            else:
                result[col] = min_target
            
            print(f"\n{col}: min={min_val:.3f}, max={max_val:.3f}, range={range_val:.3f}")
        
        return result
    
    def methode_pandas_minmax(self, data, feature_range=(0, 1)):
        """Implémentation pandas du Min-Max"""
        print("\n" + "="*60)
        print(f"🐼 MÉTHODE PANDAS - MIN-MAX [{feature_range[0]}, {feature_range[1]}]")
        print("="*60)
        
        result = data.copy()
        colonnes = data.select_dtypes(include=[np.number]).columns
        min_target, max_target = feature_range
        
        for col in colonnes:
            min_val = data[col].min()
            max_val = data[col].max()
            range_val = max_val - min_val
            
            if range_val != 0:
                result[col] = (data[col] - min_val) / range_val
                result[col] = result[col] * (max_target - min_target) + min_target
            else:
                result[col] = min_target
        
        return result
    
    def methode_sklearn_minmax(self, data, feature_range=(0, 1)):
        """Implémentation scikit-learn du Min-Max"""
        print("\n" + "="*60)
        print(f"🤖 MÉTHODE SCIKIT-LEARN - MIN-MAX [{feature_range[0]}, {feature_range[1]}]")
        print("="*60)
        
        from sklearn.preprocessing import MinMaxScaler
        
        result = data.copy()
        colonnes = data.select_dtypes(include=[np.number]).columns
        
        scaler = MinMaxScaler(feature_range=feature_range)
        result[colonnes] = scaler.fit_transform(data[colonnes])
        
        print("\nParamètres du scaler:")
        for i, col in enumerate(colonnes):
            print(f"{col}: min_={scaler.data_min_[i]:.3f}, scale_={scaler.scale_[i]:.3f}")
        
        return result
    
    # ==================== MAX-ABS NORMALIZATION ====================
    
    def methode_manuelle_maxabs(self, data):
        """Implémentation manuelle du MaxAbs"""
        print("\n" + "="*60)
        print("🔧 MÉTHODE MANUELLE - MAX-ABS [-1, 1]")
        print("="*60)
        
        result = data.copy()
        colonnes = data.select_dtypes(include=[np.number]).columns
        
        for col in colonnes:
            max_abs = np.max(np.abs(data[col]))
            
            if max_abs != 0:
                result[col] = data[col] / max_abs
            else:
                result[col] = 0
            
            print(f"\n{col}: max_abs={max_abs:.3f}")
        
        return result
    
    def methode_pandas_maxabs(self, data):
        """Implémentation pandas du MaxAbs"""
        print("\n" + "="*60)
        print("🐼 MÉTHODE PANDAS - MAX-ABS [-1, 1]")
        print("="*60)
        
        result = data.copy()
        colonnes = data.select_dtypes(include=[np.number]).columns
        
        for col in colonnes:
            max_abs = np.max(np.abs(data[col]))
            if max_abs != 0:
                result[col] = data[col] / max_abs
            else:
                result[col] = 0
        
        return result
    
    def methode_sklearn_maxabs(self, data):
        """Implémentation scikit-learn du MaxAbs"""
        print("\n" + "="*60)
        print("🤖 MÉTHODE SCIKIT-LEARN - MAX-ABS [-1, 1]")
        print("="*60)
        
        from sklearn.preprocessing import MaxAbsScaler
        
        result = data.copy()
        colonnes = data.select_dtypes(include=[np.number]).columns
        
        scaler = MaxAbsScaler()
        result[colonnes] = scaler.fit_transform(data[colonnes])
        
        print("\nParamètres du scaler:")
        for i, col in enumerate(colonnes):
            print(f"{col}: scale_={scaler.scale_[i]:.3f}, max_abs_={scaler.max_abs_[i]:.3f}")
        
        return result
    
    # ==================== ROBUST NORMALIZATION ====================
    
    def methode_manuelle_robust(self, data):
        """Implémentation manuelle de la normalisation robuste"""
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
            
            if iqr != 0:
                result[col] = (data[col] - mediane) / iqr
            else:
                result[col] = 0
            
            print(f"\n{col}: médiane={mediane:.3f}, Q1={q1:.3f}, Q3={q3:.3f}, IQR={iqr:.3f}")
        
        return result
    
    def methode_pandas_robust(self, data):
        """Implémentation pandas de la normalisation robuste"""
        print("\n" + "="*60)
        print("🐼 MÉTHODE PANDAS - ROBUST")
        print("="*60)
        
        result = data.copy()
        colonnes = data.select_dtypes(include=[np.number]).columns
        
        for col in colonnes:
            mediane = data[col].median()
            iqr = data[col].quantile(0.75) - data[col].quantile(0.25)
            
            if iqr != 0:
                result[col] = (data[col] - mediane) / iqr
            else:
                result[col] = 0
        
        return result
    
    def methode_sklearn_robust(self, data):
        """Implémentation scikit-learn de la normalisation robuste"""
        print("\n" + "="*60)
        print("🤖 MÉTHODE SCIKIT-LEARN - ROBUST")
        print("="*60)
        
        from sklearn.preprocessing import RobustScaler
        
        result = data.copy()
        colonnes = data.select_dtypes(include=[np.number]).columns
        
        scaler = RobustScaler()
        result[colonnes] = scaler.fit_transform(data[colonnes])
        
        print("\nParamètres du scaler:")
        for i, col in enumerate(colonnes):
            print(f"{col}: center_={scaler.center_[i]:.3f}, scale_={scaler.scale_[i]:.3f}")
        
        return result
    
    # ==================== QUANTILE NORMALIZATION ====================
    
    def methode_manuelle_quantile(self, data, output_distribution='uniform'):
        """Approche manuelle approximative pour quantile"""
        print("\n" + "="*60)
        print(f"🔧 MÉTHODE MANUELLE - QUANTILE ({output_distribution})")
        print("="*60)
        print("(Version simplifiée - pour la vraie transformation, utilisez sklearn)")
        
        result = data.copy()
        colonnes = data.select_dtypes(include=[np.number]).columns
        
        for col in colonnes:
            # Tri des valeurs
            sorted_vals = np.sort(data[col])
            n = len(sorted_vals)
            
            # Rangs (percentiles)
            ranks = np.argsort(np.argsort(data[col])) + 1
            percentiles = ranks / (n + 1)
            
            if output_distribution == 'uniform':
                # Distribution uniforme [0, 1]
                result[col] = percentiles
            else:  # 'normal'
                # Approximation de la distribution normale
                from scipy import stats
                result[col] = stats.norm.ppf(percentiles)
            
            print(f"\n{col}: transformation {output_distribution} appliquée")
        
        return result
    
    def methode_pandas_quantile(self, data, output_distribution='uniform'):
        """Implémentation pandas pour quantile"""
        print("\n" + "="*60)
        print(f"🐼 MÉTHODE PANDAS - QUANTILE ({output_distribution})")
        print("="*60)
        
        result = data.copy()
        colonnes = data.select_dtypes(include=[np.number]).columns
        
        for col in colonnes:
            # Rangs
            ranks = data[col].rank(method='average')
            percentiles = (ranks - 1) / (len(data[col]) - 1)
            
            if output_distribution == 'uniform':
                result[col] = percentiles
            else:  # 'normal'
                from scipy import stats
                result[col] = stats.norm.ppf(percentiles.clip(0.001, 0.999))
            
            print(f"\n{col}: transformation {output_distribution} appliquée")
        
        return result
    
    def methode_sklearn_quantile(self, data, output_distribution='uniform'):
        """Implémentation scikit-learn de quantile"""
        print("\n" + "="*60)
        print(f"🤖 MÉTHODE SCIKIT-LEARN - QUANTILE ({output_distribution})")
        print("="*60)
        
        from sklearn.preprocessing import QuantileTransformer
        
        result = data.copy()
        colonnes = data.select_dtypes(include=[np.number]).columns
        
        transformer = QuantileTransformer(output_distribution=output_distribution, random_state=42)
        result[colonnes] = transformer.fit_transform(data[colonnes])
        
        print(f"\n✅ Transformation quantile ({output_distribution}) appliquée")
        print(f"n_quantiles_: {transformer.n_quantiles_}")
        
        return result
    
    # ==================== DECIMAL SCALING ====================
    
    def methode_manuelle_decimal(self, data):
        """Implémentation manuelle du Decimal Scaling"""
        print("\n" + "="*60)
        print("🔧 MÉTHODE MANUELLE - DECIMAL SCALING")
        print("="*60)
        
        result = data.copy()
        colonnes = data.select_dtypes(include=[np.number]).columns
        
        for col in colonnes:
            max_abs = np.max(np.abs(data[col]))
            if max_abs > 0:
                k = np.ceil(np.log10(max_abs))
                result[col] = data[col] / (10 ** k)
                print(f"\n{col}: max_abs={max_abs:.3f}, k={int(k)}")
            else:
                result[col] = 0
        
        return result
    
    def methode_pandas_decimal(self, data):
        """Implémentation pandas du Decimal Scaling"""
        print("\n" + "="*60)
        print("🐼 MÉTHODE PANDAS - DECIMAL SCALING")
        print("="*60)
        
        result = data.copy()
        colonnes = data.select_dtypes(include=[np.number]).columns
        
        for col in colonnes:
            max_abs = np.max(np.abs(data[col]))
            if max_abs > 0:
                k = np.ceil(np.log10(max_abs))
                result[col] = data[col] / (10 ** k)
            else:
                result[col] = 0
        
        return result
    
    def methode_sklearn_decimal(self, data):
        """Pas de classe directe dans sklearn, implémentation personnalisée"""
        print("\n" + "="*60)
        print("🤖 MÉTHODE SCIKIT-LEARN - DECIMAL SCALING")
        print("="*60)
        print("(Pas de classe directe, implémentation personnalisée)")
        
        result = data.copy()
        colonnes = data.select_dtypes(include=[np.number]).columns
        
        for col in colonnes:
            max_abs = np.max(np.abs(data[col]))
            if max_abs > 0:
                k = np.ceil(np.log10(max_abs))
                result[col] = data[col] / (10 ** k)
                print(f"\n{col}: division par 10^{int(k)}")
        
        return result
    
    def executer_methode(self, methode_id, approche_id):
        """Exécute une méthode spécifique avec l'approche choisie"""
        
        # Mapping des méthodes
        methodes = {
            '1': {  # Min-Max
                '1': self.methode_manuelle_minmax,
                '2': self.methode_pandas_minmax,
                '3': self.methode_sklearn_minmax
            },
            '2': {  # MaxAbs
                '1': self.methode_manuelle_maxabs,
                '2': self.methode_pandas_maxabs,
                '3': self.methode_sklearn_maxabs
            },
            '3': {  # Robust
                '1': self.methode_manuelle_robust,
                '2': self.methode_pandas_robust,
                '3': self.methode_sklearn_robust
            },
            '4': {  # Quantile
                '1': self.methode_manuelle_quantile,
                '2': self.methode_pandas_quantile,
                '3': self.methode_sklearn_quantile
            },
            '5': {  # Decimal Scaling
                '1': self.methode_manuelle_decimal,
                '2': self.methode_pandas_decimal,
                '3': self.methode_sklearn_decimal
            }
        }
        
        # Afficher les informations de la méthode
        self.afficher_infos_methode(methode_id)
        
        # Paramètres spécifiques pour certaines méthodes
        feature_range = (0, 1)
        output_distribution = 'uniform'
        
        if methode_id == '1':  # Min-Max
            print("\nIntervalle cible:")
            print("1. [0, 1] (défaut)")
            print("2. [-1, 1]")
            print("3. Personnalisé")
            range_choix = input("Choix (1-3) [1]: ").strip() or "1"
            
            if range_choix == '2':
                feature_range = (-1, 1)
            elif range_choix == '3':
                min_val = float(input("Borne inférieure: "))
                max_val = float(input("Borne supérieure: "))
                feature_range = (min_val, max_val)
        
        elif methode_id == '4':  # Quantile
            print("\nDistribution de sortie:")
            print("1. Uniforme [0, 1] (défaut)")
            print("2. Normale")
            dist_choix = input("Choix (1-2) [1]: ").strip() or "1"
            output_distribution = 'uniform' if dist_choix == '1' else 'normal'
        
        # Exécuter la méthode
        print(f"\n📊 AVANT NORMALISATION:")
        colonnes = self.data.select_dtypes(include=[np.number]).columns
        print(self.data[colonnes].describe().round(3))
        
        # Appliquer la transformation
        if methode_id == '1':
            resultat = methodes[methode_id][approche_id](self.data, feature_range)
        elif methode_id == '4':
            resultat = methodes[methode_id][approche_id](self.data, output_distribution)
        else:
            resultat = methodes[methode_id][approche_id](self.data)
        
        print(f"\n📊 APRÈS NORMALISATION:")
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
                axes[0, i].axvline(avant[col].min(), color='red', linestyle='--', label=f'Min: {avant[col].min():.2f}')
                axes[0, i].axvline(avant[col].max(), color='green', linestyle='--', label=f'Max: {avant[col].max():.2f}')
                axes[0, i].set_title(f'{col} - AVANT')
                axes[0, i].legend()
                
                # Après
                axes[1, i].hist(apres[col], bins=20, alpha=0.7, color='orange')
                axes[1, i].axvline(apres[col].min(), color='red', linestyle='--', label=f'Min: {apres[col].min():.2f}')
                axes[1, i].axvline(apres[col].max(), color='green', linestyle='--', label=f'Max: {apres[col].max():.2f}')
                axes[1, i].set_title(f'{col} - APRÈS')
                axes[1, i].legend()
            
            plt.tight_layout()
            plt.show()
    
    def menu_principal(self):
        """Affiche le menu principal"""
        while True:
            print("\n" + "="*70)
            print("🔷 SYSTÈME DE NORMALISATION - MENU PRINCIPAL")
            print("="*70)
            
            print("\n📋 DONNÉES ACTUELLES:")
            if self.data is not None:
                print(f"   Shape: {self.data.shape}")
                print(f"   Colonnes: {list(self.data.columns)}")
                print("\n   Aperçu (5 premières lignes):")
                print(self.data.head().to_string())
            else:
                print("   ❌ Aucune donnée chargée")
            
            print("\n" + "-"*70)
            print("1. 📂 Charger/Saisir des données")
            print("2. 🔍 Choisir une méthode de normalisation")
            print("3. ℹ️  Informations sur les méthodes")
            print("4. 🔄 Réinitialiser les données")
            print("5. 📊 Comparer toutes les méthodes")
            print("6. 🚪 Quitter")
            
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
                if self.data is None:
                    print("❌ Veuillez d'abord charger des données!")
                    continue
                self.comparer_toutes_methodes()
            elif choix == '6':
                print("\n👋 Au revoir!")
                sys.exit(0)
            else:
                print("❌ Choix invalide")
    
    def menu_methodes(self):
        """Menu pour choisir la méthode de normalisation"""
        print("\n" + "="*70)
        print("🔍 CHOIX DE LA MÉTHODE DE NORMALISATION")
        print("="*70)
        
        print("\nMéthodes disponibles:")
        print("1. 📊 Min-Max Normalization [0,1] ou [-1,1]")
        print("2. 📈 MaxAbs Normalization [-1,1]")
        print("3. 🛡️ Robust Normalization")
        print("4. 🔮 Quantile Normalization (uniforme/normale)")
        print("5. 🔟 Decimal Scaling Normalization")
        print("6. 🔙 Retour")
        
        choix_methode = input("\n👉 Choisissez une méthode (1-6): ").strip()
        
        if choix_methode == '6':
            return
        
        if choix_methode not in ['1', '2', '3', '4', '5']:
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
        print("\n" + "="*70)
        print("ℹ️ INFORMATIONS SUR LES MÉTHODES DE NORMALISATION")
        print("="*70)
        
        for i in range(1, 6):
            self.afficher_infos_methode(str(i))
            if i < 5:
                input("\nAppuyez sur Entrée pour continuer...")
    
    def comparer_toutes_methodes(self):
        """Compare toutes les méthodes sur les données actuelles"""
        print("\n" + "="*70)
        print("📊 COMPARAISON DE TOUTES LES MÉTHODES")
        print("="*70)
        
        colonnes = self.data.select_dtypes(include=[np.number]).columns
        
        # Dictionnaire pour stocker tous les résultats
        tous_resultats = {'original': self.data[colonnes]}
        
        print("\nApplication de toutes les méthodes (approche sklearn)...")
        
        # Min-Max [0,1]
        result_minmax = self.methode_sklearn_minmax(self.data)
        tous_resultats['minmax'] = result_minmax[colonnes]
        
        # MaxAbs
        result_maxabs = self.methode_sklearn_maxabs(self.data)
        tous_resultats['maxabs'] = result_maxabs[colonnes]
        
        # Robust
        result_robust = self.methode_sklearn_robust(self.data)
        tous_resultats['robust'] = result_robust[colonnes]
        
        # Quantile uniforme
        result_quantile = self.methode_sklearn_quantile(self.data, 'uniform')
        tous_resultats['quantile'] = result_quantile[colonnes]
        
        # Decimal Scaling
        result_decimal = self.methode_sklearn_decimal(self.data)
        tous_resultats['decimal'] = result_decimal[colonnes]
        
        # Créer un tableau comparatif
        print("\n" + "="*70)
        print("TABLEAU COMPARATIF - PREMIÈRE LIGNE DE DONNÉES")
        print("="*70)
        
        table_comparative = pd.DataFrame()
        for nom, df in tous_resultats.items():
            table_comparative[nom] = df.iloc[0].values
        
        print(table_comparative.round(4).to_string())
        
        # Statistiques par méthode
        print("\n" + "="*70)
        print("STATISTIQUES PAR MÉTHODE")
        print("="*70)
        
        for nom, df in tous_resultats.items():
            print(f"\n▶ {nom.upper()}:")
            print(f"   min={df.min().min():.3f}, max={df.max().max():.3f}")
            print(f"   μ={df.mean().mean():.3f}, σ={df.std().mean():.3f}")
        
        # Visualisation
        choix = input("\n📈 Voulez-vous visualiser la comparaison? (o/n) [n]: ").strip().lower()
        if choix == 'o':
            self.visualiser_comparaison(tous_resultats)
    
    def visualiser_comparaison(self, tous_resultats):
        """Visualise la comparaison de toutes les méthodes"""
        n_methodes = len(tous_resultats)
        fig, axes = plt.subplots(2, (n_methodes + 1)//2, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (nom, df) in enumerate(tous_resultats.items()):
            if idx < len(axes):
                # Prendre la première colonne pour l'exemple
                col = df.columns[0]
                axes[idx].hist(df[col], bins=20, alpha=0.7)
                axes[idx].set_title(f'{nom}')
                axes[idx].axvline(df[col].min(), color='red', linestyle='--', label=f'Min: {df[col].min():.2f}')
                axes[idx].axvline(df[col].max(), color='green', linestyle='--', label=f'Max: {df[col].max():.2f}')
                axes[idx].legend()
        
        # Cacher les axes vides
        for idx in range(len(tous_resultats), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.show()

def main():
    """Fonction principale"""
    print("\n" + "="*80)
    print("🌟 BIENVENUE DANS LE SYSTÈME COMPLET DE NORMALISATION 🌟")
    print("="*80)
    print("\nCe programme vous permet de:")
    print("• Comparer 5 méthodes de normalisation")
    print("• Utiliser 3 approches différentes (manuelle, pandas, sklearn)")
    print("• Visualiser les résultats avant/après")
    print("• Comprendre les cas d'utilisation de chaque méthode")
    print("\nMéthodes disponibles:")
    print("1. Min-Max Normalization [0,1] ou [-1,1]")
    print("2. MaxAbs Normalization [-1,1]")
    print("3. Robust Normalization")
    print("4. Quantile Normalization")
    print("5. Decimal Scaling")
    
    # Créer le gestionnaire
    manager = NormalisationManager()
    
    # Démarrer le menu
    manager.menu_principal()

if __name__ == "__main__":
    main()