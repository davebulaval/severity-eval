# Plan : severity-eval — A Severity-Aware Evaluation Metric for AI Systems

## 1. Positionnement

### Contribution principale
Une librairie Python (`severity-eval`) compatible HuggingFace `evaluate` qui implémente le cadre actuariel fréquence-sévérité pour l'évaluation de systèmes IA. Accompagnée d'expériences empiriques à large échelle sur 5+ domaines démontrant que l'accuracy seule masque des écarts de risque financier de 10x--100x.

### Gap dans la littérature
**Aucun dataset existant n'annote le coût économique par erreur d'un système IA.** Les benchmarks mesurent la *présence* d'erreurs (accuracy, F1) ou parfois leur *type* (hallucination, factualité), mais jamais leur *impact financier*. Ce travail comble ce gap en :
1. Fournissant un outil de calcul standardisé (librairie)
2. Proposant des taxonomies de sévérité calibrées par domaine
3. Démontrant empiriquement la divergence accuracy vs. risque financier

### Venue cible
- **EMNLP 2027** (System Demonstrations track) ou
- **NeurIPS 2027** (Datasets & Benchmarks track) ou
- **AIES 2027** (AI Ethics and Society)
- **FAccT 2027**

### Relation avec le position paper (NeurIPS 2026)
Le position paper pose le cadre théorique et l'argument conceptuel. Ce second article fournit :
- L'implémentation open-source
- La validation empirique multi-domaines
- Les annotations de sévérité économique (contribution de données)

---

## 2. Architecture de la librairie

### 2.1 API principale

```python
import severity_eval

metric = severity_eval.load(
    cost_levels=[100, 1_000, 10_000, 100_000],
    severity_labels=["negligible", "minor", "major", "critical"],
)

results = metric.compute(
    predictions=predictions,
    references=references,
    severity_annotations=severity_labels,  # par erreur
    n_queries=10_000,
    n_simulations=100_000,
)

# results = {
#     "accuracy": 0.95,
#     "error_rate": 0.05,
#     "pi": [0.371, 0.285, 0.232, 0.112],
#     "mu_X": 13842.0,
#     "expected_loss": 6_920_000,
#     "VaR_95": 8_185_000,
#     "TVaR_95": 8_525_000,
#     "pure_premium": 692.0,
#     "lundberg_reserve": 9_700_000,
#     "lundberg_R": 4.77e-07,
# }
```

### 2.2 Modules

```
severity_eval/
├── __init__.py
├── metric.py              # HuggingFace evaluate compatible
├── compound_loss.py       # Compound Poisson/Binomial simulation
├── risk_measures.py       # VaR, TVaR, expected loss
├── ruin.py                # Lundberg coefficient, reserve requirements
├── routing.py             # HITL routing analysis (retention limit d)
├── severity_taxonomy.py   # Predefined taxonomies by domain
├── visualization.py       # Plots (distribution, surplus, sensitivity)
├── sensitivity.py         # Sensitivity analysis (pi, p, epsilon)
└── domains/
    ├── finance.py         # Cost levels for financial QA
    ├── medical.py         # Cost levels for clinical NLP
    ├── legal.py           # Cost levels for legal AI
    ├── code_security.py   # Cost levels mapped from CVSS
    └── moderation.py      # Cost levels mapped from regulatory fines
```

### 2.3 Intégration HuggingFace evaluate

```python
# Usage via HuggingFace
import evaluate
metric = evaluate.load("severity-eval/compound_loss")
```

### 2.4 Noyau computationnel

Le `simulate.py` du position paper contient déjà :
- Simulation Monte Carlo du modèle composé (Binomial × Categorical)
- Calcul VaR/TVaR par quantiles empiriques
- Coefficient d'ajustement de Lundberg (scipy.optimize.brentq)
- Analyse de routage (retention limit, coût total)
- Export de données pour visualisation

À ajouter :
- Interface `evaluate.Metric` compatible
- Support pour distributions continues (log-normal, Pareto)
- Intervalles de confiance bootstrap
- Comparaison statistique entre deux systèmes (test de dominance stochastique)

---

## 3. Expériences empiriques

### 3.1 Protocole expérimental

Pour chaque dataset :
1. Évaluer 5--10 LLMs (GPT-4o, Claude, Llama, Mistral, Gemini, etc.)
2. Classifier chaque erreur par sévérité (annotation humaine ou proxy)
3. Calculer les métriques actuarielles (E[S], VaR, TVaR, Lundberg reserve)
4. Comparer le classement par accuracy vs. classement par expected loss
5. Mesurer le tau de Kendall entre les deux classements

### 3.2 Hypothèse centrale
> Le classement des systèmes par expected loss diverge significativement du classement par accuracy (tau de Kendall < 0.5), démontrant que l'accuracy est un proxy insuffisant du risque opérationnel.

### 3.3 Datasets sélectionnés

#### Tier 1 — Sévérité native ou quasi-native (expériences principales)

| # | Dataset | Domaine | Taille | Sévérité | Source de sévérité |
|---|---------|---------|--------|----------|-------------------|
| D1 | **StockBench** | Finance (trading) | 20 actions DJIA | Native | P&L direct en $ |
| D2 | **FinanceBench** | Finance (QA) | 10 231 questions SEC | À annoter | Écart numérique sur chiffres financiers |
| D3 | **MedCalc-Bench** | Médical | 11 643 instances | Quasi-native | Catégories Risk/Severity/Diagnosis |
| D4 | **CUAD** | Juridique | 13 000+ labels, 510 contrats | Quasi-native | 3 niveaux de risque contractuel |
| D5 | **CVE-Bench** | Code/Sécurité | 509 CVE | Native | Score CVSS → coût de brèche |
| D6 | **Jigsaw Rate Severity** | Modération | ~14 000 commentaires | Native | Classement pairwise de sévérité |

#### Tier 2 — Sévérité dérivable (expériences complémentaires)

| # | Dataset | Domaine | Taille | Proxy de sévérité |
|---|---------|---------|--------|-------------------|
| D7 | **FinQA** | Finance (raisonnement) | 8 281 QA, S&P 500 | Écart numérique calculable |
| D8 | **LegalBench-RAG** | Juridique | 6 858 QA | Type de clause/précédent |
| D9 | **MedHallu** | Médical | 10 000 QA | Difficulté easy/medium/hard + catégorie clinique |
| D10 | **CyberSecEval** | Code | CWE Top 25 | Mapping CWE → CVSS |
| D11 | **ToolEmu** | Agents IA | 144 cas, 311 outils | Score 0-3 natif (financial loss, property damage) |
| D12 | **WMT MQM** | Traduction | Milliers d'annotations | Minor=1, Major=5, Critical=25 |

#### Tier 3 — Agents IA (expériences exploratoires)

| # | Dataset | Domaine | Taille | Proxy de sévérité |
|---|---------|---------|--------|-------------------|
| D13 | **R-Judge** | Agents multi-domaines | 569 interactions | 10 types de risque annotés |
| D14 | **tau-bench** | Service client | ~115 tâches | Coût transactionnel (remboursement, re-booking) |
| D15 | **InvestorBench** | Finance (investissement) | Actions + crypto + ETF | CR, Sharpe, MDD natifs |

### 3.4 Taxonomies de sévérité par domaine

#### Finance (QA sur documents réglementaires)
| Niveau | Coût $c_k$ | Description | Exemples |
|--------|-----------|-------------|----------|
| Negligible | $100 | Erreur de formatage, arrondi mineur | Revenu arrondi au millier près |
| Minor | $1,000 | Information incomplète | Ratio P/E approximatif |
| Major | $10,000 | Erreur factuelle sur métriques clés | Mauvais trimestre pour un chiffre de revenu |
| Critical | $100,000 | Hallucination de chiffres financiers | Inventer un revenu ou une dette |

#### Médical (calculs cliniques)
| Niveau | Coût $c_k$ | Description | Exemples |
|--------|-----------|-------------|----------|
| Negligible | $500 | Erreur sur métrique non-critique | BMI arrondi |
| Minor | $5,000 | Erreur sur évaluation de routine | Score de Framingham approximatif |
| Major | $50,000 | Erreur sur décision thérapeutique | Mauvais score APACHE II |
| Critical | $500,000 | Erreur sur dosage/triage critique | Score de sepsis incorrect, erreur de dosage |

#### Juridique (analyse contractuelle)
| Niveau | Coût $c_k$ | Description | Exemples |
|--------|-----------|-------------|----------|
| Negligible | $200 | Clause de définition manquée | Définition de terme standard |
| Minor | $2,000 | Information générale incomplète | Date d'entrée en vigueur |
| Major | $20,000 | Clause restrictive manquée | Non-concurrence, confidentialité |
| Critical | $200,000 | Clause de revenue risk manquée | Indemnisation, limitation de responsabilité |

#### Code/Sécurité (vulnérabilités)
| Niveau | Coût $c_k$ | Mapping CVSS | Exemples |
|--------|-----------|--------------|----------|
| Negligible | $1,000 | 0.0--3.9 (Low) | Style, warnings |
| Minor | $10,000 | 4.0--6.9 (Medium) | DoS local |
| Major | $100,000 | 7.0--8.9 (High) | Privilege escalation |
| Critical | $1,000,000 | 9.0--10.0 (Critical) | RCE, SQLi, data breach |

#### Modération de contenu (risque réglementaire)
| Niveau | Coût $c_k$ | Description | Exemples |
|--------|-----------|-------------|----------|
| Negligible | $50 | Contenu mildly offensive non détecté | Insulte légère |
| Minor | $500 | Contenu toxique non détecté | Discours haineux implicite |
| Major | $5,000 | Contenu dangereux non modéré | Menaces, harcèlement ciblé |
| Critical | $50,000 | Violation réglementaire (DSA, COPPA) | CSAM, incitation à la violence |

---

## 4. Figures et résultats attendus

### Figure 1 : Accuracy vs. Expected Loss (scatter plot)
- Axe x : Accuracy du modèle
- Axe y : Expected Loss E[S]
- Un point par (modèle × domaine)
- Démonstration visuelle : la corrélation est faible (tau << 1)

### Figure 2 : Classement divergent (bump chart)
- Gauche : classement par accuracy
- Droite : classement par expected loss
- Lignes qui se croisent = inversions de classement

### Figure 3 : Distribution de sévérité par modèle
- Stacked bar chart : pour chaque modèle, proportion d'erreurs par catégorie de sévérité
- Montre que les modèles diffèrent plus en sévérité qu'en fréquence

### Figure 4 : Impact du routage par domaine
- Bar chart : réduction de E[S] après routage, par domaine
- Montre que le routage est plus bénéfique quand la queue est lourde

### Table 1 : Résultats principaux
- Colonnes : Modèle, Accuracy, E[S], VaR₀.₉₅, TVaR₀.₉₅, Lundberg u*, Rank_acc, Rank_E[S]
- Par domaine (sous-tables ou une grande table multi-panneaux)

### Table 2 : Corrélation accuracy vs. risk measures
- Par domaine : tau de Kendall entre classement accuracy et classement par E[S], VaR, TVaR
- Hypothèse : tau < 0.5 dans les domaines à queue lourde

---

## 5. Structure de l'article

```
1. Introduction
   - Le problème : accuracy ≠ risque opérationnel
   - La solution : severity-eval, première librairie de métriques actuarielles pour l'IA
   - Contributions : (1) librairie, (2) taxonomies, (3) expériences multi-domaines

2. Related Work
   - Métriques d'évaluation IA (accuracy, F1, BLEU, BERTScore, etc.)
   - Cost-sensitive learning (mais pas cost-sensitive evaluation)
   - Risk-aware AI (EU AI Act, NIST AI RMF, mais pas de métriques quantitatives)
   - Position paper (NeurIPS 2026) — cadre théorique que nous implémentons

3. The severity-eval Library
   3.1 Core API and Design Principles
   3.2 Compound Loss Model (Binomial × Categorical)
   3.3 Risk Measures (VaR, TVaR, Lundberg)
   3.4 HITL Routing Analysis
   3.5 Domain-Specific Severity Taxonomies
   3.6 HuggingFace evaluate Integration

4. Experimental Setup
   4.1 Datasets and Severity Annotation Protocol
   4.2 Models Evaluated
   4.3 Evaluation Protocol

5. Results
   5.1 Accuracy vs. Expected Loss: Rank Divergence
   5.2 Severity Profiles Across Models
   5.3 Tail Risk and Reserve Requirements
   5.4 Impact of HITL Routing
   5.5 Sensitivity to Severity Calibration

6. Discussion
   - When does severity-aware evaluation matter most?
   - Limitations of the cost calibration
   - Recommendations for practitioners

7. Conclusion

Appendix A: Severity Annotation Guidelines
Appendix B: Full Results Tables
Appendix C: Library Documentation
```

---

## 6. Timeline estimée

| Phase | Tâches | Livrables |
|-------|--------|-----------|
| **Phase 1** | Librairie core + tests | Package pip installable, tests unitaires |
| **Phase 2** | Taxonomies de sévérité | 5 fichiers de taxonomie par domaine |
| **Phase 3** | Annotation de sévérité | Scripts d'annotation, guidelines, données annotées |
| **Phase 4** | Expériences Tier 1 | Résultats sur D1--D6 |
| **Phase 5** | Expériences Tier 2+3 | Résultats sur D7--D15 |
| **Phase 6** | Rédaction | Article complet |
| **Phase 7** | HuggingFace | Publication métrique sur HuggingFace Hub |

---

## 7. Références clés

### Datasets avec sévérité native
- StockBench: https://github.com/ChenYXxxx/stockbench (Apache 2.0)
- CVE-Bench: NAACL 2025, 509 CVE avec scores CVSS
- Jigsaw Rate Severity: https://www.kaggle.com/competitions/jigsaw-toxic-severity-rating
- MedCalc-Bench: NeurIPS 2024 Oral, https://github.com/ncbi-nlp/MedCalc-Bench (CC-BY-SA-4.0)
- CUAD: https://www.atticusprojectai.org/cuad (CC BY 4.0)
- ToolEmu: ICLR 2024 Spotlight, score de sévérité 0-3 (Apache 2.0)
- R-Judge: EMNLP 2024, 10 types de risque annotés
- WMT MQM: Minor=1, Major=5, Critical=25

### Datasets financiers avec P&L
- InvestorBench: ACL 2025, https://github.com/felis33/INVESTOR-BENCH
- AI-Trader: https://github.com/HKUDS/AI-Trader
- FinRL-Meta: NeurIPS 2022, https://github.com/AI4Finance-Foundation/FinRL (MIT)
- CryptoTrade: EMNLP 2024, CC BY-NC-SA

### Datasets QA haute-valeur
- FinanceBench: https://huggingface.co/datasets/PatronusAI/financebench (CC-BY-NC-4.0)
- FinQA: https://huggingface.co/datasets/ibm-research/finqa (MIT)
- LegalBench-RAG: https://github.com/zeroentropy-ai/legalbenchrag
- MedHallu: https://github.com/MedHallu/MedHallu (MIT)

### Hallucination benchmarks
- RAGTruth: ACL 2024, intensité annotée au niveau mot (MIT)
- HaluBench: https://huggingface.co/datasets/PatronusAI/HaluBench (Apache 2.0)
- RAGChecker: NeurIPS 2024, 4 162 questions, 10 domaines (Apache 2.0)
- RAGEval: ACL 2025, framework génératif (MIT)
- CRAG: Meta, NeurIPS 2024, 4 409 QA (CC-BY-SA-4.0)

### Agents IA
- tau-bench: ICLR 2025, service client retail + airline (MIT)
- ST-WebAgentBench: 375 tâches, 6 dimensions de conformité (Apache 2.0)
- WorkArena: ServiceNow, 19 912 instances (Apache 2.0)
- WebArena: 812 tâches e-commerce/CMS/GitLab (Apache 2.0)

### Travaux connexes sur le coût des erreurs IA
- "AI and Operational Losses" (Boston Fed / Richmond Fed, 2025) — fréquence-sévérité pour pertes bancaires liées à l'IA
- "How Should AI Safety Benchmarks Benchmark Safety?" (2025) — sur 210 benchmarks, seulement 36 distinguent la sévérité
- "The Hallucination Tax" — coût moyen par incident >550k$ (Gartner)
- Air Canada chatbot (2024) — 812$ remboursement pour hallucination
- Hallucination TVA Pologne — perte contrat 4M EUR

### Cas réels pour calibration des coûts
| Cas | Coût | Source |
|-----|------|--------|
| Air Canada chatbot (deuil) | 812 $ | BC Civil Resolution Tribunal, 2024 |
| Citations juridiques hallucinations | Jusqu'à 31 100 $ | Multiple US courts, 2023-2024 |
| Hallucination TVA (appel d'offres) | 4M EUR (contrat perdu) | vatcalc.com |
| Coût moyen incident AI (Gartner) | >550 000 $ | Gartner 2024 |
| Data breach moyen (IBM/Ponemon) | 4,45M $ | IBM Cost of a Data Breach 2023 |
