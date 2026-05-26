# Revue ACL/EMNLP - focus faiblesses

Manuscrit : `paper/main.tex`
Posture : reviewer ACL/EMNLP, brouillon (review mode).

## Résumé du papier

Évaluation de LLM via le modèle actuariel fréquence-sévérité de Beauchemin et al. (2026, position paper). Annonce 23 modèles × 11 benchmarks × 4 domaines, avec une bibliothèque `severity-eval`. Cinq RQ structurent l'analyse.

## Faiblesses majeures

### 1. Décalage critique entre la promesse expérimentale et l'exécution
L'abstract et l'intro annoncent « 23 language models across 11 benchmarks », mais la Section 6.0 (« Sample size caveat ») admet que seul FinanceBench atteint la couverture multi-modèles ; les autres domaines ont 1 modèle. Pour un papier dont le titre est « An Empirical Study », c'est rédhibitoire. Les conclusions de 4 des 5 RQ reposent sur 1 domaine, voire 1 modèle (RQ4 : « insurance 99.9 % single model »). À retravailler avant resoumission ou à recadrer comme « preliminary methodology paper ».

### 2. Les résultats empiriques contredisent la motivation
- RQ1 : Kendall τ = 0,96 (p = 5×10⁻⁴) entre accuracy et E[S].
- RQ3 : 0 inversion de classement.
- RQ5 : ρ minimal = 1,00 sous perturbation ±20 %.

Autrement dit, sur la seule donnée disponible, **accuracy est un proxy fidèle de E[S]**. C'est le contraire du « gap between accuracy and economic risk » qui justifie le papier (intro, §1, lignes 113-123). La Section 6.1 reconnaît cette observation en une phrase mais ne réinterprète pas la thèse centrale. Un reviewer attendra soit (a) des résultats sur les domaines à queue plus lourde (médical, assurance) où la divergence devrait apparaître ; soit (b) un repositionnement honnête (« quand est-ce que la décomposition apporte de la valeur ? »).

### 3. RQ2 ne sauve pas le cadre
La variance-decomposition donne 61 % de variance due à π, 39 % à p. Mais si les rangs coïncident (RQ1), la décomposition explique une variance dont l'ordre est déjà capturé par accuracy. Le mécanisme E[S] = np·μ_X est arithmétique, pas empirique : démontrer que μ_X varie entre modèles ne démontre pas que cette information change une décision. Cette ambiguïté doit être traitée frontalement.

### 4. Annotation de sévérité : κ = 0,46 est faible pour un papier méthodologique
Cohen's κ = 0,46 est qualifié de « moderate agreement » (Section 8 limitations). Pour une contribution dont l'un des 4 piliers est « severity-annotated benchmarks », c'est en deçà du seuil habituellement attendu (κ ≥ 0,6). De plus :
- L'annotateur de validation est Claude Sonnet 4 (un LLM), pas un expert humain. Valider un système de règles par un LLM zero-shot, avec la même rubrique, expose à une circularité (corrélation par construction sur les features superficielles).
- L'« adjacent agreement 94 % » sur 4 niveaux est trivial : un random uniforme donnerait déjà ~62,5 % adjacent agreement. Il faut comparer à une baseline.
- La validation est faite sur FinanceBench uniquement (150 instances). Les 10 autres benchmarks n'ont pas de validation rapportée.

### 5. Le modèle de coût est faiblement justifié et non-stress-testé
- Table 2 : les cost vectors couvrent ~3 ordres de magnitude par domaine, choix non motivé empiriquement. Pourquoi médical c₄ = 500 k$ et finance c₄ = 100 k$ ? Une mortalité évitée vs une décision de trading ne se comparent pas en dollars de cette manière, et la limite éthique est mentionnée en passant (§8 Ethics).
- La sensibilité ±20 % (RQ5) est largement insuffisante : l'incertitude réelle sur c₄ couvre facilement 1-2 ordres de magnitude (cf. la dispersion ORX, IBM/Ponemon, settlements). Un stress test devrait inclure des perturbations log-uniformes sur [c_k/10, 10·c_k].
- La calibration en appendice (§A) repose sur des sources hétérogènes (SAB 99, FDA, malpractice settlements) sans modèle d'agrégation explicite. Reviewer demandera : « comment passe-t-on de FINRA fines à c₃ = 10 k$ ? »

### 6. Incohérences chiffres et formulations
- Section 3.3 : « $10^5$ replications ».
- Section 4.4 (Risk computation, item 3) : « $10^5$ Monte Carlo replications ».
- Section 6 ouverture : « 20,000 Monte Carlo replications ».
- Appendix §D.1 : `n_sim=100_000`.

Quatre valeurs distinctes mentionnées dans le même papier. À harmoniser.

### 7. La bibliothèque comme contribution est mince
`severity-eval` est listé comme contribution majeure (#3), mais le contenu décrit (Section 5, Appendix §D) est essentiellement un wrapper autour de scipy.stats / numpy : Monte Carlo binomial-catégoriel, empirique VaR/TVaR, Brent pour Lundberg. Une revue ACL/EMNLP demandera : quel est le research artifact ? Si la bibliothèque est l'artefact principal, il faudrait :
- une étude d'utilisabilité ou un benchmark de performance ;
- une comparaison à une implémentation naïve (combien de lignes économisées ?) ;
- une intégration HuggingFace effective (vous reconnaissez qu'elle n'est pas publiée sur le Hub : « not published on the HuggingFace Hub as a loadable module in this release » ; affaiblit la contribution).

### 8. Hypothèses du modèle non discutées
- **i.i.d. binomial** : errors are independent. Faux dans la pratique : erreurs corrélées par sujet (un calculateur médical mal compris produit plusieurs erreurs corrélées). Aucune correction.
- **n = 10 000 outputs/période** : choix arbitraire. Pour RAG Insurance (82 instances), n=10 000 surestime massivement les queues. La sensibilité à n n'est pas analysée.
- **Severity profile π̂ estimé à partir des mêmes erreurs utilisées pour E[S]** : pas de hold-out. Risque de double-dipping ; intervalles de confiance bootstrap mentionnés (Appendix §D) mais pas rapportés dans les résultats.

### 9. Comparaison à MQM trop superficielle
La Section 7.5 différencie le cadre de MQM sur 3 points, mais l'un d'eux (« costs are calibrated against external evidence ») est précisément ce que vos limitations admettent comme faible. Une baseline directe « weighted accuracy » avec poids {1, 5, 25, 125} (MQM-style) devrait être ajoutée et comparée à E[S] pour montrer ce que le compound model apporte au-delà d'une moyenne pondérée.

### 10. Pas de tests d'incertitude inter-modèles
Aucun intervalle de confiance, aucun test, sur les différences d'E[S] entre modèles. Avec p estimé par bootstrap sur quelques milliers d'instances, l'erreur standard sur E[S] est calculable et devrait être rapportée. Sinon, on ne sait pas si la « divergence » est statistiquement détectable.

### 11. Section Discussion (§7) trop programmatique
La discussion alterne entre « ce que le cadre ajoute » et « limitations » sans véritablement interpréter les résultats. Les sections 7.1-7.5 redonnent le cadre théorique plutôt que d'expliquer ce que les données ont montré. Pour un papier empirique, c'est inversé.

### 12. Dépendance circulaire à un position paper non publié
Le manuscrit cite ~15 fois Beauchemin et al. 2026 (NeurIPS position paper). À l'évaluation ACL/EMNLP, un reviewer ne pourra pas lire ce papier. Le cadre doit être suffisamment auto-contenu, ce qui n'est pas le cas actuellement (la Section 3 est explicitement une « recapitulation »). Anonymisation aussi à vérifier en review mode (vous citez vos thèses 2024, 2025 : « beauchemin2024quebec », « beauchemin2025thesis » : risque de désanonymisation).

## Faiblesses mineures

- **Ligne 73** : « up to 23 language models » : formulation hedgée qui doit disparaître ; soit 23, soit le chiffre réellement évalué.
- **Section 4.3 scoring cascade** : seuils (4 caractères, 3 mots, 5 % tolerance) non justifiés ; impact sur correctness rate non discuté.
- **Lundberg inequality** : ψ(u) ≤ e^(-Ru) est une borne, mais u* = -ln(ψ*)/R suppose l'égalité (Section 3.4). C'est une approximation conservative ; à signaler.
- **« Reusable evaluation protocol »** (contribution #4) : ne semble pas distinct de la bibliothèque ; risque de double-comptage des contributions.
- **Table 1, RAG Insurance** : 82 instances, déjà publiées dans beauchemin2024quebec. Vraie nouveauté ?
- **Section 6.5 (RQ5)** : « minimum Spearman correlation between the base and perturbed rankings is 1.00 » : ρ=1.00 indique zéro changement de rang. Avec τ=0.96 baseline, l'invariance n'est pas surprenante ; presque vide d'information.
- **Appendix §E** : la mesure d'agreement reproductible « can be applied to additional datasets to extend the table » : l'auteur reconnaît que le travail reste à faire. Reviewer demandera : pourquoi ne pas l'avoir fait ?

## Questions pour clarification

1. Quel serait le résultat attendu sur médical/assurance si on avait la couverture multi-modèles ? Avez-vous un argument formel pour anticiper une rupture du τ ≈ 1 ?
2. Pourquoi ne pas comparer à une baseline « MQM-weighted accuracy » avec poids exponentiels {1, 10, 100, 1000} ?
3. La routing analysis suppose une oracle severity au moment de l'inférence : comment cela se traduit-il en pratique sans un classificateur de sévérité ?
4. Les cost vectors par domaine sont-ils calibrés indépendamment, ou normalisés pour que μ_X soit comparable inter-domaine ?
5. Pourquoi n = 10 000 et pas le n du dataset ? Le scaling change les queues.

## Recommandation

**Borderline → Reject** dans l'état actuel pour ACL/EMNLP main track. Le papier promet « an empirical study » mais l'évaluation empirique n'est pas complétée ; les résultats disponibles tendent à invalider la thèse de divergence accuracy/E[S] sans que ce constat soit assumé. Le cadre est intéressant mais doit être (a) testé sur des domaines à queue lourde où la divergence est attendue, ou (b) reformulé comme un « companion methodology paper » avec une thèse différente (e.g., « quantifying when severity matters »).

Si une resoumission est envisagée :
1. Compléter la couverture sur médical et assurance (les deux domaines aux queues les plus lourdes).
2. Ajouter une baseline MQM-weighted.
3. Stress test des cost vectors sur log-échelle, pas ±20 %.
4. Reformuler abstract/intro pour aligner avec l'évidence empirique disponible.
5. Validation d'annotation par expert humain sur ≥ 2 domaines.

## TODO de reprise

- [ ] Compléter runs médical + assurance (multi-modèles) avant resoumission.
- [ ] Ajouter baseline weighted-accuracy (MQM-style) dans `experiments/analysis.py`.
- [ ] Étendre RQ5 à perturbations log-échelle.
- [ ] Harmoniser le chiffre de Monte Carlo (10⁵ partout, ou justifier les variantes).
- [ ] Anonymiser citations beauchemin2024/2025 pour review mode.
- [ ] Validation annotation humaine sur FinanceBench + 1 domaine non-financier.
- [ ] Ajouter bootstrap CI sur E[S] dans `results/table_main.tex`.
- [ ] Repositionner intro/abstract si la divergence accuracy/E[S] ne se matérialise pas dans les nouveaux runs.
