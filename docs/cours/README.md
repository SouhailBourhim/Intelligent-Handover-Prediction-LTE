# Cours — Prédiction intelligente du handover dans les réseaux LTE

Ce cours explique le projet **de zéro** : d'abord les télécommunications (qu'est-ce qu'un réseau LTE, un handover, le RSRP…), puis l'apprentissage automatique (formulation, modèles, évaluation), puis les expériences et l'industrialisation. Il se termine par une préparation à l'entretien technique.

📄 **Version PDF** (tout le cours en un document, avec les diagrammes) : [`cours_projet_handover_lte.pdf`](cours_projet_handover_lte.pdf)

## Sommaire

### Partie A — Télécommunications

| # | Chapitre | Vous saurez… |
|---|---|---|
| 1 | [Réseaux mobiles et architecture LTE](01_reseaux_mobiles_et_lte.md) | dessiner l'architecture EPS (eNodeB, MME, S-GW…), situer le handover |
| 2 | [Couche physique et propagation radio](02_couche_physique_et_propagation.md) | expliquer OFDMA, la grille de ressources, path loss / shadowing / fading, le SINR |
| 3 | [Les KPI radio : RSRP, RSRQ, SINR, CQI](03_kpi_radio.md) | définir chaque KPI et **justifier pourquoi ce sont eux qu'on utilise** |
| 4 | [Mobilité et handover](04_mobilite_et_handover.md) | dérouler la procédure X2, A3/TTT/hystérésis, les échecs, le MRO |
| 5 | [Le simulateur du projet](05_simulateur.md) | relier chaque concept au code, connaître les corrections v4 et les limites |

### Partie B — Apprentissage automatique

| # | Chapitre | Vous saurez… |
|---|---|---|
| 6 | [Le problème de ML : label, données, variables](06_probleme_ml_et_donnees.md) | formuler la cible, expliquer les 83 variables, prouver l'absence de fuite |
| 7 | [Les modèles](07_modeles.md) | expliquer régression logistique, forêt, XGBoost, LSTM, stacking, et le déséquilibre |
| 8 | [Évaluation et résultats](08_evaluation_et_resultats.md) | lire F1 / ROC / PR, choisir un seuil, présenter les résultats et SHAP |

### Partie C — Aller plus loin

| # | Chapitre | Vous saurez… |
|---|---|---|
| 9 | [Boucle fermée et généralisation](09_boucle_fermee_et_generalisation.md) | montrer si la prédiction **améliore vraiment le réseau** |
| 10 | [MLOps, tests et organisation du code](10_mlops_et_code.md) | expliquer DVC, MLflow, la CI, les tests et l'inférence |
| 11 | [Préparer l'entretien](11_preparer_entretien.md) | pitch de 2 minutes, chiffres clés, 40 questions-réponses, glossaire |

## Comment lire ce cours

- **Une semaine avant l'entretien :** lisez tout dans l'ordre, une partie par jour.
- **La veille :** relisez les encadrés 🎯 de chaque chapitre et le chapitre 11 en entier.
- **Le jour J :** le pitch et le tableau des chiffres clés du chapitre 11.

Les encadrés :
- 🎯 point clé à savoir dire en entretien ;
- 💡 intuition ou justification ;
- ⚠️ piège ou nuance à connaître ;
- 🐛 bug trouvé et corrigé dans le projet (excellente matière pour « parlez-moi d'une difficulté rencontrée ») ;
- 🔧 simplification du simulateur par rapport à la réalité.

## Régénérer les figures et le PDF

```bash
python scripts/make_figures.py          # figures du README (docs/img/)
python scripts/make_course_figures.py   # figures du cours (docs/cours/img/)
python scripts/build_course_pdf.py      # PDF (nécessite Edge ou Chrome)
```
