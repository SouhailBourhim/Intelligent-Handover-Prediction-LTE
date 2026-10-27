# Chapitre 11 — Préparer l'entretien

> **Objectif :** être capable de présenter le projet en 2 minutes, de citer les chiffres clés sans hésiter, et de répondre aux questions techniques probables, en télécom comme en ML.

---

## 11.1 Le pitch en 2 minutes

> « Dans un réseau LTE, quand un utilisateur passe d'une cellule à l'autre, l'eNodeB déclenche un **handover** sur la base des mesures radio que le terminal lui remonte : principalement le **RSRP**, comparé entre la cellule de service et la voisine (événement **A3**). Si le handover est déclenché trop tard, le lien casse avant la fin de la procédure : c'est une **coupure**.
>
> Mon projet prédit, **3 secondes à l'avance**, qu'un handover réussi va avoir lieu, à partir uniquement des mesures standardisées que l'eNodeB reçoit déjà : RSRP, RSRQ, SINR, CQI, leurs tendances, la vitesse et la charge.
>
> Comme il n'existe pas de jeu de données public, j'ai écrit un **simulateur LTE** fondé sur les modèles 3GPP : propagation UMa, LOS/NLOS spatialement cohérent, masquage, évanouissement rapide, RSRP par élément de ressource, filtrage L3, A3 avec TTT, échecs, ping-pong. J'ai entraîné et comparé **six modèles et une règle de référence** (régression logistique, forêt aléatoire, XGBoost, LSTM, GRU, stacking), avec un découpage temporel, des seuils et une sélection faits sur la validation, dans un pipeline reproductible (DVC, MLflow, CI, 36 tests).
>
> Le champion, XGBoost, atteint une **ROC-AUC de 0,92** et anticipe **83 % des handovers à coût de fausses alarmes égal**, contre 68 % pour la règle A3. Enfin, je l'ai mis **dans la boucle de décision** du simulateur : il réduit les échecs de handover de **30 %**, là où un témoin aléatoire ne fait rien. Mais un simple TTT court fait presque aussi bien, et les deux augmentent les ping-pongs. Prédire le handover ne suffit donc pas à battre ce compromis : il faudrait prédire les ping-pongs. »

## 11.2 Les chiffres clés

| Sujet | Chiffre |
|---|---|
| Réseau | 4 eNodeB, 1 km², ISD 500 m, 2 GHz, 10 MHz, 46 dBm (18,2 dBm/RE) |
| UE | 15 (8 piétons, 7 véhicules), 3 600 s, 54 000 lignes |
| Label | handover **réussi** dans les **3 s** ; **6,4 %** de positifs |
| Variables | **83** (8 signaux × 10 transformations + vitesse, rsrp_diff, charge) |
| Découpage | temporel 70 / 15 / 15 |
| Handovers | 1 434 tentatives / h ; 16 % d'échecs ; 20 % de ping-pong |
| Champion | **XGBoost**, seuil 0,80 (choisi sur validation) |
| Test | F1 **0,50**, ROC-AUC **0,92**, PR-AUC **0,43** (règle A3 : 0,40 / 0,85 / 0,26) |
| Événements | **89 %** des handovers anticipés, **3,4 s** d'avance en moyenne |
| À budget égal | **83 %** (ML) contre **68 %** (règle) à ≤ 5 % de temps de fausse alarme |
| Boucle fermée | échecs **14,9 % → 10,5 %** ; témoin aléatoire 14,2 % ; TTT court 11,0 % ; ping-pong 21,7 % → 29,4 % |
| Généralisation | graines inédites : AUC 0,93, PR-AUC 0,43–0,45 ; véhicules : AUC 0,87 |
| Reproductibilité | 2 exécutions complètes → métriques identiques ; pipeline en ~7 min |

## 11.3 Questions télécom

**1. Qu'est-ce qu'un handover et pourquoi est-il critique ?**
Le transfert d'une communication en cours vers une autre cellule. En LTE il est « dur » (*break-before-make*) : s'il échoue, le lien radio casse (RLF) et la communication est coupée le temps du rétablissement.

**2. Qui décide du handover en LTE ?**
L'eNodeB source, sur la base des *Measurement Reports* RRC de l'UE. L'UE mesure, le réseau décide. En veille (RRC_IDLE), c'est l'UE qui fait de la resélection, sans handover.

**3. Décrivez la procédure de handover X2.**
Mesure et filtrage L3 → événement A3 vrai pendant le TTT → *Measurement Report* → décision de la source → *X2 Handover Request* → contrôle d'admission de la cible → *Request Ack* → *RRC Reconfiguration* (ordre de HO) → l'UE fait un RACH sur la cible → *Reconfiguration Complete* → *Path Switch* vers le MME/S-GW → libération du contexte dans la source.

**4. Différence entre RSRP et RSRQ ?**
RSRP = puissance d'un RE de signal de référence : couverture, indépendante de la charge. RSRQ = N·RSRP/RSSI : part de la cellule dans tout ce qui est reçu, donc sensible à l'interférence et à la charge. Même RSRP, RSRQ très différent selon l'interférence.

**5. Pourquoi le RSRP est-il défini par élément de ressource ?**
Pour être indépendant de la largeur de bande et de la charge : on compare des cellules de configurations différentes sur une base commune. Conséquence dans le projet : 46 dBm sur 600 sous-porteuses → 18,2 dBm par RE. La v3 l'oubliait (RSRP trop élevés de 27,8 dB).

**6. Quelle est la plage de valeurs du RSRP ?**
De −140 à −44 dBm (TS 36.133). En pratique, au-dessus de −80 c'est excellent, en dessous de −110 c'est le bord de couverture.

**7. Qu'est-ce que l'événement A3 ?**
La voisine devient meilleure que la cellule de service d'un décalage (offset + hystérésis + CIO), pendant au moins le TTT. C'est le déclencheur standard du handover intra-fréquence. Dans le projet : marge de 3 dB, RSRP filtré L3.

**8. À quoi servent l'hystérésis et le TTT ?**
À éviter les déclenchements sur des fluctuations passagères, donc les ping-pongs. Mais un TTT ou une hystérésis trop grands rendent le handover tardif : c'est le compromis de la mobilité.

**9. Pourquoi un TTT plus court pour les UE rapides ?**
Ils traversent la zone de recouvrement en quelques secondes : avec un TTT long, ils sortent de la couverture de la source avant que le handover parte. La norme prévoit le *speed-dependent scaling* ; le projet utilise TTT = max(1, 3 − ⌊v/7⌋) s.

**10. Qu'est-ce que le filtrage L3 ?**
Une moyenne exponentielle des mesures dans la couche RRC : F = (1−a)·F + a·M, avec a = 1/2^(k/4). Avec k = 4 (défaut), a = 0,5, la valeur du projet. Il lisse le fast fading avant la comparaison aux seuils.

**11. Citez les types d'échec de mobilité.**
Trop tardif (RLF avant/pendant le HO), trop précoce (RLF juste après, retour vers la source), mauvaise cellule (rétablissement dans une 3ᵉ cellule), ping-pong (aller-retour rapide). Le MRO les compte pour ajuster CIO, hystérésis et TTT.

**12. Qu'est-ce qu'une RLF ?**
*Radio Link Failure* : après N310 indications de perte de synchronisation, le temporisateur T310 démarre ; à son expiration, l'UE déclare la RLF et tente un *RRC Re-establishment*. Dans le projet : 3 s de RLF puis rétablissement sur la meilleure cellule.

**13. Pourquoi le SINR chute-t-il avant un handover ?**
En approchant de la frontière, le signal utile baisse et celui de la voisine (qui est de l'interférence, même fréquence) monte : double effet. C'est aussi pourquoi un handover tardif échoue : l'ordre de handover ne passe plus.

**14. Pourquoi le modèle 3GPP UMa ?**
Il représente un macro-réseau urbain (antennes en toiture, ISD ~500 m) et c'est le modèle de référence des études 3GPP (TR 36.873, puis 38.901 en 5G). Formules différentes en LOS et NLOS.

**15. Quelles sont les trois échelles de variation du canal ?**
Path loss (distance, centaines de m), masquage log-normal (obstacles, ~100 m, σ 4–6 dB), fast fading (multi-trajets, λ/2 = 7,5 cm à 2 GHz). Le terminal moyenne le fast fading sur sa période de mesure.

**16. Qu'est-ce que le CQI ?**
Un indice de 1 à 15 envoyé par l'UE, qui indique la modulation et le codage les plus efficaces décodables avec moins de 10 % d'erreurs de bloc. L'eNodeB s'en sert pour l'adaptation de lien. C'est un SINR quantifié.

**17. En quoi votre approche se rapproche-t-elle de la 5G / O-RAN ?**
La 5G NR utilise les mêmes mesures (SS-RSRP, SS-RSRQ, SS-SINR) et les mêmes événements A3/A5. O-RAN prévoit des applications d'IA (xApps) dans un RIC quasi temps réel, proche des stations de base : c'est exactement l'endroit où un prédicteur de handover se déploierait.

## 11.4 Questions ML

**18. Pourquoi une classification et pas une régression du « temps avant handover » ?**
La décision opérationnelle est binaire (préparer ou non), et le temps avant le prochain handover est mal défini quand aucun n'arrive (censure). Une formulation de survie ou de régression serait une extension possible.

**19. Pourquoi 3 secondes ?**
Assez pour agir (préparer la cible, avancer le HO) et assez court pour que les signes radio soient visibles. À 10 s, la mobilité devient imprévisible et le label trop bruité.

**20. Comment avez-vous géré le déséquilibre (6 % de positifs) ?**
Pondération des classes (class_weight, scale_pos_weight, sampler équilibré pour les LSTM), une seule par modèle. Puis seuil choisi sur la validation et métriques adaptées (F1, PR-AUC). J'ai d'ailleurs corrigé une double pondération dans les LSTM, qui faisait chuter leur précision.

**21. Pourquoi pas de SMOTE ?**
Le suréchantillonnage synthétique interpole entre exemples : sur des séries temporelles avec des variables de lag, il crée des trajectoires physiquement incohérentes. La pondération atteint le même objectif sans fabriquer de données.

**22. Comment avez-vous évité les fuites de données ?**
Fenêtres glissantes décalées d'un pas ; split temporel, pas aléatoire ; scaler appris sur le train seulement ; aucune variable inobservable (position, LOS) ; seuils et champion choisis sur la validation ; un test automatique vérifie que modifier le futur ne change pas le passé.

**23. Pourquoi un split temporel ?**
Deux secondes consécutives sont presque identiques. Un split aléatoire mettrait l'une en train et l'autre en test : le modèle « reconnaîtrait » le test, et les scores seraient gonflés.

**24. Pourquoi ne pas utiliser la position GPS, qui serait très informative ?**
Le réseau ne la connaît pas en temps réel (vie privée, batterie, précision en intérieur). L'utiliser rendrait le modèle non déployable. La v4 l'a retirée, pour un coût mesuré d'environ 0,01 de F1 (sur la v3).

**25. Pourquoi PR-AUC plutôt que ROC-AUC ?**
Avec des positifs rares, la ROC-AUC reste élevée même avec beaucoup de fausses alarmes (le dénominateur des FPR, ce sont les négatifs, très nombreux). La PR-AUC mesure la précision des alarmes, que l'opérateur paie. Son niveau de hasard vaut le taux de positifs.

**26. Pourquoi choisir le seuil sur la validation ?**
Les modèles pondérés ne sont pas calibrés : 0,5 n'a pas de sens. Choisir le seuil sur le test serait une fuite : le score de test deviendrait optimiste.

**27. Pourquoi XGBoost gagne-t-il ? Est-ce significatif ?**
Il a la meilleure AUC et PR-AUC de validation. Mais les modèles appris sont à quelques millièmes les uns des autres : je ne surinterprète pas le classement. Le message solide est l'écart avec la règle A3.

**28. Pourquoi les LSTM ne font-ils pas mieux que les arbres ?**
Les variables de lag, delta et fenêtre glissante donnent déjà la dynamique aux modèles tabulaires. La séquence brute de 10 s n'ajoute pas grand-chose. Les arbres sont aussi plus robustes avec peu de données et des variables hétérogènes.

**29. Bagging vs boosting ?**
Le bagging (forêt aléatoire) entraîne des arbres indépendants sur des échantillons bootstrap et moyenne : il réduit la variance. Le boosting (XGBoost) ajoute des arbres séquentiellement, chacun corrigeant le gradient des erreurs précédentes : il réduit le biais, avec un learning rate et un arrêt anticipé contre le sur-apprentissage.

**30. Pourquoi l'arrêt anticipé sur la PR-AUC ?**
Sur la log-loss, XGBoost continuait jusqu'à ~900 arbres et sa PR-AUC de validation baissait. La log-loss est dominée par les 94 % de négatifs ; la PR-AUC mesure ce qui compte. Avec elle, il s'arrête vers 120 arbres et la PR-AUC monte à 0,47.

**31. Comment fonctionne votre stacking, et quel piège avez-vous évité ?**
Une régression logistique combine les probabilités de XGBoost, RF et LSTM, entraînée sur la validation. Pour la comparer équitablement aux autres modèles sur cette même validation, j'utilise des prédictions hors échantillon (validation croisée en 5 blocs temporels) ; sinon son score serait optimiste.

**32. Qu'est-ce que SHAP, et que montre-t-il ici ?**
Une décomposition additive de chaque prédiction en contributions des variables (valeurs de Shapley). Pour XGBoost : la vitesse (17,7 %), l'écart service/voisine `rsrp_diff` (14,9 %), le SINR et le RSRQ dominent ; les variables temporelles pèsent ~37 % au total. La vitesse sert d'a priori (les véhicules changent de cellule 4 fois plus souvent) ; `rsrp_diff` reproduit la logique d'A3.

**33. Vos métriques « événementielles », pourquoi ?**
Un opérateur raisonne en handovers anticipés et en coût des alarmes, pas en secondes bien classées. Je mesure le rappel des handovers à 3 s, le délai d'anticipation et le temps de fausse alarme, et je compare les modèles à budget de fausses alarmes égal.

**34. Le F1 de 0,5 n'est-il pas faible ?**
Le label est intrinsèquement bruité : prédire 3 s à l'avance un événement en partie aléatoire (masquage, transitions LOS). L'AUC de 0,92 montre un bon classement ; en événements, 89 % des handovers sont anticipés. Et le gain sur la règle (+0,10 de F1, PR-AUC ×1,6) est le vrai indicateur de valeur.

## 11.5 Questions sur la démarche

**35. Comment savez-vous que la prédiction améliore le réseau ?**
Je l'ai mise dans la boucle du simulateur, avec les mêmes trajectoires pour toutes les politiques et un témoin aléatoire à taux d'intervention égal. Résultat : −30 % d'échecs là où le témoin ne fait rien. Mais un TTT court fait presque aussi bien, et les ping-pongs augmentent : c'est la limite actuelle.

**36. Quelle est la principale limite du projet ?**
Les données sont simulées : la vraie validation serait des traces opérateur. Ensuite : pas de temps d'1 s, transitions LOS franches qui gonflent les handovers piétons, réseau de 4 cellules. Et la boucle fermée montre qu'il faut aussi prédire les ping-pongs.

**37. Que feriez-vous avec plus de temps ?**
Prédire les ping-pongs (ou la qualité du handover) en plus du handover ; ré-entraîner sous la politique prédictive ; une approche par apprentissage par renforcement pour régler TTT/CIO ; valider sur des traces réelles ; prédire aussi la cellule cible.

**38. Racontez une difficulté rencontrée.**
En revoyant le projet, j'ai trouvé que `predict.py` rejetait 100 % des lignes d'un fichier brut, et que le dashboard affichait les risques sur les mauvaises lignes quand le fichier était trié par temps. J'ai unifié l'inférence dans un seul module, rattaché les scores par index, et écrit un test qui mélange l'ordre des lignes. Autre exemple : le RSRP calculé avec la puissance totale au lieu de la puissance par RE.

**39. Comment garantissez-vous la reproductibilité ?**
Graines fixées partout (simulateur, scikit-learn, XGBoost, PyTorch, sampler) ; pipeline DVC ; rapports JSON versionnés ; CI qui relance tout. Deux exécutions complètes donnent des métriques identiques.

**40. Pourquoi avoir écrit des tests sur un projet de data science ?**
Parce que les erreurs les plus graves sont silencieuses : une fuite de données ou un décalage de lignes donnent quand même des chiffres, juste faux. Les tests figent les propriétés critiques (pas de fuite, parité en ligne/hors ligne, ordre des lignes sans effet, physique cohérente), et plusieurs reproduisent des bugs réels de la v3.

## 11.6 Glossaire

| Terme | Signification |
|---|---|
| 3GPP | organisme de normalisation des réseaux mobiles (normes TS / TR) |
| A3, A4, A5 | événements de mesure déclenchant un rapport (voisine > service + offset, etc.) |
| CIO | *Cell Individual Offset* : décalage par voisine pour avancer/retarder les HO |
| CQI | *Channel Quality Indicator*, 1 à 15 |
| CRS | *Cell-specific Reference Signal*, sur lequel on mesure le RSRP |
| eNodeB | station de base LTE |
| EPC / E-UTRAN | cœur de réseau / réseau d'accès LTE |
| Handover (HO) | transfert intercellulaire |
| ISD | *Inter-Site Distance* |
| L1 / L3 | couche physique / couche RRC (filtrage des mesures) |
| LOS / NLOS | avec / sans trajet direct |
| MME, S-GW, P-GW, HSS | entités du cœur EPC |
| MRO | *Mobility Robustness Optimisation* (fonction SON) |
| OFDMA | accès multiple par répartition en fréquences orthogonales |
| PR-AUC | aire sous la courbe précision-rappel (*average precision*) |
| RB / RE | *Resource Block* (12 sous-porteuses × 1 slot) / *Resource Element* |
| RLF | *Radio Link Failure* |
| RRC | *Radio Resource Control* (couche 3 radio) |
| RSRP / RSRQ / RSSI | puissance de référence / qualité / puissance totale reçue |
| SHAP | méthode d'explicabilité fondée sur les valeurs de Shapley |
| SINR | rapport signal sur interférence plus bruit |
| SON | *Self-Organizing Networks* |
| TTT | *Time-To-Trigger* |
| UE | *User Equipment* (terminal) |
| UMa | *Urban Macro* (scénario de propagation 3GPP) |
| X2 / S1 | interface eNodeB–eNodeB / eNodeB–cœur |

---

⬅️ [Chapitre 10](10_mlops_et_code.md) · [Retour au sommaire](README.md)
