# Délai de séjour des conteneurs — Port de Casablanca

Application Streamlit de prédiction du délai de séjour des conteneurs, à partir d'un
modèle CatBoost entraîné sur des données portuaires (import de conteneurs, terminaux,
marchandises, importateurs).

## Lancer en local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Déployer sur Streamlit Community Cloud

1. [share.streamlit.io](https://share.streamlit.io) → New app
2. Sélectionner ce dépôt, branche `main`, fichier principal `app.py`
3. Déployer

## Contenu

- `app.py` — application Streamlit (prédiction, tableau de bord, exploration, limites du modèle)
- `model_bundle.joblib` — modèle CatBoost entraîné + tables de dimension par catégorie (généré hors ligne, données brutes non incluses dans ce dépôt)
- `requirements.txt`
