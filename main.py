import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# Titre de l'application
st.title("Détection de Visages avec Viola-Jones")
st.markdown("---")

# Instructions
st.markdown("""
### Instructions :
1. **Téléchargez une image** : Utilisez le bouton ci-dessous pour télécharger une image depuis votre appareil.
2. **Paramètres de détection** :
   - Ajustez le **Scale Factor** (facteur d'échelle) pour contrôler la réduction de la taille de l'image à chaque échelle.
   - Ajustez le **Min Neighbors** pour contrôler le nombre minimum de voisins qu'un rectangle doit avoir pour être retenu.
3. **Couleur du rectangle** : Choisissez la couleur des rectangles autour des visages détectés.
4. **Résultats** : L'image avec les visages détectés s'affichera automatiquement.
5. **Enregistrement** : Cliquez sur "Enregistrer l'image" pour télécharger l'image avec les détections sur votre appareil.
""")

# Chargement du classificateur Viola-Jones
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Téléchargement de l'image
uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convertir l'image téléchargée en format OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Sidebar pour les paramètres
    st.sidebar.markdown("### Paramètres de Détection")

    # Paramètre scaleFactor
    scale_factor = st.sidebar.slider(
        "Scale Factor",
        min_value=1.01,
        max_value=1.5,
        value=1.1,
        step=0.01,
        help="Facteur par lequel l'image est réduite à chaque échelle. Une valeur plus petite détecte plus de visages mais est plus lente."
    )

    # Paramètre minNeighbors
    min_neighbors = st.sidebar.slider(
        "Min Neighbors",
        min_value=0,
        max_value=10,
        value=4,
        step=1,
        help="Nombre minimum de voisins qu'un rectangle doit avoir pour être retenu. Une valeur plus élevée réduit les fausses détections."
    )

    # Choix de la couleur du rectangle
    rect_color = st.sidebar.color_picker(
        "Couleur du rectangle",
        "#00FF00",
        help="Choisissez la couleur des rectangles autour des visages détectés."
    )

    # Conversion de la couleur hex en BGR (format OpenCV)
    rect_color = tuple(int(rect_color.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))

    # Détection des visages
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(30, 30)
    )

    # Dessiner les rectangles autour des visages
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), rect_color, 2)

    # Afficher l'image avec les détections
    st.markdown("### Résultat de la Détection")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image_rgb, channels="RGB", use_column_width=True)

    # Bouton pour enregistrer l'image
    if st.button("Enregistrer l'image"):
        # Créer un fichier temporaire
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_path = tmp_file.name
            cv2.imwrite(tmp_path, image)

        # Lire le fichier temporaire et le proposer en téléchargement
        with open(tmp_path, "rb") as file:
            st.download_button(
                label="Télécharger l'image",
                data=file,
                file_name="visages_detectes.jpg",
                mime="image/jpeg"
            )

        # Supprimer le fichier temporaire
        os.unlink(tmp_path)
else:
    st.warning("Veuillez télécharger une image pour commencer la détection.")
