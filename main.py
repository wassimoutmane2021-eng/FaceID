import cv2
import streamlit as st
import os
from datetime import datetime
import numpy as np
from PIL import Image

# Charger le classificateur de cascade de visage
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(scale_factor, min_neighbors, rect_color):
    # Initialiser la webcam
    cap = cv2.VideoCapture(0)

    # Créer un dossier pour sauvegarder les images si nécessaire
    if not os.path.exists('captured_faces'):
        os.makedirs('captured_faces')

    # Placeholder pour afficher la vidéo dans Streamlit
    frame_placeholder = st.empty()

    # Bouton pour sauvegarder l'image (hors de la boucle)
    save_col, stop_col = st.columns(2)
    save_button = save_col.button("📸 Sauvegarder l'image")
    stop_button = stop_col.button("⏹ Arrêter la détection")

    while cap.isOpened():
        # Lire les images de la webcam
        ret, frame = cap.read()
        if not ret:
            st.error("Erreur lors de la capture de la vidéo")
            break

        # Convertir les images en niveaux de gris
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Détecter les visages
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(30, 30)
        )

        # Dessiner des rectangles autour des visages détectés
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), rect_color, 2)

        # Convertir l'image BGR en RGB pour Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Afficher la vidéo dans Streamlit
        frame_placeholder.image(frame_rgb, channels="RGB")

        # Sauvegarder l'image si le bouton est cliqué
        if save_button:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = f"captured_faces/face_{timestamp}.jpg"
            cv2.imwrite(image_path, frame)
            st.success(f"✅ Image sauvegardée : {image_path}")

        # Arrêter la détection si le bouton est cliqué
        if stop_button:
            break

    # Libérer la webcam
    cap.release()
    st.success("Détection arrêtée.")

def app():
    st.title("👤 Détection de visage avec Viola-Jones")

    # Instructions
    st.markdown("""
    ### **Instructions :**
    1. Ajustez les paramètres de détection dans la barre latérale.
    2. Cliquez sur **"Démarrer la détection"** pour activer la webcam.
    3. Utilisez **"📸 Sauvegarder l'image"** pour enregistrer une capture.
    4. Cliquez sur **"⏹ Arrêter la détection"** pour terminer.
    """)

    # Paramètres de détection (dans la sidebar)
    st.sidebar.header("⚙️ Paramètres")

    # Choix de la couleur du rectangle
    rect_color_hex = st.sidebar.color_picker("Couleur des rectangles", "#00FF00")
    rect_color = tuple(int(rect_color_hex.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))  # BGR

    # Paramètre scaleFactor
    scale_factor = st.sidebar.slider(
        "Scale Factor (1.01–2.0)",
        min_value=1.01,
        max_value=2.0,
        value=1.3,
        step=0.01,
        help="Ajuste la sensibilité à la taille des visages."
    )

    # Paramètre minNeighbors
    min_neighbors = st.sidebar.slider(
        "Min Neighbors (0–10)",
        min_value=0,
        max_value=10,
        value=5,
        step=1,
        help="Nombre minimum de voisins pour valider un visage."
    )

    # Bouton pour démarrer la détection
    if st.sidebar.button("🎥 Démarrer la détection"):
        detect_faces(scale_factor, min_neighbors, rect_color)

if __name__ == "__main__":
    app()
