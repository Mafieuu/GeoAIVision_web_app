import streamlit as st 
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import boto3
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
import tempfile
import os 
from utils.create_shapefile import predict_masks, merge_masks_to_shapefile, streamlit_visualisation
from utils.decoup_raster import create_directory_structure, extract_raster_info, cut_raster



# Titre de l'application
st.title("Chargeur et Visualisateur de Raster")


st.write("Bouton de téléchargement du raster" ) 
uploaded_file = st.file_uploader(
    "Choisissez un fichier raster", 
    type=['tif', 'tiff']
)


if uploaded_file is not None:
    # Sauvegarde temporaire du fichier
    temp_dir = tempfile.mkdtemp()
    input_path = os.path.join(temp_dir, uploaded_file.name)
    original_folder = os.path.dirname(uploaded_file.name)
    
    with open(input_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    # Chemins de modèle et étapes de traitement
    model_path = os.path.join(original_folder,'GeoAIVision_trained_model.h5')
    raster_path = original_folder
    output_base_path = os.path.join(original_folder, 'raster')
    sub_rasters = output_base_path
    mask_paths = os.path.join(original_folder, 'mask')
    output_shapefile = os.path.join(original_folder, 'shap')

    
    # Étapes de traitement
    sub_rasters = cut_raster(raster_path, output_base_path, window_size=256, overlap=0)
    mask_paths = predict_masks(sub_rasters, model_path)
    
    # Génération du shapefile
    output_shapefile = os.path.splitext(input_path)[0] + '.shp'
    shapefile_path = merge_masks_to_shapefile(mask_paths, output_shapefile)
    
    # Visualisation
    streamlit_visualisation(input_path, shapefile_path)