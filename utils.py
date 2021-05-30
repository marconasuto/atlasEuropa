# %%
import zipfile
import requests
import os
import io
from io import BytesIO
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
from matplotlib.cm import ScalarMappable
import seaborn as sns
import skimage
from skimage import io
from spectral import *
import rasterio
from rasterio.plot import show
from rasterio.warp import calculate_default_transform, reproject
from rasterio.enums import Resampling
from rasterio.coords import BoundingBox
from rasterio import windows
from rasterio import warp
import geopandas as gpd
from shapely.geometry import box
import folium
import branca
import random
import pickle
from tqdm import tqdm
import time
import datetime
import shutil
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import confusion_matrix
import scipy as sp
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory
import tensorflow_addons as tfa
from tensorflow_addons.metrics import MultiLabelConfusionMatrix
import cv2
from keras_applications.imagenet_utils import _obtain_input_shape
from keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Activation, concatenate, Dropout, Flatten, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, ReduceLROnPlateau
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.utils import get_source_inputs
from tensorflow.keras.utils import get_file
import tempfile

# %%
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

# %%
def select_channels(string):
    '''Channels selector:
       Input: 'all' or list among following: 'Blue', 'Green', 'Red', 'Red edge 1', 'Red edge 2', 'Red edge 3', 'NIR', 'Red edge 4', 'SWIR 1', 'SWIR 2'''

    channels = []
    if string == 'all':
        channels = list(np.arange(10))
    else:
        _dict = {
            'Blue': 0,
            'Green': 1,
            'Red': 2,
            'Red edge 1': 3,
            'Red edge 2': 4,
            'Red edge 3': 5,
            'NIR': 6,
            'Red edge 4': 7,
            'SWIR 1': 8,
            'SWIR 2': 9
        }
        channels = list(map(_dict.get, string))

    return channels

# %%
def params(
        extension,
        epochs,
        pcg_dataset=1,
        batch_size=128,
        size=64,
        parse_bands_verbose=False,
        inspect_raster=False,
        channels='all',
        preprocess=False,
        horizontal_flip=False,
        vertical_flip=False,
        rotation_range=0,
        shear_range=0,
        seed=random.seed(123),
        columns=[
            'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
            'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River',
            'SeaLake'
        ],
        loss_type='categorical_crossentropy',
        opt_type='Adam',
        learning_rate=1e-4,
        momentum=0.9,
        regularization=False,
        rlronplateau=False,
        checkpoint=True,
        no_imbalanced=True,
        trainable='Full',
        pcg_unfreeze=0,
        data_path='./data',
        reports_path='./reports',
        tif_path='./data/raw/eurosat/ds/images/remote_sensing/otherDatasets/sentinel_2/tif',
        jpg_path='./data/raw/eurosat/2750'):
    ''' extension: jpg or tif
        channels: 'all' means all channels in reference table;
                     as alternative, select channels by name, i.e.:
                     
                     channels = ['Blue','Green', 'Red', 'NIR', 'SWIR2']
                     
                                B02 - Blue         10          490
                                B03 - Green        10          560
                                B04 - Red          10          665
                                B05 - Red edge 1   20          705
                                B06 - Red edge 2   20          740
                                B07 - Red edge 3   20          783
                                B08 - NIR          10          842
                                B08A - Red edge 4  20          865
                                B11 - SWIR 1       20          1610
                                B12 - SWIR 2       20          2190 '''

    raw_data_path = os.path.join(data_path, 'raw')
    data_path_jpg = os.path.join(data_path, 'jpg')
    data_path_tif = os.path.join(data_path, 'tif')
    processed_path = os.path.join(data_path, 'processed')
    eurosat_path = os.path.join(raw_data_path, 'eurosat')
    assets_path = os.path.join(reports_path, 'assets')
    pickled_tif_path = os.path.join(processed_path, 'tif')
    reprojected_path = os.path.join(processed_path, 'reprojected')
    reprojected_path_tif = os.path.join(reprojected_path, 'tif')
    reports_maps_path = os.path.join(reports_path, 'maps')
    reports_map_eda_path = os.path.join(reports_maps_path, 'eda')
    reports_map_classifier_path = os.path.join(reports_maps_path,
                                               'classifier')
    train_data_dir_jpg = os.path.join(data_path_jpg, 'train')
    val_data_dir_jpg = os.path.join(data_path_jpg, 'val')
    test_data_dir_jpg = os.path.join(data_path_jpg, 'test')
    train_data_dir_tif = os.path.join(data_path_tif, 'train')
    val_data_dir_tif = os.path.join(data_path_tif, 'val')
    test_data_dir_tif = os.path.join(data_path_tif, 'test')
    log_folder = os.path.join(reports_path, 'logs')
    log_gradient_tape_path = os.path.join(log_folder, 'gradient_tape')
    log_cm_path = os.path.join(log_folder, 'cm')
    weights_path = os.path.join(data_path, 'weights')
    num_classes = len(columns)

    channels = select_channels(channels)

    if extension == 'jpg':
        num_channels = 3
    elif extension == 'tif':
        num_channels = len(channels)
    else:
        print(
            'Error extension format: specify correct exentsion, either \'jpg\' or \'tif\''
        )

    subdirs_raw = os.listdir(jpg_path)
    filenames_raw = []
    for subdir in subdirs_raw:
        imgs_raw = os.listdir(os.path.join(jpg_path, subdir))
        random_sampled = random.sample(imgs_raw, 2000)
        if no_imbalanced:
            sub_path_imgs = [
                os.path.join(subdir, img) for img in random_sampled
            ]
        else:
            sub_path_imgs = [os.path.join(subdir, img) for img in imgs_raw]
        filenames_raw.append(sub_path_imgs)
    filenames = [
        os.path.join(data_path_jpg, f) for sublist in filenames_raw
        for f in sublist if f.endswith('.jpg')
    ]

    pcg_total_files = int(pcg_dataset * len(filenames))
    filenames = filenames[:pcg_total_files]

    train_val_files_length = int(
        0.9 * len(filenames))  # 10% for testing, 90% for val and train
    test_files_length = len(filenames) - train_val_files_length

    train_files_length = int(
        0.7 * train_val_files_length)  # 70% for train, 30% for val
    val_files_length = train_val_files_length - train_files_length

    params = AttrDict({
        'num_channels':
        num_channels,
        'extension':
        extension,
        'num_images_train':
        train_files_length,
        'num_images_val':
        val_files_length,
        'num_images_test':
        test_files_length,
        'num_classes':
        num_classes,
        'parse_bands_verbose':
        parse_bands_verbose,
        'inspect_raster':
        inspect_raster,
        'channels':
        channels,
        'num_epochs':
        epochs,
        'learning_rate':
        learning_rate,
        'momentum':
        momentum,
        'checkpoint':
        checkpoint,
        'trainable':
        trainable,
        'pcg_dataset':
        pcg_dataset,
        'pcg_unfreeze':
        pcg_unfreeze,
        'preprocess':
        preprocess,
        'horizontal_flip':
        horizontal_flip,
        'vertical_flip':
        vertical_flip,
        'rotation_range':
        rotation_range,
        'shear_range':
        shear_range,
        'no_imbalanced':
        no_imbalanced,
        'batch_size':
        batch_size,
        'size':
        size,
        'seed':
        seed,
        'columns':
        columns,
        'regularization':
        regularization,
        'rlronplateau':
        rlronplateau,
        'num_classes':
        num_classes,
        'loss_type':
        loss_type,
        'opt_type':
        opt_type,
        'loss_obj':
        loss_obj(loss_type),
        'optimizer_obj':
        optimizer(learning_rate, momentum, opt_type),
        'raw_jpg_path':
        jpg_path,
        'raw_tif_path':
        tif_path,
        'raw_data_path':
        raw_data_path,
        'data_path_jpg':
        data_path_jpg,
        'data_path_tif':
        data_path_tif,
        'weights_path':
        weights_path,
        'processed_path':
        processed_path,
        'pickled_tif_path':
        pickled_tif_path,
        'eurosat_path':
        eurosat_path,
        'assets_path':
        assets_path,
        'reprojected_path':
        reprojected_path,
        'reprojected_path_tif':
        reprojected_path_tif,
        'reports_maps_path':
        reports_maps_path,
        'reports_map_eda_path':
        reports_map_eda_path,
        'reports_map_classifier_path':
        reports_map_classifier_path,
        'train_data_dir_jpg':
        train_data_dir_jpg,
        'val_data_dir_jpg':
        val_data_dir_jpg,
        'test_data_dir_jpg':
        test_data_dir_jpg,
        'train_data_dir_tif':
        train_data_dir_tif,
        'val_data_dir_tif':
        val_data_dir_tif,
        'test_data_dir_tif':
        test_data_dir_tif,
        'log_folder':
        log_folder,
        'log_gradient_tape_path':
        log_gradient_tape_path,
        'log_cm_path':
        log_cm_path,
        'num_classes':
        len(columns)
    })

    return params

# %%
def resample(path):
    '''Resamples img and returns bands upsampled'''
    upscale_factor = 2
    # upsample channels to 2x
    image = rasterio.open(path)
    b01, b02, b03, b04, b05, b06, b07, b08, b08A, b09, b10, b11, b12 = image.read(
        out_shape=(image.count, int(image.height * upscale_factor),
                   int(image.width * upscale_factor)),
        resampling=Resampling.bilinear)

    return  # bands that were resampled from 20m to 10m

# %%
def parse_bands(img, params):
    '''Parse tif Sentinel-2A images into 13 bands. 
    Returns: coord_bb,
             channels = [b02, b03, b04, b05, b06, b07, b08, b08A, b11, b12] with b05, b06, b07, b08A, b11, b12 upsampled to 10m '''
    satdat = rasterio.open(img)
    if img.split('/')[-1].endswith('.tif'):
        b01, b02, b03, b04, b05, b06, b07, b08, b08A, b09, b10, b11, b12 = satdat.read(
        )
        channels = [b02, b03, b04, b05, b06, b07, b08, b08A, b11, b12
                    ]  # filter out b01, b09, b10 intended for atmosphere study
    elif img.split('/')[-1].endswith('.jpg'):
        b, g, r = satdat.read()
        channels = [b, g, r]

    # Get resolution, in map units (meters)
    xres = (satdat.bounds.right - satdat.bounds.left) / satdat.width
    yres = (satdat.bounds.top - satdat.bounds.bottom) / satdat.height
    coord_bb = [
        satdat.bounds.left, satdat.bounds.bottom, satdat.bounds.right,
        satdat.bounds.top
    ]  # coordinate bounding box [left, bottom, right, top]
    # geo coordinates [left-long, bottom-lat, right-long, top-lat]
    if params.parse_bands_verbose:
        print('W resolution (m): {}; H resolution: {}'.format(xres, yres))
        print("Are the pixels square: {}".format(xres == yres))
        print(satdat.profile)

    return coord_bb, channels

# %%
def transform_reproj(img, params):
    '''Apply affine transformation to array (satdat) and save to file (.tif or .jpg) 
    path = './data/processed/reprojected/filename; 
    filename format: rerpoj_{image_name})')'''
    target_crs = 'epsg:4326'
    satdat = rasterio.open(img)
    # calculate a transform and new dimensions using our dataset's current CRS and dimensions
    transform, width, height = calculate_default_transform(
        satdat.crs, target_crs, satdat.width, satdat.height, *satdat.bounds)
    # Copy the metadata
    metadata = satdat.meta.copy()
    # Change the CRS, transform, and dimensions in metadata to match our desired output dataset
    metadata.update({
        'crs': target_crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    # apply the transform & metadata to perform the reprojection
    dst = os.path.join(params.reprojected_path_tif,
                       'reproj_' + img.split('/')[-1])

    with rasterio.open(dst, 'w', **metadata) as reprojected:
        for band in range(1, satdat.count + 1):
            reproject(source=rasterio.band(satdat, band),
                      destination=rasterio.band(reprojected, band),
                      src_transform=satdat.transform,
                      src_crs=satdat.crs,
                      dst_transform=transform,
                      dst_crs=target_crs)

    return dst

# %%
def inspect_raster(satdat, img):
    '''Inspect raster (after rescaling)'''
    fig, ax = plt.subplots(1, 1, dpi=100)
    show(satdat.read([4, 3, 2]) * 255 / 65535, ax=ax)
    plt.title(img.split('/')[-1])

# %%
def mkdir(path):
    new_dir = path
    if not os.path.exists(path):
        os.mkdir(path)

# %%
def percentage(count_tags):
    _sum = sum(count_tags.values())
    return [(el / _sum) * 100 for el in count_tags.values()]

# %%
def cmap_rescale(elements):
    result = []
    if isinstance(elements, dict):
        _max = max(elements.values())
        _min = min(elements.values())
        result = [(el - _min) / (_max - _min) for el in elements.values()]
    if isinstance(elements, list):
        _max = np.max(elements.values())
        _min = np.min(elements.values())
        result = [(el - _min) / (_max - _min) for el in elements.values()]
    return result

# %%
def convert_to_hex(rgba_color):
    red = str(hex(int(rgba_color[0] * 255)))[2:].capitalize()
    green = str(hex(int(rgba_color[1] * 255)))[2:].capitalize()
    blue = str(hex(int(rgba_color[2] * 255)))[2:].capitalize()

    if blue == '0':
        blue = '00'
    if red == '0':
        red = '00'
    if green == '0':
        green = '00'

    return '#' + red + green + blue

# %%
def dirs2df(img_path):
    '''From img directory to dataframe. 
    input path images folder
    return df
    ------------------------------------
    img directory tree: |images
                        |      --> labels
                        |               --> .tif or .jpg'''
    dirs_path = []
    dirs = []
    dirs = os.listdir(img_path)
    dirs_path = [os.path.join(img_path, _dir) for _dir in dirs]

    imgdict = {}
    img_names = []
    img_paths = []
    for _dir in dirs_path:
        if _dir.split('/')[-1] != '.DS_Store':
            nameslist = os.listdir(_dir)
        for el in nameslist:
            if el.endswith('.jpg') | el.endswith('.tif'):
                img_names.append(el)
                img_paths.append(os.path.join(_dir, el))
        imgdict['image_name'] = img_names
    df = pd.DataFrame.from_dict(imgdict)
    df['label'] = df['image_name'].apply(lambda x: x.split('_')[0])
    return df, img_paths

# %%
def create_filenames(df, params):
    # pcg_dataset = percentage of total files to use: i.e. 30% of 40479 samples = 12143 samples
    # empty data dirs
    if df['image_name'].iloc[0].endswith('.jpg'):
        print('Format: jpg')
        train_data_dir = params.train_data_dir_jpg
        val_data_dir = params.val_data_dir_jpg
        test_data_dir = params.test_data_dir_jpg
        raw_data_dir = params.raw_jpg_path
        endswith = '.jpg'

    if df['image_name'].iloc[0].endswith('.tif'):
        print('Format: tif')
        train_data_dir = params.train_data_dir_tif
        val_data_dir = params.val_data_dir_tif
        test_data_dir = params.test_data_dir_tif
        raw_data_dir = params.raw_tif_path
        endswith = '.tif'

    data_dirs = [train_data_dir, val_data_dir, test_data_dir]
    for data_dir in data_dirs:
        for file in os.listdir(data_dir):
            os.remove(os.path.join(data_dir, file))
    # create lists of filenames for train, val, test sets
    # copy lists of images from raw folder to train, val, test folders using lists of filenames

    pcg_total_files = int(params.pcg_dataset * len(df))

    subdirs_raw = os.listdir(raw_data_dir)
    filenames_raw = []
    for subdir in subdirs_raw:
        imgs_raw = os.listdir(os.path.join(raw_data_dir, subdir))
        random_sampled = random.sample(imgs_raw, 2000)
        if params.no_imbalanced:
            sub_path_imgs = [
                os.path.join(subdir, img) for img in random_sampled
            ]
        else:
            sub_path_imgs = [os.path.join(subdir, img) for img in imgs_raw]
        filenames_raw.append(sub_path_imgs)
    filenames = [
        os.path.join(raw_data_dir, f) for sublist in filenames_raw
        for f in sublist if f.endswith(endswith)
    ]

    seed = random.seed(123)
    filenames.sort()
    random.shuffle(filenames)

    filenames = filenames[:pcg_total_files]

    split_train_test = int(
        0.9 * len(filenames))  # 10% for testing, 90% for val and train
    train_filenames_raw = filenames[:split_train_test]
    test_filenames_raw = filenames[split_train_test:]

    split_train_val = int(
        0.7 * len(train_filenames_raw))  # 70% for train, 30% for val
    val_filenames_raw = train_filenames_raw[split_train_val:]
    train_filenames_raw = train_filenames_raw[:split_train_val]

    train_val_test = [
        train_filenames_raw, val_filenames_raw, test_filenames_raw
    ]
    dest_dirs = [train_data_dir, val_data_dir, test_data_dir]

    for filename_dir, dest_dir in tqdm(zip(train_val_test, dest_dirs)):
        if len(os.listdir(dest_dir)) != len(
                filename_dir):  #check if directory is empty
            for filename in filename_dir:
                shutil.copy(filename, dest_dir)

    # get lists of filenames with new path (i.e. '.data/jpg/train/img_name.jpg')
    train_filenames = []
    val_filenames = []
    test_filenames = []

    for filename_dir, dest_dir in tqdm(zip(train_val_test, dest_dirs)):
        for filename in filename_dir:
            if dest_dir == train_data_dir:
                train_filenames.append(
                    os.path.join(dest_dir,
                                 filename.split('/')[-1]))
            elif dest_dir == val_data_dir:
                val_filenames.append(
                    os.path.join(dest_dir,
                                 filename.split('/')[-1]))
            elif dest_dir == test_data_dir:
                test_filenames.append(
                    os.path.join(dest_dir,
                                 filename.split('/')[-1]))

    train_val_test = [train_filenames, val_filenames, test_filenames]

    #get names of images for each set
    train_filenames_img = [el.split('/')[-1] for el in train_filenames_raw]
    val_filenames_img = [el.split('/')[-1] for el in val_filenames_raw]
    test_filenames_img = [el.split('/')[-1] for el in test_filenames_raw]

    data_filenames_img = [
        train_filenames_img, val_filenames_img, test_filenames_img
    ]

    print(
        'Total number of samples (train + val + test) (%d %% of original dataset) : %d'
        % (params.pcg_dataset * 100, len(filenames)))
    print('Training set - number of samples: %d' % len(train_filenames_raw))
    print('Validation set - number of samples: %d' % len(val_filenames_raw))
    print('Test set - number of samples: %d' % len(test_filenames_raw))

    print('Training set - number of samples in .data/train: %d' %
          len(os.listdir(train_data_dir)))
    print('Validation set - number of samples .data/val: %d' %
          len(os.listdir(val_data_dir)))
    print('Test set - number of samples .data/test: %d' %
          len(os.listdir(test_data_dir)))

    return train_val_test, data_filenames_img

# %%
def loss_obj(loss_type):
    if loss_type == 'categorical_crossentropy':
        loss_obj = tf.keras.losses.CategoricalCrossentropy()
    return loss_obj

# %%
def optimizer(learning_rate, momentum, opt_type):
    if opt_type == 'SGD_momentum':
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate,
                                      momentum=momentum)
    if opt_type == 'Adam':
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                       decay=0.0001)
    return opt

# %%
def load_data_using_keras(folders, df, data_filenames_img, params):
    image_generator = {}
    data_generator = {}

    for _dir, _filenames in zip(folders, data_filenames_img):
        end = _dir.split('/')[-1]
        if params.preprocess:

            if end == 'train':
                image_generator['train'] = ImageDataGenerator(
                    horizontal_flip=params.horizontal_flip,
                    vertical_flip=params.vertical_flip,
                    rotation_range=params.rotation_range,
                    shear_range=params.shear_range)

                data_generator['train'] = image_generator[
                    'train'].flow_from_dataframe(
                        dataframe=df[df['image_name'].isin(_filenames)],
                        x_col='image_name',
                        y_col=params.columns,
                        batch_size=params.batch_size,
                        directory=_dir,
                        seed=params.seed,
                        shuffle=True,
                        target_size=(64, 64),
                        class_mode='raw',
                        color_mode='rgb')

            if end == 'val':
                image_generator['val'] = ImageDataGenerator()
                data_generator['val'] = image_generator[
                    'val'].flow_from_dataframe(
                        dataframe=df[df['image_name'].isin(_filenames)],
                        x_col='image_name',
                        y_col=params.columns,
                        batch_size=params.batch_size,
                        directory=_dir,
                        seed=params.seed,
                        shuffle=False,
                        target_size=(64, 64),
                        class_mode='raw',
                        color_mode='rgb')

            if end == 'test':
                image_generator['test'] = ImageDataGenerator()
                data_generator['test'] = image_generator[
                    'test'].flow_from_dataframe(
                        dataframe=df[df['image_name'].isin(_filenames)],
                        x_col='image_name',
                        y_col=params.columns,
                        batch_size=len(df[df['image_name'].isin(_filenames)]),
                        directory=_dir,
                        seed=params.seed,
                        shuffle=False,
                        target_size=(64, 64),
                        class_mode='raw',
                        color_mode='rgb')

        else:
            if end == 'train':
                image_generator['train'] = ImageDataGenerator(
                    horizontal_flip=params.horizontal_flip,
                    vertical_flip=params.vertical_flip,
                    rotation_range=params.rotation_range,
                    shear_range=params.shear_range,
                    rescale=1. / 255)

                data_generator['train'] = image_generator[
                    'train'].flow_from_dataframe(
                        dataframe=df[df['image_name'].isin(_filenames)],
                        x_col='image_name',
                        y_col=params.columns,
                        batch_size=params.batch_size,
                        directory=_dir,
                        seed=params.seed,
                        shuffle=True,
                        target_size=(64, 64),
                        class_mode='raw',
                        color_mode='rgb')

            if end == 'val':
                image_generator['val'] = ImageDataGenerator(rescale=1. / 255)
                data_generator['val'] = image_generator[
                    'val'].flow_from_dataframe(
                        dataframe=df[df['image_name'].isin(_filenames)],
                        x_col='image_name',
                        y_col=params.columns,
                        batch_size=params.batch_size,
                        directory=_dir,
                        seed=params.seed,
                        shuffle=False,
                        target_size=(64, 64),
                        class_mode='raw',
                        color_mode='rgb')

            if end == 'test':
                image_generator['test'] = ImageDataGenerator(rescale=1. / 255)
                data_generator['test'] = image_generator[
                    'test'].flow_from_dataframe(
                        dataframe=df[df['image_name'].isin(_filenames)],
                        x_col='image_name',
                        y_col=params.columns,
                        batch_size=len(df[df['image_name'].isin(_filenames)]),
                        directory=_dir,
                        seed=params.seed,
                        shuffle=False,
                        target_size=(64, 64),
                        class_mode='raw',
                        color_mode='rgb')

    return data_generator

# %%
def normalize_band(band):
    # min-max norm
    MinMax = MinMaxScaler()
    band_norm = MinMax.fit_transform(band)
    return band_norm

# %%
def tif2sets(train_val_test_tif, dataset_tif, params):
    '''This function parses tiff images from images path, returns train, val, test set with upsampled bands'''
    # initialize
    if params.channels == 'all':
        X_train = np.zeros([len(train_val_test_tif[0]), 64, 64, 10],
                           dtype="float32")
        X_val = np.zeros([len(train_val_test_tif[1]), 64, 64, 10],
                         dtype="float32")
        X_test = np.zeros([len(train_val_test_tif[2]), 64, 64, 10],
                          dtype="float32")
        y_train = np.zeros([len(train_val_test_tif[0]), 10])
        y_val = np.zeros([len(train_val_test_tif[1]), 10])
        y_test = np.zeros([len(train_val_test_tif[2]), 10])

    else:
        X_train = np.zeros(
            [len(train_val_test_tif[0]), 64, 64,
             len(params.channels)],
            dtype="float32")
        X_val = np.zeros(
            [len(train_val_test_tif[1]), 64, 64,
             len(params.channels)],
            dtype="float32")
        X_test = np.zeros(
            [len(train_val_test_tif[2]), 64, 64,
             len(params.channels)],
            dtype="float32")
        y_train = np.zeros([len(train_val_test_tif[0]), len(params.channels)])
        y_val = np.zeros([len(train_val_test_tif[1]), len(params.channels)])
        y_test = np.zeros([len(train_val_test_tif[2]), len(params.channels)])

    sets = [(X_train, y_train), (X_val, y_val), (X_test, y_test)]

    for folder, _set in zip(train_val_test_tif, sets):
        X_set, y_set = _set
        for i in range(len(_set[0])):
            X_set[i, :, :, :] = dataset_tif[folder[i].split('/')
                                            [-1]]['X_array']
            y_set[i, :] = dataset_tif[folder[i].split('/')[-1]]['y_array']

    print('Number of samples in train set: {}'.format(len(sets[0][0])))
    print('Number of labels in train set: {}'.format(len(sets[0][-1])))
    print('\nNumber of samples in val set: {}'.format(len(sets[1][0])))
    print('Number of labels in val set: {}'.format(len(sets[1][-1])))
    print('\nNumber of samples in test set: {}'.format(len(sets[-1][0])))
    print('Number of labels in test set: {}'.format(len(sets[-1][-1])))
    print('\nTotal number of samples: {}'.format(
        len(sets[0][0]) + len(sets[1][0]) + len(sets[2][0])))
    return sets

# %%
def load_data_using_keras_tif(train_val_test_tif, dataset_tif, params):
    data_generators = {}

    train_val_test_sets = tif2sets(train_val_test_tif, dataset_tif, params)
    X_train, y_train = train_val_test_sets[0]
    X_val, y_val = train_val_test_sets[1]
    X_test, y_test = train_val_test_sets[2]

    image_generator = ImageDataGenerator(
        horizontal_flip=params.horizontal_flip,
        vertical_flip=params.vertical_flip,
        rotation_range=params.rotation_range,
        shear_range=params.shear_range)

    data_generators['train'] = image_generator.flow(
        X_train, y_train, batch_size=params.batch_size, seed=params.seed)

    data_generators['val'] = image_generator.flow(X_val,
                                                  y_val,
                                                  batch_size=params.batch_size,
                                                  seed=params.seed)

    data_generators['test'] = image_generator.flow(
        X_test, y_test, batch_size=params.batch_size, seed=params.seed)

    return data_generators

# %%
def spectral_module(x, spectral_id, squeeze=16, expand_1x1=96, expand_3x3=32):
    sq1x1 = "squeeze1x1"
    exp1x1 = "expand1x1"
    exp3x3 = "expand3x3"
    relu = "relu_"
    s_id = 'spectral' + str(spectral_id) + '/'

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    x = Conv2D(squeeze, (1, 1),
               padding='same',
               name=s_id + sq1x1,
               kernel_initializer='glorot_uniform',
               activation='relu')(x)

    left = Conv2D(expand_1x1, (1, 1),
                  padding='same',
                  name=s_id + exp1x1,
                  kernel_initializer='glorot_uniform')(x)
    left = Activation('relu', name=s_id + relu + exp1x1)(left)

    right = Conv2D(expand_3x3, (3, 3),
                   padding='same',
                   name=s_id + exp3x3,
                   kernel_initializer='glorot_uniform')(x)
    right = Activation('relu', name=s_id + relu + exp3x3)(right)

    x = concatenate([left, right], axis=channel_axis, name=s_id + 'concat')
    return x

# %%
def SpectralNet(params, input_shape=(64, 64, 10), classes=10):
    """Implementation of SpectralNet architecture - Jacob J. Senecal, John W. Sheppard, Joseph A. Shaw
                                                  - Gianforte School of Computing and Dept. Elec & Computer Engineering
                                                  - Montana State University, Bozeman, USA
                                
    paper: https://www.cs.montana.edu/sheppard/pubs/ijcnn-2019c.pdf
    
    modifing SqueezeNet implementation in Keras: https://github.com/rcmalli/keras-squeezenet
    """
    if params.extension == 'jpg':
        input_shape = (64, 64, 3)
    elif params.extension == 'tif':
        input_shape = (64, 64, 10)

    inputs = tf.keras.Input(shape=input_shape)
    x = Conv2D(96, (2, 2),
               strides=(2, 2),
               padding='same',
               name='conv1',
               activation='relu',
               kernel_initializer='glorot_uniform')(inputs)
    x = spectral_module(x,
                        spectral_id=2,
                        squeeze=16,
                        expand_1x1=96,
                        expand_3x3=32)
    x = spectral_module(x,
                        spectral_id=3,
                        squeeze=16,
                        expand_1x1=96,
                        expand_3x3=32)
    x = spectral_module(x,
                        spectral_id=4,
                        squeeze=32,
                        expand_1x1=192,
                        expand_3x3=64)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool4')(x)

    x = spectral_module(x,
                        spectral_id=5,
                        squeeze=32,
                        expand_1x1=192,
                        expand_3x3=64)
    x = spectral_module(x,
                        spectral_id=6,
                        squeeze=48,
                        expand_1x1=288,
                        expand_3x3=96)
    x = spectral_module(x,
                        spectral_id=7,
                        squeeze=48,
                        expand_1x1=288,
                        expand_3x3=96)
    x = spectral_module(x,
                        spectral_id=8,
                        squeeze=64,
                        expand_1x1=385,
                        expand_3x3=128)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool8')(x)

    x = spectral_module(x,
                        spectral_id=9,
                        squeeze=64,
                        expand_1x1=385,
                        expand_3x3=128)

    x = Conv2D(classes, (1, 1),
               padding='same',
               name='conv10',
               activation='relu',
               kernel_initializer='glorot_uniform')(x)
    x = GlobalAveragePooling2D()(x)
    softmax = Activation("softmax", name='softmax')(x)
    model = tf.keras.Model(inputs, softmax)
    model.compile(loss=params.loss_obj,
                  optimizer=params.optimizer_obj,
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])

    return model

# %%
def create_model(version, params):

    if version == 'v1.0':
        # Baseline
        inputs = Input(shape=(params.size, params.size, params.num_channels))
        x = Conv2D(32, 3, activation='relu', padding='same')(inputs)
        x = Conv2D(32, 3, activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(64, 3, activation='relu', padding='same')(x)
        x = Conv2D(64, 3, activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)

        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)

        outputs = Dense(params.num_classes, activation='softmax')(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        model.compile(loss=params.loss_obj,
                      optimizer=params.optimizer_obj,
                      metrics=[tf.keras.metrics.CategoricalAccuracy()])

    if version == 'v1.1':
        # v1.0 with 128 units in FC layer w.r.t 64
        inputs = Input(shape=(params.size, params.size, params.num_channels))
        x = Conv2D(32, 3, activation='relu', padding='same')(inputs)
        x = Conv2D(32, 3, activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(64,
                   3,
                   activation='relu',
                   kernel_initializer='he_uniform',
                   padding='same')(x)
        x = Conv2D(64,
                   3,
                   activation='relu',
                   kernel_initializer='he_uniform',
                   padding='same')(x)
        x = MaxPooling2D((2, 2))(x)

        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)

        outputs = Dense(params.num_classes, activation='softmax')(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        model.compile(loss=params.loss_obj,
                      optimizer=params.optimizer_obj,
                      metrics=[tf.keras.metrics.CategoricalAccuracy()])

    if version == 'v1.2':
        # v1.3 with dropout layers after each block
        inputs = Input(shape=(params.size, params.size, params.num_channels))
        x = Conv2D(32, 3, activation='relu', padding='same')(inputs)
        x = Conv2D(32, 3, activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(64, 3, activation='relu', padding='same')(x)
        x = Conv2D(64, 3, activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(128, 3, activation='relu', padding='same')(x)
        x = Conv2D(128, 3, activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)

        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)

        outputs = Dense(params.num_classes, activation='softmax')(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        model.compile(loss=params.loss_obj,
                      optimizer=params.optimizer_obj,
                      metrics=[tf.keras.metrics.CategoricalAccuracy()])

    if version == 'v1.3':
        inputs = Input(shape=(params.size, params.size, params.num_channels))
        x = Conv2D(32, 3, activation='relu', padding='same')(inputs)
        x = Conv2D(32, 3, activation='relu', padding='same')(x)
        x = Conv2D(32, 3, activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(64, 3, activation='relu', padding='same')(x)
        x = Conv2D(64, 3, activation='relu', padding='same')(x)
        x = Conv2D(64, 3, activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(128, 3, activation='relu', padding='same')(x)
        x = Conv2D(128, 3, activation='relu', padding='same')(x)
        x = Conv2D(128, 3, activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)

        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)

        outputs = Dense(params.num_classes, activation='softmax')(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        model.compile(loss=params.loss_obj,
                      optimizer=params.optimizer_obj,
                      metrics=[tf.keras.metrics.CategoricalAccuracy()])

    if version == 'v1.4':
        inputs = Input(shape=(params.size, params.size, params.num_channels))
        x = Conv2D(32, 3, activation='relu', padding='same')(inputs)
        x = Conv2D(32, 3, activation='relu', padding='same')(x)
        x = Conv2D(32, 3, activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(64, 3, activation='relu', padding='same')(x)
        x = Conv2D(64, 3, activation='relu', padding='same')(x)
        x = Conv2D(64, 3, activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(128, 3, activation='relu', padding='same')(x)
        x = Conv2D(128, 3, activation='relu', padding='same')(x)
        x = Conv2D(128, 3, activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)

        x = Flatten()(x)
        x = Dense(1024, activation='relu')(x)

        outputs = Dense(params.num_classes, activation='softmax')(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        model.compile(loss=params.loss_obj,
                      optimizer=params.optimizer_obj,
                      metrics=[tf.keras.metrics.CategoricalAccuracy()])

    return model

# %%
def create_resnet(params):
    if params.trainable == True:
        print('\n Unfreezing ResNet {}% top layers'.format(
            params.pcg_unfreeze * 100))
        layers_to_freeze = 175 - int(
            175 * params.pcg_unfreeze
        )  #resnet has 175 layers; this is the number of layers to freeze
        base_model = tf.keras.applications.ResNet50(
            input_shape=(params.size, params.size, params.num_channels),
            include_top=False,
            weights='imagenet')

        for layer in base_model.layers[:layers_to_freeze]:
            layer.trainable = False
        for layer in base_model.layers[layers_to_freeze:]:
            layer.trainable = True

        if params.regularization:
            base_model = add_regularization(
                base_model, regularizer=tf.keras.regularizers.l2(0.0001))
            print('L2 regularization added')

        if params.preprocess:
            inputs = tf.keras.Input(shape=(params.size, params.size,
                                           params.num_channels))
            x = tf.keras.applications.resnet.preprocess_input(inputs)
            x = base_model(x, training=False)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = Dropout(0.2)(x)
            outputs = Dense(params.num_classes, activation='softmax')(x)
            model = tf.keras.Model(inputs, outputs)

            model.compile(loss=params.loss_obj,
                          optimizer=params.optimizer_obj,
                          metrics=[tf.keras.metrics.CategoricalAccuracy()])

        else:
            inputs = tf.keras.Input(shape=(params.size, params.size,
                                           params.num_channels))
            x = base_model(inputs, training=False)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = Dropout(0.2)(x)
            outputs = Dense(params.num_classes, activation='softmax')(x)
            model = tf.keras.Model(inputs, outputs)

            model.compile(loss=params.loss_obj,
                          optimizer=params.optimizer_obj,
                          metrics=[tf.keras.metrics.CategoricalAccuracy()])

    elif (params.trainable == 'Full'):

        print('\n Using Resnet - Full training'.format(params.pcg_unfreeze))
        base_model = tf.keras.applications.ResNet50(
            input_shape=(params.size, params.size, params.num_channels),
            include_top=False,
            weights='imagenet')

        if params.preprocess:
            print('\n Using Keras preprocess_input')
            base_model.trainable = True
            if params.regularization:
                base_model = add_regularization(
                    base_model, regularizer=tf.keras.regularizers.l2(0.0001))
                print('L2 regularization added')
            inputs = tf.keras.Input(shape=(params.size, params.size,
                                           params.num_channels))
            x = tf.keras.applications.resnet.preprocess_input(inputs)
            x = base_model(x, training=False)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = Dropout(0.2)(x)
            outputs = Dense(params.num_classes, activation='softmax')(x)
            model = tf.keras.Model(inputs, outputs)

            model.compile(loss=params.loss_obj,
                          optimizer=params.optimizer_obj,
                          metrics=[tf.keras.metrics.CategoricalAccuracy()])

        else:
            base_model.trainable = True
            if params.regularization:
                base_model = add_regularization(
                    base_model, regularizer=tf.keras.regularizers.l2(0.0001))
                print('L2 regularization added')
            inputs = tf.keras.Input(shape=(params.size, params.size,
                                           params.num_channels))
            x = tf.keras.applications.resnet.preprocess_input(inputs)
            x = base_model(x, training=False)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = Dropout(0.2)(x)
            outputs = Dense(params.num_classes, activation='softmax')(x)
            model = tf.keras.Model(inputs, outputs)

            model.compile(loss=params.loss_obj,
                          optimizer=params.optimizer_obj,
                          metrics=[tf.keras.metrics.CategoricalAccuracy()])

    else:
        print('\n Using Resnet as feature extractor'.format(
            params.pcg_unfreeze))
        base_model = tf.keras.applications.ResNet50(
            input_shape=(params.size, params.size, params.num_channels),
            include_top=False,
            weights='imagenet')

        if params.preprocess:
            base_model.trainable = False
            if params.regularization:
                print('L2 regularization added')
                base_model = add_regularization(
                    base_model, regularizer=tf.keras.regularizers.l2(0.0001))
            inputs = tf.keras.Input(shape=(params.size, params.size,
                                           params.num_channels))
            x = tf.keras.applications.mobilenet.preprocess_input(inputs)
            x = base_model(x, training=False)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = Dropout(0.2)(x)
            outputs = Dense(params.num_classes, activation='softmax')(x)
            model = tf.keras.Model(inputs, outputs)

            model.compile(loss=params.loss_obj,
                          optimizer=params.optimizer_obj,
                          metrics=[tf.keras.metrics.CategoricalAccuracy()])

        else:
            base_model.trainable = False
            inputs = tf.keras.Input(shape=(params.size, params.size,
                                           params.num_channels))
            x = base_model(inputs, training=False)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = Dropout(0.2)(x)
            outputs = Dense(params.num_classes, activation='softmax')(x)
            model = tf.keras.Model(inputs, outputs)

            model.compile(loss=params.loss_obj,
                          optimizer=params.optimizer_obj,
                          metrics=[tf.keras.metrics.CategoricalAccuracy()])

    return model

# %%
# credits to Thalles Silva: https://gist.github.com/sthalles
def add_regularization(model, regularizer=tf.keras.regularizers.l2(0.0001)):

    if not isinstance(regularizer, tf.keras.regularizers.Regularizer):
        print(
            "Regularizer must be a subclass of tf.keras.regularizers.Regularizer"
        )
        return model

    for layer in model.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)

    # Save the weights before reloading the model.
    config_json = model.to_json()
    tmp_weights_path = os.path.join(tempfile.gettempdir(),
                                    'tmp_weights_resnet.h5')
    model.save_weights(tmp_weights_path)

    model = tf.keras.models.model_from_json(config_json)
    # Reload the model weights
    model.load_weights(tmp_weights_path, by_name=True)

    return model

# %%
def run_models_generator(versions,
                         data_generator,
                         test_dataset,
                         test_labels,
                         train_params,
                         experiment=''):
    v_outputs = {}
    log_folder = train_params.log_folder
    log_cm_path = train_params.log_cm_path
    for i, version in enumerate(versions):
        v = []
        v_history = []
        v_loss = []
        v_grid = []
        v_dict = {}

        version_folder = os.path.join(log_folder, version + experiment)
        mkdir(log_cm_path)
        v, v_history, v_loss, v_grid = run_baseline_model_generator(
            version, data_generator, test_dataset, test_labels, train_params,
            version_folder)
        shutil.copytree(log_cm_path, os.path.join(version_folder, 'cm'))
        shutil.rmtree(log_cm_path)

        v_meta = {
            'channels': train_params.channels,
            'image_size': train_params.size,
            'num_images_train': train_params.num_images_train,
            'num_images_val': train_params.num_images_val,
            'num_images_test': train_params.num_images_test,
            'channels': train_params.num_channels,
            'epochs': train_params.num_epochs,
            'batch_size': train_params.batch_size,
            'loss_type': train_params.loss_type,
            'opt_type': train_params.opt_type,
            'learning_rate': train_params.learning_rate,
            'momentum': train_params.momentum,
            'regularization': train_params.regularization,
            'horizontal_flip': train_params.horizontal_flip,
            'vertical_flip': train_params.vertical_flip,
            'rotation_range': train_params.rotation_range,
            'shear_range': train_params.shear_range
        }

        v_dict['meta'] = v_meta
        v_dict['model'] = v
        v_dict['history'] = v_history
        v_dict['loss'] = v_loss
        v_dict['grid'] = v_grid

        v_outputs[version] = v_dict

    return v_outputs

# %%
def run_baseline_model_generator(version, data_generator, test_dataset,
                                 test_labels, train_params, version_folder):
    if version.startswith('ResNet'):
        model = create_resnet(train_params)
        print('Version: Resnet model - {}'.format(
            version_folder.split('/')[-1]))

    elif version.startswith('SpectralNet'):
        model = SpectralNet(train_params)
        print('Version: SpectralNet model - {}'.format(
            version_folder.split('/')[-1]))
    else:
        model = create_model(version, train_params)
        print('Version: {}'.format(version_folder.split('/')[-1]))

    # History

    if train_params.rlronplateau:
        print('RLRonPlateau: active\n')
        cm_callback = ConfusionMatrixCallback(test_dataset, test_labels,
                                              train_params)
        ReduceLRonPLateau_callback = ReduceLROnPlateau(monitor='loss',
                                                       factor=0.1,
                                                       patience=3,
                                                       mode='min',
                                                       min_lr=0.000001)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(
                version_folder,
                datetime.datetime.now().strftime("%Y%m%d-%H%M%S")),
            histogram_freq=1)

        history = model.fit_generator(
            data_generator['train'],
            steps_per_epoch=train_params.num_images_train //
            train_params.batch_size,
            epochs=train_params.num_epochs,
            validation_data=data_generator['val'],
            validation_steps=train_params.num_images_val //
            train_params.batch_size,
            callbacks=[
                tensorboard_callback, cm_callback, ReduceLRonPLateau_callback
            ])

    else:
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(
                version_folder,
                datetime.datetime.now().strftime("%Y%m%d-%H%M%S")),
            histogram_freq=1)

        # Confusion matrix
        cm_callback = ConfusionMatrixCallback(test_dataset, test_labels,
                                              train_params)
        history = model.fit_generator(
            data_generator['train'],
            steps_per_epoch=train_params.num_images_train //
            train_params.batch_size,
            epochs=train_params.num_epochs,
            validation_data=data_generator['val'],
            validation_steps=train_params.num_images_val //
            train_params.batch_size,
            callbacks=[tensorboard_callback, cm_callback])

    loss, val_loss, categorical_accuracy, val_categorical_accuracy = learning_curves(
        history, version)
    grid = perf_grid(test_dataset,
                     test_labels,
                     train_params.columns,
                     model,
                     n_thresh=100)

    return model, history, loss, grid

# %%
def results_to_file(versions, experiment):
    # save
    assets_path = './reports/assets/'
    saved_models_dir = './reports/saved_models'

    save_path = os.path.join(assets_path,
                             list(versions.keys())[0] + experiment)
    save_meta_csv_path = os.path.join(
        save_path,
        list(versions.keys())[0] + experiment + '_meta_.csv')
    save_grid_csv_path = os.path.join(
        save_path,
        list(versions.keys())[0] + experiment + '_grid_.csv')
    mkdir(save_path)

    df_meta = pd.DataFrame(versions['ResNet']['meta']).iloc[0]
    df_grid = pd.DataFrame(versions['ResNet']['grid'])

    # save meta and grid to csv
    pd.DataFrame.to_csv(df_meta, save_meta_csv_path, index=False)
    pd.DataFrame.to_csv(df_grid, save_grid_csv_path, index=False)

    # save model
    versions['ResNet']['model'].save(
        os.path.join(saved_models_dir,
                     list(versions.keys())[0] + experiment))

# %%
class ConfusionMatrixCallback(tf.keras.callbacks.Callback):
    def __init__(self, X_test, y_test, params):
        self.X_test = X_test
        self.y_test = y_test
        self.params = params

    def on_epoch_end(self, epoch, logs=None):
        train_params = params('jpg', 1)
        log_folder = './reports/logs'
        log_cm_path = os.path.join(log_folder, 'cm')
        cm_writer = tf.summary.create_file_writer(log_cm_path)
        test_pred = self.model.predict(self.X_test)
        # Calculate the confusion matrix using sklearn.metrics
        cm = tfa.metrics.MultiLabelConfusionMatrix(
            num_classes=(train_params.num_classes))(self.y_test,
                                                    np.where(
                                                        test_pred > 0.5, 1, 0))
        figure = plot_confusion_matrix(cm, train_params.columns)
        cm_image = plot_to_image(figure)

        # Log the confusion matrix as an image summary.
        with cm_writer.as_default():
            tf.summary.image("Confusion Matrix", cm_image, step=epoch)

# %%
def perf_grid(dataset, labels, columns, model, n_thresh=100):
    """Computes the performance table containing target, label names,
    label frequencies, thresholds between 0 and 1, number of tp, fp, fn,
    precision, recall and f-score metrics for each label.
    
    Args:
        dataset (tf.data.Datatset): contains the features array
        labels (numpy array): target matrix of shape (BATCH_SIZE, N_LABELS)
        tags (list of strings): column names in target matrix
        model (tensorflow keras model): model to use for prediction
        n_thresh (int) : number of thresholds to try
        
    Returns:
        grid (Pandas dataframe): performance table 
    """

    # Get predictions
    y_hat_val = model.predict(dataset)
    # Define target matrix
    y_val = np.array(labels)
    # Find label frequencies in the validation set
    label_freq = np.array(labels).sum(axis=0)
    # Get label indexes
    label_index = [i for i in range(len(columns))]
    # Define thresholds
    thresholds = np.linspace(0, 1, n_thresh + 1).astype(np.float32)

    # Compute all metrics for all labels
    ids, labels, freqs, tps, fps, fns, precisions, recalls, f1s = [], [], [], [], [], [], [], [], []
    for l in label_index:
        for thresh in thresholds:
            ids.append(l)
            labels.append(columns[l])
            freqs.append(round(label_freq[l] / len(y_val), 2))
            y_hat = y_hat_val[:, l]
            y = y_val[:, l]
            y_pred = y_hat > thresh
            tp = np.count_nonzero(y_pred * y)
            fp = np.count_nonzero(y_pred * (1 - y))
            fn = np.count_nonzero((1 - y_pred) * y)
            precision = tp / (tp + fp + 1e-16)
            recall = tp / (tp + fn + 1e-16)
            f1 = tp / (tp + (fn + fp) * 0.5 + 1e-16)
            tps.append(tp)
            fps.append(fp)
            fns.append(fn)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

    # Create the performance dataframe
    grid = pd.DataFrame({
        'id': ids,
        'label': np.array(labels),
        'freq': freqs,
        'threshold': list(thresholds) * len(label_index),
        'tp': tps,
        'fp': fps,
        'fn': fns,
        'precision': precisions,
        'recall': recalls,
        'f1': f1s
    })

    grid = grid[[
        'id', 'label', 'freq', 'threshold', 'tp', 'fn', 'fp', 'precision',
        'recall', 'f1'
    ]]

    return grid

# %%
# Modified versions of functions implemented by Ashref Maiza
def learning_curves(history, version):
    """Plot the learning curves of loss and macro f1 score 
    for the training and validation datasets.
    
    Args:
        history: history callback of fitting a tensorflow keras model 
    """
    path_assets = './reports/assets/{}'.format(version)
    mkdir(path_assets)
    title_loss = 'Training and Validation Loss - Model {}'.format(version)
    title_f1_score = 'Training and Validation Categorical Accuracy - Model {}'.format(
        version)
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    categorical_accuracy = history.history['categorical_accuracy']
    val_categorical_accuracy = history.history['val_categorical_accuracy']

    epochs = len(loss)

    style.use("bmh")
    plt.figure(figsize=(8, 8))

    plt.subplot(2, 1, 1)
    plt.plot(range(1, epochs + 1), loss, label='Training Loss')
    plt.plot(range(1, epochs + 1), val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Loss')
    plt.title(title_loss)
    plt.tight_layout()

    plt.savefig('./reports/assets/{}/{}.png'.format(version, title_loss))

    plt.subplot(2, 1, 2)
    plt.plot(range(1, epochs + 1),
             categorical_accuracy,
             label='Training categorical accuracy')
    plt.plot(range(1, epochs + 1),
             val_categorical_accuracy,
             label='Validation categorical accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Categorical accuracy')
    plt.title(title_f1_score)
    plt.xlabel('epoch')
    plt.tight_layout()
    plt.savefig('./reports/assets/{}/{}.png'.format(version, title_f1_score))

    plt.show()

    return loss, val_loss, categorical_accuracy, val_categorical_accuracy

# %%
def plot_confusion_matrix(cm, columns):
    fig = plt.figure(figsize=(10, 20))
    for i, (label, matrix) in enumerate(zip(columns, cm)):
        ax = plt.subplot(6, 3, i + 1)
        labels = [f'not_{label}', label]
        sns.heatmap(matrix,
                    ax=ax,
                    annot=True,
                    square=True,
                    fmt='.0f',
                    cbar=False,
                    cmap='Blues',
                    xticklabels=labels,
                    yticklabels=labels,
                    linecolor='black',
                    linewidth=1)
        plt.title(labels[1], size=8)
        plt.subplots_adjust(wspace=5, hspace=5)
        ax.set_yticklabels(labels, va='center', position=(0, 0.28), size=8)
        ax.set_xticklabels(labels, ha='center', position=(0.28, 0), size=8)
        plt.xlabel('PREDICTED CLASS', labelpad=10)
        plt.ylabel('TRUE CLASS', labelpad=10)
        plt.tight_layout()

    return fig

# %%
def plot_to_image(figure):
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    """

    buf = BytesIO()

    # Use plt.savefig to save the plot to a PNG in memory.
    plt.savefig(buf, format='png')

    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)

    # Use tf.image.decode_png to convert the PNG buffer
    # to a TF image. Make sure you use 4 channels.
    image = tf.image.decode_png(buf.getvalue(), channels=4)

    # Use tf.expand_dims to add the batch dimension
    image = tf.expand_dims(image, 0)

    return image