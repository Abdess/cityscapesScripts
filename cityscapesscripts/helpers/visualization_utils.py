import cv2
import numpy as np
import os
import scipy.misc
from glob import glob
from moviepy.editor import ImageSequenceClip


def print_segmentation_onto_image(image, prediction, color_map):
    """
    Affiche une segmentation sur une image de taille égale en fonction d'une
    carte de couleurs.

    Arguments :
        image (sous forme de tableau) : Une image à 3 canaux sur laquelle
            imprimer la segmentation de `prediction`.
        prediction (analogue à un tableau) : Un tableau de rang 4 qui est la
            prédiction de la segmentation avec les mêmes dimensions spatiales
            que `image`. Le dernier axe contient les classes de segmentation
            dans un format à un coup.
        color_map (dictionnaire) : Un dictionnaire Python dont les clés sont
            des entiers non négatifs représentant les classes de segmentation
            et dont les valeurs sont des tuples 1D (ou listes, tableaux Numpy)
            de longueur 4 qui représentent les valeurs de couleur RGBA dans
            lesquelles les classes respectives doivent être annotées.
            Par exemple, si le dictionnaire contient la paire clé-valeur
            `{1 : (0, 255, 0, 127)}`, cela signifie que tous les pixels de la
            prédiction qui appartiennent à la classe de segmentation 1 seront
            colorés en vert avec une transparence de 50% dans l'image d'entrée.

    Retourne :
        Une copie de l'image d'entrée avec la segmentation imprimée dessus.

    Renvoie :
        ValueError si les dimensions spatiales de `image` et `prediction` ne
        correspondent pas.
    """

    if (image.shape[0] != prediction.shape[1]) or (image.shape[1] !=
                                                   prediction.shape[2]):
        raise ValueError(
            "L''image' et la 'prediction' doivent avoir la même hauteur et la \
            même largeur, mais l'image a des dimensions spatiales ({}, {}) \
            et la prédiction a des dimensions spatiales ({}, {}).".format(
                image.shape[0], image.shape[1], prediction.shape[1],
                prediction.shape[2]))

    image_size = image.shape

    # Créer un modèle de forme `(image_height, image_width, 4)`
    # pour stocker les valeurs RGBA.
    mask = np.zeros(shape=(image_size[0], image_size[1], 4), dtype=np.uint8)
    segmentation_map = np.squeeze(np.argmax(prediction, axis=-1))

    # Boucle sur toutes les classes de segmentation qui doivent être annotées
    # et place leur valeur de couleur au pixel d'image respectif.
    for segmentation_class, color_value in color_map.items():
        mask[segmentation_map == segmentation_class] = color_value

    mask = scipy.misc.toimage(mask, mode="RGBA")

    output_image = scipy.misc.toimage(image)
    output_image.paste(
        mask, box=None, mask=mask
    )  # Voir http://effbot.org/imagingbook/image.htm#tag-Image.Image.paste
    # pour plus de détails.

    return output_image


def create_split_view(target_size, images, positions, sizes, captions=[]):
    """
    Place les images sur un canevas rectangulaire pour créer une vue
    fractionnée.

    Arguments :
        target_size (tuple) : La taille cible du canevas de sortie au format
            (hauteur, largeur). Le canevas de sortie aura toujours trois
            canaux de couleur.
        images (liste) : Une liste contenant les images à placer sur le
            canevas de sortie. Les images peuvent varier en taille et peuvent
            avoir un ou trois canaux de couleur.
        positions (liste) : Une liste contenant les positions souhaitées du
            coin supérieur gauche des images dans le canevas de sortie au
            format (y, x), où x se réfère à la coordonnée horizontale et y se
            réfère à la coordonnée verticale et les deux sont des entiers non
            négatifs.
        sizes (liste) : Une liste contenant des tuples avec les tailles
            souhaitées des images dans le format (hauteur, largeur).
        captions (list, facultatif) : Une liste contenant soit une chaîne de
            légende, soit `None` pour chaque image. La liste doit avoir la
            même longueur que `images`. La valeur par défaut est une liste
            vide, c'est-à-dire qu'aucune légende ne sera ajoutée.

    Retourne :
        L'image de la vue fractionnée de taille `target_size`.
    """

    assert len(images) == len(positions) == len(
        sizes
    ), "Les `images`, `positions` et `sizes` doivent avoir la même longueur, \
    mais c'est `len(images) == {}`, `len(poisitons) = {}`, `len(sizes) == \
    {}`.".format(len(images), len(positions), len(sizes))

    y_max, x_max = target_size
    canvas = np.zeros((y_max, x_max, 3), dtype=np.uint8)

    for i, img in enumerate(images):

        # Redimensionner l'image
        if img.shape[0] != sizes[i][0] | img.shape[1] != sizes[i][1]:
            img = scipy.misc.imresize(img, sizes[i])

        # Placer l'image redimensionnée sur le canevas
        y, x = positions[i]
        h, w = sizes[i]
        # Si img est en niveaux de gris, la méthode de diffusion Numpy mettra
        # la même valeur d'intensité pour chacun des canaux R, G, et B.
        # L'indexation ci-dessous protège contre les problèmes d'index
        # hors de portée.
        canvas[y:min(y + h, y_max),
        x:min(x + w, x_max), :] = img[:min(h, y_max -
                                           y), :min(w, x_max - x)]

        # Affichez les légendes sur le canvas s'il y en a.
        if captions and (captions[i] is not None):
            cv2.putText(canvas,
                        "{}".format(captions[i]), (x + 10, y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.8,
                        color=(255, 255, 255),
                        thickness=2,
                        lineType=cv2.LINE_AA)

    return canvas


def create_video_from_images(video_output_name,
                             image_input_dir,
                             frame_rate=30.0,
                             image_file_extension='png'):
    """
    Crée une vidéo MP4 à partir des images contenues dans un répertoire donné.

    Arguments :
        nom_sortie_vidéo (chain) : Le chemin complet et le nom de la vidéo de
            sortie, sans l'extension de fichier. La vidéo de sortie sera au
            format MP4.
        image_input_dir (string) : Le répertoire qui contient les images
            d'entrée.
        frame_rate (float, facultatif) : Le nombre d'images par seconde.
        image_file_extension : L'extension de fichier des images sources.
            Seules les images dont l'extension de fichier correspond seront
            incluses dans la vidéo. La valeur par défaut est 'png'.
    """

    image_paths = glob(
        os.path.join(image_input_dir, '*.' + image_file_extension))
    image_paths = sorted(image_paths)

    video = ImageSequenceClip(image_paths, fps=frame_rate)
    video.write_videofile("{}.mp4".format(video_output_name))
