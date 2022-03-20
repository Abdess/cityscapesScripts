import cv2
import imageio
import numpy as np
import os
import pathlib
import random
import sys
from glob import glob
from math import ceil
from tqdm import trange

from cityscapesscripts.helpers.ground_truth_conversion_utils import convert_IDs_to_IDs, \
    convert_IDs_to_one_hot, convert_between_IDs_and_colors, \
    convert_IDs_to_IDs_partial


class BatchDataGenAug:

    def __init__(self,
                 image_dirs,
                 image_file_extension='png',
                 ground_truth_dirs=None,
                 image_name_split_separator=None,
                 ground_truth_suffix=None,
                 check_existence=True,
                 num_classes=None,
                 root_dir=None,
                 export_dir=None):
        """
        Arguments :
            image_dirs ( list ) : Une liste de chemins de répertoires, chacun
                d'entre eux contenant des images soit directement, soit dans
                une hiérarchie de sous-répertoires. Les chemins de répertoire
                donnés servent de répertoires racines et le générateur
                chargera les images de tous les sous-répertoires. Cela vous
                permet de combiner plusieurs ensembles de données de manière
                aléatoire. Toutes les images doivent avoir 3 canaux.
            image_file_extension (string, facultatif) : L'extension de fichier
                des images dans les ensembles de données. Doit être identique
                pour toutes les images dans tous les ensembles de données dans
                `datasets`. La valeur par défaut est `png`.
            ground_truth_dirs ( list, facultatif ) : `None` ou une liste de
                chemins de répertoire, chacun d'entre eux contenant les images
                de vérité de terrain qui correspondent aux chemins de
                répertoire respectifs dans `datasets'. Les images de vérité
                terrain doivent avoir 1 canal qui code les classes de
                segmentation numérotées consécutivement de 0 à `n`, où `n` est
                un nombre entier.
            image_name_split_separator (string, facultatif) : N'est pertinent
                que si `ground_truth_dirs` contient au moins un élément. Une
                chaîne de caractères par laquelle les noms d'images seront
                divisés en une partie gauche et une partie droite, dont la
                partie gauche (c'est-à-dire le début du nom du fichier image)
                sera utilisée pour obtenir le nom du fichier image de vérité
                terrain correspondant. Plus précisément, tous les caractères à
                gauche de la chaîne séparatrice constitueront le début du nom
                de fichier de l'image de vérité terrain correspondante.
            ground_truth_suffix (string, facultatif) : Le suffixe ajouté à la
                partie gauche d'une chaîne de nom d'image
                (voir `image_name_split_separator`) afin de composer le nom du
                fichier d'image de vérité terrain correspondant. Le suffixe
                doit exclure l'extension du fichier.
            check_existence (bool, facultatif) : Seulement pertinent si les
                images de vérité terrain sont données. Si `True`, le
                constructeur vérifie pour chaque chemin d'image de vérité
                terrain si le fichier respectif existe réellement et lance un
                `DataError` si ce n'est pas le cas. La valeur par défaut est
                `True`.
            num_classes (int, optionnel) : Le nombre de classes de
                segmentation dans les données de vérité terrain. N'est
                pertinent que si vous voulez que le générateur convertisse les
                étiquettes numériques en un format à un coup, sinon vous
                pouvez laisser cette option `None`.
            root_dir (string, facultatif) : Le répertoire racine du jeu de
                données. Ceci n'est pertinent que si vous voulez utiliser le
                générateur pour enregistrer les données traitées sur le disque
                en plus de les restituer, c'est-à-dire si vous voulez faire un
                traitement hors ligne. Dans ce cas, le générateur reproduira
                la hiérarchie des répertoires des données sources dans le
                répertoire cible dans lequel il enregistrera les données
                traitées. Pour ce faire, il doit connaître le répertoire
                racine de l'ensemble de données.
            export_dir (chaîne de caractères, facultatif) : Ce paramètre n'est
                pertinent que si vous souhaitez utiliser le générateur pour
                enregistrer les données traitées sur le disque en plus de les
                restituer, c'est-à-dire si vous souhaitez effectuer un
                traitement hors ligne. Il s'agit du répertoire dans lequel les
                données traitées seront écrites. Le générateur reproduira la
                hiérarchie des répertoires des données sources dans ce
                répertoire.
        """

        self.image_dirs = image_dirs
        self.ground_truth_dirs = ground_truth_dirs
        self.root_dir = root_dir  # Le répertoire racine du jeu de données.
        self.export_dir = export_dir
        self.image_paths = [
        ]  # La liste des images dans lesquelles le générateur va puiser.
        self.ground_truth_paths = {
        }  # Le dictionnaire des images de la vérité du terrain qui...
        # ...correspondent aux images.
        self.num_classes = num_classes
        self.dataset_size = 0
        # Si des images de vérité de terrain ont été données ou non.
        self.ground_truth = False

        if (not self.ground_truth_dirs is None) and (len(
                self.image_dirs) != len(self.ground_truth_dirs)):
            raise ValueError(
                "`image_dirs` et `ground_truth_dirs` doivent contenir le même\
                nombre d'éléments.")

        image_file_extension = image_file_extension.lower()

        for i, image_dir in enumerate(
                image_dirs
        ):  # Itérer sur tous les ensembles de données fournis.

            for image_dir_path, subdir_list, file_list in os.walk(
                    image_dir, topdown=True
            ):  # Itère sur tous les sous-répertoires de ce répertoire de jeux
                # de données.

                image_paths = glob(
                    os.path.join(image_dir_path, '*.' + image_file_extension)
                )  # Obtenir toutes les images dans ce répertoire

                if len(
                        image_paths
                ) > 0:  # S'il y a des images, il s'agit de les ajouter à la
                    # liste des images.

                    self.image_paths += image_paths

                    # S'il existe des données de vérité terrain, on les ajoute
                    # à la liste de vérité terrain.
                    if not ground_truth_dirs is None:
                        # Permet d'obtenir le chemin du répertoire de la
                        # vérité terrain qui correspond à ce répertoire
                        # d'image.
                        ground_truth_dir = ground_truth_dirs[
                            i]  # Prends le sommet.
                        ground_truth_subdir = os.path.basename(
                            os.path.normpath(image_dir_path)
                        )  # Obtenir le sous-répertoire dans lequel nous nous
                        # trouvons actuellement.
                        ground_truth_dir_path = os.path.join(
                            ground_truth_dir, ground_truth_subdir)

                        # Boucle sur tous les chemins d'image pour collecter
                        # les chemins d'image de vérité de terrain
                        # correspondants.
                        for image_path in image_paths:
                            # Construire le nom de l'image de vérité du
                            # terrain à partir du nom de l'image.
                            image_name = os.path.basename(image_path)
                            left_part = image_name.split(
                                image_name_split_separator, 1
                            )[0]  # Prends la partie gauche de la séparation.
                            # Le nom de l'image de vérité terrain qui
                            # correspond à cette image.
                            ground_truth_image_name = left_part + \
                                                      ground_truth_suffix + '.' + image_file_extension
                            # Créer le chemin complet vers cette image de
                            # vérité du terrain.
                            ground_truth_path = os.path.join(
                                ground_truth_dir_path, ground_truth_image_name)

                            if check_existence and not os.path.isfile(
                                    ground_truth_path):
                                raise DataError(
                                    "Le jeu de données contient un fichier\
                                     image '{}' pour lequel le fichier image\
                                      de vérité de terrain correspondant\
                                       n'existe pas à '{}'.".format(
                                        image_path, ground_truth_path))

                            # Ajouter la paire
                            # `image_name : ground_truth_path` au dictionnaire
                            self.ground_truth_paths[
                                image_name] = ground_truth_path

        self.dataset_size = len(self.image_paths)

        if self.dataset_size == 0:
            raise DataError(
                "Aucune image avec l'extension de fichier '{}' n'a été\
                 trouvée dans les répertoires d'images spécifiés.".format(
                    image_file_extension))

        if (not ground_truth_dirs is None) and (len(self.ground_truth_paths) !=
                                                self.dataset_size):
            raise DataError(
                'Les répertoires de vérité terrain ont été donnés, mais le\
                  nombre d\'images de vérité terrain trouvées ne correspond\
                  pas au nombre d\'images. Nombre d\'images : {}. Nombre\
                  d\'images de vérité terrain : {}'.format(
                    self.dataset_size, len(self.ground_truth_paths)))

        if len(self.ground_truth_paths) > 0:
            self.ground_truth = True

    def get_num_files(self):
        """
        Renvoie le nombre total de fichiers d'image
        (ou de paires de fichiers d'image/de vérité de terrain si des données
        de vérité de terrain ont été fournies) contenus dans tous les
        répertoires de jeux de données transmis au constructeur de
        BatchDataGenAug.
        """
        return self.dataset_size

    def generate(self,
                 batch_size,
                 convert_colors_to_ids=False,
                 convert_ids_to_ids=False,
                 convert_to_one_hot=True,
                 void_class_id=None,
                 random_crop=False,
                 crop=False,
                 resize=False,
                 brightness=False,
                 flip=False,
                 translate=False,
                 scale=False,
                 gray=False,
                 to_disk=False,
                 shuffle=True):
        """

        Avec l'une des transformations d'images ci-dessous, les images de
        référence respectives, si elles sont données, seront transformées en
        conséquence.

        Arguments :
            batch_size (int) : Le nombre d'images
                (ou de paires image/vérité de terrain) à générer par lot.
            convert_colors_to_ids (dict, facultatif) : `False` ou un
                dictionnaire dans lequel les clés sont 3-tuples de
                `dtype uint8` représentant des valeurs de couleur à 3 canaux
                et les valeurs sont des entiers représentant l'ID de la classe
                de segmentation associée à une valeur de couleur donnée. Si
                les images de vérité terrain d'entrée sont des images couleur
                à 3 canaux et qu'un dictionnaire de conversion est passé, les
                images de vérité terrain seront converties en images à un seul
                canal avec les ID de classe correspondants au lieu des valeurs
                de couleur. Il est recommandé d'effectuer la conversion
                couleur-ID hors ligne.
            convert_ids_to_ids (array or dict, facultatif) : `False` ou un
                tableau 1D Numpy ou un dictionnaire Python qui représente une
                carte selon laquelle le générateur convertira les identifiants
                de classe actuels des données de vérité terrain en
                identifiants de classe souhaités. Dans le cas d'un tableau,
                les indices du tableau représentent les identifiants actuels
                et les valeurs entières du tableau représentent les
                identifiants souhaités à convertir. Le tableau doit contenir
                une carte pour tous les identifiants de classe actuels uniques
                possibles. Dans le cas d'un dictionnaire, les clés et les
                valeurs doivent être des entiers. Les clés sont les
                identifiants actuels et les valeurs sont les identifiants
                souhaités à convertir. Il n'est pas nécessaire que le
                dictionnaire contienne un mappage pour tous les identifiants
                uniques possibles de la classe actuelle. Pour la conversion de
                tous les IDs, un tableau permettra une conversion beaucoup
                plus rapide qu'un dictionnaire.
            convert_to_one_hot (bool, facultatif) : Si `True`, les données de
                vérité terrain seront converties au format one-hot.
            void_class_id (int, facultatif) : L'ID de classe d'une classe
                'void' ou 'background'. Uniquement pertinent si l'une des
                transformations `random_crop`, `translate`, ou `scale` est
                utilisée sur les données de vérité terrain. Détermine la
                valeur du pixel de l'espace vide de l'image qui pourrait se
                produire par les transformations susmentionnées.
            random_crop (tuple, facultatif) : `False` ou un tuple de deux
                entiers, `(height, width)`, où `height` et `width` sont la
                hauteur et la largeur du patch qui doit être découpé à une
                position aléatoire dans l'image d'entrée. Notez que `height`
                et `width` peuvent être arbitraires - ils sont autorisés à
                être plus grands que la hauteur et la largeur de l'image,
                auquel cas l'image originale sera placée de manière aléatoire
                sur un canevas de fond noir de taille `(height, width)`. La
                valeur par défaut est `False`.
            crop (tuple, facultatif) : `False` ou un tuple de quatre entiers,
                `(crop_top, crop_bottom, crop_left, crop_right)`, avec le
                nombre de pixels à rogner de chaque côté des images.
                Remarque : le recadrage a lieu après le recadrage aléatoire.
            resize (tuple, facultatif) : `False` ou un tuple de 2 entiers pour
                la taille de sortie désirée des images en pixels. Le format
                attendu est `(height, width)`. Remarque : Le redimensionnement
                a lieu après le recadrage aléatoire et le recadrage.
            brightness (tuple, facultatif) : `False` ou un tuple contenant
                trois flottants, `(min, max, prob)`. Met à l'échelle la
                luminosité de l'image par un facteur choisi aléatoirement dans
                une distribution uniforme dans les limites de `[min, max]`.
                Les valeurs min et max doivent être >=0.
            flip (float, facultatif) : `False` ou un float dans [0,1], voir
                `prob` ci-dessus. Retourne l'image horizontalement.
            translate (tuple, facultatif) : `False` ou un tuple, dont les deux
                premiers éléments sont des tuples contenant chacun deux
                entiers, et le troisième élément est un flottant :
                `((min, max), (min, max), prob)`. Le premier tuple fournit la
                plage en pixels pour le décalage horizontal de l'image, le
                second tuple pour le décalage vertical. Le nombre de pixels
                dont il faut décaler l'image est distribué uniformément dans
                les limites de `[min, max]`, c'est-à-dire que `min` est le
                nombre de pixels dont l'image est déplacée au minimum. Les
                valeurs `min` et `max` doivent être >=0.
            scale (tuple, facultatif) : `False` ou un tuple contenant trois
                flottants, `(min, max, prob)`. Met à l'échelle l'image par un
                facteur choisi aléatoirement dans une distribution uniforme
                dans les limites de `[min, max]`. Les valeurs de `min` et
                `max` doivent être >=0.
            gray (bool, facultatif) : Si `True`, convertit les images en
                niveaux de gris. Notez que les images en niveaux de gris
                résultantes ont la forme `(height, width, 1)`.
            to_disk (bool, facultatif) : Si `True`, les lots générés sont
                enregistrés dans `export_dir` (voir constuctor) en plus d'être
                cédés. Cela peut être utilisé pour le traitement hors ligne
                des jeux de données.
            shuffle (bool, optionnel) : Si `True`, le jeu de données sera
                mélangé avant chaque nouvelle passe.

        Retourne :
            Soit un tableau Numpy 4D de forme
            `(batch_size, img_height, img_with, num_channels)` avec les images
            générées, soit, si les chemins vers les données de vérité terrain
            ont été passés dans le constructeur, deux tableaux Numpy, le
            premier est le même que dans le premier cas et le second a la
            forme `(batch_size, img_height, img_with)` et contient les images
            de vérité terrain générées.
        """
        if (convert_to_one_hot or (not convert_colors_to_ids is False) or
            (not convert_ids_to_ids is False)) and not self.ground_truth:
            raise ValueError(
                "Impossible de convertir les données de vérité terrain : \
                Aucune donnée de vérité terrain n'est donnée.")

        if convert_to_one_hot and self.num_classes is None:
            raise ValueError(
                "La conversion à chaud exige que vous passiez une valeur \
                entière pour `num_classes` dans le constructeur, mais \
                `num_classes` est `None`.")

        if shuffle:
            random.shuffle(self.image_paths)

        current = 0

        while True:

            # Stocker le nouveau lot ici
            images = []
            gt_images = []

            # Mélange des données après chaque passage complet
            if current >= len(self.image_paths):
                if shuffle:
                    random.shuffle(self.image_paths)
                current = 0

            # Charger les images et les images de vérité terrain pour ce lot
            for image_path in self.image_paths[
                              current:current +
                                      batch_size]:  # Attention : Cela fonctionne avec Python,
                # mais peut provoquer une erreur 'index out of bounds' dans
                # d'autres langages si `current+batch_size > len(image_paths)`.

                # Charger l'image
                image = imageio.imread(image_path)
                img_height, img_width, img_ch = image.shape

                # Si au moins un répertoire de vérité de terrain a été donné,
                # charger les images de vérité de terrain.
                if self.ground_truth:

                    gt_image_path = self.ground_truth_paths[os.path.basename(
                        image_path)]
                    gt_image = imageio.imread(gt_image_path)
                    gt_dtype = gt_image.dtype

                    if not convert_colors_to_ids is False:
                        gt_image = convert_between_IDs_and_colors(
                            gt_image, convert_colors_to_ids, gt_dtype=gt_dtype)

                    if not convert_ids_to_ids is False:
                        if isinstance(convert_ids_to_ids, np.ndarray):
                            gt_image = convert_IDs_to_IDs(
                                gt_image, convert_ids_to_ids)
                        if isinstance(convert_ids_to_ids, dict):
                            gt_image = convert_IDs_to_IDs_partial(
                                gt_image, convert_ids_to_ids)

                # Traiter peut-être les images et les images de vérité du sol.

                if random_crop:
                    # Calculer l'espace dont nous disposons dans les deux
                    # dimensions pour effectuer un recadrage aléatoire.
                    # Un nombre négatif signifie que nous voulons recadrer une
                    # partie plus grande que l'image originale dans la
                    # dimension respective, dans ce cas, nous créerons une
                    # canvas de fond noire sur laquelle nous placerons l'image
                    # de manière aléatoire.
                    y_range = img_height - random_crop[0]
                    x_range = img_width - random_crop[1]

                    # Choisir une position de coupe aléatoire parmi les
                    # positions de coupe possibles.
                    if y_range >= 0:
                        crop_ymin = np.random.randint(
                            0, y_range + 1
                        )  # Il y a y_range + 1 positions possibles pour le
                        # crop dans la dimension verticale.
                    else:
                        crop_ymin = np.random.randint(
                            0, -y_range + 1
                        )  # Les positions possibles de l'image sur le canevas
                        # d'arrière-plan dans la dimension verticale.
                    if x_range >= 0:
                        crop_xmin = np.random.randint(
                            0, x_range + 1
                        )  # Il y a x_plage + 1 positions possibles pour la
                        # coupe dans la dimension horizontale.
                    else:
                        crop_xmin = np.random.randint(
                            0, -x_range + 1
                        )  # Les positions possibles de l'image sur le canevas
                        # d'arrière-plan dans la dimension horizontale.
                    # Effectuer la coupe
                    # Si le patch à recadrer est plus petit que l'image
                    # originale dans les deux dimensions, nous effectuons
                    # simplement un recadrage normal.
                    if y_range >= 0 and x_range >= 0:
                        # Recadrer l'image
                        image = np.copy(
                            image[crop_ymin:crop_ymin + random_crop[0],
                            crop_xmin:crop_xmin + random_crop[1]])
                        # Fais de même pour l'image de vérité terrain.
                        if self.ground_truth:
                            gt_image = np.copy(
                                gt_image[crop_ymin:crop_ymin + random_crop[0],
                                crop_xmin:crop_xmin + random_crop[1]])
                    # Si le recadrage est plus grand que l'image originale
                    # dans la dimension horizontale seulement,...
                    elif y_range >= 0 and x_range < 0:
                        # Recadrer l'image
                        patch_image = np.copy(
                            image[crop_ymin:crop_ymin + random_crop[0]]
                        )  # ...recadrer la dimension verticale
                        # comme avant,...
                        canvas = np.zeros(
                            shape=(random_crop[0], random_crop[1],
                                   patch_image.shape[2]),
                            dtype=np.uint8
                        )  # ...générer une image de fond vierge sur laquelle
                        # placer le patch,...
                        canvas[:, crop_xmin:crop_xmin +
                                            img_width] = patch_image  # ...et placer le
                        # patch sur le canevas à la position aléatoire
                        # `crop_xmin` calculée ci-dessus.
                        image = canvas
                        # Fais de même pour l'image de vérité terrain.
                        if self.ground_truth:
                            patch_gt_image = np.copy(
                                gt_image[crop_ymin:crop_ymin + random_crop[0]]
                            )  # ...recadrer la dimension verticale
                            # comme avant,...
                            canvas = np.full(
                                shape=random_crop,
                                fill_value=void_class_id,
                                dtype=gt_dtype
                            )  # ...générer une image de fond vierge sur
                            # laquelle placer le patch,...
                            canvas[:, crop_xmin:crop_xmin +
                                                img_width] = patch_gt_image  # ...et placer
                            # le patch sur le canevas à la position aléatoire
                            # `crop_xmin` calculée ci-dessus.
                            gt_image = canvas
                    # Si le recadrage est plus grand que l'image originale
                    # dans la dimension verticale uniquement,...
                    elif y_range < 0 <= x_range:
                        # Recadrer l'image
                        patch_image = np.copy(
                            image[:, crop_xmin:crop_xmin + random_crop[1]]
                        )  # ...recadrer la dimension horizontale comme dans
                        # le premier cas,...
                        canvas = np.zeros(
                            shape=(random_crop[0], random_crop[1],
                                   patch_image.shape[2]),
                            dtype=np.uint8
                        )  # ...générer une image de fond vierge sur laquelle
                        # placer le patch,...
                        canvas[
                        crop_ymin:crop_ymin +
                                  img_height, :] = patch_image  # ...et placer le
                        # patch sur le canevas à la position aléatoire
                        # `crop_ymin` calculée ci-dessus.
                        image = canvas
                        # Fais de même pour l'image de référence.
                        if self.ground_truth:
                            patch_gt_image = np.copy(
                                gt_image[:,
                                crop_xmin:crop_xmin + random_crop[1]]
                            )  # ...recadrer la dimension horizontale comme
                            # dans le premier cas,...
                            canvas = np.full(
                                shape=random_crop,
                                fill_value=void_class_id,
                                dtype=gt_dtype
                            )  # ...générer une image de fond vierge sur
                            # laquelle placer le patch,...
                            canvas[
                            crop_ymin:crop_ymin +
                                      img_height, :] = patch_gt_image  # ...et
                            # placer le patch sur le canevas à la position
                            # aléatoire `crop_ymin` calculée ci-dessus.
                            gt_image = canvas
                    else:  # Si le recadrage est plus grand que l'image
                        # originale dans les deux dimensions,...
                        patch_image = np.copy(image)
                        canvas = np.zeros(
                            shape=(random_crop[0], random_crop[1],
                                   patch_image.shape[2]),
                            dtype=np.uint8
                        )  # ...générer une image de fond vierge sur laquelle
                        # placer le patch,...
                        canvas[
                        crop_ymin:crop_ymin + img_height,
                        crop_xmin:crop_xmin +
                                  img_width] = patch_image  # ...et placer le patch
                        # sur le canevas à la position aléatoire
                        # `(crop_ymin, crop_xmin)` calculée ci-dessus.
                        image = canvas
                        # Fais de même pour l'image de référence.
                        if self.ground_truth:
                            patch_gt_image = np.copy(gt_image)
                            canvas = np.full(
                                shape=random_crop,
                                fill_value=void_class_id,
                                dtype=gt_dtype
                            )  # ...générer une image de fond vierge sur
                            # laquelle placer le patch,...
                            canvas[
                            crop_ymin:crop_ymin + img_height,
                            crop_xmin:crop_xmin +
                                      img_width] = patch_gt_image  # ...et placer le
                            # patch sur le canevas à la position aléatoire
                            # `(crop_ymin, crop_xmin)` calculée ci-dessus.
                            gt_image = canvas
                    # Mettre à jour les valeurs de la hauteur et de la largeur.
                    img_height, img_width = random_crop

                if crop:
                    image = np.copy(image[crop[0]:img_height - crop[1],
                                    crop[2]:img_width - crop[3]])
                    gt_image = np.copy(gt_image[crop[0]:img_height - crop[1],
                                       crop[2]:img_width - crop[3]])

                if resize:
                    image = cv2.resize(image,
                                       dsize=(resize[1], resize[0]),
                                       interpolation=cv2.INTER_LINEAR)
                    if self.ground_truth:
                        gt_image = cv2.resize(gt_image,
                                              dsize=(resize[1], resize[0]),
                                              interpolation=cv2.INTER_NEAREST)
                    # Il n'est pas nécessaire de les mettre à jour à ce stade,
                    # mais c'est une source d'erreur en moins si cette méthode
                    # est étendue à l'avenir.
                    img_height, img_width = resize

                if brightness:
                    p = np.random.uniform(0, 1)
                    if p >= (1 - brightness[2]):
                        image = _brightness(image,
                                            min=brightness[0],
                                            max=brightness[1])

                if flip:
                    p = np.random.uniform(0, 1)
                    if p >= (1 - flip):
                        image = cv2.flip(image, 1)  # Retournement horizontal
                        if self.ground_truth:
                            gt_image = cv2.flip(gt_image,
                                                1)  # Retournement horizontal

                if translate:
                    p = np.random.uniform(0, 1)
                    if p >= (1 - translate[2]):
                        # Sélection aléatoire des valeurs de décalage
                        # horizontal et vertical.
                        x = np.random.randint(translate[0][0],
                                              translate[0][1] + 1)
                        y = np.random.randint(translate[1][0],
                                              translate[1][1] + 1)
                        x_shift = random.choice([-x, x])
                        y_shift = random.choice([-y, y])
                        # Calculer la matrice de déformation pour les valeurs
                        # sélectionnées.
                        translation_matrix = np.float32([[1, 0, x_shift],
                                                         [0, 1, y_shift]])
                        # Déformer l'image et peut-être l'image de base.
                        image = cv2.warpAffine(src=image,
                                               M=translation_matrix,
                                               dsize=(img_width, img_height))
                        if self.ground_truth:
                            gt_image = cv2.warpAffine(
                                src=gt_image,
                                M=translation_matrix,
                                dsize=(img_width, img_height),
                                borderValue=void_class_id)

                if scale:
                    p = np.random.uniform(0, 1)
                    if p >= (1 - scale[2]):
                        scaling_factor = np.random.uniform(scale[0], scale[1])
                        scaled_height = int(img_height * scaling_factor)
                        scaled_width = int(img_width * scaling_factor)
                        y_offset = abs(int((img_height - scaled_height) / 2))
                        x_offset = abs(int((img_width - scaled_width) / 2))

                        # Mise à l'échelle de l'image.
                        patch_image = cv2.resize(
                            image,
                            dsize=(scaled_width, scaled_height),
                            interpolation=cv2.INTER_LINEAR)
                        if scaling_factor <= 1:
                            canvas = np.zeros(shape=(img_height, img_width,
                                                     img_ch),
                                              dtype=np.uint8)
                            canvas[y_offset:y_offset + scaled_height,
                            x_offset:x_offset +
                                     scaled_width] = patch_image
                            image = canvas
                        if scaling_factor > 1:
                            image = np.copy(
                                patch_image[y_offset:img_height + y_offset,
                                x_offset:img_width + x_offset])

                        # Mise à l'échelle de l'image de vérité terrain.
                        if self.ground_truth:
                            patch_gt_image = cv2.resize(
                                gt_image,
                                dsize=(scaled_width, scaled_height),
                                interpolation=cv2.INTER_NEAREST)
                            if scaling_factor <= 1:
                                canvas = np.full(shape=(img_height, img_width),
                                                 fill_value=void_class_id,
                                                 dtype=gt_dtype)
                                canvas[y_offset:y_offset + scaled_height,
                                x_offset:x_offset +
                                         scaled_width] = patch_gt_image
                                gt_image = canvas
                            if scaling_factor > 1:
                                gt_image = np.copy(
                                    patch_gt_image[y_offset:img_height +
                                                            y_offset,
                                    x_offset:img_width +
                                             x_offset])

                if gray:
                    image = np.expand_dims(cv2.cvtColor(
                        image, cv2.COLOR_RGB2GRAY),
                        axis=2)

                # Convertir les identifications de la vérité terrain en one-hot.
                if convert_to_one_hot:
                    gt_image = convert_IDs_to_one_hot(gt_image,
                                                      self.num_classes)

                if to_disk:  # Si les données traitées doivent être écrites
                    # sur le disque au lieu d'être lues.
                    # Créer le répertoire (y compris les parents) s'il
                    # n'existe pas déjà.
                    image_save_file_path = os.path.join(
                        self.export_dir,
                        os.path.relpath(image_path, start=self.root_dir))
                    image_save_directory_path = os.path.dirname(
                        image_save_file_path)
                    pathlib.Path(image_save_directory_path).mkdir(
                        parents=True, exist_ok=True)
                    # Enregistrer l'image.
                    imageio.imsave(image_save_file_path, image)
                    if self.ground_truth:
                        # Créer le répertoire (y compris les parents)
                        # s'il n'existe pas déjà.
                        gt_image_save_file_path = os.path.join(
                            self.export_dir,
                            os.path.relpath(gt_image_path,
                                            start=self.root_dir))
                        gt_image_save_directory_path = os.path.dirname(
                            gt_image_save_file_path)
                        pathlib.Path(gt_image_save_directory_path).mkdir(
                            parents=True, exist_ok=True)
                        # Sauvegarder l'image de vérité terrain.
                        imageio.imsave(gt_image_save_file_path, gt_image)

                # Ajouter l'image traitée
                # (et peut-être l'image de vérité terrain) à ce lot.
                images.append(image)
                if self.ground_truth:
                    gt_images.append(gt_image)

            current += batch_size

            if self.ground_truth:
                yield np.array(images), np.array(gt_images)
            else:
                yield np.array(images)

    def process_all(self,
                    convert_colors_to_ids=False,
                    convert_ids_to_ids=False,
                    convert_to_one_hot=False,
                    void_class_id=None,
                    random_crop=False,
                    crop=False,
                    resize=False,
                    brightness=False,
                    flip=False,
                    translate=False,
                    scale=False,
                    gray=False,
                    to_disk=True,
                    shuffle=False,
                    batch_size=1):
        """
        Traite le jeu de données entier par lots de `batch_size` et enregistre
        les résultats dans `export_dir` (voir constructeur).

        Il s'agit simplement d'une enveloppe autour de la méthode `generate()`
        qui itère sur l'ensemble du jeu de données (ou des jeux de données,
        dans le cas où plusieurs ont été passés dans le constructeur).

        Pour la documentation des arguments, voir `generate()`. Retourne void.
        """

        preprocessor = self.generate(
            batch_size=batch_size,
            convert_colors_to_ids=convert_colors_to_ids,
            convert_ids_to_ids=convert_ids_to_ids,
            convert_to_one_hot=convert_to_one_hot,
            void_class_id=void_class_id,
            random_crop=random_crop,
            crop=crop,
            resize=resize,
            brightness=brightness,
            flip=flip,
            translate=translate,
            scale=scale,
            gray=gray,
            to_disk=to_disk,
            shuffle=shuffle)

        num_batches = ceil(self.dataset_size / batch_size)

        tr = trange(num_batches, file=sys.stdout)
        tr.set_description('Traitement des images')

        for batch in tr:
            next(preprocessor)


def _brightness(image, min=0.5, max=2.0):
    """
    Change aléatoirement la luminosité de l'image d'entrée.

    Protégé contre le débordement.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    random_br = np.random.uniform(min, max)

    # Pour se protéger contre le débordement : Calculer un masque pour tous
    # les pixels où le réglage de la luminosité dépasserait la valeur maximale
    # de la luminosité et fixer la valeur au maximum sur ces pixels.
    mask = hsv[:, :, 2] * random_br > 255
    v_channel = np.where(mask, 255, hsv[:, :, 2] * random_br)
    hsv[:, :, 2] = v_channel

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


class DataError(Exception):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
