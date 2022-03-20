import numpy as np


def convert_IDs_to_IDs(input_array, id_map_array):
    """
    Convertit un tableau d'entiers en un tableau de même forme de n'importe
    quel type de données numériques avec des éléments du tableau d'entrée
    remplacés selon une carte.

    Nécessite que `id_map_array` contienne un mapping pour toutes les valeurs
    possibles de `input_array`.

    Arguments :
        input_array (tableau) : Un tableau Numpy nD de type entier non signé.
        id_map_array (tableau) : Un tableau Numpy 1D de longueur `k` qui sert
            de carte entre les valeurs du tableau d'entrée et les valeurs du
            tableau retourné. `k` doit être la plus grande valeur possible
            pour `input_array`. Les indices de `id_map_array` représentent les
            valeurs de `input_array` et les valeurs de `id_map_array`
            représentent les valeurs souhaitées du tableau retourné.

    Retourne :
        Un tableau Numpy de la même forme que `input_array` avec les valeurs
        selon `id_map_array`.
    """
    return id_map_array[input_array]


def convert_IDs_to_IDs_partial(image, id_map_dict):
    """
    Convertit un tableau d'entiers en un tableau de même forme de n'importe
    quel type de données numériques avec des éléments du tableau d'entrée
    remplacés selon une carte.

    C'est beaucoup plus lent que `conver_IDs_to_IDs()`, mais il n'est pas
    nécessaire que `id_map_dict` contienne une correspondance pour toutes les
    valeurs possibles de `input_array`, mais seulement pour les valeurs qui
    doivent être remplacées.

    Arguments :
        input_array (tableau) : Un tableau Numpy nD de type entier non signé.
        id_map_dict (tableau) : Un dictionnaire Python qui sert de
            correspondance entre les valeurs du tableau d'entrée et les
            valeurs du tableau retourné. Les clés de `id_map_dict`
            représentent les valeurs de `input_array` et les valeurs de
            `id_map_dict` représentent les valeurs souhaitées du tableau
            retourné.

    Retourne :
        Un tableau Numpy de la même forme que `input_array` avec les valeurs
        selon `id_map_dict`.
    """
    canvas = np.copy(image)

    for key, value in id_map.items():
        canvas[image == key] = value

    return canvas


def convert_between_IDs_and_colors(image, color_map_dict, gt_dtype=np.uint8):
    if len(np.squeeze(image).shape) == 3:
        canvas = np.zeros(shape=(image.shape[0], image.shape[1]),
                          dtype=gt_dtype)
        for key, value in color_map_dict.items():
            canvas[np.all(image == key, axis=2)] = value
    else:
        canvas = np.zeros(shape=(image.shape[0], image.shape[1], 3),
                          dtype=np.uint8)
        for key, value in color_map_dict.items():
            canvas[image == key] = value

    return canvas


def convert_IDs_to_colors(image, color_map_array):
    """
    Convertit un tableau d'entiers non négatifs de forme `(k, ..., m)`
    (comme dans une image monocanal 2D avec dtype uint8) en un tableau de
    forme `(k, ..., m, color_array.shape[1])`, avec le dernier axe contenant
    les valeurs de `color_map_array`.

    Pour convertir les ID des classes de segmentation en couleurs à 3 canaux,
    cette fonction est beaucoup plus rapide que
    `convert_between_IDs_and_colors()`.
    """

    return color_map_array[image]


def convert_one_hot_to_IDs(one_hot):
    return np.squeeze(np.argmax(one_hot, axis=-1))


def convert_IDs_to_one_hot(image, num_classes):
    unity_vectors = np.eye(num_classes, dtype=np.bool)

    return unity_vectors[image]
