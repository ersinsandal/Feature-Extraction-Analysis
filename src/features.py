import numpy as np
from skimage.feature import hog
from skimage.feature import local_binary_pattern


def extract_raw_pixels(images):
    """
    28x28 görüntüyü 784 uzunluklu düz sayı listesine çevirir.
    """
    n_samples = images.shape[0]
    return images.reshape(n_samples, -1)

def extract_hog_features(images):
    """
    HOG yöntemiyle görüntülerin çizgi/kenar yönü özelliklerini çıkarır.
    """
    hog_features = []

    for image in images:
        features = hog(
            image,
            orientations=9,
            pixels_per_cell=(4, 4),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True,
        )
        hog_features.append(features)

    return np.array(hog_features)

def extract_lbp_features(images):
    """
    Multi-Scale Spatial LBP

    Amaç:
    - Texture bilgisini daha güçlü yakalamak
    - Sadece tüm görüntünün genel histogramını değil,
      görüntünün farklı bölgelerindeki texture bilgisini de temsil etmek

    Kullanılan mantık:
    - Farklı scale'lerde LBP çıkarılır
    - Görüntü 4x4 parçalara bölünür
    - Her parçadan LBP histogramı alınır
    - Tüm histogramlar birleştirilir
    """

    lbp_features = []

    # Popüler ve dengeli multi-scale LBP ayarları
    lbp_settings = [
        {"P": 8, "R": 1},
        {"P": 16, "R": 2},
        {"P": 24, "R": 3},
    ]

    grid_size = 4

    for image in images:
        image_uint8 = (image * 255).astype("uint8")
        final_feature_vector = []

        for setting in lbp_settings:
            P = setting["P"]
            R = setting["R"]
            n_bins = P + 2

            lbp = local_binary_pattern(
                image_uint8,
                P=P,
                R=R,
                method="uniform",
            )

            h, w = lbp.shape
            cell_h = h // grid_size
            cell_w = w // grid_size

            for i in range(grid_size):
                for j in range(grid_size):
                    cell = lbp[
                        i * cell_h:(i + 1) * cell_h,
                        j * cell_w:(j + 1) * cell_w,
                    ]

                    hist, _ = np.histogram(
                        cell.ravel(),
                        bins=np.arange(0, n_bins + 1),
                        range=(0, n_bins),
                        density=True,
                    )

                    final_feature_vector.extend(hist)

        lbp_features.append(final_feature_vector)

    return np.array(lbp_features)