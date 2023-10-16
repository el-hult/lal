import os
import pathlib
import gzip
from typing import Union
import urllib.request
import shutil
from pandas.core.frame import DataFrame
import tarfile
import numpy as np
import pandas as pd


def cache_download(path, url):
    """Downloads url to path, if nonexistent"""
    # https://stackoverflow.com/questions/7243750/download-file-from-web-in-python-3
    dir = pathlib.Path(path).parent
    dir.mkdir(parents=True, exist_ok=True)
    if not os.path.isfile(path):
        with urllib.request.urlopen(url) as response, open(path, "wb") as out_file:
            shutil.copyfileobj(response, out_file)


def load_uci(dataset: str, cache_path="data") -> Union[DataFrame, dict[str, DataFrame]]:
    """Load some UCI dataset

    Args:
        cache_path: a path to a folder to cache the files in
    """
    from enum import Enum

    Dataset = Enum("Dataset", ["Wine", "Superconductivity", "Airfoil"])
    try:
        dataset = Dataset[dataset]
    except KeyError:
        raise ValueError(
            f'The dataset "{dataset}" is not recognized. Must be one of {[a.name for a in Dataset]}'
        )

    dir = pathlib.Path(cache_path)
    os.makedirs(dir, exist_ok=True)
    if dataset == Dataset.Wine:
        fname = dir / "wine.dat"
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
        cache_download(
            fname,
            url,
        )
        names = [
            "Cultivar",
            "Alcohol",
            "Malic acid",
            "Ash",
            "Alcalinity of ash",
            "Magnesium",
            "Total phenols",
            "Flavanoids",
            "Nonflavanoid phenols",
            "Proanthocyanins",
            "Color intensity",
            "Hue",
            "OD280/OD315 of diluted wines",
            "Proline",
        ]
        df = pd.read_csv(fname, names=names)
    elif dataset == Dataset.Airfoil:
        fname = dir / "airfoil.dat"
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat"
        cache_download(
            fname,
            url,
        )
        names = [
            "Frequency [Hz]",
            "Angle of attack [deg]",
            "Chord length [m]",
            "Free-stream velocity [m/s]",
            "Suction side displacement thickness [m]",
            "Scaled sound pressure level [dB]",
        ]
        df = pd.read_csv(fname, names=names, sep="\t")
    elif dataset == Dataset.Superconductivity:
        path_to_train = dir / "train.csv"
        path_to_unique = dir / "unique_m.csv"
        if not path_to_train.is_file() or not path_to_unique.is_file():
            path_to_zip_file = dir / "superconduct.zip"
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00464/superconduct.zip"
            cache_download(
                path_to_zip_file,
                url,
            )
            import zipfile

            with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
                zip_ref.extractall(dir)
        df1 = pd.read_csv(path_to_train)
        df1 = df1.astype("double")
        df2 = pd.read_csv(path_to_unique)
        df2.iloc[:, :87] = df2.iloc[:, :87].astype("double")
        df2.iloc[:, 87] = df2.iloc[:, 87].astype("string")
        df = {
            "train": df1,
            "unique_m": df2,
        }
    else:
        raise ValueError(f"The dataset {dataset} is not recognized")
    return df


def mnist(cache_path="data"):
    """Load MNIST into numpy arrays.

    Cache the results, so it goes FAST to load.
    """
    dir = cache_path
    os.makedirs(dir, exist_ok=True)
    fnames = [
        "train-images-idx3-ubyte",
        "t10k-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
        "t10k-labels-idx1-ubyte",
    ]
    paths = [os.path.join(cache_path, fname + ".npy") for fname in fnames]
    gzpaths = [os.path.join(cache_path, fname + ".gz") for fname in fnames]

    #
    # Download and unzip if needed
    #
    for fname, path, gzpath in zip(fnames, paths, gzpaths):
        if not os.path.isfile(path):
            url = f"http://yann.lecun.com/exdb/mnist/{fname}.gz"
            cache_download(
                gzpath,
                url,
            )
            with gzip.open(gzpath, "rb") as f:
                if "idx3" in fname:
                    # for the X-arrays,
                    # First 16 bytes are magic_number, n_imgs, n_rows, n_cols
                    pixels = np.frombuffer(f.read(), "B", offset=16)
                    X = pixels.reshape(-1, 784).astype("float32") / 255
                    np.save(path, X)
                elif "idx1" in fname:
                    # for the y-arrays,
                    # First 8 bytes are magic_number, n_labels
                    integer_labels = np.frombuffer(f.read(), "B", offset=8)
                    y = integer_labels
                    np.save(path, y)
                else:
                    raise RuntimeError("F'ed up")

    X_train, X_test, y_train, y_test = [np.load(path) for path in paths]
    assert X_train.shape == (60000, 784), X_train.shape
    assert X_test.shape == (10000, 784), X_test.shape
    assert y_train.shape == (60000,), y_train.shape
    assert y_test.shape == (10000,), y_test.shape
    assert np.all(y_train < 10)
    assert np.all(y_train >= 0)
    assert np.all(y_test < 10)
    assert np.all(y_test >= 0)
    return X_train, X_test, y_train, y_test


def palmer_penguins():
    """Python port of the code in https://github.com/allisonhorst/palmerpenguins/blob/master/data-raw/penguins.R"""
    pkl_path = os.path.join("data", "penguins.pkl")
    if not os.path.isfile(pkl_path):
        sources = [
            {
                "part": "adelie",
                "uri": "https://portal.edirepository.org/nis/dataviewer?packageid=knb-lter-pal.219.3&entityid=002f3893385f710df69eeebe893144ff",
                "doi": "10.6073/pasta/abc50eed9138b75f54eaada0841b9b86",
            },
            {
                "part": "gentoo",
                "uri": "https://portal.edirepository.org/nis/dataviewer?packageid=knb-lter-pal.220.3&entityid=e03b43c924f226486f2f0ab6709d2381",
                "doi": "10.6073/pasta/2b1cff60f81640f182433d23e68541ce",
            },
            {
                "part": "chinstrap",
                "uri": "https://portal.edirepository.org/nis/dataviewer?packageid=knb-lter-pal.221.2&entityid=fe853aa8f7a59aa84cdd3197619ef462",
                "doi": "10.6073/pasta/409c808f8fc9899d02401bdb04580af7",
            },
        ]

        dfs = []
        for source in sources:
            fname = os.path.join("data", source["part"] + "_raw.csv")
            cache_download(fname, source["uri"])
            dfs.append(pd.read_csv(fname))

        df = pd.concat(dfs)
        df["Species"] = df["Species"].str.extract("^([\w]+)").astype("category")
        df["Island"] = df["Island"].astype("category")
        df["Sex"] = df["Sex"].str.lower().replace(".", np.nan).astype("category")
        df["Year"] = df["Date Egg"].str.slice(0, 4).astype(int)
        df["Flipper Length (mm)"] = df["Flipper Length (mm)"].astype(
            pd.Int16Dtype()
        )  # nullable int type
        df["Body Mass (g)"] = df["Body Mass (g)"].astype(pd.Int16Dtype())
        df["Bill Length (mm)"] = (
            df["Culmen Length (mm)"].round().astype(pd.Int16Dtype())
        )
        df["Bill Depth (mm)"] = df["Culmen Depth (mm)"].round().astype(pd.Int16Dtype())

        df = df[
            [
                "Species",
                "Island",
                "Bill Length (mm)",
                "Bill Depth (mm)",
                "Flipper Length (mm)",
                "Body Mass (g)",
                "Sex",
                "Year",
            ]
        ]
        df = df.reset_index(drop=True)
        df.to_pickle(pkl_path)

        return df
    else:
        return pd.read_pickle(pkl_path)


def quake_data():
    """The number of global earthquakes per month during the 120 months 2012-2022, with Richter 5+"""
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query.csv?starttime=2012-01-01%2000:00:00&endtime=2022-01-01%2000:00:00&minmagnitude=5&orderby=time"
    fname = "data/quakes.csv"
    cache_download(fname, url)
    df = pd.read_csv(fname, parse_dates=["time"])
    return df["time"].groupby([df["time"].dt.year, df["time"].dt.month]).agg("count")


def load_cal_housing(cache_path="data"):
    """ Pace, R. Kelley and Ronald Barry, Sparse Spatial Autoregressions, Statistics and Probability Letters, 33 (1997) 291-297.
    
    https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html

    S&P Letters Data
    We collected information on the variables using all the block groups in California from the 1990 Census.
    In this sample a block group on average includes 1425.5 individuals living in a geographically co mpact area.
    Naturally, the geographical area included varies inversely with the population density.
    We computed distances among the centroids of each block group as measured in latitude and longitude.
    We excluded all the block groups reporting zero entries for the independent and dependent variables.
    The final data contained 20,640 observations on 9 variables. The dependent variable is ln(median house value).
    """
    cache_path="data"
    dir = pathlib.Path(cache_path)
    os.makedirs(dir, exist_ok=True)
    gzpath = dir / "cal_housing.tgz"
    url = r"https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.tgz"
    cache_download(gzpath, url)
    data_path = dir / "cal_housing.data"
    if not data_path.is_file():
        with tarfile.open(gzpath, "r:gz") as tar,open(data_path,'wb') as data_file :
            file_inside_tar = r'CaliforniaHousing/cal_housing.data'
            shutil.copyfileobj(tar.extractfile(file_inside_tar), data_file)
    names = [
        "longitude",
        "latitude",
        "housingMedianAge", 
        "totalRooms", 
        "totalBedrooms", 
        "population", 
        "households", 
        "medianIncome", 
        "medianHouseValue", 
    ]
    df = pd.read_csv(data_path, names=names)
    assert len(df) == 20640
    return df