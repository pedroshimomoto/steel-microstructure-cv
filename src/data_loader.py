"""Data loading utilities for the steel microstructure (Bainite / Martensite) dataset.

Dataset: Zenodo DOI 10.5281/zenodo.19851246 (CC-BY 4.0).

Expected layout (not versioned, see README):
    data/raw/Dataset_Patches_Microscope_Metadata.xlsx
    data/raw/Dataset/<Sample>_<Phase>_<n>/<FileNameHarm>.png

Grouping note
-------------
In the metadata, ``ID`` identifies a single *acquisition* (one image taken with one
microscope at one magnification, 474 in total). The physical specimen is
``SampleIDHarm`` (20 specimens, A-T), and the same specimen is imaged by every
microscope. Splitting by ``ID`` would therefore put the same physical region in both
train and test. All splits here are grouped by ``SampleIDHarm``.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
IMAGE_DIR = RAW_DIR / "Dataset"
METADATA_PATH = RAW_DIR / "Dataset_Patches_Microscope_Metadata.xlsx"

EXPECTED_COLUMNS = [
    "FileNameHarm", "ID", "SampleIDHarm", "Magnification", "ImageSizePx",
    "MicroscopeID", "EtchingID", "ExposureHarm", "ApertureHarm", "ObjectivHarm",
    "FilterHarm", "Phase", "PatchID",
]
CATEGORICAL_STR_COLUMNS = [
    "SampleIDHarm", "Magnification", "ExposureHarm", "ApertureHarm",
    "ObjectivHarm", "FilterHarm", "Phase",
]
CLASSES = ["Bainite", "Martensite"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
GROUP_COL = "SampleIDHarm"

# Files whose name on disk differs from FileNameHarm in the spreadsheet.
# disk stem -> metadata FileNameHarm
KNOWN_FILENAME_FIXES = {
    # Stored in D_Bainite_1/ and listed as sample D in the metadata.
    "228_C_Bainite_01": "228_D_Bainite_01",
}


def load_metadata(path: Path | str = METADATA_PATH) -> pd.DataFrame:
    """Read the metadata spreadsheet (xlsx or csv) and normalize it lightly."""
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        sheets = pd.ExcelFile(path).sheet_names
        if len(sheets) > 1:
            warnings.warn(f"Metadata has {len(sheets)} sheets {sheets}; reading only '{sheets[0]}'.")
        df = pd.read_excel(path, sheet_name=sheets[0]) #transforma o excel em um df
    else:
        df = pd.read_csv(path)

    df.columns = df.columns.str.strip()
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns] #se tiver coluna faltando das que estão predefinidas
    if missing:
        warnings.warn(f"Metadata is missing expected columns: {missing}")

    for col in CATEGORICAL_STR_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    if "Phase" in df.columns: #normalizar os valores das colunas
        df["Phase"] = df["Phase"].str.capitalize()
    if "FileNameHarm" in df.columns:
        df["FileNameHarm"] = df["FileNameHarm"].astype(str).str.strip()
    return df


def index_images(image_dir: Path | str = IMAGE_DIR) -> dict[str, Path]:
    """Map metadata file name (stem) -> image path, searching ``image_dir`` recursively."""
    index: dict[str, Path] = {}
    for p in Path(image_dir).rglob("*.png"): #extrai o nome da imagem e o caminho até ela de todas da pasta
        stem = KNOWN_FILENAME_FIXES.get(p.stem, p.stem) # tenta aplicar a mudança no arquivo com o valor errado, se não for ele só retorna o nome do arquivo normal
        if stem in index:
            warnings.warn(f"Duplicate image name '{stem}': {index[stem]} and {p}")
        index[stem] = p
    return index


def build_dataframe(
    metadata_path: Path | str = METADATA_PATH,
    image_dir: Path | str = IMAGE_DIR,
    drop_missing: bool = True,
) -> pd.DataFrame:
    """Join the metadata with image paths on disk and add an integer ``label`` column."""
    df = load_metadata(metadata_path)
    index = index_images(image_dir)

    df["path"] = df["FileNameHarm"].map(index) #aqui ele 'junta' o df com o index a partir da coluna 'FileNameHarm', que como é igual a coluna stem, vai juntar as colunas da imagem com o path dela
    n_missing = df["path"].isna().sum()
    if n_missing:
        examples = df.loc[df["path"].isna(), "FileNameHarm"].head(5).tolist()
        warnings.warn(f"{n_missing} metadata rows have no image on disk (e.g. {examples}).")
    unused = set(index) - set(df["FileNameHarm"])
    if unused:
        warnings.warn(f"{len(unused)} images on disk are not in the metadata (e.g. {sorted(unused)[:5]}).")

    if drop_missing:
        df = df.dropna(subset=["path"]).reset_index(drop=True)
    df["label"] = df["Phase"].map(CLASS_TO_IDX)
    if df["label"].isna().any():
        warnings.warn(f"Unknown phases: {sorted(df.loc[df['label'].isna(), 'Phase'].unique())}")
    return add_region_id(df)


ROI_KEY = ["SampleIDHarm", "MicroscopeID", "Magnification", "EtchingID"]


def add_region_id(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``roi_id`` and ``region_id``: the same physical area across acquisition settings.

    The same ROI is photographed once per setting (aperture, filter, exposure, objective),
    and each photo gets a new, *consecutive* ``ID``. So a run of consecutive IDs within
    ``ROI_KEY`` is one ROI, and ``(ROI, Phase, PatchID)`` is one physical region.
    Checked on the images: same PatchID across settings has pixel correlation ~0.8,
    different patches or different ROIs ~0.
    """
    ids = df[ROI_KEY + ["ID"]].drop_duplicates("ID").sort_values("ID")
    ids["roi_block"] = ids.groupby(ROI_KEY)["ID"].transform(lambda s: (s.diff() != 1).cumsum())
    df = df.merge(ids[["ID", "roi_block"]], on="ID", how="left")

    df["roi_id"] = (
        df["SampleIDHarm"] + "_m" + df["MicroscopeID"].astype(str) + "_" + df["Magnification"]
        + "_e" + df["EtchingID"].astype(str) + "_b" + df["roi_block"].astype(str)
    )
    df["region_id"] = df["roi_id"] + "_" + df["Phase"] + "_" + df["PatchID"].astype(str).str.zfill(2)
    df = df.drop(columns="roi_block")

    # One patch per region per photo; otherwise the key does not identify a unique area.
    assert not df.duplicated(["region_id", "ID"]).any(), "region_id is not unique within an acquisition"
    return df


def _as_list(value) -> list | None:
    if value is None:
        return None
    if isinstance(value, (str, int, np.integer)):
        return [value]
    return list(value)


def filter_dataset(
    df: pd.DataFrame,
    microscope_id: int | Iterable[int] | None = None,
    magnification: str | Iterable[str] | None = None,
) -> pd.DataFrame:
    """Keep only the given microscope(s) and/or magnification(s) (e.g. ``"20x"``)."""
    mask = pd.Series(True, index=df.index) #tudo como true
    if (ids := _as_list(microscope_id)) is not None: #identifica se tem microscope_id, se tiver atualiza a coluna de mascara usando booleanos True e False
        mask &= df["MicroscopeID"].isin(ids)
    if (mags := _as_list(magnification)) is not None: #mesma coisa que microscope_id, mas aqui é com magnification
        mask &= df["Magnification"].isin(mags)
    return df[mask].reset_index(drop=True)


def _stratified_group_holdout(
    df: pd.DataFrame, size: float, group_col: str, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out roughly ``size`` of the groups, keeping the Phase ratio as close as possible."""
    n_splits = max(2, round(1 / size)) # calcula a quantidade de divisões será feita de acordo com a fração aproximada 
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed) # tenta dividir em proporções iguais, mas mantendo o mesmo corpo de prova
    rest_idx, held_idx = next(sgkf.split(df, df["label"], groups=df[group_col]))
    return df.iloc[rest_idx].reset_index(drop=True), df.iloc[held_idx].reset_index(drop=True)


def make_splits(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    group_col: str = GROUP_COL,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train / val / test with no ``group_col`` value shared between splits.

    Sizes are fractions of *groups*, not of patches. With only 20 specimens, the
    realized fractions and class balance are approximate; always print them.
    """
    train_val, test = _stratified_group_holdout(df, test_size, group_col, seed) # train_val, valores que vão ser separados para validação e treino depois. Test já é a parte separada para teste
    train, val = _stratified_group_holdout(train_val, val_size / (1 - test_size), group_col, seed) # como aqui o tamanho é 85% do dataset original, validação tem que ser 0,176 para corresponder 15% do dataset original
    # train é o que sobrou (85% do dataset) 
    groups = {name: set(part[group_col]) for name, part in # aqui cria um dicionário com (train, test, val): (groups, amostras A, B, C ...)
              [("train", train), ("val", val), ("test", test)]}
    assert not groups["train"] & groups["val"], "Group leakage between train and val" #valida se teve vazamento de dados
    assert not groups["train"] & groups["test"], "Group leakage between train and test"
    assert not groups["val"] & groups["test"], "Group leakage between val and test"
    if "region_id" in df.columns:
        regions = [set(part["region_id"]) for part in (train, val, test)]
        assert not (regions[0] & regions[1] or regions[0] & regions[2] or regions[1] & regions[2]), \
            "Same physical region in more than one split"
    return train, val, test


def make_cross_microscope_splits(
    df: pd.DataFrame,
    train_microscope: int,
    test_microscope: int,
    magnification: str | None = None,
    val_size: float = 0.15,
    test_size: float = 0.15,
    group_col: str = GROUP_COL,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Splits for the domain-shift experiment.

    Specimens are split once; then:
      - train / val / test: specimens of each split, imaged by ``train_microscope``
      - test_cross: the *same test specimens*, imaged by ``test_microscope``
    so ``test`` vs ``test_cross`` isolates the effect of changing the microscope.
    """
    base = filter_dataset(df, magnification=magnification)
    source = filter_dataset(base, microscope_id=train_microscope)
    train, val, test = make_splits(source, val_size, test_size, group_col, seed)

    target = filter_dataset(base, microscope_id=test_microscope)
    test_cross = target[target[group_col].isin(set(test[group_col]))].reset_index(drop=True) #separa as amostras (A, B, C ...) do microscópio de treino e usa as mesmas amostras, mas com imagens diferentes para o microscópio de treino
    assert not set(test_cross[group_col]) & (set(train[group_col]) | set(val[group_col]))
    return {"train": train, "val": val, "test": test, "test_cross": test_cross}


def load_image(path: Path | str, grayscale: bool = True, image_size: int | None = None) -> np.ndarray:
    """Load an image as float32 in [0, 1], shape (H, W, C)."""
    img = Image.open(path).convert("L" if grayscale else "RGB")
    if image_size is not None and img.size != (image_size, image_size):
        img = img.resize((image_size, image_size), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr[..., None] if arr.ndim == 2 else arr


def compute_normalization_stats(
    train_df: pd.DataFrame, grayscale: bool = True, image_size: int | None = 224
) -> tuple[list[float], list[float]]:
    """Per-channel mean/std. Call it on the TRAIN split only to avoid leakage."""
    total = sq_total = None
    n_pixels = 0
    for path in train_df["path"]:
        arr = load_image(path, grayscale, image_size).astype(np.float64)
        flat = arr.reshape(-1, arr.shape[-1]) #aplica essa transformação para cada pixel da imagem ter seu valor de RGB definido (ex 224 x 224 -> 50.176 linhas)
        total = flat.sum(0) if total is None else total + flat.sum(0) #sum(0) soma todas as colunas da imagem de agora, total + sum(0) soma a anterior...
        sq_total = (flat ** 2).sum(0) if sq_total is None else sq_total + (flat ** 2).sum(0) #mesma coisa, mas com ^2, serve pra calcular variância
        n_pixels += flat.shape[0] #shape[0] da o numero de linhas, consequentemente o número de pixels de cada imagem
    mean = total / n_pixels # média dos valores de cada canal
    std = np.sqrt(sq_total / n_pixels - mean ** 2)
    return mean.tolist(), std.tolist()


class SteelMicrostructureDataset(Dataset):
    """Returns ``(image_tensor [C, H, W], label)``.

    ``out_channels=3`` with ``grayscale=True`` repeats the gray channel, for backbones
    pretrained on RGB (ResNet, EfficientNet). ``transform`` runs on the normalized
    tensor (use it for augmentation).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_size: int | None = 224,
        grayscale: bool = True,
        out_channels: int | None = None,
        mean: Iterable[float] | None = None,
        std: Iterable[float] | None = None,
        transform=None,
    ):
        self.paths = df["path"].tolist() #armazena caminho
        self.labels = df["label"].astype(int).tolist() #armazena a fase
        self.image_size = image_size
        self.grayscale = grayscale
        self.out_channels = out_channels or (1 if grayscale else 3)
        self.mean = None if mean is None else torch.tensor(list(mean)).view(-1, 1, 1)
        self.std = None if std is None else torch.tensor(list(std)).view(-1, 1, 1)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths) #quantidade de imagens 

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]: #carrega uma imagem
        arr = load_image(self.paths[idx], self.grayscale, self.image_size)
        x = torch.from_numpy(arr).permute(2, 0, 1).contiguous() #transforma a array em um tensor pytorch e troca os eixos
        if self.transform is not None:
            x = self.transform(x)        
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / self.std
        if x.shape[0] == 1 and self.out_channels == 3: #cinza para 3 dimensões
            x = x.expand(3, -1, -1).contiguous()
        return x, self.labels[idx]


def describe_split(name: str, part: pd.DataFrame, group_col: str = GROUP_COL) -> str:
    counts = part["Phase"].value_counts().to_dict() #contar quantas fases tem em cada conjunto
    groups = sorted(part[group_col].unique())
    return f"{name:<10} {len(part):>5} patches | {counts} | {len(groups)} specimens {groups}"


if __name__ == "__main__":
    df = build_dataframe()
    print(f"Rows with image: {len(df)}")
    print(f"Microscopes: {sorted(df['MicroscopeID'].unique())}")
    print(f"Magnifications: {sorted(df['Magnification'].unique())}")
    print(f"Phases: {df['Phase'].value_counts().to_dict()}")
    print("\nPatches per microscope x magnification:")
    print(pd.crosstab(df["MicroscopeID"], df["Magnification"]))

    versions = df.groupby("region_id").size()
    print(f"\nPhysical regions: {len(versions)} | versions per region: {versions.value_counts().sort_index().to_dict()}")

    print("\nClean baseline split (microscope 0, 20x):")
    subset = filter_dataset(df, microscope_id=0, magnification="20x")
    train, val, test = make_splits(subset)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        print(describe_split(name, part) + f" | {part['region_id'].nunique()} regions")

    print("\nCross-microscope split (train on 0, test on 2, 20x):")
    splits = make_cross_microscope_splits(df, train_microscope=0, test_microscope=2, magnification="20x")
    for name, part in splits.items():
        print(describe_split(name, part))

    ds = SteelMicrostructureDataset(train, image_size=224, grayscale=True)
    x, y = ds[0]
    print(f"\nSample tensor: shape={tuple(x.shape)} dtype={x.dtype} range=[{x.min():.3f}, {x.max():.3f}] label={y}")
