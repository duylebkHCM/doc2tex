import enum
import datetime


class LMDB_CONST(enum.Enum):
    HEIGHT = "height"
    WIDTH = "width"
    N_SAMPLES = "num-samples"
    IMAGE = "image"
    PATH = "name"
    LABEL = "label"
    ARROW_OBJ = "obj"


class LABEL_KEY(enum.Enum):
    IMAGE_ID = "id"
    LABEL = "label"
    DELIMITER = "\t"
    DATETIME_FMT = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
