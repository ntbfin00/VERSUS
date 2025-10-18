import logging 

logger = logging.getLogger("VERSUS")
def setup_logging(level=logging.INFO):
    logging.basicConfig(level=logging.CRITICAL, 
                        format='%(asctime)s | %(levelname)s | %(filename)s | %(funcName)s | Ln%(lineno)d | %(message)s',
                        datefmt='%H:%M:%S')
    # logger = logging.getLogger(__name__)
    logger.setLevel(level)

from .sphericalvoids import SphericalVoids
from .meshbuilder import DensityMesh
from .modelling import SizeBias
__all__ = ["setup_logging", "SphericalVoids", "DensityMesh", "SizeBias"]
