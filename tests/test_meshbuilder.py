import pytest
import numpy as np
from astropy.io import fits
from astropy.table import Table
from VERSUS.meshbuilder import DensityMesh

@pytest.fixture
def sample_data():
    """Generates a valid numpy positions and weights array."""
    N = 10000
    return np.random.rand(N,4)

@pytest.mark.parametrize("use_weights", [False, True])
def test_load_from_array(sample_data, use_weights):
    weights = sample_data[:, 3] if use_weights else None
    mesh = DensityMesh(sample_data[:,:3], data_weights=weights)
    assert (mesh.data_positions == sample_data[:,:3]).all()
    if use_weights:
        assert (mesh.data_weights == weights).all()
    else:
        assert mesh.data_weights is None

@pytest.mark.parametrize("use_weights", [False, True])
def test_load_from_npy_file(tmp_path, sample_data, use_weights):
    weights = sample_data[:, 3] if use_weights else None

    file_path = str(tmp_path / "data.npy")
    np.save(file_path, sample_data if use_weights else sample_data[:,:3])

    mesh = DensityMesh(file_path)
    assert (mesh.data_positions == sample_data[:,:3]).all()
    if use_weights:
        assert (mesh.data_weights == weights).all()
    else:
        assert mesh.data_weights is None

@pytest.mark.parametrize("use_weights", [False, True])
@pytest.mark.parametrize(
    "labels_true, labels_input",
    [
        (["X", "Y", "Z", "W"], "XYZW"),
        (["X", "Y", "Z", "Weights"], ["X", "Y", "Z", "Weights"]),
        (["x_coord", "y_coord", "z_coord", "weights"], ["x_coord", "y_coord", "z_coord", "weights"]),
    ],
)
def test_load_from_fits_file(tmp_path, sample_data, use_weights, labels_true, labels_input):

    x, y, z, weights = sample_data.T
    table = Table({labels_true[0]: x, labels_true[1]: y, labels_true[2]: z})

    if use_weights:
        table[labels_true[3]] = weights
        data_cols = labels_input
    else:
        data_cols = labels_input[:3]

    file_path = str(tmp_path / "data.fits")
    table.write(file_path)

    mesh = DensityMesh(file_path, data_cols=data_cols)
    assert (mesh.data_positions == sample_data[:,:3]).all()
    if use_weights:
        assert (mesh.data_weights == weights).all()
    else:
        assert mesh.data_weights is None

@pytest.mark.parametrize("labels", ["xy", "xyzwq", ["x", "y"], ["x", "y", "z", "w", "q"]])
def test_load_incorrect_data_cols(tmp_path, sample_data, labels):

    data = np.random.rand(10000)
    table = Table()

    for l in labels:
        table[l] = data 

    file_path = str(tmp_path / "data.fits")
    table.write(file_path)

    with pytest.raises(ValueError, match=r"Expected a list of data column headers with length.*"):
        DensityMesh(file_path, data_cols=labels)

@pytest.mark.parametrize("shape", [(3, 10), (10, 3, 1), (10, 5)])
def test_load_incorrect_shape_from_array(tmp_path, sample_data, shape):
    invalid_data = np.random.random(shape)

    file_path = str(tmp_path / "data.npy")
    np.save(file_path, invalid_data)

    with pytest.raises(ValueError, match=r"Expected positions with shape \(N, 3\).*"):
        DensityMesh(invalid_data)

    with pytest.raises(ValueError, match=r"Expected positions with shape \(N, 3\).*"):
        DensityMesh(sample_data[:, :3], random_positions=invalid_data)

    with pytest.raises(ValueError, match=r"Expected positions with shape \(N, 3\).*"):
        DensityMesh(file_path)
