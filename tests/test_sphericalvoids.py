import pytest
import numpy as np
from VERSUS import SphericalVoids

@pytest.fixture
def sample_data():
    """Generates a valid numpy positions array with holes."""
    N = 10000
    data = np.random.rand(N,3) * 400 - 200
    mask = ((data < -25) | (data > 25)).all(axis=1)
    return data[mask]

@pytest.fixture
def initialise_vf(sample_data):
    """Initialise the SphericalVoids class with data positions"""
    return SphericalVoids(data_positions=sample_data)

@pytest.fixture
def valid_data_file(tmp_path, sample_data):
    """Saves valid numpy data to a temporary file."""
    file_path = tmp_path / "data.npy"
    np.save(file_path, sample_data)
    return file_path

def test_data_not_provided():
    with pytest.raises(Exception, match='Either data_positions or delta_mesh must be provided'):
        SphericalVoids()

def test_is_box(initialise_vf):
    vf = initialise_vf
    assert vf.box_like is True
    assert vf.nmesh == [100, 100, 100]
    assert vf.delta.min() >= -1.
    assert vf.delta.max() > 0.

def test_is_survey(sample_data):
    vf = SphericalVoids(data_positions=sample_data, 
                        random_positions=sample_data)
    assert vf.box_like is False 

def test_void_finding(initialise_vf):
    vf = initialise_vf
    vf.run_voidfinding()
    assert vf.vf_type == 'void'

def test_peak_finding(initialise_vf):
    vf = initialise_vf
    vf.run_voidfinding(void_delta=2.1)
    assert vf.vf_type == 'peak'

def test_output_dimensions(initialise_vf): 
    vf = initialise_vf
    vf.run_voidfinding()
    assert len(vf.position) > 0
    assert len(vf.position) == len(vf.radius) == len(vf.id) == len(np.unique(vf.cell_membership)) - 1
    assert len(vf.position) == vf.counts.sum()
    assert len(vf.input_radii) == len(vf.counts) == vf.size_function.shape[1] + 1
    assert vf.cell_membership.shape == tuple(vf.nmesh)

def test_void_position_boundaries(initialise_vf, sample_data):
    vf = initialise_vf
    vf.run_voidfinding()
    assert np.isclose(vf.position.max(), sample_data.max(), atol=vf.input_radii.min())
    assert np.isclose(vf.position.min(), sample_data.min(), atol=vf.input_radii.min())
