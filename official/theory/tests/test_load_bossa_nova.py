import pytest
from theory.core.loaders import loader
from theory.core.backend import backend


@pytest.mark.parametrize("path", ["../../training/readout/5.0/bossa_nova"])
def test_nsys_loader(
    path: str,
):
    nsys_loader = loader.load('bossa_nova', cfg={'path': path,})