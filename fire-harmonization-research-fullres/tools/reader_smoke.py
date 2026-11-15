"""Synthetic HDF4/HDF5 file I/O check; does not validate NASA calibration/alignment."""
import json
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import h5py
from pyproj import Transformer
from firefullres.core import write_json
from firefullres.native import gdal_module,h5_field


def main():
    gdal=gdal_module();(ROOT/'.cache').mkdir(exist_ok=True)
    expected=np.arange(64,dtype=np.uint16).reshape(8,8)
    with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as temp:
        hdf=Path(temp)/'synthetic.hdf'
        dataset=gdal.GetDriverByName('HDF4Image').Create(str(hdf),8,8,1,gdal.GDT_UInt16)
        dataset.GetRasterBand(1).WriteArray(expected);dataset=None
        dataset=gdal.Open(str(hdf));actual=dataset.ReadAsArray();dataset=None
        np.testing.assert_array_equal(actual.squeeze(),expected)
        h5=Path(temp)/'synthetic.nc'
        with h5py.File(h5,'w') as archive:archive.create_dataset('geophysical_data/fire mask',data=expected)
        np.testing.assert_array_equal(h5_field(h5,'fire mask'),expected)
        x,y=Transformer.from_crs('EPSG:4326','EPSG:3857',always_xy=True).transform(-120,50)
        assert np.isfinite([x,y]).all()
    report={'passed':True,'HDF4_uint16_roundtrip':True,'HDF5_nested_field_roundtrip':True,
            'PROJ_transform':True,'gdal_version':gdal.VersionInfo('--version'),
            'scope':'synthetic file I/O only; no native NASA calibration, naming, QA or alignment validation'}
    write_json('results/windows_reader_smoke.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
