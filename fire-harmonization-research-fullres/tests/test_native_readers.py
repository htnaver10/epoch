"""Reader selection regressions without requiring native NASA files or GDAL."""
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock,patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from firefullres.native import sds


class SubdatasetTests(unittest.TestCase):
    def select(self,name,descriptions):
        root=Mock();root.GetSubDatasets.return_value=[(str(i),d) for i,d in enumerate(descriptions)]
        gdal=Mock();gdal.Open.side_effect=[root,'selected']
        with patch('firefullres.native.gdal_module',return_value=gdal):
            result=sds('data/raw/example.hdf',name)
        self.assertEqual(result,'selected')
        return gdal.Open.call_args.args[0]

    def test_modis_radiance_excludes_uncertainty_and_prefixed_names(self):
        self.assertEqual(self.select('EV_1KM_Emissive',[
            '[16x2030x1354] EV_1KM_Emissive_Uncert_Indexes (8-bit unsigned integer)',
            '[16x2030x1354] EV_1KM_Emissive (16-bit unsigned integer)',
            '[16x2030x1354] Other_EV_1KM_Emissive (16-bit unsigned integer)',
        ]),'1')

    def test_fire_mask_accepts_space_or_underscore(self):
        for name in ('fire mask','fire_mask'):
            self.assertEqual(self.select('fire mask',[
                '[2030x1354] fire_mask_quality (8-bit unsigned integer)',
                f'[2030x1354] {name} (8-bit unsigned integer)',
            ]),'1')

    def test_ambiguous_names_still_fail(self):
        with self.assertRaisesRegex(ValueError,'found 2'):
            self.select('Latitude',['[2x2] Latitude (Float32)','[2x2] Latitude (Float32)'])

    def test_hidden_modis_geolocation_uses_raw_sds_fallback(self):
        root=Mock();root.GetSubDatasets.return_value=[('HDF4_EOS:angles','[8x8] SensorZenith (Int16)')]
        expanded=Mock();expanded.GetSubDatasets.return_value=[
            ('HDF4_EOS:latitude','[8x8] Latitude (Float32)'),
            ('HDF4_SDS:raw_latitude','[8x8] Latitude (Float32)'),
            ('HDF4_SDS:raw_longitude','[8x8] Longitude (Float32)')]
        gdal=Mock();gdal.Open.side_effect=[root,'selected'];gdal.OpenEx.return_value=expanded
        with patch('firefullres.native.gdal_module',return_value=gdal):
            self.assertEqual(sds('data/raw/example.hdf','Latitude'),'selected')
        self.assertEqual(gdal.Open.call_args.args[0],'HDF4_SDS:raw_latitude')
        self.assertEqual(gdal.OpenEx.call_args.kwargs['open_options'],['LIST_SDS=YES'])


if __name__=='__main__':unittest.main()
