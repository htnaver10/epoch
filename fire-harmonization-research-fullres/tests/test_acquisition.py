"""Ensure provider version metadata never silently selects the wrong collection."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from firefullres import core
from firefullres.acquisition import acquire


class AcquisitionTests(unittest.TestCase):
    def test_cmr_versions_are_distinct_from_filename_collection_codes(self):
        search=Mock(return_value=[]);(ROOT/'.cache').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as temp:
            with patch.object(core,'ROOT',Path(temp)),patch.dict(sys.modules,earthaccess=SimpleNamespace(search_data=search)):
                with contextlib.redirect_stdout(io.StringIO()):
                    acquire('2023-07-01T20:00:00Z','2023-07-01T23:59:59Z',[-125,50,-110,65],'inventory.jsonl')
        actual={call.kwargs['short_name']:call.kwargs['version'] for call in search.call_args_list}
        self.assertEqual(actual,{'MYD021KM':'6.1','MYD03':'6.1','MYD14':'061',
            'MYD02HKM':'6.1','MYD02QKM':'6.1','VNP14IMG':'002','VNP03IMG':'2'})
        self.assertTrue(all(call.kwargs['count']==51 for call in search.call_args_list))

    def test_inventory_retains_both_cmr_and_file_collection_versions(self):
        granule={'umm':{'GranuleUR':'MYD021KM.example.061.hdf',
            'TemporalExtent':{'RangeDateTime':{'BeginningDateTime':'2023-07-01T20:00:00Z',
                                             'EndingDateTime':'2023-07-01T20:05:00Z'}}},
            'meta':{'concept-id':'G-example','collection-concept-id':'C-example'}}
        search=Mock(return_value=[granule]);(ROOT/'.cache').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as temp:
            with patch.object(core,'ROOT',Path(temp)),patch.dict(sys.modules,earthaccess=SimpleNamespace(search_data=search)):
                with contextlib.redirect_stdout(io.StringIO()):
                    acquire('2023-07-01T20:00:00Z','2023-07-01T23:59:59Z',[-125,50,-110,65],
                        'inventory.jsonl',products=['MYD021KM'])
                row=core.read_jsonl('inventory.jsonl')[0]
                self.assertEqual(row['version'],'6.1');self.assertEqual(row['filename_collection'],'061')
                self.assertIsNone(row['local_path'])


if __name__=='__main__':unittest.main()
