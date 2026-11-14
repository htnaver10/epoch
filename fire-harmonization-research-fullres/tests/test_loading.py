import collections
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from firefullres import core
from firefullres.demo import make_demo
from firefullres.loading import TileStore,WindowSampler,tile_shapes


class SamplingTests(unittest.TestCase):
    def rows(self):
        return [{'native_shape':[32,32] if i<30 else [64,64],
                 'query_shape':[33+i%3,48+i%2]} for i in range(40)]

    def test_buckets_keep_every_tile_exactly_once_and_separate_native_shapes(self):
        rows=self.rows();sampler=WindowSampler(rows,8,32,True,42)
        windows=list(sampler)
        self.assertEqual(sorted(i for window in windows for i in window),list(range(40)))
        self.assertEqual(len(windows),len(sampler))
        for window in windows:
            self.assertLessEqual(len(window),8)
            self.assertEqual(len({tuple(rows[i]['native_shape']) for i in window}),1)
            self.assertEqual(len({tuple((n+31)//32 for n in rows[i]['query_shape']) for i in window}),1)

    def test_epoch_shuffle_is_reproducible_including_direct_resume(self):
        a=WindowSampler(self.rows(),8,32,True,42)
        first=list(a);a.set_epoch(3);third=list(a)
        b=WindowSampler(self.rows(),8,32,True,42);b.set_epoch(3)
        self.assertEqual(third,list(b));self.assertNotEqual(first,third)

    def test_invalid_or_empty_sampler(self):
        self.assertEqual(list(WindowSampler([],8)),[])
        with self.assertRaises(ValueError):WindowSampler(self.rows(),0)
        with self.assertRaises(ValueError):tile_shapes({'native_shape':[0,32],'query_shape':[64,64]})


class TileLoadingTests(unittest.TestCase):
    def setUp(self):
        (ROOT/'.cache').mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'.cache');self.old=core.ROOT
        core.ROOT=Path(self.temp.name)
        make_demo('manifest.jsonl');self.rows=core.read_jsonl('manifest.jsonl')
        self.stats=core.statistics('manifest.jsonl','stats.json',True)

    def tearDown(self):
        core.ROOT=self.old;self.temp.cleanup()

    def test_archive_members_materialized_once_and_values_preserved(self):
        counts=collections.Counter();original=np.load
        class Tracked:
            def __init__(self,*args,**kwargs):self.archive=original(*args,**kwargs)
            def __enter__(self):return self
            def __exit__(self,*args):self.archive.close()
            def __getitem__(self,key):counts[key]+=1;return self.archive[key]
        with patch('firefullres.loading.np.load',Tracked):
            value=TileStore(self.rows,self.stats)[0]
        self.assertEqual(counts,collections.Counter({key:1 for key in core.REQUIRED_KEYS}))
        with original(core.path(self.rows[0]['tile'])) as z:
            for key in core.INPUT_KEYS:
                expected=core.normalize(key,z[key],self.stats) if key!='query_grid' else z[key]
                np.testing.assert_array_equal(value['inputs'][key],expected)
            np.testing.assert_array_equal(value['weight'],z['weight'])
            self.assertEqual(value['inputs']['query_grid'].dtype,np.float32)
            self.assertEqual(z['query_lat'].dtype,np.float64)

    def test_cache_is_bounded_reused_and_evicted(self):
        store=TileStore(self.rows,self.stats,cache_mib=1)
        first=store[0];self.assertIs(first,store[0])
        for i in range(1,len(self.rows)):store[i]
        self.assertLessEqual(store.cache_bytes,store.limit)
        self.assertNotIn((self.rows[0]['tile'],0),store.cache)
        self.assertIsNot(first,store[0])

    def test_hash_tampering_is_rejected_even_on_cache_hit(self):
        store=TileStore(self.rows,self.stats,cache_mib=1);store[0]
        filename=core.path(self.rows[0]['tile'])
        filename.write_bytes(filename.read_bytes()+b'tamper')
        with self.assertRaisesRegex(ValueError,'changed'):store[0]

    def test_old_manifests_need_only_headers_and_shape_claims_are_checked(self):
        row=self.rows[0];old={k:v for k,v in row.items() if k not in ('native_shape','query_shape')}
        self.assertEqual(tile_shapes(old),((32,32),(64,64)))
        bad={**row,'query_shape':[32,64]}
        with self.assertRaisesRegex(ValueError,'shape'):TileStore([bad],self.stats)[0]


if __name__=='__main__':unittest.main()
