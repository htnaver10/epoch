"""Credential lifetime and pilot orchestration without accounts or downloads."""
import contextlib
import io
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import download_real_data as pilot


class PilotLauncherTests(unittest.TestCase):
    def test_prompted_credentials_are_only_present_during_download(self):
        with patch.dict(os.environ,{},clear=True),patch('builtins.input',return_value='local-user'),\
             patch.object(pilot.getpass,'getpass',return_value='dummy-password'),\
             patch.object(sys.stdin,'isatty',return_value=True):
            with pilot.local_credentials():
                self.assertEqual(os.environ['EARTHDATA_USERNAME'],'local-user')
                self.assertEqual(os.environ['EARTHDATA_PASSWORD'],'dummy-password')
            self.assertTrue(all(name not in os.environ for name in pilot.CREDENTIAL_KEYS))

    def test_existing_token_is_preserved_and_does_not_prompt(self):
        with patch.dict(os.environ,{'EARTHDATA_TOKEN':'dummy-token'},clear=True),\
             patch('builtins.input') as prompt:
            with pilot.local_credentials(False):self.assertEqual(os.environ['EARTHDATA_TOKEN'],'dummy-token')
            self.assertEqual(os.environ['EARTHDATA_TOKEN'],'dummy-token');prompt.assert_not_called()

    def test_inventory_needs_no_credentials_and_cannot_download_or_pair(self):
        with patch.object(pilot,'local_credentials') as credentials,\
             patch.object(pilot,'acquire',return_value={'granules':16}) as acquire,\
             patch.object(pilot,'propose_pairs') as pairs,contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(pilot.main(['--inventory-only','--tag','test']),0)
            self.assertFalse(acquire.call_args.kwargs['download'])
            credentials.assert_not_called();pairs.assert_not_called()

    def test_additional_acquisition_range_is_passed_to_inventory(self):
        with patch.object(pilot,'acquire',return_value={'granules':8}) as acquire,\
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(pilot.main(['--inventory-only','--tag','second',
                '--start','2022-07-01T20:00:00Z','--end','2022-07-01T23:59:59Z',
                '--bbox','-125','50','-110','65']),0)
            self.assertEqual(acquire.call_args.args[:3],('2022-07-01T20:00:00Z','2022-07-01T23:59:59Z',[-125.,50.,-110.,65.]))

    def test_missing_noninteractive_credentials_fail_before_network(self):
        with patch.dict(os.environ,{},clear=True),patch.object(pilot,'acquire') as acquire,\
             contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(pilot.main(['--no-prompt','--tag','test']),2)
            acquire.assert_not_called()

    def test_credentials_restore_after_failure_without_printing_remote_secret(self):
        output=io.StringIO()
        with patch.dict(os.environ,{'EARTHDATA_TOKEN':'dummy-token'},clear=True),\
             patch.object(pilot,'acquire',side_effect=RuntimeError('signed URL dummy-token')),\
             contextlib.redirect_stdout(output),contextlib.redirect_stderr(output):
            self.assertEqual(pilot.main(['--no-prompt','--tag','test']),2)
            self.assertEqual(os.environ['EARTHDATA_TOKEN'],'dummy-token')
        self.assertNotIn('dummy-token',output.getvalue())

    def test_download_proposes_pairs_without_assigning_reviewed_splits(self):
        with patch.dict(os.environ,{'EARTHDATA_TOKEN':'dummy-token'},clear=True),\
             patch.object(pilot,'acquire',return_value={'granules':16}) as acquire,\
             patch.object(pilot,'propose_pairs',return_value={'proposed_pairs':1}) as pairs,\
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(pilot.main(['--no-prompt','--tag','test']),0)
            self.assertTrue(acquire.call_args.kwargs['download'])
            pairs.assert_called_once_with('data/manifests/test_downloads.jsonl','data/manifests/test_proposed_bundles.jsonl')


if __name__=='__main__':unittest.main()
