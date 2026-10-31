import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import publish_release as p

HERE=Path(__file__).resolve().parent
class ReleaseTests(unittest.TestCase):
    def test_paths(self):
        for name in ('../x','a/../b','/absolute','a\\b',''):
            self.assertFalse(p.safe_name(name))
        self.assertTrue(p.safe_name('paper/manuscript.pdf'))
    def test_digest(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'x';f.write_bytes(b'abc')
            self.assertEqual(p.digest(f),hashlib.sha256(b'abc').hexdigest())
    def test_reject_wrong_size(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'x';f.write_bytes(b'abc')
            with self.assertRaises(ValueError):p.check_file(f,{'bytes':4,'sha256':'0'*64})
    def test_reject_wrong_hash(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'x';f.write_bytes(b'abc')
            with self.assertRaises(ValueError):p.check_file(f,{'bytes':3,'sha256':'0'*64})
    def test_exact_commit_required(self):
        with self.assertRaises(ValueError):p.publish({},'main',HERE)
    def test_auth_failure_makes_no_release(self):
        with patch.object(p.shutil,'which',return_value='/fake/gh'),patch.object(p,'gh',side_effect=RuntimeError('not logged in')) as cli:
            with self.assertRaises(RuntimeError):p.publish({},'a'*40,HERE)
            self.assertEqual(cli.call_args_list[0].args,('auth','status','--hostname','github.com'))
            self.assertEqual(cli.call_count,1)
    def test_mock_full_publication_and_idempotence(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);f=folder/'test.json';f.write_bytes(b'{"test":true}')
            files={f.name:f};state={'release':None};calls=[]
            def fake(*args,**kwargs):
                calls.append(args)
                if args[:2]==('auth','status'):return ''
                if args[0]=='api' and '/commits/' in args[1]:return json.dumps({'sha':'a'*40})
                if args[0]=='api':return json.dumps(state['release']) if state['release'] else None
                if args[:2]==('release','create'):
                    state['release']={'tag_name':p.TAG,'target_commitish':'a'*40,'assets':[],'draft':True,'html_url':'https://github.com/example/release'};return ''
                if args[:2]==('release','upload'):
                    state['release']['assets']=[{'name':f.name,'browser_download_url':'https://example.invalid/test.json'}];return ''
                if args[:2]==('release','download'):
                    (Path(args[args.index('--dir')+1])/f.name).write_bytes(f.read_bytes());return ''
                if args[:2]==('release','edit'):state['release']['draft']=False;return ''
                raise AssertionError(args)
            with patch.object(p.shutil,'which',return_value='/fake/gh'),patch.object(p,'gh',side_effect=fake),patch.object(p,'public_hash',return_value=p.digest(f)):
                first=p.publish(files,'a'*40,folder)
                self.assertEqual(first['status'],'published_prerelease_all_assets_download_hash_verified')
                second=p.publish(files,'a'*40,folder)
                self.assertEqual(first,second)
                self.assertEqual(sum(c[:2]==('release','create') for c in calls),1)
                self.assertEqual(sum(c[:2]==('release','edit') for c in calls),1)
    def test_refuse_existing_incomplete_public_release(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'missing';f.write_text('x')
            responses=['',json.dumps({'sha':'a'*40}),json.dumps({'tag_name':p.TAG,'target_commitish':'a'*40,'draft':False,'assets':[]})]
            with patch.object(p.shutil,'which',return_value='/fake/gh'),patch.object(p,'gh',side_effect=responses) as cli:
                with self.assertRaises(ValueError):p.publish({f.name:f},'a'*40,Path(d))
                self.assertEqual(cli.call_count,3)

if __name__=='__main__':unittest.main()
