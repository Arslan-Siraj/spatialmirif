import importlib.util, json, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
p=Path(__file__).resolve().parents[1] / 'spatialmirif' / 'workflows' / 'MIRHyperspectralDataset.py'
spec=importlib.util.spec_from_file_location('hyp',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
root=Path(tempfile.mkdtemp())
source=root/'manifest.csv'
pd.DataFrame({'patch_id':['p1','n1'],'x0':[0,2],'y0':[0,2],'training_label':[1,0],'patch_size':[2,2]}).to_csv(source,index=False)
for k in ['zarr','info','reference']:(root/k).touch()
cfg={'sample_name':'S1_N3','paths':{'patch_manifest_csv':str(source),'mir_zarr':str(root/'zarr'),'mir_info':str(root/'info'),'mir_reference_nrrd':str(root/'reference'),'output_dir':str(root/'output')},'parameters':{'patch_size':2},'output_options':{'save_figures':True,'run_full_dataset_exploration':True,'run_registered_comparison':False,'run_regional_analysis':False,'continue_on_optional_error':False}}
config=root/'config.json';config.write_text(json.dumps(cfg))
cube=np.arange(6*6*3,dtype=np.float32).reshape(6,6,3);valid=np.ones((6,6),bool);axis=np.array([1200.,1500.,1650.])
m.read_reference=lambda path:(None,np.zeros((6,6)))
m.open_reader=lambda cfg,shape:(cube,axis,valid)
assert not (root/'output').exists()
assert m.run(str(config),check_only=True)['inputs']['patch_manifest_csv']['exists']
assert not (root/'output').exists()
res=m.run(str(config))
assert res['dataset_shape']==[2,2,2,3]
assert np.array_equal(np.load(root/'output/spectral_patches.npy')[0],cube[0:2,0:2])
assert (root/'output/run.log').is_file()
assert json.loads((root/'output/run_status.json').read_text())['status']=='success'
assert 'EXTRACTION' in (root/'output/run.log').read_text()
assert 'patch_overview' in (root/'output/run.log').read_text()
res=m.run(str(config),mode='figures-only')
assert json.loads((root/'output/run_status.json').read_text())['steps']['extraction']=='reused_existing_dataset'
# optional failure that is allowed should yield partial_success
cfg['output_options']['continue_on_optional_error']=True;config.write_text(json.dumps(cfg))
m.plot_patch_overview=lambda *args: (_ for _ in ()).throw(RuntimeError('deliberate plot failure'))
res=m.run(str(config),mode='figures-only')
assert json.loads((root/'output/run_status.json').read_text())['status']=='partial_success'
assert res['errors']['patch_overview']['message']=='deliberate plot failure'
# failed required input
cfg['paths']['mir_info']=str(root/'missing.info');config.write_text(json.dumps(cfg))
try:m.run(str(config),mode='figures-only')
except FileNotFoundError:pass
else:raise AssertionError('missing input did not fail')
assert json.loads((root/'output/run_status.json').read_text())['status']=='failed'
assert 'Traceback' in (root/'output/run.log').read_text()
print('PASSED: read-only preflight, extraction, logging, figures-only, optional failure, required-input failure')
