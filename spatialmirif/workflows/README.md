# `spatialmirif` IF-positive plaque detection: direct JSON execution

```powershell
python spatialmirif/workflows/IFPositivePlaqueDetection.py --config-json spatialmirif/workflows/configs/config_IF_positive_plaques_workflow.json --check-inputs
python spatialmirif/workflows/IFPositivePlaqueDetection.py --config-json spatialmirif/workflows/configs/config_IF_positive_plaques_workflow.json
```

You can also call it from Python:

```python
from spatialmirif.workflows import IFPositivePlaqueDetection
result = IFPositivePlaqueDetection.RunFromConfig(
    r'spatialmirif/workflows/configs/S1_N3_IF_positive_plaques.json'
)
print(result['counts']['final_accepted'])
# TODO:
# will check later if need a files name directly for other workflow run after this? 
# so all-in-1 workflows might possible for each section
```
