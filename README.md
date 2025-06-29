### GET STARTED
activate venv with
```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd ~/noise_scaling_laws/src/lib/geneformer
pip install -e .
cd ~/noise_scaling_laws/scaling_laws
pip install -e .
```
recreate experiment by running:
```bash
python run_larry_whole.py
python run_merfish_whole.py
python run_pbmc_whole.py
python run_shendure_whole.py
```
if starting from scratch uncommnet code to make train/test/val and tokenize the dataset.
additionaly you need to run hvg code to recompute the PCA HVGs in hvg.ipynb