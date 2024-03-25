## Security Assessment of Hierarchical Federated Deep Learning

This code reposetory investigates and assesses the security of Hierarchical Federated Deep Learning (HFL) using a novel methodology by focusing on its resilience against inference-time and training-time adversarial attacks. This code evaluate a series of extensive experiments across diverse datasets and attack scenarios.

## Results

#Baseline performance: HFL model under no attacks

Train 2-level FL
```bash
python /path/to/directory/exp_code_11222/start.py -config /path/to/directory/exp_code_11222/exp_settings/2L-FL.toml
```
Train 3-level FL
```bash
python /path/to/directory/exp_code_11222/start.py -config /path/to/directory/exp_code_11222/exp_settings/3L-HFL.toml
```
Train 4-level FL
```bash
python /path/to/directory/exp_code_11222/start.py -config /path/to/directory/exp_code_11222/exp_settings/4L-HFL.toml
```

#Models performance under Inference-time attacks and defense

Evalute Inference-time attacks and defense

```bash
python /path/to/directory/exp_code_11222/ITA_eval.py -config /path/to/directory/exp_code_11222/exp_settings/ITA_eval.toml
```

#Models performance under Training-time attacks and defense

Train 2-level FL under label flipping attack
```bash
python /path/to/directory/exp_code_11222/start.py -config /path/to/directory/exp_code_11222/exp_settings/2L-1malnode.toml
python /path/to/directory/exp_code_11222/start.py -config /path/to/directory/exp_code_11222/exp_settings/2L-5malnode.toml
python /path/to/directory/exp_code_11222/start.py -config /path/to/directory/exp_code_11222/exp_settings/2L-10malnode.toml

```
Train 3-level FL under label flipping attack
```bash
python /path/to/directory/exp_code_11222/start.py -config /path/to/directory/exp_code_11222/exp_settings/3L-1malnode.toml
python /path/to/directory/exp_code_11222/start.py -config /path/to/directory/exp_code_11222/exp_settings/3L-5malnode.toml
python /path/to/directory/exp_code_11222/start.py -config /path/to/directory/exp_code_11222/exp_settings/3L-10malnode.toml
```
Train 4-level FL under label flipping attack
```bash
python /path/to/directory/exp_code_11222/start.py -config /path/to/directory/exp_code_11222/exp_settings/4L-1malnode.toml
python /path/to/directory/exp_code_11222/start.py -config /path/to/directory/exp_code_11222/exp_settings/4L-5malnode.toml
python /path/to/directory/exp_code_11222/start.py -config /path/to/directory/exp_code_11222/exp_settings/4L-10malnode.toml
```


TOML file in the exp_setting dirctory can be edited to manage the experiment settings
