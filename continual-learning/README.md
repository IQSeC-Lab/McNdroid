## Continual learning experiments on McNdroid
In this section, we experiment with McNdroid for continual learning settings i) Domain incremental and ii) Class incremental learning

To run the domain incremental learning, use the following script:

```bash
./run_exp_cl.sh
```

Next we also did the experiment for class incremental learning:

First, we prepare the dataset for class incremental learning setup, considering 100 malware families.

Use this script to prepare the dataset:
```python
python mcndroid_class_inc_dataset.py
```

Then, we run the class incremental learning:

```bash
./run_exp.sh
```
