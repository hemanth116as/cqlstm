

**Dependencies**

> 1. pytorch>=1.12
> 2. allennlp==2.10
> 3. complexPyTorch=0.4

## Our train command

### CQLSTM
```cmd
allennlp train config/CQLSTM.jsonnet --include-package work -s ./result/cr_CQLSTM -f
```

## Dataset
Please modify the variable "task_name" in model.jsonnet to change different datasets.
