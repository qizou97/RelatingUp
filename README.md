<!--
 * -------------------------------------------------
 * @FilePath        : /RelatingUp/README.md
 * @Author          : Qi Zou
 * @Email           : qizou@mail.sdu.edu.cn
 * @Date            : 2024-01-25 18:08:46
 * -------------------------------------------------
 * Change Activity :
 * @  LastEditTime  : 2024-01-25 20:31:16
 * @  LastEditors   : Qi Zou & qizou@mail.sdu.edu.cn
 * -------------------------------------------------
 * @Description     : 
 * -------------------------------------------------
-->

# Relating-Up: Advancing Graph Neural Networks through Inter-Graph Relationships
The Offical Code of **Relating-Up: Advancing Graph Neural Networks through Inter-Graph Relationships**

## Requirements
The code has been implemented and tested with Python 3.10.0. To install the required packages:

```bash
$ pip install -r requirements.txt
```

## Usage
### Training Commands

```python
### Evaluation Commands
python main.py --dataset DATASET     Name of dataset
  --model {GCN,GIN,GCNRU,GINRU}
                        Name of model
  --seed SEED           Random seed (default: 2023)
  --n_splits N_SPLITS   Number of splits
  --n_repeats N_REPEATS
                        Number of times cross-validation needs to be repeated
  --batch_size BATCH_SIZE
                        Input batch size for training (default: 128)
  --lr LR               Learning rate (default: 0.001)
  --weight_decay WEIGHT_DECAY
                        Weight decay (L2 penalty) (default: 5e-4)
  --gradient_clip_val GRADIENT_CLIP_VAL
                        The value at which to clip gradients
  --patience PATIENCE   Number of validation epochs with no improvement after which training will be stopped
  --min_epochs MIN_EPOCHS
                        Force training for at least `min_epochs` epochs
  --max_epochs MAX_EPOCHS
                        Stop training once `max_epochs` is reached
  --hidden_dim HIDDEN_DIM
                        Number of hidden units (default: 128)
  --num_layers NUM_LAYERS
                        Number of layers (default: 5)
  --alpha ALPHA         The parameter controls the balance between the Cross Entropy loss and the distillation loss
  --beta BETA           Weight of representation hints loss
  --temp TEMP           Temperature to smooth the logits
  --cuda
```
