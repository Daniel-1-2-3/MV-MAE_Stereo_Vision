## Setup

- **Python version:** `Python 3.12.5`
- All required packages are listed in `requirements.txt`.
- Install dependencies by running:

```bash
pip install -r requirements.txt
```

---

## Run Training

1. Run the `trainer_pipeline.py` file in the base directory:

```bash
python trainer_pipeline.py
```

2. **Hyperparameters:**
   - Currently, you **cannot** specify training hyperparameters from the terminal.
   - To modify parameters, edit the `trainer_pipeline.py` file directly.
   - In the `model = SAC(...)` declaration, you can adjust:
     - `buffer_size`
     - `batch_size`
     - `learning_starts`
     - Any other hyperparameters supported by **SB3 SAC**.

3. **MVMAE Hyperparameters:**
   - Currently **cannot** be modified via configuration.
   - Fixed settings:
     - Encoder dimensions: `767`
     - Decoder dimensions: `512`
     - Attention heads: `16` (both encoder & decoder)

4. **Image Size:**
   - Fixed at `128`.
   - Image size may be hardcoded in a few places for debugging purposes (currently being refactored).

---

## Evaluate Model

- Training logs are saved to `log.csv` and include:
  - Actor loss
  - MVMAE reconstruction loss
  - Critic loss
  - Reward

- Visualize metrics in Colab:  
  [Colab Visualization Notebook](https://colab.research.google.com/drive/16gPPI8HYgLcdplrTIn5KxCcgwm1OSq_e#scrollTo=Jysv5pMR-6PQ)

**Usage:**
1. Run the first cell to load the `log.csv` file.
2. Run the second cell to plot the metrics.
