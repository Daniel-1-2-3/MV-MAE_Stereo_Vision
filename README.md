# OUTDATED
## Setup

- **Python version:** `Python 3.12.5`
- All required packages are listed in `requirements.txt`.
- Install dependencies by running:

```bash
git clone https://github.com/Daniel-1-2-3/MV-MAE_Stereo_Vision
cd MV-MAE_Stereo_Vision
pip install -r requirements.txt
```

---

## Run Training

1. Run the `trainer_pipeline.py` file in the base directory on terminal:

```bash
python trainer_pipeline.py
```

2. **Parameters (CLI flags & defaults):**

   **General / Runtime**
   - `--render_mode` (str, default: `rgb_array`) — Render mode for env (`human` to show a window, `rgb_array` for offscreen).
   - `--total_timesteps` (int, default: `5000000`) — Total env steps to train.
   - `--episode_horizon` (int, default: `300`) — Max steps per episode before reset.
   
   **SAC / RL**
   - `--learning_rate` (float, default: `3e-4`) — Optimizer learning rate.
   - `--buffer_size` (int, default: `1000000`) — Replay buffer capacity.
   - `--learning_starts` (int, default: `50000`) — Steps collected before updates begin.
   - `--batch_size` (int, default: `256`) — Mini-batch size for updates.
   - `--gamma` (float, default: `0.99`) — Discount factor.
   - `--n_steps` (int, default: `1`) — Gradient steps per environment step.
   - `--ent_coef` (str/float, default: `auto`) — Entropy coefficient; set a float to fix it, or `auto` to tune.
   - `--target_entropy` (str/float, default: `auto`) — Target entropy; float for manual target or `auto`.
   - `--verbose` (int, default: `0`) — Verbosity level for SB3 (0/1/2).
   
   **MV-MAE / Vision**
   - `--nviews` (int, default: `2`) — Number of camera views fused in observations.
   - `--mvmae_patch_size` (int, default: `6`) — Patch size for MV-MAE patch embeddings.
   - `--mvmae_encoder_embed_dim` (int, default: `768`) — Encoder embedding dim.
   - `--mvmae_decoder_embed_dim` (int, default: `512`) — Decoder embedding dim.
   - `--mvmae_encoder_heads` (int, default: `16`) — Encoder attention heads.
   - `--mvmae_decoder_heads` (int, default: `16`) — Decoder attention heads.
   - `--masking_coef` (float, default: `0.75`) — The amount of all views that is masked. 
   - `--in_channels` (int, default: `3`) — Image channels (RGB=3).
   - `--img_h_size` (int, default: `84`) — Height of a single view.
   - `--img_w_size` (int, default: `84`) — Width of a single view.
   - `--coef_mvmae` (float, default: `0.1`) — Total loss = actor loss * coef_mvmae * mvmae_recon_loss
   
   **Example**
   ```bash
   python trainer_pipeline.py --gamma 0.98 --img_h_size 84 --img_w_size 84 --mvmae_patch_size 8  --mvmae_encoder_embed_dim 768 --mvmae_decoder_embed_dim 512
   ```
## Evaluate Model

- Training logs are saved to `log.csv` and include actor loss, MVMAE reconstruction loss, critic loss and reward for each step
- Visualize metrics in Colab (run cells in order):  
  [Colab Visualization Notebook](https://colab.research.google.com/drive/16gPPI8HYgLcdplrTIn5KxCcgwm1OSq_e#scrollTo=Jysv5pMR-6PQ)
