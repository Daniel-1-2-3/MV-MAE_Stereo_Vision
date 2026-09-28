"""MV-MAE masking/patching invariants and a DrQ-v2 update smoke test (CPU, small model)."""

import torch

from agent.drqv2 import Batch, MVMAEDrQV2Agent
from agent.utils import RandomShiftsAug
from config import AgentConfig, MVMAEConfig
from mvmae.model import MVMAE


def small_model(**kw):
    args = dict(img_hw=(32, 32), num_views=2, num_frames=3, patch_size=8, embed_dim=32, encoder_depth=2,
                encoder_heads=4, decoder_dim=32, decoder_depth=1, decoder_heads=4, mask_ratio=0.95)
    args.update(kw)
    return MVMAE(**args)


def test_patchify_roundtrip_and_token_order():
    m = small_model()
    x = torch.randint(0, 256, (2, 3, 2, 3, 32, 32)).float()
    t = m.patchify(x)
    assert t.shape == (2, m.L, 8 * 8 * 3) and m.L == 3 * 2 * 16
    assert torch.equal(m.unpatchify(t), x)
    # token index (frame, view, row, col) holds exactly that image block
    f, v, r, c = 2, 1, 3, 1
    idx = ((f * 2) + v) * 16 + r * 4 + c
    block = x[0, f, v, :, r * 8:(r + 1) * 8, c * 8:(c + 1) * 8].permute(1, 2, 0).reshape(-1)
    assert torch.equal(t[0, idx], block)


def test_view_mask_properties():
    torch.manual_seed(0)
    for ratio in (0.5, 0.75, 0.95):
        m = small_model(mask_ratio=ratio)
        keep_idx, mask = m.view_mask(64, torch.device("cpu"))
        k = keep_idx.shape[1]
        assert k == max(1, round((1 - ratio) * m.L))
        assert (mask.sum(1) == m.L - k).all()
        per = mask.reshape(64, m.F, m.V, m.N)
        # every frame has exactly one fully hidden view (with two views, visible tokens come from the other)
        fully_hidden = (per.sum(-1) == m.N)
        assert (fully_hidden.sum(-1) >= 1).all()
        visible_views = (per.sum(-1) < m.N).sum(-1)
        assert (visible_views <= m.V - 1).all()
        assert torch.equal(torch.gather(mask, 1, keep_idx), torch.zeros_like(keep_idx, dtype=mask.dtype))


def test_losses_backward_and_encode_shapes():
    torch.manual_seed(0)
    m = small_model()
    x = torch.randint(0, 256, (4, 3, 2, 3, 32, 32), dtype=torch.uint8)
    z = m.encode(x)
    assert z.shape == (4, m.L, 32)
    recon, rew, out = m.losses(x, torch.randn(4, 3), torch.tensor([[False, True, True]] * 4))
    assert out["pred"].shape == out["target"].shape == (4, m.L, 192) and out["reward_pred"].shape == (4, 3)
    (recon + rew).backward()
    assert m.stem.net[0].weight.grad is not None and m.reward_head[0].weight.grad is not None
    assert torch.isfinite(recon) and torch.isfinite(rew)
    # normalise / denormalise round trip
    assert torch.equal(m.denormalize(m.normalize(x)), x)


def test_random_shift_keeps_views_aligned():
    aug = RandomShiftsAug(4)
    x = torch.rand(8, 3 * 2 * 3, 32, 32)
    x[:, 9:] = x[:, :9]  # make the "right view" channels identical to the left
    y = aug(x)
    assert y.shape == x.shape
    assert torch.allclose(y[:, :9], y[:, 9:])  # same shift for every channel of a sample


def small_agent():
    a = AgentConfig(batch_size=8, hidden_dim=64, feature_dim=16, encoder_warmup_updates=2, amp=False, bc_coef=0.4)
    m = MVMAEConfig(frame_stack=3, patch_size=8, embed_dim=32, encoder_depth=2, encoder_heads=4, decoder_dim=32,
                    decoder_depth=1, decoder_heads=4)
    return MVMAEDrQV2Agent(a, m, (32, 32), 2, 7, "cpu")


def fake_batch(b=8, demo=True):
    return Batch(
        obs=torch.randint(0, 256, (b, 3, 2, 3, 32, 32), dtype=torch.uint8),
        action=torch.rand(b, 7) * 2 - 1,
        reward=torch.randn(b),
        discount=torch.full((b,), 0.99 ** 3),
        next_obs=torch.randint(0, 256, (b, 3, 2, 3, 32, 32), dtype=torch.uint8),
        is_demo=torch.tensor([demo] * (b // 4) + [False] * (b - b // 4)),
        frame_reward=torch.randn(b, 3),
        frame_reward_valid=torch.ones(b, 3, dtype=torch.bool),
    )


def test_agent_update_changes_all_parts():
    torch.manual_seed(0)
    agent = small_agent()
    before = {name: [p.detach().clone() for p in mod.parameters()]
              for name, mod in (("mvmae", agent.mvmae), ("actor", agent.actor), ("critic", agent.critic),
                                ("target", agent.critic_target))}
    for _ in range(3):
        metrics = agent.update(fake_batch(), env_step=0)
    for key in ("critic/loss", "actor/loss", "actor/bc_loss", "mvmae/recon_loss", "mvmae/reward_loss", "grad/encoder"):
        assert key in metrics and torch.isfinite(metrics[key])
    for name, mod in (("mvmae", agent.mvmae), ("actor", agent.actor), ("critic", agent.critic),
                      ("target", agent.critic_target)):
        changed = any(not torch.equal(a, b) for a, b in zip(before[name], mod.parameters()))
        assert changed, f"{name} parameters did not change"
    act = agent.act(fake_batch().obs, env_step=0, eval_mode=False)
    assert act.shape == (8, 7) and act.abs().max() <= 1.0
    m = agent.update_mae_only(fake_batch())
    assert torch.isfinite(m["mvmae/recon_loss"])
    img = agent.reconstruction_image(fake_batch().obs)
    assert img.shape == (32 * 3, 32 * 6, 3) and img.dtype == torch.uint8
    # checkpoint round trip
    other = small_agent()
    other.load_state_dict(agent.state_dict())
    x = fake_batch().obs
    assert torch.allclose(agent.act(x, 0, True), other.act(x, 0, True))


def test_encoder_only_learns_from_mae_when_critic_grad_disabled():
    torch.manual_seed(0)
    agent = small_agent()
    agent.cfg.critic_grad_to_encoder = False
    agent.cfg.mae_coef = 0.0
    before = [p.detach().clone() for p in agent.mvmae.parameters()]
    agent.update(fake_batch(demo=False), env_step=0)
    assert all(torch.equal(a, b) for a, b in zip(before, agent.mvmae.parameters()))


def test_critic_encoder_grad_schedule():
    from agent import utils

    assert utils.schedule("linear(1.0,0.0,400000)", 0) == 1.0
    assert abs(utils.schedule("linear(1.0,0.0,400000)", 200000) - 0.5) < 1e-9
    assert utils.schedule("linear(1.0,0.0,400000)", 900000) == 0.0
    assert utils.schedule("0.1", 123) == 0.1


def test_frozen_encoder_stays_fixed_while_actor_and_critic_learn():
    torch.manual_seed(0)
    agent = small_agent()
    agent.update(fake_batch(), env_step=0)  # Adam has momentum for the encoder before freezing
    agent.freeze_encoder()
    enc = [p.detach().clone() for p in agent.mvmae.parameters()]
    actor = [p.detach().clone() for p in agent.actor.parameters()]
    critic = [p.detach().clone() for p in agent.critic.parameters()]
    for _ in range(3):
        m = agent.update(fake_batch(), env_step=0)
    assert float(m["train/encoder_frozen"]) == 1.0 and "mvmae/recon_loss" not in m
    assert all(torch.equal(a, b) for a, b in zip(enc, agent.mvmae.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(actor, agent.actor.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(critic, agent.critic.parameters()))
    other = small_agent()
    other.load_state_dict(agent.state_dict())
    assert other.encoder_frozen
