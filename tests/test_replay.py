"""Replay buffer vs. a slow reference: n-step returns, episode boundaries, frame stacking, ring wrap-around."""

import random

import torch

from agent.replay import FrameStacker, ReplayBuffer

GAMMA = 0.9


def make_stream(num_envs, steps, seed):
    """Random episodes; each obs encodes (env, global step) so samples can be traced back."""
    g = torch.Generator().manual_seed(seed)
    rows = []
    first = torch.ones(num_envs, dtype=torch.bool)
    ep_len = torch.zeros(num_envs, dtype=torch.long)
    for t in range(steps):
        obs = torch.zeros(num_envs, 2, 3, 4, 4, dtype=torch.uint8)
        obs[:, 0, 0, 0, 0] = torch.arange(num_envs, dtype=torch.uint8)
        obs[:, 0, 0, 0, 1] = t % 256
        obs[:, 0, 0, 0, 2] = t // 256
        reward = torch.randn(num_envs, generator=g)
        ep_len += 1
        term = torch.rand(num_envs, generator=g) < 0.05
        trunc = (ep_len >= 12) & ~term
        trunc |= (torch.rand(num_envs, generator=g) < 0.03) & ~term  # e.g. evaluation cut-offs
        action = torch.randn(num_envs, 3, generator=g)
        rows.append(dict(obs=obs, action=action, reward=reward, term=term, trunc=trunc, first=first.clone()))
        done = term | trunc
        ep_len[done] = 0
        first = done
    return rows


def decode(obs):  # (..., 2, 3, 4, 4) -> (env, t)
    return obs[..., 0, 0, 0, 0].long(), obs[..., 0, 0, 0, 1].long() + 256 * obs[..., 0, 0, 0, 2].long()


def reference(rows, e, t, n, frames):
    """Slow n-step target and frame stacks for start (t, e)."""
    ret, disc, m, terminal = 0.0, 1.0, n, False
    for k in range(n):
        r = rows[t + k]
        te, tr = bool(r["term"][e]), bool(r["trunc"][e]) and not bool(r["term"][e])
        if tr:
            m = k
            break
        ret += disc * float(r["reward"][e])
        disc *= GAMMA
        if te:
            m, terminal = k + 1, True
            break

    def stack(s):
        idx, cur = [s], s
        for _ in range(frames - 1):
            if not bool(rows[cur]["first"][e]):
                cur -= 1
            idx.append(cur)
        return idx[::-1]

    obs_idx = stack(t)
    frame_valid = [not bool(rows[c]["first"][e]) for c in obs_idx]
    frame_rew = [float(rows[c - 1]["reward"][e]) if v else None for c, v in zip(obs_idx, frame_valid)]
    return dict(ret=ret, disc=0.0 if terminal else disc, m=m, obs_idx=obs_idx, next_idx=stack(t + m),
                frame_valid=frame_valid, frame_rew=frame_rew)


def check_buffer(num_envs, steps, capacity, n=3, frames=3, seed=0):
    rows = make_stream(num_envs, steps, seed)
    buf = ReplayBuffer(capacity, num_envs, (2, 3, 4, 4), 3, frames, n, GAMMA, "cpu")
    for r in rows:
        buf.add(r["obs"], r["action"], r["reward"], r["term"], r["trunc"], r["first"])
    torch.manual_seed(seed)
    b = buf.sample(512)
    envs, t0 = decode(b.obs[:, -1])
    oldest = max(0, steps - buf.R)
    assert (t0 >= oldest + frames - 1).all() and (t0 <= steps - 1 - n).all()
    for i in range(512):
        e, t = int(envs[i]), int(t0[i])
        assert not bool(rows[t]["trunc"][e]) or bool(rows[t]["term"][e]), "a truncated transition was sampled"
        ref = reference(rows, e, t, n, frames)
        assert ref["m"] > 0
        assert abs(float(b.reward[i]) - ref["ret"]) < 1e-5
        assert abs(float(b.discount[i]) - ref["disc"]) < 1e-6
        assert torch.equal(b.action[i], rows[t]["action"][e])
        _, ot = decode(b.obs[i])
        assert ot.tolist() == ref["obs_idx"]
        ne, nt = decode(b.next_obs[i])
        assert (ne == e).all() and nt.tolist() == ref["next_idx"]
        assert b.frame_reward_valid[i].tolist() == [v and c - 1 >= oldest for v, c in zip(ref["frame_valid"], ref["obs_idx"])]
        for j, fr in enumerate(ref["frame_rew"]):
            if b.frame_reward_valid[i, j]:
                assert abs(float(b.frame_reward[i, j]) - fr) < 1e-6
    return buf, rows


def test_nstep_and_stacking_without_wrap():
    check_buffer(num_envs=4, steps=200, capacity=4 * 1000)


def test_nstep_and_stacking_with_wrap():
    buf, _ = check_buffer(num_envs=3, steps=500, capacity=3 * 64, seed=1)
    assert buf.R == 64 and buf.t == 500


def test_mark_last_truncated_and_demo_buffer():
    rows = make_stream(2, 40, seed=2)
    buf = ReplayBuffer(1000, 2, (2, 3, 4, 4), 3, 3, 3, GAMMA, "cpu")
    for r in rows:
        buf.add(r["obs"], r["action"], r["reward"], r["term"], r["trunc"], r["first"])
    buf.mark_last_truncated()
    assert buf.trunc[39].all()
    # static demo buffer from a flat episode list
    data = dict(obs=torch.cat([r["obs"][:1] for r in rows]), action=torch.cat([r["action"][:1] for r in rows]),
                reward=torch.cat([r["reward"][:1] for r in rows]), terminated=torch.cat([r["term"][:1] for r in rows]),
                truncated=torch.cat([r["trunc"][:1] for r in rows]), first=torch.cat([r["first"][:1] for r in rows]))
    demo = ReplayBuffer.from_episodes(data, 3, 3, GAMMA, "cpu")
    batch = demo.sample(64, is_demo=True)
    assert batch.is_demo.all() and batch.obs.shape == (64, 3, 2, 3, 4, 4)


def test_frame_stacker_matches_replay_padding():
    st = FrameStacker(2, 3, (1,), "cpu")
    o = lambda v: torch.tensor([[v], [v + 100]], dtype=torch.uint8)  # noqa: E731
    s = st.reset(o(1))
    assert s[:, :, 0].tolist() == [[1, 1, 1], [101, 101, 101]]
    s = st.step(o(2), torch.tensor([False, False]))
    s = st.step(o(3), torch.tensor([False, True]))  # env 1 starts a new episode
    assert s[:, :, 0].tolist() == [[1, 2, 3], [103, 103, 103]]
    s = st.step(o(4), torch.tensor([False, False]))
    assert s[:, :, 0].tolist() == [[2, 3, 4], [103, 103, 104]]


if __name__ == "__main__":
    random.seed(0)
    test_nstep_and_stacking_without_wrap()
    test_nstep_and_stacking_with_wrap()
    print("ok")
