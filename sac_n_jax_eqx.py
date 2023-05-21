# Implementation of SAC-N on equinox framework.
# A little more lengthy than on flax, but a lot easier to reason about.
import jax
import d4rl
import gym
import wandb
import optax
import uuid
import distrax
import numpy as np
import jax.numpy as jnp
import pyrallis
import dataclasses
import equinox as eqx
import equinox.nn as nn

from tqdm.auto import trange
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple
from jaxtyping import PyTree


@dataclass
class Config:
    # wandb params
    project: str = "SAC-N-JAX"
    group: str = "SAC-N"
    name: str = "sac-n-jax-eqx"
    # model params
    hidden_dim: int = 256
    num_critics: int = 10
    gamma: float = 0.99
    tau: float = 5e-3
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    # training params
    env_name: str = "halfcheetah-medium-v2"
    batch_size: int = 256
    num_epochs: int = 3000
    num_updates_on_epoch: int = 1000
    # evaluation params
    eval_episodes: int = 10
    eval_every: int = 50
    # general params
    train_seed: int = 10
    eval_seed: int = 42

    def __post_init__(self):
        self.name = f"{self.name}-{self.env_name}-{str(uuid.uuid4())[:8]}"


# Unfortunately, distrax is not compatible by default with equinox, so some hacks are needed
# see: https://github.com/patrick-kidger/equinox/issues/269#issuecomment-1446586093
class TanhNormal(distrax.Transformed):
    def __init__(self, loc, scale):
        normal_dist = distrax.Normal(loc, scale)
        tanh_bijector = distrax.Tanh()
        super().__init__(distribution=normal_dist, bijector=tanh_bijector)

    def mean(self):
        return self.bijector.forward(self.distribution.mean())


class FixedDistrax(eqx.Module):
    cls: type
    args: PyTree[Any]
    kwargs: PyTree[Any]

    def __init__(self, cls, *args, **kwargs):
        self.cls = cls
        self.args = args
        self.kwargs = kwargs

    def sample_and_log_prob(self, *, seed):
        return self.cls(*self.args, **self.kwargs).sample_and_log_prob(seed=seed)

    def sample(self, *, seed):
        return self.cls(*self.args, **self.kwargs).sample(seed=seed)

    def log_prob(self, x):
        return self.cls(*self.args, **self.kwargs).log_prob(x)

    def mean(self):
        return self.cls(*self.args, **self.kwargs).mean()


class ReplayBuffer(eqx.Module):
    data: Dict[str, jax.Array]

    @staticmethod
    def create_from_d4rl(dataset_name: str) -> "ReplayBuffer":
        d4rl_data = d4rl.qlearning_dataset(gym.make(dataset_name))
        buffer = {
            "obs": jnp.asarray(d4rl_data["observations"], dtype=jnp.float32),
            "actions": jnp.asarray(d4rl_data["actions"], dtype=jnp.float32),
            "rewards": jnp.asarray(d4rl_data["rewards"], dtype=jnp.float32),
            "next_obs": jnp.asarray(d4rl_data["next_observations"], dtype=jnp.float32),
            "dones": jnp.asarray(d4rl_data["terminals"], dtype=jnp.float32)
        }
        return ReplayBuffer(data=buffer)

    @property
    def size(self):
        # WARN: do not use __len__ here! It will use len of the dataclass, i.e. number of fields.
        return self.data["obs"].shape[0]

    def sample_batch(self, key: jax.random.PRNGKey, batch_size: int) -> Dict[str, jax.Array]:
        indices = jax.random.randint(key, shape=(batch_size,), minval=0, maxval=self.size)
        batch = jax.tree_map(lambda arr: arr[indices], self.data)
        return batch


class TrainState(eqx.Module):
    model: eqx.Module
    optim: optax.GradientTransformation
    optim_state: optax.OptState

    @classmethod
    def create(cls, *, model, optim, **kwargs):
        optim_state = optim.init(eqx.filter(model, eqx.is_array))
        return cls(model, optim, optim_state, **kwargs)

    def apply_updates(self, grads):
        updates, new_optim_state = self.optim.update(grads, self.optim_state)
        new_model = eqx.apply_updates(self.model, updates)
        return dataclasses.replace(
            self,
            model=new_model,
            optim_state=new_optim_state
        )


class CriticTrainState(TrainState):
    target_model: eqx.Module

    def soft_update(self, tau):
        model_params = eqx.filter(self.model, eqx.is_array)
        target_model_params, target_model_static = eqx.partition(self.target_model, eqx.is_array)

        new_target_params = optax.incremental_update(model_params, target_model_params, tau)
        return dataclasses.replace(
            self,
            target_model=eqx.combine(new_target_params, target_model_static)
        )


class Critic(eqx.Module):
    layers: nn.Sequential

    def __init__(self, obs_dim, action_dim, hidden_dim, *, key):
        keys = jax.random.split(key, num=4)
        self.layers = nn.Sequential([
            nn.Linear(obs_dim + action_dim, hidden_dim, key=keys[0]),
            nn.Lambda(jax.nn.relu),
            nn.Linear(hidden_dim, hidden_dim, key=keys[1]),
            nn.Lambda(jax.nn.relu),
            nn.Linear(hidden_dim, hidden_dim, key=keys[2]),
            nn.Lambda(jax.nn.relu),
            nn.Linear(hidden_dim, 1, key=keys[3])
        ])

    def __call__(self, obs, action):
        state_action = jnp.concatenate([obs, action], axis=-1)
        out = self.layers(state_action).squeeze(-1)
        return out


class Actor(eqx.Module):
    layers: nn.Sequential

    def __init__(self, obs_dim, action_dim, hidden_dim, *, key):
        keys = jax.random.split(key, num=4)
        self.layers = nn.Sequential([
            nn.Linear(obs_dim, hidden_dim, key=keys[0]),
            nn.Lambda(jax.nn.relu),
            nn.Linear(hidden_dim, hidden_dim, key=keys[1]),
            nn.Lambda(jax.nn.relu),
            nn.Linear(hidden_dim, hidden_dim, key=keys[2]),
            nn.Lambda(jax.nn.relu),
            nn.Linear(hidden_dim, action_dim * 2, key=keys[3])
        ])

    def __call__(self, obs):
        mu, log_sigma = jnp.split(self.layers(obs), 2, axis=-1)
        # clipping params from EDAC paper, not as in SAC paper (-20, 2)
        log_sigma = jnp.clip(log_sigma, -5, 2)
        dist = FixedDistrax(TanhNormal, mu, jnp.exp(log_sigma))
        return dist


class Alpha(eqx.Module):
    value: jax.Array

    def __init__(self, init_value=1.0):
        self.value = jnp.array([jnp.log(init_value)])

    def __call__(self):
        return jnp.exp(self.value)


@eqx.filter_vmap(in_axes=dict(obs=None, action=None), out_axes=0)
def ensemble_predict(ensemble, obs, action):
    return eqx.filter_vmap(ensemble)(obs, action)


def update_actor(
        key: jax.random.PRNGKey,
        actor: TrainState,
        critic: TrainState,
        alpha: TrainState,
        batch: Dict[str, jax.Array]
) -> Tuple[TrainState, Dict[str, Any]]:
    def actor_loss_fn(actor):
        dist = eqx.filter_vmap(actor)(batch["obs"])
        actions, actions_logp = dist.sample_and_log_prob(seed=key)

        q_values = ensemble_predict(critic.model, batch["obs"], actions).min(0)
        loss = (alpha.model() * actions_logp.sum(-1) - q_values).mean()

        batch_entropy = -actions_logp.sum(-1).mean()

        return loss, batch_entropy

    (loss, batch_entropy), grads = eqx.filter_value_and_grad(actor_loss_fn, has_aux=True)(actor.model)
    new_actor = actor.apply_updates(grads)

    info = {
        "batch_entropy": batch_entropy,
        "actor_loss": loss
    }
    return new_actor, info


def update_alpha(
        alpha: TrainState,
        entropy: jax.Array,
        target_entropy: float,
) -> Tuple[TrainState, Dict[str, Any]]:
    def alpha_loss_fn(alpha):
        return (alpha() * (entropy - target_entropy)).mean()

    loss, grads = eqx.filter_value_and_grad(alpha_loss_fn)(alpha.model)
    new_alpha = alpha.apply_updates(grads)
    info = {
        "alpha": alpha.model(),
        "alpha_loss": loss
    }
    return new_alpha, info


def update_critic(
    key: jax.random.PRNGKey,
    actor: TrainState,
    critic: CriticTrainState,
    alpha: TrainState,
    batch: Dict[str, jax.Array],
    gamma: float,
    tau: float,
) -> Tuple[CriticTrainState, Dict[str, Any]]:
    next_actions_dist = eqx.filter_vmap(actor.model)(batch["next_obs"])
    next_actions, next_actions_logp = next_actions_dist.sample_and_log_prob(seed=key)

    next_q = ensemble_predict(critic.target_model, batch["next_obs"], next_actions).min(0)
    next_q = next_q - alpha.model() * next_actions_logp.sum(-1)
    target_q = batch["rewards"] + (1 - batch["dones"]) * gamma * next_q

    def critic_loss_fn(critic):
        q_values = ensemble_predict(critic, batch["obs"], batch["actions"])
        # [num_critics, batch_size] - [1, batch_size]
        loss = ((q_values - target_q[None, ...]) ** 2).mean(1).sum(0)
        return loss

    loss, grads = eqx.filter_value_and_grad(critic_loss_fn)(critic.model)
    new_critic = critic.apply_updates(grads).soft_update(tau)
    info = {
        "critic_loss": loss,
    }
    return new_critic, info


@eqx.filter_jit
def eval_actions_jit(actor: Actor, obs: jax.Array) -> jax.Array:
    dist = actor(obs)
    action = dist.mean()
    return action


def make_env(env_name: str, seed: int) -> gym.Env:
    env = gym.make(env_name)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def evaluate(env: gym.Env, actor: Actor, num_episodes: int, seed: int) -> np.ndarray:
    env.seed(seed)

    returns = []
    for _ in trange(num_episodes, leave=False):
        obs, done = env.reset(), False
        total_reward = 0.0
        while not done:
            action = eval_actions_jit(actor, obs)
            obs, reward, done, _ = env.step(np.asarray(jax.device_get(action)))
            total_reward += reward
        returns.append(total_reward)

    return np.array(returns)


@pyrallis.wrap()
def main(config: Config):
    wandb.init(
        config=asdict(config),
        project=config.project,
        group=config.group,
        name=config.name,
        id=str(uuid.uuid4()),
        save_code=True
    )
    buffer = ReplayBuffer.create_from_d4rl(config.env_name)

    eval_env = make_env(config.env_name, seed=config.eval_seed)
    target_entropy = -np.prod(eval_env.action_space.shape)
    obs_dim = eval_env.observation_space.shape[-1]
    action_dim = eval_env.action_space.shape[-1]

    key = jax.random.PRNGKey(seed=config.train_seed)
    key, actor_key, critic_key = jax.random.split(key, 3)

    actor = TrainState.create(
        model=Actor(obs_dim, action_dim, config.hidden_dim, key=actor_key),
        optim=optax.adam(learning_rate=config.actor_learning_rate)
    )
    alpha = TrainState.create(
        model=Alpha(),
        optim=optax.adam(learning_rate=config.alpha_learning_rate)
    )

    @eqx.filter_vmap
    def init_ensemble(key):
        return Critic(obs_dim, action_dim, config.hidden_dim, key=key)

    critic = CriticTrainState.create(
        model=init_ensemble(jax.random.split(critic_key, config.num_critics)),
        target_model=init_ensemble(jax.random.split(critic_key, config.num_critics)),
        optim=optax.adam(learning_rate=config.critic_learning_rate)
    )

    def update_networks(key, actor, critic, alpha, batch):
        actor_key, critic_key = jax.random.split(key)

        new_actor, actor_info = update_actor(actor_key, actor, critic, alpha, batch)
        new_alpha, alpha_info = update_alpha(alpha, actor_info["batch_entropy"], target_entropy)
        new_critic, critic_info = update_critic(critic_key, new_actor, critic, new_alpha, batch, config.gamma, config.tau)

        return new_actor, new_critic, new_alpha, {**actor_info, **critic_info, **alpha_info}

    @eqx.filter_jit
    def update_epoch(key, actor, critic, alpha, buffer):
        init_actor_params, init_actor_static = eqx.partition(actor, eqx.is_array)
        init_critic_params, init_critic_static = eqx.partition(critic, eqx.is_array)
        init_alpha_params, init_alpha_static = eqx.partition(alpha, eqx.is_array)

        def update_step(carry, _):
            key, actor_params, critic_params, alpha_params, info = carry

            key, update_key, batch_key = jax.random.split(key, 3)
            batch = buffer.sample_batch(batch_key, config.batch_size)

            new_actor, new_critic, new_alpha, new_info = update_networks(
                key=update_key,
                actor=eqx.combine(actor_params, init_actor_static),
                critic=eqx.combine(critic_params, init_critic_static),
                alpha=eqx.combine(alpha_params, init_alpha_static),
                batch=batch
            )
            new_actor_params, _ = eqx.partition(new_actor, eqx.is_array)
            new_critic_params, _ = eqx.partition(new_critic, eqx.is_array)
            new_alpha_params, _ = eqx.partition(new_alpha, eqx.is_array)
            info = jax.tree_map(lambda c, u: c + u, info, new_info)

            return (key, new_actor_params, new_critic_params, new_alpha_params, info), None

        init_info = {
            "critic_loss": jnp.array([0.0]),
            "actor_loss": jnp.array([0.0]),
            "alpha_loss": jnp.array([0.0]),
            "alpha": jnp.array([0.0]),
            "batch_entropy": jnp.array([0.0])
        }
        init_carry = (key, init_actor_params, init_critic_params, init_alpha_params, init_info)

        (key, actor_params, critic_params, alpha_params, info), _ = jax.lax.scan(
            f=update_step,
            init=init_carry,
            xs=None,
            length=config.num_updates_on_epoch
        )
        actor = eqx.combine(actor_params, init_actor_static)
        critic = eqx.combine(critic_params, init_critic_static)
        alpha = eqx.combine(alpha_params, init_alpha_static)

        return key, actor, critic, alpha, info

    for epoch in trange(config.num_epochs):
        key, actor, critic, alpha, info = update_epoch(key, actor, critic, alpha, buffer)

        info = jax.tree_map(lambda v: v.item() / config.num_updates_on_epoch, info)
        wandb.log({"epoch": epoch, **info})

        if epoch % config.eval_every == 0 or epoch == config.num_epochs - 1:
            eval_returns = evaluate(eval_env, actor.model, config.eval_episodes, seed=config.eval_seed)
            normalized_score = eval_env.get_normalized_score(eval_returns) * 100.0

            wandb.log({
                "epoch": epoch,
                "eval/normalized_score_mean": np.mean(normalized_score),
                "eval/normalized_score_std": np.std(normalized_score)
            })


if __name__ == "__main__":
    main()
