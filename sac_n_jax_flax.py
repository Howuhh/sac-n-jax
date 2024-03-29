import copy
import gym
import d4rl
import pyrallis
import numpy as np
import wandb
import uuid

import jax
import chex
import optax
import distrax
import jax.numpy as jnp

import flax
import flax.linen as nn

from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Any
from tqdm.auto import trange

from flax.training.train_state import TrainState


@dataclass
class Config:
    # wandb params
    project: str = "SAC-N-JAX"
    group: str = "SAC-N"
    name: str = "sac-n-jax-flax"
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


@chex.dataclass(frozen=True)
class ReplayBuffer:
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


class CriticTrainState(TrainState):
    target_params: flax.core.FrozenDict

    def soft_update(self, tau):
        new_target_params = optax.incremental_update(self.params, self.target_params, tau)
        return self.replace(target_params=new_target_params)


# SAC-N networks
class TanhNormal(distrax.Transformed):
    def __init__(self, loc, scale):
        normal_dist = distrax.Normal(loc, scale)
        tanh_bijector = distrax.Tanh()
        super().__init__(distribution=normal_dist, bijector=tanh_bijector)

    def mean(self):
        return self.bijector.forward(self.distribution.mean())


def uniform_init(bound: float):
    def _init(key, shape, dtype):
        return jax.random.uniform(key, shape=shape, minval=-bound, maxval=bound, dtype=dtype)
    return _init


# WARN: only for [-1, 1] action bounds, scaling/unscaling is left as an exercise for the reader :D
class Actor(nn.Module):
    action_dim: int
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, state):
        net = nn.Sequential([
            nn.Dense(self.hidden_dim, kernel_init=nn.initializers.he_uniform(), bias_init=nn.initializers.constant(0.1)),
            nn.relu,
            nn.Dense(self.hidden_dim, kernel_init=nn.initializers.he_uniform(), bias_init=nn.initializers.constant(0.1)),
            nn.relu,
            nn.Dense(self.hidden_dim, kernel_init=nn.initializers.he_uniform(), bias_init=nn.initializers.constant(0.1)),
            nn.relu,
        ])
        log_sigma_net = nn.Dense(self.action_dim, kernel_init=uniform_init(1e-3), bias_init=uniform_init(1e-3))
        mu_net = nn.Dense(self.action_dim, kernel_init=uniform_init(1e-3), bias_init=uniform_init(1e-3))

        trunk = net(state)
        mu, log_sigma = mu_net(trunk), log_sigma_net(trunk)
        log_sigma = jnp.clip(log_sigma, -5, 2)

        dist = TanhNormal(mu, jnp.exp(log_sigma))
        return dist


class Critic(nn.Module):
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, state, action):
        network = nn.Sequential([
            nn.Dense(self.hidden_dim, kernel_init=nn.initializers.he_uniform(), bias_init=nn.initializers.constant(0.1)),
            nn.relu,
            nn.Dense(self.hidden_dim, kernel_init=nn.initializers.he_uniform(), bias_init=nn.initializers.constant(0.1)),
            nn.relu,
            nn.Dense(self.hidden_dim, kernel_init=nn.initializers.he_uniform(), bias_init=nn.initializers.constant(0.1)),
            nn.relu,
            nn.Dense(1, kernel_init=uniform_init(3e-3), bias_init=uniform_init(3e-3))
        ])
        state_action = jnp.hstack([state, action])
        out = network(state_action).squeeze(-1)
        return out


class EnsembleCritic(nn.Module):
    hidden_dim: int = 256
    num_critics: int = 10

    @nn.compact
    def __call__(self, state, action):
        ensemble = nn.vmap(
            target=Critic,
            in_axes=None,
            out_axes=0,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            axis_size=self.num_critics
        )
        q_values = ensemble(self.hidden_dim)(state, action)
        return q_values


class Alpha(nn.Module):
    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        log_alpha = self.param("log_alpha", lambda key: jnp.array([jnp.log(self.init_value)]))
        return jnp.exp(log_alpha)


# SAC-N losses
def update_actor(
        key: jax.random.PRNGKey,
        actor: TrainState,
        critic: TrainState,
        alpha: TrainState,
        batch: Dict[str, jax.Array]
) -> Tuple[TrainState, Dict[str, Any]]:
    def actor_loss_fn(actor_params):
        actions_dist = actor.apply_fn(actor_params, batch["obs"])
        actions, actions_logp = actions_dist.sample_and_log_prob(seed=key)

        q_values = critic.apply_fn(critic.params, batch["obs"], actions).min(0)
        loss = (alpha.apply_fn(alpha.params) * actions_logp.sum(-1) - q_values).mean()

        batch_entropy = -actions_logp.sum(-1).mean()
        return loss, batch_entropy

    (loss, batch_entropy), grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(actor.params)
    new_actor = actor.apply_gradients(grads=grads)
    info = {
        "batch_entropy": batch_entropy,
        "actor_loss": loss
    }
    return new_actor, info


def update_alpha(
        alpha: TrainState,
        entropy: float,
        target_entropy: float
) -> Tuple[TrainState, Dict[str, Any]]:
    def alpha_loss_fn(alpha_params):
        alpha_value = alpha.apply_fn(alpha_params)
        loss = (alpha_value * (entropy - target_entropy)).mean()
        return loss

    loss, grads = jax.value_and_grad(alpha_loss_fn)(alpha.params)
    new_alpha = alpha.apply_gradients(grads=grads)
    info = {
        "alpha": alpha.apply_fn(alpha.params),
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
) -> Tuple[TrainState, Dict[str, Any]]:
    next_actions_dist = actor.apply_fn(actor.params, batch["next_obs"])
    next_actions, next_actions_logp = next_actions_dist.sample_and_log_prob(seed=key)

    next_q = critic.apply_fn(critic.target_params, batch["next_obs"], next_actions).min(0)
    next_q = next_q - alpha.apply_fn(alpha.params) * next_actions_logp.sum(-1)
    target_q = batch["rewards"] + (1 - batch["dones"]) * gamma * next_q

    def critic_loss_fn(critic_params):
        # [N, batch_size] - [1, batch_size]
        q = critic.apply_fn(critic_params, batch["obs"], batch["actions"])
        loss = ((q - target_q[None, ...]) ** 2).mean(1).sum(0)
        return loss

    loss, grads = jax.value_and_grad(critic_loss_fn)(critic.params)
    new_critic = critic.apply_gradients(grads=grads).soft_update(tau=tau)
    info = {
        "critic_loss": loss
    }
    return new_critic, info


# evaluation
@jax.jit
def eval_actions_jit(actor: TrainState, obs: jax.Array) -> jax.Array:
    dist = actor.apply_fn(actor.params, obs)
    action = dist.mean()
    return action


def make_env(env_name: str, seed: int) -> gym.Env:
    env = gym.make(env_name)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def evaluate(env: gym.Env, actor: TrainState, num_episodes: int, seed: int) -> np.ndarray:
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

    key = jax.random.PRNGKey(seed=config.train_seed)
    key, actor_key, critic_key, alpha_key = jax.random.split(key, 4)
    init_state = jnp.asarray(eval_env.observation_space.sample())
    init_action = jnp.asarray(eval_env.action_space.sample())

    actor_module = Actor(action_dim=np.prod(eval_env.action_space.shape), hidden_dim=config.hidden_dim)
    actor = TrainState.create(
        apply_fn=actor_module.apply,
        params=actor_module.init(actor_key, init_state),
        tx=optax.adam(learning_rate=config.actor_learning_rate),
    )

    critic_module = EnsembleCritic(hidden_dim=config.hidden_dim, num_critics=config.num_critics)
    critic = CriticTrainState.create(
        apply_fn=critic_module.apply,
        params=critic_module.init(critic_key, init_state, init_action),
        target_params=critic_module.init(critic_key, init_state, init_action),
        tx=optax.adam(learning_rate=config.critic_learning_rate),
    )

    alpha_module = Alpha()
    alpha = TrainState.create(
        apply_fn=alpha_module.apply,
        params=alpha_module.init(alpha_key),
        tx=optax.adam(learning_rate=config.alpha_learning_rate)
    )

    def update_networks(key, actor, critic, alpha, batch):
        actor_key, critic_key = jax.random.split(key)

        new_actor, actor_info = update_actor(actor_key, actor, critic, alpha, batch)
        new_alpha, alpha_info = update_alpha(alpha, actor_info["batch_entropy"], target_entropy)
        new_critic, critic_info = update_critic(critic_key, new_actor, critic, new_alpha, batch, config.gamma, config.tau)

        return new_actor, new_critic, new_alpha, {**actor_info, **critic_info, **alpha_info}

    @jax.jit
    def update_step(_, carry):
        key, update_key, batch_key = jax.random.split(carry["key"], 3)
        batch = carry["buffer"].sample_batch(batch_key, batch_size=config.batch_size)

        actor, critic, alpha, update_info = update_networks(
            key=update_key,
            actor=carry["actor"],
            critic=carry["critic"],
            alpha=carry["alpha"],
            batch=batch,
        )
        update_info = jax.tree_map(lambda c, u: c + u, carry["update_info"], update_info)
        carry.update(key=key, actor=actor, critic=critic, alpha=alpha, update_info=update_info)

        return carry

    update_carry = {
        "key": key,
        "actor": actor,
        "critic": critic,
        "alpha": alpha,
        "buffer": buffer,
    }
    for epoch in trange(config.num_epochs):
        # metrics for accumulation during epoch and logging to wandb, we need to reset them every epoch
        update_carry["update_info"] = {
            "critic_loss": jnp.array([0.0]),
            "actor_loss": jnp.array([0.0]),
            "alpha_loss": jnp.array([0.0]),
            "alpha": jnp.array([0.0]),
            "batch_entropy": jnp.array([0.0])
        }
        update_carry = jax.lax.fori_loop(
            lower=0,
            upper=config.num_updates_on_epoch,
            body_fun=update_step,
            init_val=update_carry
        )
        # log mean over epoch for each metric
        update_info = jax.tree_map(lambda v: v.item() / config.num_updates_on_epoch, update_carry["update_info"])
        wandb.log({"epoch": epoch, **update_info})

        if epoch % config.eval_every == 0 or epoch == config.num_epochs - 1:
            eval_returns = evaluate(eval_env, update_carry["actor"], config.eval_episodes, seed=config.eval_seed)
            normalized_score = eval_env.get_normalized_score(eval_returns) * 100.0

            wandb.log({
                "epoch": epoch,
                "eval/normalized_score_mean": np.mean(normalized_score),
                "eval/normalized_score_std": np.std(normalized_score)
            })


if __name__ == "__main__":
    main()
