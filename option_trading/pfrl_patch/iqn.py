import numpy as np
import torch
from torch import nn

from typing import Any, Sequence

from pfrl.action_value import QuantileDiscreteActionValue
from pfrl.agents import dqn
from pfrl.nn import Recurrent
from pfrl.utils.contexts import evaluating
from pfrl.utils.recurrent import one_step_forward, pack_and_forward


def cosine_basis_functions(x, n_basis_functions=64):
    """Cosine basis functions used to embed quantile thresholds.

    Args:
        x (torch.Tensor): Input.
        n_basis_functions (int): Number of cosine basis functions.

    Returns:
        ndarray: Embedding with shape of (x.shape + (n_basis_functions,)).
    """
    # Equation (4) in the IQN paper has an error stating i=0,...,n-1.
    # Actually i=1,...,n is correct (personal communication)
    i_pi = (
        torch.arange(1, n_basis_functions + 1, dtype=torch.float, device=x.device)
        * np.pi
    )
    embedding = torch.cos(x[..., None] * i_pi)
    assert embedding.shape == x.shape + (n_basis_functions,)
    return embedding


class CosineBasisLinear(nn.Module):

    """Linear layer following cosine basis functions.

    Args:
        n_basis_functions (int): Number of cosine basis functions.
        out_size (int): Output size.
    """

    def __init__(self, n_basis_functions, out_size):
        super().__init__()
        self.linear = nn.Linear(n_basis_functions, out_size)
        self.n_basis_functions = n_basis_functions
        self.out_size = out_size

    def forward(self, x):
        """Evaluate.

        Args:
            x (torch.Tensor): Input.

        Returns:
            torch.Tensor: Output with shape of (x.shape + (out_size,)).
        """
        h = cosine_basis_functions(x, self.n_basis_functions)
        h = h.reshape(-1, self.n_basis_functions)
        out = self.linear(h)
        out = out.reshape(*x.shape, self.out_size)
        return out


def _evaluate_psi_x_with_quantile_thresholds(psi_x, phi, f, taus):
    assert psi_x.ndim == 2
    batch_size, hidden_size = psi_x.shape
    assert taus.ndim == 2
    assert taus.shape[0] == batch_size
    n_taus = taus.shape[1]
    phi_taus = phi(taus)
    assert phi_taus.ndim == 3
    assert phi_taus.shape == (batch_size, n_taus, hidden_size)
    h = psi_x.unsqueeze(1) * phi_taus
    h = h.reshape(-1, hidden_size)
    assert h.shape == (batch_size * n_taus, hidden_size)
    h = f(h)
    assert h.ndim == 2
    assert h.shape[0] == batch_size * n_taus
    n_actions = h.shape[-1]
    h = h.reshape(batch_size, n_taus, n_actions)
    return QuantileDiscreteActionValue(h)


class ImplicitQuantileQFunction(nn.Module):

    """Implicit quantile network-based Q-function.

    Args:
        psi (torch.nn.Module): Callable module
            (batch_size, obs_size) -> (batch_size, hidden_size).
        phi (torch.nn.Module): Callable module
            (batch_size, n_taus) -> (batch_size, n_taus, hidden_size).
        f (torch.nn.Module): Callable module
            (batch_size * n_taus, hidden_size)
            -> (batch_size * n_taus, n_actions).

    Returns:
        QuantileDiscreteActionValue: Action values.
    """

    def __init__(self, psi, phi, f):
        super().__init__()
        self.psi = psi
        self.phi = phi
        self.f = f

    def forward(self, x):
        """Evaluate given observations.

        Args:
            x (torch.Tensor): Batch of observations.
        Returns:
            callable: (batch_size, taus) -> (batch_size, taus, n_actions)
        """
        batch_size = x.shape[0]
        psi_x = self.psi(x)
        assert psi_x.ndim == 2
        assert psi_x.shape[0] == batch_size

        def evaluate_with_quantile_thresholds(taus):
            return _evaluate_psi_x_with_quantile_thresholds(
                psi_x, self.phi, self.f, taus
            )

        return evaluate_with_quantile_thresholds


class RecurrentImplicitQuantileQFunction(Recurrent, nn.Module):

    """Recurrent implicit quantile network-based Q-function.

    Args:
        psi (torch.nn.Module): Module that implements
            `pfrl.nn.Recurrent`.
            (batch_size, obs_size) -> (batch_size, hidden_size).
        phi (torch.nn.Module): Callable module
            (batch_size, n_taus) -> (batch_size, n_taus, hidden_size).
        f (torch.nn.Module): Callable module
            (batch_size * n_taus, hidden_size)
            -> (batch_size * n_taus, n_actions).

    Returns:
        ImplicitQuantileDiscreteActionValue: Action values.
    """

    def __init__(self, psi, phi, f):
        super().__init__()
        self.psi = psi
        self.phi = phi
        self.f = f

    def forward(self, x, recurrent_state):
        """Evaluate given observations.

        Args:
            x (object): Batched sequences of observations.
            recurrent_state (object): Batched recurrent states.

        Returns:
            callable: (batch_size, taus) -> (batch_size, taus, n_actions)
            object: new recurrent states
        """
        psi_x, recurrent_state = self.psi(x, recurrent_state)
        # unwrap PackedSequence
        assert isinstance(psi_x, nn.utils.rnn.PackedSequence)
        psi_x = psi_x.data
        assert psi_x.ndim == 2

        def evaluate_with_quantile_thresholds(taus):
            return _evaluate_psi_x_with_quantile_thresholds(
                psi_x, self.phi, self.f, taus
            )

        return evaluate_with_quantile_thresholds, recurrent_state


def compute_eltwise_huber_quantile_loss(y, t, taus):
    """Compute elementwise Huber losses for quantile regression.

    This is based on Algorithm 1 of https://arxiv.org/abs/1806.06923.

    This function assumes that, both of the two kinds of quantile thresholds,
    taus (used to compute y) and taus_prime (used to compute t) are iid samples
    from U([0,1]).

    Args:
        y (torch.Tensor): Quantile prediction from taus as a
            (batch_size, N)-shaped array.
        t (torch.Tensor or ndarray): Target values for quantile regression
            as a (batch_size, N_prime)-array.
        taus (ndarray): Quantile thresholds used to compute y as a
            (batch_size, N)-shaped array.

    Returns:
        torch.Tensor: Loss (batch_size, N, N_prime)
    """
    assert y.shape == taus.shape
    # (batch_size, N) -> (batch_size, N, 1)
    y = y.unsqueeze(2)
    # (batch_size, N_prime) -> (batch_size, 1, N_prime)
    t = t.unsqueeze(1)
    # (batch_size, N) -> (batch_size, N, 1)
    taus = taus.unsqueeze(2)
    # Broadcast to (batch_size, N, N_prime)
    y, t, taus = torch.broadcast_tensors(y, t, taus)
    I_delta_lt_0 = (t < y).float()
    eltwise_huber_loss = nn.functional.smooth_l1_loss(y, t, reduction="none")
    eltwise_loss = torch.abs(taus - I_delta_lt_0) * eltwise_huber_loss
    return eltwise_loss


def compute_value_loss(eltwise_loss, batch_accumulator="mean"):
    """Compute a loss for value prediction problem.

    Args:
        eltwise_loss (Variable): Element-wise loss per example
        batch_accumulator (str): 'mean' or 'sum'. 'mean' will use the mean of
            the loss values in a batch. 'sum' will use the sum.
    Returns:
        (Variable) scalar loss
    """
    assert batch_accumulator in ("mean", "sum")
    assert eltwise_loss.ndim == 3

    if batch_accumulator == "sum":
        # mean over N_prime, then sum over (batch_size, N)
        loss = eltwise_loss.mean(2).sum()
    else:
        # mean over (batch_size, N_prime), then sum over N
        loss = eltwise_loss.mean((0, 2)).sum()

    return loss


def compute_weighted_value_loss(eltwise_loss, weights, batch_accumulator="mean"):
    """Compute a loss for value prediction problem.

    Args:
        eltwise_loss (Variable): Element-wise loss per example
        weights (ndarray): Weights for y, t.
        batch_accumulator (str): 'mean' will divide loss by batchsize
    Returns:
        (Variable) scalar loss
    """
    batch_size = eltwise_loss.shape[0]
    assert batch_accumulator in ("mean", "sum")
    assert eltwise_loss.ndim == 3
    # eltwise_loss is (batchsize, n , n') array of losses
    # weights is an array of shape (batch_size)
    # apply weights per example in batch
    loss_sum = torch.matmul(eltwise_loss.mean(2).sum(1), weights)
    if batch_accumulator == "mean":
        loss = loss_sum / batch_size
    elif batch_accumulator == "sum":
        loss = loss_sum
    return loss


class IQN(dqn.DQN):

    """Implicit Quantile Networks.

    See https://arxiv.org/abs/1806.06923.

    Args:
        quantile_thresholds_N (int): Number of quantile thresholds used in
            quantile regression.
        quantile_thresholds_N_prime (int): Number of quantile thresholds used
            to sample from the return distribution at the next state.
        quantile_thresholds_K (int): Number of quantile thresholds used to
            compute greedy actions.
        act_deterministically (bool): IQN's action selection is by default
            stochastic as it samples quantile thresholds every time it acts,
            even for evaluation. If this option is set to True, it uses
            equally spaced quantile thresholds instead of randomly sampled ones
            for evaluation, making its action selection deterministic.

    For other arguments, see pfrl.agents.DQN.
    """

    def __init__(self, *args, **kwargs):
        # N=N'=64 and K=32 were used in the IQN paper's experiments
        # (personal communication)
        self.quantile_thresholds_N = kwargs.pop("quantile_thresholds_N", 64)
        self.quantile_thresholds_N_prime = kwargs.pop("quantile_thresholds_N_prime", 64)
        self.quantile_thresholds_K = kwargs.pop("quantile_thresholds_K", 32)
        self.act_deterministically = kwargs.pop("act_deterministically", False)
        self.cvar_alpha = kwargs.pop('cvar_alpha',1.0)
        self.cvar_static = kwargs.pop('cvar_static',False)
        super().__init__(*args, **kwargs)

        self.cutoff_threshold=None
        self.z_count=0
        self.z0_count=0
        print('alpha',self.cvar_alpha)
        print('static',self.cvar_static)

    def _compute_target_values(self, exp_batch):
        """Compute a batch of target return distributions.

        Returns:
            torch.Tensor: (batch_size, N_prime).
        """
        batch_next_state = exp_batch["next_state"]
        batch_size = len(exp_batch["reward"])
        taus_tilde = torch.rand(
            batch_size,
            self.quantile_thresholds_K,
            device=self.device,
            dtype=torch.float,
        )

        if self.recurrent:
            target_next_tau2av, _ = pack_and_forward(
                self.target_model,
                batch_next_state,
                exp_batch["next_recurrent_state"],
            )
        else:
            target_next_tau2av = self.target_model(batch_next_state)



            
        if self.cvar_static: # and False:
            av=target_next_tau2av(taus_tilde)
            av.quantiles=torch.clamp(av.quantiles-exp_batch['cutoff'].view(-1,1,1),max=0)
            greedy_actions = av.greedy_actions
        else:
            greedy_actions = target_next_tau2av(taus_tilde*self.cvar_alpha).greedy_actions
        taus_prime = torch.rand(
            batch_size,
            self.quantile_thresholds_N_prime,
            device=self.device,
            dtype=torch.float,
        )
        target_next_maxz = target_next_tau2av(taus_prime).evaluate_actions_as_quantiles(
            greedy_actions
        )

        batch_rewards = exp_batch["reward"]
        batch_terminal = exp_batch["is_state_terminal"]
        batch_discount = exp_batch["discount"]
        assert batch_rewards.shape == (batch_size,)
        assert batch_terminal.shape == (batch_size,)
        assert batch_discount.shape == (batch_size,)
        batch_rewards = batch_rewards.unsqueeze(-1)
        batch_terminal = batch_terminal.unsqueeze(-1)
        batch_discount = batch_discount.unsqueeze(-1)

        return (
            batch_rewards + batch_discount * (1.0 - batch_terminal) * target_next_maxz
        )

    def _compute_y_and_taus(self, exp_batch):
        """Compute a batch of predicted return distributions.

        Returns:
            torch.Tensor: Predicted return distributions.
                (batch_size, N).
        """

        batch_size = exp_batch["reward"].shape[0]

        # Compute Q-values for current states
        batch_state = exp_batch["state"]

        # (batch_size, n_actions, n_atoms)
        if self.recurrent:
            tau2av, _ = pack_and_forward(
                self.model,
                batch_state,
                exp_batch["recurrent_state"],
            )
        else:
            tau2av = self.model(batch_state)
        taus = torch.rand(
            batch_size,
            self.quantile_thresholds_N,
            device=self.device,
            dtype=torch.float,
        )
        av = tau2av(taus)
        batch_actions = exp_batch["action"]
        y = av.evaluate_actions_as_quantiles(batch_actions)

        self.q_record.extend(av.q_values.detach().cpu().numpy().ravel())

        return y, taus

    def _compute_loss(self, exp_batch, errors_out=None):
        """Compute a loss.

        Returns:
            Returns:
                torch.Tensor: Scalar loss.
        """
        y, taus = self._compute_y_and_taus(exp_batch)
        with torch.no_grad():
            t = self._compute_target_values(exp_batch)

        eltwise_loss = compute_eltwise_huber_quantile_loss(y, t, taus)
        if errors_out is not None:
            del errors_out[:]
            with torch.no_grad():
                delta = eltwise_loss.mean((1, 2))
                errors_out.extend(delta.detach().cpu().numpy())

        if "weights" in exp_batch:
            return compute_weighted_value_loss(
                eltwise_loss,
                exp_batch["weights"],
                batch_accumulator=self.batch_accumulator,
            )
        else:
            return compute_value_loss(
                eltwise_loss, batch_accumulator=self.batch_accumulator
            )

    def _evaluate_model_and_update_recurrent_states(self, batch_obs):
        self.z_count+=1
        
        batch_xs = self.batch_states(batch_obs, self.device, self.phi)
        if self.recurrent:
            if self.training:
                self.train_prev_recurrent_states = self.train_recurrent_states
                tau2av, self.train_recurrent_states = one_step_forward(
                    self.model, batch_xs, self.train_recurrent_states
                )
            else:
                tau2av, self.test_recurrent_states = one_step_forward(
                    self.model, batch_xs, self.test_recurrent_states
                )
        else:
            tau2av = self.model(batch_xs)
        if not self.training and self.act_deterministically:
            # Instead of uniform sampling, use a deterministic sequence of
            # equally spaced numbers from 0 to 1 as quantile thresholds.
            taus_tilde = torch.linspace(
                start=0,
                end=1,
                steps=self.quantile_thresholds_K,
                device=self.device,
                dtype=torch.float,
            ).repeat(len(batch_obs), 1)
        else:
            taus_tilde = torch.rand(
                len(batch_obs),
                self.quantile_thresholds_K,
                device=self.device,
                dtype=torch.float,
            )
            
            
        if self.cutoff_threshold is None:
            self.z0_count+=1
            if self.cvar_static:
                self.cutoff_thresholds=tau2av(torch.tensor([[self.cvar_alpha]],device=self.device,dtype=torch.float)).quantiles[0,0].detach().cpu()
            return tau2av(taus_tilde*self.cvar_alpha)
        else:
            av=tau2av(taus_tilde)
            av.quantiles=torch.clamp(av.quantiles-self.cutoff_threshold,max=0)
            return av

    
    def batch_act(self, batch_obs: Sequence[Any]) -> Sequence[Any]:
        with torch.no_grad(), evaluating(self.model):
            batch_av = self._evaluate_model_and_update_recurrent_states(batch_obs)
            batch_argmax = batch_av.greedy_actions.detach().cpu().numpy()
            if self.cvar_static and self.cutoff_threshold is None:
                self.cutoff_threshold=self.cutoff_thresholds[batch_argmax.item()].item()
                    

        if self.training:
            batch_action = [
                self.explorer.select_action(
                    self.t,
                    lambda: batch_argmax[i],
                    action_value=batch_av[i : i + 1],
                )
                for i in range(len(batch_obs))
            ]
            self.batch_last_obs = list(batch_obs)
            self.batch_last_action = list(batch_action)
        else:
            batch_action = batch_argmax
        return batch_action




    def _batch_observe_train(
        self,
        batch_obs: Sequence[Any],
        batch_reward: Sequence[float],
        batch_done: Sequence[bool],
        batch_reset: Sequence[bool],
    ) -> None:

        for i in range(len(batch_obs)):
            self.t += 1
            self._cumulative_steps += 1
            # Update the target network
            if self.t % self.target_update_interval == 0:
                self.sync_target_network()
            if self.batch_last_obs[i] is not None:
                assert self.batch_last_action[i] is not None
                # Add a transition to the replay buffer
                transition = {
                    "state": self.batch_last_obs[i],
                    "action": self.batch_last_action[i],
                    "reward": batch_reward[i],
                    "next_state": batch_obs[i],
                    "next_action": None,
                    "is_state_terminal": batch_done[i],
                }
                if self.cvar_static:
                    transition["cutoff"]=self.cutoff_threshold
                if self.recurrent:
                    transition["recurrent_state"] = recurrent_state_as_numpy(
                        get_recurrent_state_at(
                            self.train_prev_recurrent_states, i, detach=True
                        )
                    )
                    transition["next_recurrent_state"] = recurrent_state_as_numpy(
                        get_recurrent_state_at(
                            self.train_recurrent_states, i, detach=True
                        )
                    )
                self.replay_buffer.append(env_id=i, **transition)
                if batch_reset[i] or batch_done[i]:
                    self.batch_last_obs[i] = None
                    self.batch_last_action[i] = None
                    self.replay_buffer.stop_current_episode(env_id=i)
            self.replay_updater.update_if_necessary(self.t)

        if self.recurrent:
            # Reset recurrent states when episodes end
            self.train_prev_recurrent_states = None
            self.train_recurrent_states = (
                _batch_reset_recurrent_states_when_episodes_end(  # NOQA
                    batch_done=batch_done,
                    batch_reset=batch_reset,
                    recurrent_states=self.train_recurrent_states,
                )
            )

            
    def batch_observe(
        self,
        batch_obs: Sequence[Any],
        batch_reward: Sequence[float],
        batch_done: Sequence[bool],
        batch_reset: Sequence[bool],
    ) -> None:

        if self.training:
            self._batch_observe_train(
                batch_obs, batch_reward, batch_done, batch_reset
            )
        else:
            self._batch_observe_eval(
                batch_obs, batch_reward, batch_done, batch_reset
            )

        # if self.z_count>10000:
        #     import ipdb; ipdb.set_trace()
            
        if self.cvar_static:
            for i in range(len(batch_obs)):
                if batch_reset[i] or batch_done[i]:
                    self.cutoff_threshold=None
                else:
                    self.cutoff_threshold=(self.cutoff_threshold-batch_reward[i])/self.gamma
        
        
