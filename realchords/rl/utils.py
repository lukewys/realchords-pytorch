"""Utility functions for RL algorithms."""

from typing import Optional, Union, List, Dict, Tuple, Any
import torch
import torch.nn.functional as F
from pathlib import Path


def is_rl_checkpoint(path: Union[str, Path]) -> bool:
    """Return True if the given path looks like an RL checkpoint.

    Currently we consider RL checkpoints to be PyTorch ``.pth`` files
    (e.g., ``actor.pth`` or ``critic.pth``).
    """
    return ".pth" in str(path)


def compute_full_kl(
    logits: torch.Tensor,
    logits_base: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
    use_reverse_kl: bool = False,
) -> torch.Tensor:
    """
    Compute the full KL divergence between two distributions.

    Args:
        logits: Logits of the new distribution, shape (batch_size, T, num_actions).
        logits_base: Logits of the base distribution, shape (batch_size, T, num_actions).
        action_mask: Mask for actions, shape (batch_size, T, num_actions) or (batch_size, T).
        use_reverse_kl: Whether to compute the reverse KL divergence (KL(Q || P)).

    Returns:
        The computed KL divergence as a tensor.
    """
    # Compute log probabilities for both distributions
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_base = F.log_softmax(logits_base, dim=-1)

    # Apply action mask if provided
    if action_mask is not None:
        if action_mask.dim() == 2:
            action_mask = action_mask.unsqueeze(-1)
        log_probs = log_probs * action_mask
        log_probs_base = log_probs_base * action_mask

    # Compute probabilities from log probabilities
    probs = log_probs.exp()
    probs_base = log_probs_base.exp()

    # Compute KL divergence
    if use_reverse_kl:
        # Reverse KL: KL(Q || P) = ∑ Q(x) * (log(Q(x)) - log(P(x)))
        kl_divergence = torch.sum(
            probs_base * (log_probs_base - log_probs), dim=-1
        )
    else:
        # Forward KL: KL(P || Q) = ∑ P(x) * (log(P(x)) - log(Q(x)))
        kl_divergence = torch.sum(probs * (log_probs - log_probs_base), dim=-1)

    return kl_divergence


def assign_reward_to_last_token(
    r: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    """Assign the reward to the last token of the sequence.

    Args:
        r: (B) tensor of rewards.
        action_mask: (B, S) tensor of action masks.

    Returns:
        (B, S) tensor of rewards with the reward assigned to the last token.
    """
    eos_indices = (
        action_mask.size(1)
        - 1
        - action_mask.long().fliplr().argmax(dim=1, keepdim=True)
    )

    last_reward = (
        torch.zeros_like(action_mask)
        .float()
        .scatter_(dim=1, index=eos_indices, src=r.unsqueeze(1).float())
    )

    return last_reward


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy


class PreparedModels:
    """Container for prepared models with named access."""

    def __init__(self):
        self._trainable: Dict[str, Tuple[Any, Any, Any]] = {}
        self._models: Dict[str, Any] = {}
        self._model_lists: Dict[str, List[Any]] = {}

    def set_trainable(
        self, name: str, model: Any, optimizer: Any, scheduler: Any
    ) -> None:
        """Set a trainable model (model, optimizer, scheduler) tuple."""
        self._trainable[name] = (model, optimizer, scheduler)

    def set_model(self, name: str, model: Any) -> None:
        """Set a single model."""
        self._models[name] = model

    def set_model_list(self, name: str, models: List[Any]) -> None:
        """Set a list of models."""
        self._model_lists[name] = models

    def get_trainable(self, name: str) -> Tuple[Any, Any, Any]:
        """Get a trainable model tuple."""
        if name not in self._trainable:
            raise KeyError(f"Trainable model '{name}' not found")
        return self._trainable[name]

    def get_model(self, name: str) -> Any:
        """Get a single model."""
        if name not in self._models:
            raise KeyError(f"Model '{name}' not found")
        return self._models[name]

    def get_model_list(self, name: str) -> List[Any]:
        """Get a list of models."""
        if name not in self._model_lists:
            raise KeyError(f"Model list '{name}' not found")
        return self._model_lists[name]


class ModelPreparer:
    """Helper class to prepare models/optimizers for distributed training.

    This class provides a cleaner interface for preparing multiple models,
    avoiding manual index tracking and making the code more maintainable.

    Example:
        >>> preparer = ModelPreparer(strategy)
        >>> preparer.add_trainable("actor", actor, actor_optim, actor_scheduler)
        >>> preparer.add_trainable("critic", critic, critic_optim, critic_scheduler)
        >>> preparer.add_model("initial_model", initial_model)
        >>> preparer.add_model_list("reward_models", reward_models)
        >>>
        >>> models = preparer.prepare(is_rlhf=True)
        >>>
        >>> actor, actor_optim, actor_scheduler = models.get_trainable("actor")
        >>> critic, critic_optim, critic_scheduler = models.get_trainable("critic")
        >>> initial_model = models.get_model("initial_model")
        >>> reward_models = models.get_model_list("reward_models")
    """

    def __init__(self, strategy):
        """Initialize the ModelPreparer with a strategy.

        Args:
            strategy: The distributed training strategy (e.g., DeepspeedStrategy).
        """
        self.strategy = strategy
        self._items: List[Tuple[str, str, Any]] = []

    def add_trainable(
        self,
        name: str,
        model: Any,
        optimizer: Any,
        scheduler: Any,
    ) -> "ModelPreparer":
        """Add a trainable model with optimizer and scheduler.

        Args:
            name: Name to identify this trainable model.
            model: The model to prepare.
            optimizer: The optimizer for this model.
            scheduler: The learning rate scheduler for this model.

        Returns:
            Self for method chaining.
        """
        self._items.append(("trainable", name, (model, optimizer, scheduler)))
        return self

    def add_model(self, name: str, model: Any) -> "ModelPreparer":
        """Add a single model (evaluation only, no optimizer).

        Args:
            name: Name to identify this model.
            model: The model to prepare.

        Returns:
            Self for method chaining.
        """
        self._items.append(("model", name, model))
        return self

    def add_model_list(
        self, name: str, models: List[Any]
    ) -> "ModelPreparer":
        """Add a list of models.

        Args:
            name: Name to identify this model list.
            models: List of models to prepare.

        Returns:
            Self for method chaining.
        """
        self._items.append(("model_list", name, models))
        return self

    def prepare(self, is_rlhf: bool = False) -> PreparedModels:
        """Prepare all added models using the strategy.

        Args:
            is_rlhf: Whether this is RLHF training.

        Returns:
            PreparedModels object with named access to prepared models.
        """
        # Build the list of items to prepare
        prepare_items = []
        for item_type, _, item_value in self._items:
            if item_type == "model_list":
                prepare_items.extend(item_value)
            else:
                prepare_items.append(item_value)

        # Call strategy.prepare
        prepared_results = self.strategy.prepare(*prepare_items, is_rlhf=is_rlhf)

        # Unpack results back into named structure
        result = PreparedModels()
        current_idx = 0

        for item_type, name, item_value in self._items:
            if item_type == "trainable":
                result.set_trainable(name, *prepared_results[current_idx])
                current_idx += 1
            elif item_type == "model":
                result.set_model(name, prepared_results[current_idx])
                current_idx += 1
            elif item_type == "model_list":
                num_models = len(item_value)
                models = list(
                    prepared_results[current_idx : current_idx + num_models]
                )
                result.set_model_list(name, models)
                current_idx += num_models

        return result
