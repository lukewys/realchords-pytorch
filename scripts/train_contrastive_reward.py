"""Training of encoder decoder model."""

import argbind

from functools import partial
from pathlib import Path

from realchords.base_trainer import Trainer
from realchords.lit_module.contrastive_reward import LitContrastiveReward

GROUP = __file__
# Binding things only when this file is loaded
bind = partial(argbind.bind, group=GROUP)

Trainer = bind(Trainer, without_prefix=True)


@bind(without_prefix=True)
def main(args, save_dir: str = ""):
    # Create the lit module
    lit_module = LitContrastiveReward()

    # Get the dataloaders
    train_dataloader, val_dataloader = lit_module.get_dataloaders()

    # Train the model
    trainer = Trainer(
        args=args,
        lit_module=lit_module,
        save_dir=save_dir,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
    )
    trainer.train()


if __name__ == "__main__":
    args = argbind.parse_args(group=GROUP)
    argbind.dump_args(args, Path(args["save_dir"]) / "args.yml")
    with argbind.scope(args):
        main(args)
