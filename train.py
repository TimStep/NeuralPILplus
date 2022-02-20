import os
from typing import Callable, List, Dict

import imageio
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import dataflow.nerd as data
import nn_utils.math_utils as math_utils
import utils.training_setup_utils as train_utils
from models.nerd_net import NerdModel
from nn_utils.nerf_layers import get_full_image_eval_grid
from nn_utils.tensorboard_visualization import hdr_to_tb, horizontal_image_log, to_8b

def add_args(parser):
    parser.add_argument(
        "--log_step",
        type=int,
        default=100,
        help="frequency of tensorboard metric logging",
    )
    parser.add_argument(
        "--weights_epoch", type=int, default=10, help="save weights every x epochs"
    )
    parser.add_argument(
        "--validation_epoch",
        type=int,
        default=5,
        help="render validation every x epochs",
    )
    parser.add_argument(
        "--testset_epoch",
        type=int,
        default=300,
        help="render testset every x epochs",
    )
    parser.add_argument(
        "--video_epoch",
        type=int,
        default=300,
        help="render video every x epochs",
    )

    parser.add_argument(
        "--lrate_decay",
        type=int,
        default=250,
        help="exponential learning rate decay (in 1000s)",
    )

    parser.add_argument("--render_only", action="store_true")

    return parser

def parse_args():
    parser = add_args(
        data.add_args(
            NerdModel.add_args(
                train_utils.setup_parser(),
            ),
        ),
    )
    return train_utils.parse_args_file_without_nones(parser)

def main(args):
# Setup dataflow
        (
            hwf,
            near,
            far,
            render_poses,
            num_images,
            _,
            train_df,
            val_df,
            test_df,
        ) = data.create_dataflow(args)
        
        # Optimizer and models
        with strategy.scope():
            # Setup models
            nerd = NerdModel(num_images, args)
            lrate = train_utils.adjust_learning_rate_to_replica(args)
            if args.lrate_decay > 0:
                lrate = tf.keras.optimizers.schedules.ExponentialDecay(
                    lrate, decay_steps=args.lrate_decay * 1000, decay_rate=0.1
                )
            optimizer = tf.keras.optimizers.Adam(lrate)

            sgs_optimizer = tf.keras.optimizers.Adam(1e-3)

		# Will be 1 magnitude lower after advanced_loss_done steps
        advanced_loss_lambda = tf.Variable(1.0, dtype=tf.float32)
        color_loss_lambda = tf.Variable(1.0, dtype=tf.float32)
        # Run the actual optimization for x epochs

        for epoch in range(start_epoch + 1, args.epochs + 1):
            pbar = tf.keras.utils.Progbar(len(train_df))

            # Iterate over the train dataset
            if not args.render_only:
                with strategy.scope():
                    for dp in train_dist_df:
                        (
                            img_idx,
                            rays_o,
                            rays_d,
                            pose,
                            mask,
                            ev100,
                            wb,
                            wb_ref_image,
                            target,
                        ) = dp

                        advanced_loss_lambda.assign(
                            1 * 0.9 ** (tf.summary.experimental.get_step() / 5000)
                        )  # Starts with 1 goes to 0
                        color_loss_lambda.assign(
                            1 * 0.75 ** (tf.summary.experimental.get_step() / 1500)
                        )  # Starts with 1 goes to 0

                        # Execute train the train step
                        (
                            fine_payload,
                            _,
                            loss_per_replica,
                            coarse_losses_per_replica,
                            fine_losses_per_replica,
                        ) = strategy.run(
                            nerd.train_step,
                            (
                                rays_o,
                                rays_d,
                                pose,
                                near,
                                far,
                                img_idx,
                                ev100,
                                wb_ref_image,
                                wb,
                                optimizer,
                                target,
                                mask,
                                advanced_loss_lambda,
                                color_loss_lambda,
                                (tf.summary.experimental.get_step() < 1000),
                            ),
                        )

                        loss = strategy.reduce(
                            tf.distribute.ReduceOp.SUM, loss_per_replica, axis=None
                        )
                        coarse_losses = {}
                        for k, v in coarse_losses_per_replica.items():
                            coarse_losses[k] = strategy.reduce(
                                tf.distribute.ReduceOp.SUM, v, axis=None
                            )
                        fine_losses = {}
                        for k, v in fine_losses_per_replica.items():
                            fine_losses[k] = strategy.reduce(
                                tf.distribute.ReduceOp.SUM, v, axis=None
                            )

                        losses_for_pbar = [
                            ("loss", loss.numpy()),
                            ("coarse_loss", coarse_losses["loss"].numpy()),
                            ("fine_loss", fine_losses["loss"].numpy()),
                            ("fine_image_loss", fine_losses["image_loss"].numpy()),
                        ]

                        pbar.add(
                            1,
                            values=losses_for_pbar,
                        )
                        
            # Save when a weight epoch arrives
            if epoch % args.weights_epoch == 0:
                nerd.save(
                    tf.summary.experimental.get_step()
                )  # Step was already incremented

if __name__ == "__main__":
    args = parse_args()
    print(args)

    main(args)
