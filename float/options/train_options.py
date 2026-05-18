from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
        parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
        parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
        parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
        parser.add_argument('--optimizer', type=str, default="AdamW", help='optimizer')
        parser.add_argument('--start_from_beginning_ratio', type=float, default=0.2, help='ratio of starting from the beginning')
        parser.add_argument('--vel_loss_weight', type=float, default=1.0, help='weight of velocity loss')
        parser.add_argument('--exp_name', type=str, default="test", help='name of the experiment')
        parser.add_argument('--datasets', type=str, nargs='+', default=["RAVDESS", "HDTF"], help='list of datasets to use')
        parser.add_argument('--save_every_n_epochs', type=int, default=100, help='save every n epochs')
        parser.add_argument('--gc', type=float, default=1.0, help='gradient clipping')
        parser.add_argument('--keep_last_n_checkpoints', type=int, default=2, help='keep last n checkpoints')
        parser.add_argument('--root_dir', type=str, default="./", help='root directory of the dataset')
        parser.add_argument('--use_speaking_scores', action='store_true', help='use speaking scores')
        parser.add_argument('--save_intermediate_video', action='store_true', help='save intermediate video')
        parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='number of gradient accumulation steps')

        # option for reaction scores process
        parser.add_argument('--reaction_loss_weight', type=float, default=0.2, help='weight of reaction loss')
        return parser