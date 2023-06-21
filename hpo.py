import argparse
import time
import json
import logging
import numpy as np
import torch
import ConfigSpace as CS
from ConfigSpace import Configuration
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    CategoricalHyperparameter,
)
from utils.datasets import VideoDataset
from utils.trainer import Trainer
from datetime import datetime
from dehb import DEHB
from smac.configspace import ConfigurationSpace
from smac.facade.smac_mf_facade import SMAC4MF
from smac.scenario.scenario import Scenario
from torchsummary import summary
from models.aslr import Transformer as T
from utils.free_memory import free_gpu_cache

'''
这个超参数优化（HPO）脚本支持使用两种不同的优化方法：SMAC（Sequential Model-based Algorithm Configuration）
                                            和DEHB（Differential Evolution and Hyperband）。

你可以通过命令行参数 `--hpo_searcher` 来指定使用哪种方法。默认的方法是 SMAC，但你也可以通过将 `--hpo_searcher` 参数设置为 "dehb" 来使用 DEHB 方法。

- SMAC 是一种基于模型的优化方法，它通过构建一个概率模型来预测哪些超参数可能会给出好的结果，然后基于这个模型来选择新的超参数。
        SMAC 还使用了一种叫做 "intensification" 的过程来在探索（尝试新的超参数）和利用（使用已知的好的超参数）之间进行权衡。

- DEHB 是一种结合了差分进化（Differential Evolution）和 Hyperband 的优化方法。差分进化是一种用于实数函数优化的进化算法，
        而 Hyperband 是一种基于多臂老虎机的优化方法，用于在有限的预算下找到最优的超参数。

这两种方法都是为了在有限的计算预算下找到最优的超参数，但它们使用的策略和技术是不同的。
'''

parser = argparse.ArgumentParser(description="Arguments for the HPO script")
parser.add_argument(
    "--multisigner",
    type=bool,
    help="Boolean to specify if we use multisigner data or signer independent data.",
    default=True,
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    help="String to specify which device to train with",
)
parser.add_argument(
    "--hpo_searcher",
    type=str,
    default="smac",
    help="String to specify which HPO method to use",
)
parser.add_argument(
    "--seed",
    type=int,
    help="Integer for reproducibility",
    default=2022
)
parser.add_argument(
    "--runtime",
    type=int,
    help="Integer to specify how long we search",
    default=604800
)
args = parser.parse_args()

class HPO:
    """
    Class to perform hyperparameter Optimization.
    """

    def __init__(
            self,
            seed: int = 2022,
            method: str = "smac",
            multisigner: bool = True
    ):
        """
        Initializes the constructor
        :param seed: SEED to use
        :param multisigner: Bool to indicate which split to use
        """
        super(HPO, self).__init__()
        self.seed = seed
        rng = np.random.default_rng(self.seed)
        self.to_use = "multi" if multisigner else "si5"
        self.padding = 1296 if multisigner else 1136
        self.trg_vocab = 1297 if multisigner else 1137
        self.training_features = np.load(self.to_use + "_training_features.npy", allow_pickle=True)
        self.training_label = np.load(self.to_use + "_training_label.npy", allow_pickle=True)
        self.evaluation_features = np.load(self.to_use + "_validation_features.npy", allow_pickle=True)
        self.evaluation_label = np.load(self.to_use + "_validation_label.npy", allow_pickle=True)
        self.vocab = np.load("full_gloss_" + self.to_use + ".npy", allow_pickle=True)
        self.vocab = dict(enumerate(self.vocab.flatten()))
        self.vocab = self.vocab[0]
        self.method = method
        print("Search method is " + self.method + " on " + self.to_use + " data.")

    def __transformer_from_cfg__(self, cfg: Configuration, budget: float):
        """
        Creates an instance of the transformer model and fits the given data on it.
        This is the function-call we try to optimize. Chosen values are stored in
        the configuration (cfg).

        :param cfg: Configuration (basically a dictionary)
            configuration chosen by smac
        :param kwargs: dict
            used to set other parameters value
        :param budget: float
            used to set max iterations for the MLP
        Returns
        -------
        wer :Word Error Rate or/and runtime
        """
        learning_rate = cfg["learning_rate"] if cfg["learning_rate"] else 0.17
        batch_size = cfg["batch_size"] if cfg["batch_size"] else 4
        dropout_rate = cfg["dropout"] if cfg["dropout"] else 0.2
        num_layers = cfg["num_layers"] if cfg["num_layers"] else 6
        device = cfg["device"] if cfg["device"] else "cuda"
        torch.manual_seed(seed=self.seed)

        train_data = VideoDataset(
            feature_tab=self.training_features, label=self.training_label
        )
        val_data = VideoDataset(
            feature_tab=self.training_features, label=self.training_label
        )
        num_epochs = int(np.ceil(budget))
        model = T(
            device=torch.device(str(device).lower()),
            embed_dim=512,
            seq_length_en=300,
            seq_length_dec=300,
            src_vocab_size=300,
            target_vocab_size=self.trg_vocab,
            num_layers=num_layers,
            expansion_factor=4,
            n_heads=8,
            dropout_rate=dropout_rate,
        )
        if torch.cuda.device_count() > 1 and device.lower() == "cuda":
            # You may need to adapt this to your GPU environment
            dev_arr = np.arange(torch.cuda.device_count())
            visible_device = ""
            for dev in dev_arr:
                visible_device = visible_device + str(dev) + ","
            visible_device = visible_device[:-1]
            # os.environ["CUDA_VISIBLE_DEVICES"] = visible_device
            print("We will be training on {0} GPUs with IDs" + visible_device)
            model = torch.nn.DataParallel(model, device_ids=[0, 1])
        summary(model, device=torch.device(str(device).lower()))
        log_train_loss, log_val_loss = [], []
        old_val_loss = None
        print(
            "Used Hyperparameters: Learning rate is {0}, batch size is {1}, dropout rate is {2} "
            "and the number of layers is {3}. Model trained on {4}".format(
                learning_rate, batch_size, dropout_rate, num_layers, device,
            )
        )
        trainer = Trainer(
            model=model,
            training_set=train_data,
            validation_set=val_data,
            epochs=num_epochs,
            loss="cross entropy",
            optimizer="sgd",
            evaluate_during_training=False,
            learning_rate=learning_rate,
            batch_size=batch_size,
            device=torch.device(device),
            save_epoch=False,
            padding_character=self.padding
        )
        now = datetime.now()
        start_time = datetime.now().timestamp()
        print("Beginning time : ", now)
        trainer.train(vocab=self.vocab, remove_pad=True)
        now = datetime.now()
        print("Ending time : ", now)
        l, w = trainer.evaluate(vocab=self.vocab, remove_pad=True)
        print("WER: ", w)
        final = datetime.now().timestamp() - start_time
        res = {"fitness": w, "cost": final}
        if self.method.lower() == "dehb":
            free_gpu_cache()
            return res
        else:
            free_gpu_cache()
            return w

    def search(
            self,
            device,
            working_dir: str = "./tmp_slr",
            runtime: int = 604800,
            max_epochs: int = 100,
    ):
        logger = logging.getLogger("Optimization")
        logging.basicConfig(level=logging.INFO)
        cs = ConfigurationSpace()
        learning_rate = UniformFloatHyperparameter(
            "learning_rate", 0.01, 1.0, default_value=0.17, log=True,
        )
        dropout = UniformFloatHyperparameter("dropout", 0.2, 0.5, default_value=0.2, )
        num_layers = UniformIntegerHyperparameter("num_layers", 6, 8, default_value=6)

        batch_size = UniformIntegerHyperparameter("batch_size", 1, 4, default_value=1)
        device_hp = CategoricalHyperparameter("device", choices=[device])
        cs.add_hyperparameters(
            [learning_rate, dropout, num_layers, batch_size, device_hp]
        )
        scenario = Scenario(
            {
                "run_obj": "quality",
                "wallclock-limit": runtime,
                "cs": cs,
                "output-dir": working_dir,
                "deterministic": "True",
                "algo_runs_timelimit": runtime,
                "cutoff": 1000,
            }
        )

        intensifier_kwargs = {"initial_budget": 5, "max_budget": max_epochs, "eta": 3}
        np.random.seed(seed=self.seed)
        if self.method.lower() == "dehb":
            dehb = DEHB(
                f=self.__transformer_from_cfg__,
                cs=cs,
                dimensions=len(cs.get_hyperparameters()),
                min_budget=intensifier_kwargs["initial_budget"],
                max_budget=max_epochs,
                eta=intensifier_kwargs["eta"],
                output_path="tmp_dehb",
                seed=self.seed,
                # if client is not None and of type Client, n_workers is ignored
                # if client is None, a Dask client with n_workers is set up
                client=None,
                n_workers=1,
            )
            traj, runtime, history = dehb.run(
                total_cost=runtime, verbose=True, seed=self.seed
            )
            incumbent = dehb.vector_to_configspace(dehb.inc_config)
            opt_config = incumbent.get_dictionary()
            with open("dehb_opt_cfg.json", "w") as f:
                json.dump(opt_config, f)
            return opt_config
        elif self.method.lower() == "smac":
            smac = SMAC4MF(
                scenario=scenario,
                rng=np.random.RandomState(seed=self.seed),
                tae_runner=self.__transformer_from_cfg__,
                intensifier_kwargs=intensifier_kwargs,
                initial_design_kwargs={
                    "n_configs_x_params": 3,
                    "max_config_fracs": 0.2,
                },
            )
            try:
                incumbent = smac.optimize()
            finally:
                incumbent = smac.solver.incumbent

            opt_config = incumbent.get_dictionary()
            with open("smac_opt_cfg.json", "w") as f:
                json.dump(opt_config, f)
            return opt_config, smac.stats.ta_time_used
        else:
            raise ValueError("Method not defined. Please give smac or dehb")


if __name__ == "__main__":
    hpo = HPO(seed=args.seed, multisigner=args.multisigner, method=args.hpo_searcher)
    opt, time = hpo.search(device="cuda", runtime=args.runtime)
