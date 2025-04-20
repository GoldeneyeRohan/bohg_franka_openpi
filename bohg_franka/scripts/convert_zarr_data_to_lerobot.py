"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

import shutil
import zarr
import yaml
import tqdm
import numpy as np
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tyro
# result = {
#         “train_idx”: np.where(dataset.train_mask)[0],
#         “val_idx”: np.where(dataset.val_mask)[0],
#         “holdout_idx”: np.where(dataset.holdout_mask)[0],
#     }

class DPEpisodeDataset:
    def __init__(self, path, remap_config, exclude_episodes=None):
        self.path = path
        self._load_dataset()
        self.train_idx = None
        if exclude_episodes is not None:
            self.train_idx = exclude_episodes["train_idx"]

        exclude_episodes = np.load()
        self.image_remapping = remap_config["image"]
        self.proprio_remapping = remap_config["proprio"]
        self.action_remapping = remap_config["action"]
        self.language_instruction = remap_config["language_instruction"]

    def _load_dataset(self):
        with zarr.storage.ZipStore(self.path) as zip_store:
            src_root = zarr.group(zip_store)
            meta = dict()
            for key, value in src_root['meta'].items():
                if len(value.shape) == 0:
                    meta[key] = np.array(value)
                else:
                    meta[key] = value[:]
            self.meta = meta

            keys = src_root['data'].keys()
            data = dict()
            for key in keys:
                arr = src_root['data'][key]
                data[key] = arr[:]
            self.data = data

    @property
    def num_episodes(self):
        return len(self.meta["episode_ends"])
    
    @property
    def episode_ends(self):
        return self.meta["episode_ends"]
    
    def is_valid(self, episode_idx):
        if self.train_idx is not None:
            return episode_idx in self.train_idx
        return True
    
    def aggregate_episode(self, episode):
        agged_episode = {}
        for key, value in self.image_remapping.items():
            agged_episode[key] = episode[value]
        agged_episode["state"] = np.hstack([episode[key] for key in self.proprio_remapping])
        agged_episode["actions"] = np.hstack([episode[key] for key in self.action_remapping]) 
        ep_len = agged_episode["state"].shape[0]
        return agged_episode, ep_len
    
    def get_episode(self, idx, copy=False):
        idx = list(range(len(self.episode_ends)))[idx]
        start_idx = 0
        if idx > 0:
            start_idx = self.episode_ends[idx-1]
        end_idx = self.episode_ends[idx]
        result = self.get_steps_slice(start_idx, end_idx, copy=copy)
        result, ep_len = self.aggregate_episode(result)
        return result, ep_len

    def get_steps_slice(self, start, stop, step=None, copy=False):
        _slice = slice(start, stop, step)

        result = dict()
        for key, value in self.data.items():
            x = value[_slice]
            if copy and isinstance(value, np.ndarray):
                x = x.copy()
            result[key] = x
        return result

def main(data_path:str, output_path:str, task_name:str, remap_config:str, exclude_file:str = None, *, cmd_freq:int = 10, push_to_hub:bool = False):
    # Clean up any existing dataset in the output directory
    output_path = Path(output_path) / task_name
    if output_path.exists():
        shutil.rmtree(output_path)

    with open(remap_config, "r") as f:
        remapping = yaml.safe_load(f)
    if exclude_file is not None:
        exclude_episodes = np.load(exclude_file)
    else:
        exclude_episodes = None
    
    # load the raw dataset:
    print("Loading raw dataset...")
    raw_dataset = DPEpisodeDataset(data_path, remapping, exclude_episodes=exclude_episodes)
    dummy_episode, ep_len = raw_dataset.get_episode(0)
    print(f"Loaded {raw_dataset.num_episodes}.")

    features = {}
    for key, value in dummy_episode.items():
        if key == "state":
            features[key] = {
                "dtype": "float32",
                "shape": (value.shape[1],),
                "names": [key],
            }
        elif key == "actions":
            features[key] = {
                "dtype": "float32",
                "shape": (value.shape[1],),
                "names": [key],
            }
        else:
            features[key] = {
                "dtype": "image",
                "shape": value.shape[1:],
                "names": ["height", "width", "channel"],
            }

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=task_name,
        robot_type="panda",
        fps=cmd_freq,
        features=features,
        image_writer_threads=10,
        image_writer_processes=5,
    )
    print(f"Creating dataset {task_name} with features {features.keys()}...")

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    for episode_idx in tqdm.tqdm(range(raw_dataset.num_episodes)):
        if raw_dataset.is_valid(episode_idx):
            episode, ep_len = raw_dataset.get_episode(episode_idx)
            for i in range(ep_len):
                dataset.add_frame({key: value[i] for key, value in episode.items()})
            dataset.save_episode(task=raw_dataset.language_instruction)

    # Consolidate the dataset, skip computing stats since we will do that later
    dataset.consolidate(run_compute_stats=False)

    # # Optionally push to the Hugging Face Hub
    # if push_to_hub:
    #     dataset.push_to_hub(
    #         tags=["libero", "panda", "rlds"],
    #         private=False,
    #         push_videos=True,
    #         license="apache-2.0",
    #     )


if __name__ == "__main__":
    tyro.cli(main)
