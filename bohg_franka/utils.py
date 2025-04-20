import numpy as np

class FrankaRemapper:

    def __init__(self, remap_config):
        self.image_remapping = remap_config["image"]
        self.proprio_remapping = remap_config["proprio"]
        self.action_remapping = remap_config["action"]
        self.language_instruction = remap_config["language_instruction"]

    def remap_observation(self, observation):
        remapped_observation = {}
        for key, value in self.image_remapping.items():
            remapped_observation[key] = observation[value]
        remapped_observation["state"] = np.hstack([observation[key] for key in self.proprio_remapping])
        return remapped_observation
    
    def remap_action(self, action):
        remapped_action = {}
        remapped_action["action"] = np.hstack([action[key] for key in self.action_remapping])
        return remapped_action
    
    def remap_episode(self, episode):
        remapped_episode = self.remap_observation(episode)
        remapped_episode.update(self.remap_action(episode))
        return remapped_episode
    