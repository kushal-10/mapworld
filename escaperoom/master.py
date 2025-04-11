"""
Game Master for Escape Room
"""

from clemcore.clemgame import Player, GameMaster, GameBenchmark, DialogueGameMaster, GameScorer, GameSpec
from clemcore.clemgame import metrics as ms
from clemcore.backends import Model
from clemcore.utils import file_utils

from typing import List, Dict, Tuple
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)
stdout_logger = logging.getLogger("escaperoom.master")

class Hunt(Player):
    def __init__(self, model: Model):
        super().__init__(model)

        self.response: str = ""

    def _custom_response(self, context: Dict) -> str:
        return "HALLO!"
    
class Benji(Player):
    def __init__(self, model: Model):
        super().__init__(model)

        self.response: str = ""

    def _custom_response(self, context: Dict) -> str:
        return "SERVUS!"

class EscapeRoom(DialogueGameMaster):

    def __init__(self, name: str, path: str, experiment: Dict, player_models: List[Model]):
        super().__init__(name, path, experiment, player_models)

        self.experiment: str = experiment["name"]
        self.hunt_prompt: str = self.game_instance["hunt_prompt"]
        self.benji_prompt: str = self.game_instance["benji_prompt"]


        self.hunt = Hunt(self.player_models[0])
        self.benji = Benji(self.player_models[1])


    def _on_before_game(self):
        self.set_context_for(self.hunt, self.hunt_prompt, {'image': "https://www.ling.uni-potsdam.de/clembench/adk/images/ADE/training/urban/street/ADE_train_00016858.jpg"})

        