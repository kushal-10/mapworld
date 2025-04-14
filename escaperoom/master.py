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

class Player1(Player):
    def __init__(self, model: Model):
        super().__init__(model)

        self.response: str = ""

    def _custom_response(self, context: Dict) -> str:
        return "Test1!"
    
class Player2(Player):
    def __init__(self, model: Model):
        super().__init__(model)

        self.response: str = ""

    def _custom_response(self, context: Dict) -> str:
        return "Test2!"

class EscapeRoom(DialogueGameMaster):

    def __init__(self, name: str, path: str, experiment: Dict, player_models: List[Model]):
        super().__init__(name, path, experiment, player_models)

        self.experiment: str = experiment["name"]
        self.player1_prompt: str = self.game_instance["player1_prompt"]
        self.player2_prompt: str = self.game_instance["player2_prompt"]


        self.player1 = Player1(self.player_models[0])
        self.player2 = Player2(self.player_models[1])


    def _on_before_game(self):
        self.set_context_for(self.player1, self.player1_prompt, {'image': "https://www.ling.uni-potsdam.de/clembench/adk/images/ADE/training/urban/street/ADE_train_00016858.jpg"})

    