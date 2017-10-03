
from datetime import datetime
import time
import gc
import collections
import os

import numpy as np
import tensorflow as tf
import offshoot

import serpent.cv
from serpent.game_agent import GameAgent
from serpent.machine_learning.context_classification.context_classifiers.cnn_inception_v3_context_classifier import CNNInceptionV3ContextClassifier
from serpent.machine_learning.reinforcement_learning.ddqn import DDQN
from serpent.machine_learning.reinforcement_learning.keyboard_mouse_action_space import KeyboardMouseActionSpace
from serpent.input_controller import KeyboardKey

# from .helpers.ml import *



class SerpentSuperHexagonGameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # self.frame_handlers["PLAY"] = self.handle_play
        self.frame_handlers["PLAY"] = self.handle_play

        # self.frame_handler_setups["PLAY"] = self.setup_play
        self.frame_handler_setups["PLAY"] = self.setup_play

        self.analytics_client = None
        self._reset_game_state()

        self.game_state = None

    def setup_play(self):
        self.plugin_path = offshoot.config["file_paths"]["plugins"]

        # Context Classifier
        context_classifier_path = f"{self.plugin_path}/SerpentSuperHexagonGameAgentPlugin/files/ml_models/context_classifier.model"

        context_classifier = CNNInceptionV3ContextClassifier(input_shape=(240, 384, 3))
        context_classifier.prepare_generators()
        context_classifier.load_classifier(context_classifier_path)

        self.machine_learning_models["context_classifier"] = context_classifier

        self.current_run = 0
        self.current_run_started_at = datetime.utcnow()
        self.current_run_duration = 0

        self.last_run_duration = 0
        self.last_run = 0

        input_mapping = {
            "L": [self.input_controller.tap_key(KeyboardKey.KEY_LEFT, duration=0.1)],
            "R": [self.input_controller.tap_key(KeyboardKey.KEY_RIGHT, duration=0.1)],
        }

        action_space = KeyboardMouseActionSpace(
            directional_keys=[None, "L", "R"]
        )

        model_file_path = None

        self.dqn_movement = DDQN(
            model_file_path=model_file_path, #if os.path.isfile(model_file_path) else None,
            input_shape=(120, 192, 3),
            input_mapping=input_mapping,
            action_space=action_space,
            replay_memory_size=50000,
            max_steps=500000,
            observe_steps=10000,
            batch_size=64,
            initial_epsilon=0.1,
            final_epsilon= 0.0001,
            override_epsilon=False
            )
    def handle_play(self, game_frame):
        gc.disable()

        context = self.machine_learning_models["context_classifier"].predict(game_frame.frame)
        if context is None:
            print("No context found...")
            return

        self.display_game_agent_state(context=context)
        if context in ["Hexagon_main_menu", "Hexagon_level", "Hexagon_game_over"]:
            self.input_controller.tap_key(KeyboardKey.KEY_SPACE)
        elif self.dqn_movement.first_run:
            self.dqn_movement.first_run = False
            return None





    # def handle_play(self, game_frame):
    #     print("hello world")
    #     print(self.machine_learning_models["context_classifier"])
    #     print("DANK MEMES")
    #     context = self.machine_learning_models["context_classifier"].predict(game_frame.frame)
    #
    #     if context is None:
    #         print("No Context Found...")
    #         return
    #     if context == "Hexagon_main_menu":
    #         self.input_controller.tap_key("space")
    #     if context == "Hexagon_level":
    #         self.input_controller.tap_key("space")
    #     if context == "Hexagon_game_over":
    #         self.input_controller.tap_key("space")
    #
    #     if context == "Hexagon_game":
    #         self.input_controller.tap_key("right")
    #
    #     self.display_game_agent_state(context=context)
    #
    #     for i, game_frame in enumerate(self.game_frame_buffer.frames):
    #         self.visual_debugger.store_image_data(
    #             game_frame.frame,
    #             game_frame.frame.shape,
    #             str(i)
    #         )
    def display_game_agent_state(self, context):
        self.current_run_duration = (datetime.utcnow() - self.current_run_started_at).seconds

        print("\033c")
        print(f"GAME: Super Hexagon         PLATFORM: Steam\n")

        print("")

        print(f"CURRENT CONTEXT: {context}")

        print("")

        print(f"CURRENT RUN: {self.current_run}")
        print(f"CURRENT RUN DURATION: {self.current_run_duration} seconds")

        print("")

        print(f"LAST RUN: {self.last_run}")
        print(f"LAST RUN DURATION: {self.last_run_duration} seconds")

        print("")


    def _reset_game_state(self):
        self.game_state = {
            "seed_entered": False,
            "health": collections.deque(np.full((8,), 6), maxlen=8),
            "coins": 0,
            "game_context": None,
            "current_run": 1,
            "current_run_steps": 0,
            "average_aps": 0,
            "run_reward_movement": 0,
            "run_reward_projectile": 0,
            "run_future_rewards": 0,
            "run_predicted_actions": 0,
            "run_timestamp": datetime.utcnow(),
            "last_run_duration": 0,
            "record_time_alive": dict(),
            "record_distance_travelled": dict(),
            "record_coins_collected": dict(),
            "random_time_alive": None,
            "random_time_alives": list(),
            "random_distance_travelled": None,
            "random_boss_hps": list()
            }
