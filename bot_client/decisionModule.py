# Asyncio (for concurrency)
import asyncio
from stable_baselines3 import PPO

# Game state
from gameState import *
from env import PacbotEnv

model = PPO.load("./model")

ACTION_NAMES = ["RIGHT", "LEFT", "UP", "DOWN"]
ACTIONS = [Directions.RIGHT, Directions.LEFT, Directions.UP, Directions.DOWN]

class DecisionModule:
	'''
	Sample implementation of a decision module for high-level
	programming for Pacbot, using asyncio.
	'''

	def __init__(self, state: GameState) -> None:
		'''
		Construct a new decision module object
		'''

		# Game state object to store the game information
		self.state = state
		self.env = PacbotEnv(self.state)

	async def decisionLoop(self) -> None:
		'''
		Decision loop for Pacbot
		'''

		# Receive values as long as we have access
		while self.state.isConnected():

			'''
			WARNING: 'await' statements should be routinely placed
			to free the event loop to receive messages, or the
			client may fall behind on updating the game state!
			'''

			# If the current messages haven't been sent out yet, skip this iteration
			if len(self.state.writeServerBuf) or not self.state.received_update:
				await asyncio.sleep(0)
				continue

			# Wait for game state to (hopefully) be updated
			await asyncio.sleep(0.15)

			# Lock the game state
			self.state.lock()

			obs = self.env.get_observation()

			action, _ = model.predict(obs, deterministic=True)

			# Write back to the server, as a test (move right)
			self.state.queueAction(1, ACTIONS[action])

			# Unlock the game state
			self.state.unlock()

			# Print that a decision has been made
			print('decided', ACTION_NAMES[action])

			# Free up the event loop
			await asyncio.sleep(0)
