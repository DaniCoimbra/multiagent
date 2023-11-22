# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
import math

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        risk = 0
        if len(newGhostStates) is not 0:
            risk = min(manhattanDistance(ghost.getPosition(),newPos) for ghost in newGhostStates)
        if risk <=2:
            return float("-inf")
        foodDist = 0
        if len(newFood.asList()) is not 0:
            foodDist = min(manhattanDistance(food,newPos) for food in newFood.asList())
        scaredTime = 0
        if len(newScaredTimes) is not 0:
            scaredTime = min(newScaredTimes)
        oldFood = currentGameState.getFood().asList()
        if successorGameState.isLose():
            return float("-inf")
        return successorGameState.getScore()-currentGameState.getScore()-(len(oldFood) - len(newFood.asList()))-foodDist/10+risk/11+scaredTime

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def minMax(self, gameState, agent, depth):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), ""

        if agent == 0:
            score = -math.inf
            for action in gameState.getLegalActions(0):
                nextState = gameState.generateSuccessor(0, action)
                newScore, _ = self.minMax(nextState, 1, depth)
                if newScore > score:
                    score = newScore
                    bestAction = action
            return score, bestAction
        else:
            score = math.inf
            for action in gameState.getLegalActions(agent):
                nextState = gameState.generateSuccessor(agent, action)
                if agent == gameState.getNumAgents() - 1:
                    minScore, _ = self.minMax(nextState, 0, depth + 1)
                else:
                    minScore, _ = self.minMax(nextState, agent + 1, depth)
                if minScore < score:
                    score = minScore
                    bestAction = action
            return score, bestAction

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        score, action = self.minMax(gameState, 0, 0)
        return action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def alphaBeta(self, gameState, agent, depth, alpha, beta):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), ""
        bestAction = None
        if agent == 0:
            score = -math.inf
            for action in gameState.getLegalActions(0):
                nextState = gameState.generateSuccessor(0, action)
                newScore, _ = self.alphaBeta(nextState, 1, depth, alpha, beta)
                if newScore > score:
                    score = newScore
                    bestAction = action

                if score > beta:
                    return score, bestAction
                alpha = max(alpha, score)

            return score, bestAction
        else:
            score = math.inf
            for action in gameState.getLegalActions(agent):
                nextState = gameState.generateSuccessor(agent, action)
                if agent == gameState.getNumAgents() - 1:
                    minScore, _ = self.alphaBeta(nextState, 0, depth + 1, alpha, beta)
                else:
                    minScore, _ = self.alphaBeta(nextState, agent + 1, depth, alpha, beta)
                if minScore < score:
                    score = minScore
                    bestAction = action

                if score < alpha:
                    return score, bestAction
                beta = min(beta, score)

            return score, bestAction
    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        score, action = self.alphaBeta(gameState, 0, 0, -math.inf, math.inf)
        return action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def expectiMax(self, gameState, agent, depth):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), ""
        bestAction = None

        if agent == 0:
            score = -math.inf
            for action in gameState.getLegalActions(0):
                nextState = gameState.generateSuccessor(0, action)
                newScore, _ = self.expectiMax(nextState, 1, depth)
                if newScore > score:
                    score = newScore
                    bestAction = action

            return score, bestAction
        else:
            score = 0
            actions = gameState.getLegalActions(agent)

            if not actions:
                return self.evaluationFunction(gameState), ""

            for action in actions:
                nextState = gameState.generateSuccessor(agent, action)
                if agent == gameState.getNumAgents() - 1:
                    minScore, _ = self.expectiMax(nextState, 0, depth + 1)
                else:
                    minScore, _ = self.expectiMax(nextState, agent + 1, depth)
                score += minScore

            return score/len(actions), _

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        score, action = self.expectiMax(gameState, 0, 0)
        return action

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: get distance from closest food and the game score,
    """
    "*** YOUR CODE HERE ***"
    foodsPos = currentGameState.getFood().asList()
    pacmanPos = currentGameState.getPacmanPosition()

    # Calculate the distance to the closest food
    closestFoodDis = min(manhattanDistance(pacmanPos, foodPos) for foodPos in foodsPos) if foodsPos else 0.1

    score = currentGameState.getScore()

    # Consideration for Pac-Man staying put when multiple actions have the same evaluation
    returningDis = 1.0 / closestFoodDis + score

    return returningDis
# Abbreviation
better = betterEvaluationFunction
