import util, math, random
from collections import defaultdict
import numpy as np


# Class to succinctly apply the effect of an action
# When generate_effect is called, returns a new state after accounting for
# the effect this action can have on the state if it is successful
class ActionEffect():
    def __init__(self, category, field, mean, std):
        self.category = category
        self.field = field
        self.mean = mean
        self.std = std

    # Returns a new state based on the effect of this action
    # |state| is of the form (resources, severity)
    def generate_effect(self, state, multiplier):
        resources, severity = state
        if self.category == 'resources':
            resources[self.field] += np.random.normal(self.mean * multiplier, self.std * abs(multiplier))
        else:
            severity[self.field] += np.random.normal(self.mean * multiplier, self.std * abs(multiplier))

        return resources, severity

# IMPLEMENT LATER
class CompoundingEffect():
    def __init__(self, field, mean, std):
        self.field = field
        self.mean = mean
        self.std = std

    def generate_effect(self, state):
        _, severity = state
        # IMPLEMENT ME LATER
        return state


class DisasterMDP(util.MDP):
    # For readability, init would only take on default values for state. Any changes to the default value
    # would then be updated using set_initial_state
    def __init__(self):

        ### INITIAL STATE VARIABLES ###
        """
        The amount of resources the agent has available:
        cash: can be used to hire peronsel, get food, or send financial aid - but is generally not as effective as sending
            personnel or foodstuff directly
        personnel: can be deployed immediately to combat a variety of areas. Can be hired using cash, but takes time
        foodstuff: can be sent immediately to combat food shortages. Can be bought using cash, but takes time
        """
        self.initial_resources = {
            'cash': 100000,
            'personnel': 25,
            'foodstuff': 50
        }

        """
        A measure (scale of 1 to 10)
        Not only does this give a measure of the final reward
        """
        self.initial_severities = {
            'food_shortage': 5.0,
            'infrastructure': 5.0,
            'civil_unrest': 5.0,
            'political_tension': 5.0
        }

        """
        Dict of action: (resource needed, probability of success)
        """
        self.action_succ_rates = {
            'fundraise': (None, 1.0),
            'hire': ('cash', 1.0),
            'buy_food': ('cash', 1.0),
            'send_food': ('foodstuff', 0.9),
            'diplomacy': ('personnel', 0.7),
            'build': ('personnel', 0.8)
        }

        """
        Threshold for success
        """
        self.threshold = 3

        ### TRANSITION PROBABILITIES ###
        """
        FIRST, an ACTION TAKEN can probabilistically influence the status of each severity
        ActionEffect objects encode how a particular category of state might be effected by ActionEffect (based on a guassian
        distribution)
        """
        self.action_effects = {
            'fundraise': [ActionEffect('resources', 'cash', 20000, 3000)],
            'hire': [ActionEffect('resources', 'personnel', 5, 1)],
            'buy_food': [ActionEffect('resources', 'foodstuff', 10, 2)],
            'send_food': [ActionEffect('severities', 'food_shortage', -1, 0.2), ActionEffect('severities', 'civil_unrest', -0.8, 0.1)],
            'diplomacy': [ActionEffect('severities', 'political_tension', -1, 0.3), ActionEffect('severities', 'civil_unrest', -0.2, 0.05)],
            'build': [ActionEffect('severities', 'infrastructure', -1, 0.3), ActionEffect('severities', 'political_tension', -0.1, 0.03),
                      ActionEffect('severities', 'civil_unrest', -0.2, 0.05)]
        }

        # After the effect of the ACTION takes place, severities would then develop again based on
        # probabilistic relationships
        # IMPLEMENT LATER
        self.compounding_effects = {
            'food_shortage': [],
            'infrastructure': [],
            'civil_unrest': [],
            'political_tension': []
        }

    """
    Call to update the initial state
    |resources|: dict of resource:amount pairs signifying how much of each resource the agent starts off with
        If a key is not present, self.resources keeps the default value
        Only valid keys (cash, personell, foodstuff) can be added - this might be changed later as we add more 
        complexities
    |severities|: dict of category:severity pairs signifying how severe each scenario is the in the disaster model
        If a key is not present, self.resources keeps the default value
        Only valid keys (food_shortage, infrastructure, civil_unrest, political_tension) can be added - this might be 
        changed later as we add more complexities
    |action_succ_rates|: the probability of an action to be successful
    |threshold|: the depth to search
    """

    def set_initial_state(self, resources, severities, action_succ_rates, threshold=3):
        for key, value in resources.items():
            if key in self.initial_resources:
                self.initial_resources[key] = value

        for key, value in severities.items():
            if key in self.initial_severities:
                self.initial_severities[key] = value

        for key, value in action_succ_rates.items():
            if key in self.action_succ_rates and 0 <= value <= 1:
                self.action_succ_rates[key] = value

        self.threshold = threshold

    # Since dicts are not hashable, we have to pass state as a tuple of values
    # This function takes in a state and returns the dicts representing them
    def state_tuple_to_dict(self, state):
        resources = {}
        severities = {}
        resources_keys = list(self.initial_resources.keys())
        severities_keys = list(self.initial_severities.keys())
        for i in range(len(state[0])):
            resources[resources_keys[i]] = state[0][i]
        for i in range(len(state[1])):
            severities[severities_keys[i]] = state[1][i]

        return resources, severities

    def startState(self):
        return tuple(self.initial_resources.values()), tuple(self.initial_severities.values())

    def actions(self, state):
        return list(self.action_succ_rates.keys())

    # Given a |state| and |action|, return a list of (newState, prob, reward) tuples
    # corresponding to the states reachable from |state| when taking |action|.
    def succAndProbReward(self, state, action):
        resource_needed, prob_success = self.action_succ_rates[action]
        resources, severities = self.state_tuple_to_dict(state)

        done = True
        for severity in severities.values():
            if severity > self.threshold:
                done = False

        if done:
            return []

        multiplier = 1

        # For now, we spend half the available corresponding resource for each action
        # Our reward scales with the amount we spent
        if resource_needed is not None:
            resources_spent = resources[resource_needed] // 2
            resources[resource_needed] -= resources_spent

            # If we have too little of a resource, we can't take this action
            if resources[resource_needed] < 2:
                return [(state, 1, -1)]

            multiplier = 5 * resources_spent / self.initial_resources[resource_needed]

        fail_state = tuple(resources.values()), tuple(severities.values())
        success_state = resources, severities

        # Apply the effect of the action
        for action_effect in self.action_effects[action]:
            success_state = action_effect.generate_effect(success_state, multiplier)

        # Apply the compounding severities of the situation (IMPLEMENT LATER)

        return [(fail_state, 1 - prob_success, -1), ((tuple(success_state[0].values()), tuple(success_state[1].values())), prob_success, -1)]

    def discount(self):
        return 1


class DMDP(util.MDP):
    def __init__(self, issuesSuccCost={'fundraise': (1, 250), 'hunger': (0.9, -10), 'infrastructure': (0.8, -25),
                                       'political': (0.6, -50)}, \
                 initialSeverities={'fundraise': 0, 'hunger': 5, 'infrastructure': 5, 'political': 5},
                 initialFunds=1000, threshold=3):

        self.issuesSuccCost = issuesSuccCost
        self.initialSeverities = [initialSeverities[k] for k in initialSeverities]
        self.initialFunds = initialFunds
        self.threshold = threshold

    # Return the start state.
    def startState(self):
        return (self.initialFunds, tuple(self.initialSeverities))

    # Return set of actions possible from |state|.
    def actions(self, state):
        # IMPLEMENT ME

        """
        Ideas include: fundraise/send aid/dispatch personnel
        """
        return ['fundraise', 'hunger', 'infrastructure', 'political']

    # Given a |state| and |action|, return a list of (newState, prob, reward) tuples
    # corresponding to the states reachable from |state| when taking |action|.
    def succAndProbReward(self, state, action):
        if state[0] < self.issuesSuccCost[action][1]:  # if don't have enough
            return [(state, 1, -1)]
        done = True
        for k in state[1]:
            if k > self.threshold:
                done = False
                break
        if done:
            return []

        if action == 'fundraise':
            index = 0
        elif action == 'hunger':
            index = 1
        elif action == 'infrastructure':
            index = 2
        elif action == 'political':
            index = 3

        newSeverity = []
        for i in range(len(state[1])):
            if i == index:
                newSeverity.append(state[1][i] - 1)
            else:
                newSeverity.append(state[1][i])

        succState = (state[0] + self.issuesSuccCost[action][1], tuple(newSeverity))
        negState = (state[0] + self.issuesSuccCost[action][1], tuple([v for v in state[1]]))
        return [(succState, self.issuesSuccCost[action][0], -1), (negState, 1 - self.issuesSuccCost[action][0], -1)]

    def discount(self):
        # We might want to change, depending if we want to see short term or long term goals
        return 1


def identityFeatureExtractor(state, action):
    featureKey = (state, action)
    featureValue = 1
    return [(featureKey, featureValue)]


if __name__ == '__main__':
    # print('=' * 6, 'initialization', '=' * 6)
    model = DisasterMDP()
    # viSolver = util.ValueIteration()
    # print('=' * 6, 'solving', '=' * 6)
    # viSolver.solve(model)
    # fixedVIAlgo = FixedRLAlgorithm(viSolver.pi)
    # print('=' * 6, 'simulating', '=' * 6)
    # totalVIRewards = simulate(model, fixedVIAlgo)
    # print('Avg VI Reward:', sum(totalVIRewards)/len(totalVIRewards))
    random.seed(42)
    print('=' * 6, 'initialization', '=' * 6)
    qLearningSolver = util.QLearningAlgorithm(model.actions, 1, identityFeatureExtractor)
    print('=' * 6, 'simulating', '=' * 6)
    totalQLRewards = util.simulate(model, qLearningSolver, numTrials=250000)
    print('Avg QL Reward:', sum(totalQLRewards) / len(totalQLRewards))
