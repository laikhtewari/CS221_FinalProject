import util, math, random
import numpy as np

# Class to succinctly apply the effect of an action to a state
# When generate_effect is called, returns a new state after accounting for
# the effect this action produces - modeled as a gaussian
class ActionEffect():
    """
    Category:field refers to the specific aspect of state this action can affect, where category can be
    resources or severities
    """
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
        elif self.category == 'severities':
            severity[self.field] += np.random.normal(self.mean * multiplier, self.std * abs(multiplier))
        else:
            raise ValueError('The category of an ActionEffect object must be either \'resources\' or \'severities\'')

        return resources, severity

# IMPLEMENT LATER
# Describes how a state's severities compound upon itself and might influence other severities (i.e. high food shortages
# can lead to more civil unrest)
class CompoundingEffect():
    def __init__(self, field, target_field, mean, std):
        self.field = field
        self.target_field = target_field
        self.mean = mean
        self.std = std

    def generate_effect(self, state):
        _, severity = state
        # IMPLEMENT ME LATER
        return state


class DisasterMDP(util.MDP):
    # For readability (we don't want 25 params), init would only take on default values for state.
    # Any changes to the default value would then be updated using set_initial_state
    def __init__(self, randomize=True):

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
        A measure (scale of 1 to 10) of the severities of several of the categories of the crisis situation that should
        be addressed. Once severity levels are sufficiently lowered, the disaster is mitigated and the disaster is solved
        Additionally, high severities tend to get worse and are harder to mitigate
        """
        self.initial_severities = {
            'food_shortage': 5.0,
            'infrastructure': 5.0,
            'civil_unrest': 5.0,
            'political_tension': 5.0
        }

        """
        When an action is to be take, get the cost associated and the probability of success
        Dict of action: (resource needed, price per unit, probability of success)
        """
        self.action_cost_succ = {
            'fundraise': (None, None, 1.0),
            'hire': ('cash', 2500.0, 1.0),
            'buy_food': ('cash', 1000.0, 1.0),
            'send_food': ('foodstuff', 10.0, 0.9),
            'diplomacy': ('personnel', 5.0, 0.7),
            'build': ('personnel', 5.0, 0.8)
        }

        """
        Threshold for success
        """
        self.threshold = 1.0

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
        # IMPLEMENT LATER - should make things much more interesting
        self.compounding_effects = {
            'food_shortage': [],
            'infrastructure': [],
            'civil_unrest': [],
            'political_tension': []
        }

        self.randomize = randomize

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

    def set_initial_state(self, resources={}, severities={}, action_succ_cost={}, action_effects={}, compounding_effects={}, threshold=3.0):
        for key, value in resources.items():
            if key in self.initial_resources:
                self.initial_resources[key] = value

        for key, value in severities.items():
            if key in self.initial_severities:
                self.initial_severities[key] = value

        for key, value in action_succ_cost.items():
            if key in self.action_cost_succ and 0 <= value[2] <= 1:
                self.action_cost_succ[key] = value

        for key, value in action_effects.items():
            if key in self.action_effects:
                self.action_effects[key] = value

        for key, value in compounding_effects.items():
            if key in self.compounding_effects:
                self.compounding_effects[key] = value

        self.threshold = threshold

    def randomize_initial_state(self):
        resource_seed = random.uniform(5, 25)

        resources = {
            'cash': max(200, random.gauss(1000, 200)) * resource_seed,
            'personnel': max(0.5, random.gauss(2, 0.2)) * resource_seed,
            'foodstuff': max(0.8, random.gauss(4, 0.5)) * resource_seed
        }

        severities_seed = random.gauss(resource_seed, 2) / 3

        severities = {
            'food_shortage': max(1.0, min(10.0, random.gauss(severities_seed, 1.5))),
            'infrastructure': max(1.0, min(10.0, random.gauss(severities_seed, 1.5))),
            'civil_unrest': max(1.0, min(10.0, random.gauss(severities_seed, 1.5))),
            'political_tension': max(1.0, min(10.0, random.gauss(severities_seed, 1.5)))
        }

        self.set_initial_state(resources=resources, severities=severities)

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
        if self.randomize:
            self.randomize_initial_state()
        return tuple(self.initial_resources.values()), tuple(self.initial_severities.values())

    def actions(self, state):
        return list(self.action_cost_succ.keys())

    # Given a |state| and |action|, return a list of (newState, prob, reward) tuples
    # corresponding to the states reachable from |state| when taking |action|.
    def succAndProbReward(self, state, action):
        resource_needed, amount_per_unit, prob_success = self.action_cost_succ[action]
        resources, severities = self.state_tuple_to_dict(state)

        # If the threshold has been met for all severity values, the problem is considered solved
        # Return terminal state
        if all(severity < self.threshold for severity in severities.values()):
            return []

        multiplier = 1

        # For now, we spend half the available corresponding resource for each action
        # Our reward scales with the amount we spent
        if resource_needed is not None:
            resources_spent = resources[resource_needed] // 2
            resources[resource_needed] -= resources_spent

            multiplier = resources_spent / amount_per_unit

        # In the fail state, the resources are spent but the severities are not addressed
        fail_state = tuple(resources.values()), tuple(severities.values())

        # Build the success state
        success_state = resources, severities

        # Apply the effect of the action
        for action_effect in self.action_effects[action]:
            success_state = action_effect.generate_effect(success_state, multiplier)

        _, success_severities = success_state
        success_reward = -1
        if all(success_severity < self.threshold for success_severity in success_severities.values()):
            success_reward = 25

        # Apply the compounding severities of the situation (IMPLEMENT LATER)

        return [(fail_state, 1 - prob_success, -1), ((tuple(success_state[0].values()), tuple(success_state[1].values())), prob_success, success_reward)]

    def discount(self):
        return 0.9


# Naive extractor - does not work very well due to the
# sheer magnitude of the number of states
def identityFeatureExtractor(state, action):
    featureKey = (state, action)
    featureValue = 1
    return [(featureKey, featureValue)]


# Much better extractor
def bucket_feature_extractor(state, action):
    features = []
    resource_values, severity_values = state

    # For each resource and severity, bucketize the values
    # so that states with similar values share features
    for index, value in enumerate(resource_values):
        if index == 0:
            # Cash on hand is usually a much larger number, so we take the log then bucketize
            features.append((('resource', index, math.ceil(math.log(value)), action), 1))
        else:
            features.append((('resource', index, value // 3, action), 1))
    for index, value in enumerate(severity_values):
        # Bucketize severities based on the nearest 10th
        features.append((('severity', index, round(value), action), 1))

    # max_severity, _ = max(enumerate(severity_values), key=lambda iv: iv[1])
    # features.append((('max severity', max_severity, action), 1))
    return features


def small_bucket_feature_extractor(state, action):
    features = []
    resource_values, severity_values = state

    # For each resource and severity, bucketize the values
    # so that states with similar values share features
    for index, value in enumerate(resource_values):
        if index == 0:
            # Cash on hand is usually a much larger number, so we take the log then bucketize
            features.append((('resource', index, math.ceil(math.log(value)), action), 1))
        else:
            features.append((('resource', index, value // 3, action), 1))
    for index, value in enumerate(severity_values):
        # Bucketize severities based on the nearest 10th
        features.append((('severity', index, round(value, 1), action), 1))

    # max_severity, _ = max(enumerate(severity_values), key=lambda iv: iv[1])
    # features.append((('max severity', max_severity, action), 1))
    return features


def max_severity_extractor(state, action):
    resource_values, severity_values = state
    max_severity, _ = max(enumerate(severity_values), key=lambda iv: iv[1])
    return [(('max_severity', max_severity, action), 1)]


def bucket_max_feature_extractor(state, action):
    bucket_features = bucket_feature_extractor(state, action)
    max_features = max_severity_extractor(state, action)
    return bucket_features + max_features


def joint_feature_extractor(state, action):
    features = []
    resource_values, severity_values = state

    for index, value in enumerate(resource_values):
        if index == 0:
            # Cash on hand is usually a much larger number, so we take the log then bucketize
            resource_tuple = ('resource', index, math.ceil(math.log(value)), action)
        else:
            resource_tuple = ('resource', index, value // 5, action)

        for i, v in enumerate(severity_values):
            joint_tuple = resource_tuple + ('severity', i, round(v), action)
            features.append((joint_tuple, 1))

    return features


def joint_bucket_feature_extractor(state, action):
    bucket_features = bucket_feature_extractor(state, action)
    joint_features = joint_feature_extractor(state, action)
    return bucket_features + joint_features


def joint_bucket_max_feature_extractor(state, action):
    bucket_features = bucket_feature_extractor(state, action)
    joint_features = joint_feature_extractor(state, action)
    max_features = max_severity_extractor(state, action)
    return bucket_features + joint_features + max_features
