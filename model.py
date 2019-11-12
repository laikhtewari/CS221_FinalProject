import util, math, random
from collections import defaultdict

class DisasterMDP(util.MDP):
    def __init__(self, issuesSuccCost={'fundraise':(1, 250), 'hunger':(0.9, -10), 'infrastructure':(0.8, -25), 'political':(0.6, -50)}, \
        initialSeverities={'fundraise':0, 'hunger':5, 'infrastructure':5, 'political':5}, initialFunds=1000, threshold=2):
        """
        
        """
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
    # A few reminders:
    # * Indicate a terminal state (after quitting, busting, or running out of cards)
    #   by setting the deck to None.
    # * If |state| is an end state, you should return an empty list [].
    # * When the probability is 0 for a transition to a particular new state,
    #   don't include that state in the list returned by succAndProbReward.
    def succAndProbReward(self, state, action):
        if state[0] < self.issuesSuccCost[action][1]: #if don't have enough
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
        elif action == 'hunger' : 
            index = 1
        elif action == 'infrastructure' : 
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
    print('Avg QL Reward:', sum(totalQLRewards)/len(totalQLRewards))
