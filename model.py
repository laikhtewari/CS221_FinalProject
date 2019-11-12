import util, math, random
from collections import defaultdict

# Maybe we can have some different way to keep track of all of this and
# Create a parent class for each of these successor states

class DisasterStatus():
    def __init__(self):
        """

        """

class Personel():
    def __init__(self):
        """
        Keeps track of the staff we have available at a given time, with their own specializations
        and how they might be able to assist in specific scenariors
        """

class Resources():
    def __init__(self, personel, money=0, expected_income=0):
        """
        Takes in as input different state
        """
        self.money = money
        self.expected_income = expected_income

    def get_money(self):
        return self.money

def combine_states(state1, state2):


class DisasterMDP(util.MDP):
    def __init__(self, resources, disaster_status):
        """
        resources: available resources we can draw upon (money, food, raw materials, personel)
        disaster_status: describes the current state of the city (population, political stability)
        """

        self.resources = resources
        self.disaster_status = disaster_status


    # Return the start state.
    def startState(self):
        return (self.resources, self.disaster_status)

    # Return set of actions possible from |state|.
    def actions(self, state):
        # IMPLEMENT ME

        """
        Ideas include: fundraise/send aid/dispatch personnel
        """

        return []

    # Given a |state| and |action|, return a list of (newState, prob, reward) tuples
    # corresponding to the states reachable from |state| when taking |action|.
    # A few reminders:
    # * Indicate a terminal state (after quitting, busting, or running out of cards)
    #   by setting the deck to None.
    # * If |state| is an end state, you should return an empty list [].
    # * When the probability is 0 for a transition to a particular new state,
    #   don't include that state in the list returned by succAndProbReward.
    def succAndProbReward(self, state, action):

        #IMPLEMENT ME

        return []

    def discount(self):
        # We might want to change, depending if we want to see short term or long term goals
        return 1