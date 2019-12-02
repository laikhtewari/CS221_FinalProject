import util, model, random, sys
from matplotlib import pyplot as plt
import numpy as np


def give_intro():
    print('Welcome to disaster relief simulation! What would you like to do?')
    print('A: Run Q-Learning with randomized initial states')
    print('B: Run Q-Learning with predefined initial states')
    print('C: Run Q-Learning with randomized initial states and plot multiple features')
    print('D: Simulate custom decisions with randomized initial state')
    print('E: Simulate custom decisions with predefined initial state')
    print('F: Find distribution of weight vectors')
    print('G: Find the averages of the weight vectors')
    while True:
        response = input('Select A, B, C, or D: ')
        if response.upper() == 'A':
            return 'A'
        if response.upper() == 'B':
            return 'B'
        if response.upper() == 'C':
            return 'C'
        if response.upper() == 'D':
            return 'D'
        if response.upper() == 'E':
            return 'E'
        if response.upper() == 'F':
            return 'F'
        if response.upper() == 'G':
            return 'G'
        print('Invalid input')

def query_num_trials():
    while True:
        response = input('How many trials? ')
        if response.isnumeric():
            return int(response)
        print('Please enter a valid integer...')

def give_player_information():
    print()
    print('=' * 10, 'SIMULATION BEGIN', '=' * 10)
    input('Simulating a disaster scenario! (press enter to continue)')
    input('In this exercise, you are acting as a relief organization that is lending aid to a region facing turmoil'
          'due to natural disaster, political insurgence, or some other crisis.')
    input('You begin with a set amount of resources - cash, personnel, and foodstuff.')
    input('Additionally, you receive some measure of how severe the situation is, on a scale of 1 to 10.')
    input('Examples of severity categories include food shortage, infrastructure damage, civil unrest, etc.')
    input('At every stage, you can take several actions: fundraise, hire personnel, buy food, send food, etc.')
    input('Each action has different effects, and can either increase your resource supply or use resources to '
          'hopefully alleviate some of the severities.')
    input('Your goal is to get every severity level below a certain threshold in the minimum number of actions!')
    input('Much like the real world don\'t know how specifically each action will play out - it might even fail'
          'and not have any effect! However, each action will tend to do what it sounds like it is doing (for '
          'example, sending food will likely improve the food shortage severity, and fundraising increases'
          'the amount of money you have).')
    input('Good luck!')


def dict_diff(prev_dict, new_dict):
    result = {}
    for key in new_dict:
        if new_dict[key] != prev_dict[key]:
            result[key] = new_dict[key] - prev_dict[key]
    return result


def signed(difference):
    if difference < 0:
        return str(difference)
    return '+' + str(difference)


def run_custom_simulation(initialState='random'):
    def sample(probs):
        target = random.random()
        accum = 0
        for i, prob in enumerate(probs):
            accum += prob
            if accum >= target: return i
        raise Exception("Invalid probs: %s" % probs)

    total_reward = 0
    num_turns = 0
    mdp = model.DisasterMDP(randomize=(initialState == 'random'))
    give_player_information()
    state = mdp.startState()
    while state is not None:
        print()
        print('Turn', num_turns)
        num_turns += 1
        resources, severities = mdp.state_tuple_to_dict(state)
        print('Your resources:')
        for resource in resources:
            print(resource + ':', resources[resource])
        print('Here are the different severities:')
        for severity in severities:
            status = '===NOT SOLVED==='
            if severities[severity] < mdp.threshold:
                status = '===SOLVED==='
            print(severity + ':', severities[severity], status)
        print('Target threshold for severities:', mdp.threshold)
        print('Note - if all severities are considered solved, take any action in order to finish the simulation.')
        print()
        print('Here are the actions you can take')
        print('A: fundraise - raise money to increase your cash reserves.')
        print('B: hire - spend money to increase your personnel.')
        print('C: buy food - spend money to increase your food stocks.')
        print('D: send food - send food from your food stocks into the disaster situation.')
        print('E: diplomacy - send people to alleviate political tensions.')
        print('F: build infrastructure - send people to repair infrastructure.')
        action = ''
        while True:
            prompt = input('What action will you take? (Type the name of the action or the corresponding letter.) ')
            action = ''
            if prompt.upper() == 'A' or prompt.lower == 'fundraise':
                action = 'fundraise'
                break
            if prompt.upper() == 'B' or prompt.lower == 'hire':
                action = 'hire'
                break
            if prompt.upper() == 'C' or prompt.lower == 'buy food':
                action = 'buy_food'
                break
            if prompt.upper() == 'D' or prompt.lower == 'send food':
                action = 'send_food'
                break
            if prompt.upper() == 'E' or prompt.lower == 'diplomacy':
                action = 'diplomacy'
                break
            if prompt.upper() == 'F' or prompt.lower == 'build':
                action = 'build'
                break
            else:
                print('Please enter a valid option (A, B, C, D, E, or F).')
        transitions = mdp.succAndProbReward(state, action)
        print('You have taken the action', action + '!')
        if len(transitions) == 0:
            'Congratulations! The crisis has been averted!'
            state = None
        else:
            # Choose a random transition
            i = sample([prob for newState, prob, reward in transitions])
            newState, prob, reward = transitions[i]
            total_reward += reward
            new_resources, new_severities = mdp.state_tuple_to_dict(newState)
            resource_diff = dict_diff(resources, new_resources)
            severities_diff = dict_diff(severities, new_severities)
            if not resource_diff and not severities_diff:
                print('Oh no, your action failed! No effect took place')
            else:
                print('==== Changes to state ====')
                print()
                print('---Resources---')
                for key, val in resource_diff.items():
                    sign = ''
                    if val > 0:
                        sign = '+'
                    print(key + ':', sign, str(val))
                print('---Severities---')
                for key, val in severities_diff.items():
                    print(key + ':', val)
            state = newState


def simulate_with_feature(feature_extractor, randomize=True):
    mdp = model.DisasterMDP(randomize=randomize)
    random.seed(42)
    print('Simulating with', feature_extractor.__name__)
    print('=' * 6, 'initialization', '=' * 6)
    qLearningSolver = util.QLearningAlgorithm(mdp.actions, 1, feature_extractor)
    print('=' * 6, 'simulating', '=' * 6)
    totalQLRewards, time_list, turn_avg_list, avg_list = util.simulate(mdp, qLearningSolver, numTrials=num_trials)
    print('Avg QL Reward:', sum(totalQLRewards) / len(totalQLRewards))
    return time_list, turn_avg_list, avg_list


def find_weights_distribution():
    mdp = model.DisasterMDP()
    random.seed(42)
    print('=' * 6, 'initialization', '=' * 6)
    qLearningSolver = util.QLearningAlgorithm(mdp.actions, 1, model.joint_bucket_max_feature_extractor)
    print('=' * 6, 'simulating', '=' * 6)
    totalQLRewards, _, _, _ = util.simulate(mdp, qLearningSolver, numTrials=num_trials)
    print('Avg QL Reward:', sum(totalQLRewards) / len(totalQLRewards))
    weights = qLearningSolver.weights
    sorted_weights = sorted(weights.items(), key=lambda kv: abs(kv[1]))
    print('Here are the top 10% of weights by absolute value')
    num_keep = 100
    highest_weights = sorted_weights[-1 * num_keep:]
    labels = ['resources', 'severities', 'max', 'joint']
    counter = [0, 0, 0, 0]
    for w, _ in highest_weights:
        if 'resource' in w:
            if 'severity' in w:
                counter[3] += 1
            else:
                counter[0] += 1
        elif 'severity' in w:
            counter[1] += 1
        elif 'max_severity' in w:
            counter[2] += 1
    return labels, counter


def weight_averages():
    mdp = model.DisasterMDP()
    random.seed(42)
    print('=' * 6, 'initialization', '=' * 6)
    qLearningSolver = util.QLearningAlgorithm(mdp.actions, 1, model.joint_bucket_max_feature_extractor)
    print('=' * 6, 'simulating', '=' * 6)
    totalQLRewards, _, _, _ = util.simulate(mdp, qLearningSolver, numTrials=num_trials)
    print('Avg QL Reward:', sum(totalQLRewards) / len(totalQLRewards))
    weights = qLearningSolver.weights
    labels = ['resources', 'severities', 'max', 'joint']
    counter = [0, 0, 0, 0]
    sums = [0, 0, 0, 0]
    for w, val in weights.items():
        if 'resource' in w:
            if 'severity' in w:
                counter[3] += 1
                sums[3] += abs(val)
            else:
                counter[0] += 1
                sums[0] += abs(val)
        elif 'severity' in w:
            counter[1] += 1
            sums[1] += abs(val)
        elif 'max_severity' in w:
            counter[2] += 1
            sums[2] += abs(val)
    for i in range(len(sums)):
        sums[i] /= counter[i]
    return labels, sums


if __name__ == '__main__':
    user_input = give_intro()
    if user_input == 'A' or user_input == 'B' or user_input == 'C' or user_input == 'F' or user_input == 'G':
        num_trials = query_num_trials()
        if user_input == 'A' or user_input == 'B':
            randomize = user_input == 'A'
            time_list, turn_avg_list, _ = simulate_with_feature(model.bucket_feature_extractor, randomize=randomize)
            fig = plt.figure()
            ax = plt.axes()
            plt.title('Training History')
            plt.xlabel('Episodes')
            plt.ylabel('Turns Taken')
            ax.plot(time_list, turn_avg_list)
            plt.show()
        elif user_input == 'C':
            data = {}
            data['bucket'] = simulate_with_feature(model.bucket_feature_extractor)
            data['small_bucket'] = simulate_with_feature(model.small_bucket_feature_extractor)
            data['bucket_max'] = simulate_with_feature(model.bucket_max_feature_extractor)
            data['joint'] = simulate_with_feature(model.joint_feature_extractor)
            # data['joint_bucket'] = simulate_with_feature(model.joint_bucket_feature_extractor)
            data['joint_bucket_max'] = simulate_with_feature(model.joint_bucket_max_feature_extractor)
            plt.figure()
            plt.title('Training History (Reward)')
            plt.xlabel('Episodes')
            plt.ylabel('Turns Taken')
            for key, value in data.items():
                plt.plot(value[0], value[2], label=key)
            plt.legend()
            plt.show()
        elif user_input == 'F':
            labels, counter = find_weights_distribution()
            explode = (0.1, 0, 0, 0)
            colors = ['gold', 'yellowgreen', 'lightskyblue', 'lightcoral']
            plt.pie(counter, startangle=140, labels=labels, autopct='%1.1f%%', shadow=True, explode=explode, colors=colors)
            # plt.legend(patches, labels, loc="best")
            # plt.axis('equal')
            plt.title('Categories of top 100 features by absolute value of weight')
            # plt.tight_layout()
            plt.show()
        else:
            labels, averages = weight_averages()
            colors = ['gold', 'yellowgreen', 'lightskyblue', 'lightcoral']
            y_pos = np.arange(len(labels))
            plt.barh(y_pos, averages, align='center')
            plt.yticks(y_pos, labels=labels)
            plt.ylabel(labels)
            plt.xlabel('Average of magnitude of weights')
            plt.title('Averages of the magnitude of the weights of each feature category')
            plt.show()
        # randomize = user_input == 'A'
        # mdp = model.DisasterMDP(randomize=randomize)
        # random.seed(42)
        # print('=' * 6, 'initialization', '=' * 6)
        # qLearningSolver = util.QLearningAlgorithm(mdp.actions, 1, model.joint_feature_extractor)
        # print('=' * 6, 'simulating', '=' * 6)
        # totalQLRewards = util.simulate(mdp, qLearningSolver, numTrials=num_trials)
        # print('Avg QL Reward:', sum(totalQLRewards) / len(totalQLRewards))
        # weights = qLearningSolver.weights
        # sorted_weights = sorted(weights.items(), key=lambda kv: abs(kv[1]))
        # print('Here are the top 10 weights by absolute value')
        # print(sorted_weights[-10:])
    else:
        if user_input == 'D':
            run_custom_simulation()
        elif user_input == 'E':
            run_custom_simulation(initialState='not random')



