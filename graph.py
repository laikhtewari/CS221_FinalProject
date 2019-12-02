import matplotlib.pyplot as plt
import numpy as np
import model, util, random

random.seed(42)
num_iterations = 50000

simple_mdp = model.DisasterMDP(randomize=False)

moderate_mdp = model.DisasterMDP(randomize=False)
moderate_resources = {
    'cash': 20000,
    'personnel': 0,
    'foodstuff': 15
}

moderate_severities = {
    'food_shortage': 0.9,
    'infrastructure': 7.0,
    'civil_unrest': 5.0,
    'political_tension': 8.0
}

moderate_mdp.set_initial_state(resources=moderate_resources, severities=moderate_severities)

complex_mdp = model.DisasterMDP(randomize=False)
complex_resources = {
    'cash': 1000,
    'personnel': 0,
    'foodstuff': 0
}

complex_severities = {
    'food_shortage': 8.0,
    'infrastructure': 9.0,
    'civil_unrest': 7.0,
    'political_tension': 8.0
}
complex_mdp.set_initial_state(resources=complex_resources, severities=complex_severities)


def train(feature_extractor):
    mdp = model.DisasterMDP(randomize=True)
    qLearningSolver = util.QLearningAlgorithm(mdp.actions, 1, feature_extractor)
    util.simulate(mdp, qLearningSolver, numTrials=num_iterations)
    return qLearningSolver


mdps = [simple_mdp, moderate_mdp, complex_mdp]
solvers = {}

solvers['bucket'] = train(model.bucket_feature_extractor)
solvers['small_bucket'] = train(model.small_bucket_feature_extractor)
solvers['bucket_max'] = train(model.bucket_max_feature_extractor)
solvers['joint'] = train(model.joint_feature_extractor)
solvers['joint_bucket_max'] = train(model.joint_bucket_max_feature_extractor)

bars = {
    'human': [6.21, 9.1, 14.73]
}

names = ['human']

key_value_list = list(solvers.items())

for name, algorithm in key_value_list:
    bars[name] = []
    names.append(name)
    for i in range(len(mdps)):
        problem = mdps[i]
        _, _, turn_avg_list, _ = util.simulate(problem, algorithm, numTrials=1001, incorporate=False)
        bars[name].append(turn_avg_list[0])

barWidth = 0.12

rs = {}
first_name = names[0]
rs[first_name] = np.arange(len(bars[first_name]))

for i in range(1, len(names)):
    name = names[i]
    prev_name = names[i - 1]
    rs[name] = [x + barWidth for x in rs[prev_name]]

for name in names:
    plt.bar(rs[name], bars[name], width=barWidth, edgecolor='white', label=name)

plt.xlabel('group', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars[key_value_list[0][0]]))], ['Simple', 'Moderate', 'Complex'])

plt.title('Performance Compared to Human Agent (' + str(num_iterations) + ' Training Iterations)')
plt.ylabel('Average Turns Taken')
plt.legend()
plt.ylim(0, 75)
plt.show()

