import numpy as np
import networkx as nx
import scikits.statsmodels.api as sm

def Q_fn_gen(basis, weights):
    return lambda x: basis.expand(x) * weights

class gglearner:
    def __init__(self,
                 G0,
                 reward_function,
                 action_functions,
                 action_names,
                 basis_functions,
                 feature_functions,
                 termination_tolerance,
                 max_rows):

        self.G = G0
        self.reward_function
        self.basis = basis(basis_functions)
        self.actions = actions(action_functions,
                               action_names,
                               self.basis
                               len(feature_functions)
                               max_rows)
        self.features = features(feature_functions)
        self.termination_tolerance
        self.max_rows

        

    def episode(self, alpha, gamma, epsilon):
        Q = 0.0
        action_taken = np.random.randint(0, len(self.actions))
        for i in xrange(self.max_rows):
            feature_values = self.features.get(G)
            self.actions.get(action_taken).state.add_sample(feature_values, Q)

            reward = self.reward_function(G)

            if reward <= self.termination_tolerance: break

            Q_values = self.actions.rewards(feature_values)

            if np.random.rand() <= epsilon:
                action_taken = np.random.randint(0, len(self.actions))
            else:
                action_taken = Q_values.argmax()

            Q = (1 - alpha) * Q + alpha * (reward + gamma * Q_values[action_taken])

        for i in xrange(len(actions)):
            actions.get(i).compute_Q_fn()


class features:
    def __init__(self, feature_functions):
        self.feature_functions = feature_functions
        self.vfapply = np.vectorize(lambda f,x: f(x))

    def get(self, G):
        vfapply(feature_functions, G)

class state:
    def __init__(self, max_rows, n_features, basis):
        self.n = 0
        self.basis = basis
        self.design_matrix = np.zeros([max_rows, n_features * len(basis)])
        self.Q = np.zeros(max_rows)

    def add_sample(self, feature_values, Q):
        expanded_features = np.array(1)
        for feature in feature_values:
            expanded_features = append(expanded_features, basis.expand(feature))
        design_matrix[self.n] = expanded_features
        self.Q[n] = Q

    def regress(self):
        self.design_matrix = self.design_matrix[:n]
        self.Q = self.Q[:n]

        res = sm.OLS(Q,design_matrix)
        return res.weights

class basis:
    def __init__(self, functions):
        self.functions = np.array(functions)
        self.vfapply = np.vectorize(lambda f,x: f(x))

    def __len__(self):
        return len(self.functions)

    def expand(self, term):
        return self.vfapply(self.functions, term)

class action:
    def __init__(self, f, name, basis, num_features, max_size):
        self.f = f
        self.basis = basis
        self.name = name
        self.num_features = num_features
        self.max_size = max_size
        self.state = state(max_size, num_features, basis)
        self.Q_function = Q_fn_gen(self.basis, np.zeros(num_features))

    def execute(self, x):
        return self.f(x)

    def compute_Q_fn(self):
        weights = self.state.regress()
        self.Q_function = Q_fn_gen(self.basis, weights)
        self.state = state(self.max_size, self.num_features, self.basis)

class actions:
    def __init__(self, function_iterable, names_iterable, basis, num_features, max_size=1000):
        self.actions_list = [action(f, name, basis, num_features, max_size) for f, name in zip(function_iterable, names_iterable)]
        self.action_dict = {}
        for i,an_action in enumerate(self.actions_list):
            self.action_dict[an_action.name] = i

        self.vfapply = np.vectorize(lambda a,x: a.reward_function(x))

    def __len__(self):
        return len(self.actions_list)

    def get(self, action_id):
        if isinstance(action_id,str):
            action_id = self.action_dict[action_id]
        return self.actions_list[action_id]

    def Qs(self, feature_values):
        return self.vfapply(self.action_list, feature_values)
