import numpy as np
import numpy.random as rand

class action:
    """
    action instances provide the means to both execute an action 
    and to compute that action's value function.
    """
    def __init__(self, f, name, basis, num_features):
        self.f = f
        self.basis = basis
        self.name = name
        self.num_features = num_features
        self.w = np.zeros(len(self.basis)*num_features)
        #self.max_size = max_size
        #self.state = state(max_size, num_features, basis)
        #self.Q_function = Q_fn(self.basis, np.zeros(len(self.basis)*num_features))

    def execute(self, x):
        return self.f(x)

    def q(self, x):
        return (self.basis.array_expand(x) * self.w).sum()
    #def compute_Q_fn(self):
    #    weights = self.state.regress()
    #    self.Q_function = Q_fn(self.basis, weights)
    #    self.state = state(self.max_size, self.num_features, self.basis)

class actions:
    """
    An actions instance is a container for action instances.  
    It provides convenience functions and length semantics.
    """
    def __init__(self, function_iterable, names_iterable, basis, num_features):
        self.actions_list = [action(f, name, basis, num_features) for f, name in zip(function_iterable, names_iterable)]
        self.action_dict = {}
        for i,an_action in enumerate(self.actions_list):
            self.action_dict[i] = an_action.name

        # self.vfapply = np.vectorize(lambda a,x: a.Q_function.eval(x))

    def __len__(self):
        return len(self.actions_list)

    def get(self, action_id):
        if isinstance(action_id,str):
            action_id = self.action_dict[action_id]
        return self.actions_list[action_id]

    def Qs(self, feature_values):
        #return self.vfapply(self.actions_list, feature_values)
        #return np.array([a.Q_function.eval(feature_values) for a in self.actions_list])
        return np.array([a.q(feature_values) for a in self.actions_list])

    def max_Q_indices(self, feature_values):
        these_Qs = self.Qs(feature_values)
        max_indices = [0]
        max_value = these_Qs[0]

        for i in range(1,len(these_Qs)):
            if these_Qs[i] > max_value:
                max_indices = [i]
                max_value = these_Qs[i]
            elif these_Qs[i] == max_value:
                max_indices.append(i)

        return max_indices

    def rand_max_Q_index(self, feature_values):
        max_indices = self.max_Q_indices(feature_values)

        return max_indices[rand.randint(len(max_indices))]
        
