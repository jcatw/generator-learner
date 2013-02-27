import numpy as np


class episode:
    """
    class episode provides the machinery for parameterizing and
    running a single episode of the Q-learner.
    """
    def __init__(self, learner, alpha, gamma, epsilon):
        self.learner = learner
        self.G = None
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.iterations = 0
        

    #TODO: clean up method comments
    def run(self, n_iter, draw_steps = False, animate = None): 
        """ 
        Run the Q-learner.  If draw_steps is true, draw and 
        show the learned graph at each iteration.

        Parameters
        ----------
        draw_steps : bool
          view a graph at each iteration?

        animate : None, string
          If string, create an animated gif of network evolution, 
          storing animation and frames in directory specified by animate.
        
        """ 
        self.n_iter = n_iter
        self.actions_taken = -1 * np.ones(n_iter)
                
        #print self.learner.actions.action_dict 
        logging.info("action dictionary: %s" % (self.learner.actions.action_dict,))
        #Q = 0.0
        action_A = np.random.randint(0, len(self.learner.actions))

        for i in xrange(n_iter):
            prev_q = self.learner.actions.get(action_A).q(self.learner.features.get(self.learner.G))
            #if not i%100: print i
            logging.debug("iteration %s" % (i,))
            #print i

            #if np.random.rand() <= self.epsilon:
            #    action_A = np.random.randint(0, len(self.learner.actions))
            #else:
            #    action_A = Q_values.argmax()

            self.learner.actions.get(action_A).execute(self.learner.G)

            reward = self.learner.reward_function(self.learner.G)

                
            if draw_steps:
                #print "Action taken: %s" % (self.learner.actions.action_dict[action_A],)
                nx.draw(self.learner.G)
                plt.show()
                #raw_input("Press Enter to continue")

            if self.learner.termination_function(self.learner.G):
                
                break

            #new_feature_values = self.learner.features.get(self.learner.G)
            #self.learner.actions.get(action_A).state.add_sample(feature_values, Q)

            #print self.learner.actions.get(action_A).state.design_matrix[self.learner.actions.get(action_A).state.n-1], Q

            #if reward == 3: break
            
            

            #if self.learner.termination_function(reward): break
            #if self.learner.termination_function(G): break

            feature_vals = self.learner.features.get(self.learner.G)
            Q_values = self.learner.actions.Qs(feature_vals)

            if np.random.rand() <= self.epsilon:
                action_A = np.random.randint(0, len(self.learner.actions))
                #print "random action: %s" % (self.learner.actions.action_dict[action_A],)
                logging.debug("random action: %s" % (self.learner.actions.action_dict[action_A],))
            else:
                #action_A = Q_values.argmax()
                action_A = self.learner.actions.rand_max_Q_index(feature_vals)
                #print "optimal action: %s" % (self.learner.actions.action_dict[action_A],)
                logging.debug("optimal action: %s" % (self.learner.actions.action_dict[action_A],))
                
            self.actions_taken[i] = action_A
            #Q = (1 - self.alpha) * Q + self.alpha * (reward + self.gamma * Q_values[action_A])
            logging.debug("Q: %s" % (Q_values[action_A],))
            self.learner.actions.get(action_A).w += self.alpha * (reward + self.gamma * Q_values[action_A] - prev_q) * self.learner.basis.array_expand(feature_vals)
            #print self.learner.actions.get(action_A).w
            logging.debug("w: %s" % (self.learner.actions.get(action_A).w,))            

        #for i in xrange(len(self.learner.actions)):
        #    self.learner.actions.get(i).compute_Q_fn()


        self.iterations = i
        self.actions_taken = self.actions_taken[:self.iterations]
        self.G = self.learner.G

     
