#!/usr/bin/env python
"""
Plot the distribution of actions by iteration.  

First argument is an iteration,action file produced by parse-actions.awk.
Second argument is the title of the plot.
Third argument is the output filename of the plot.

"""
import matplotlib.pyplot as plt
import numpy as np
import sys

iterations = []
actions = []

actionrecords = open(sys.argv[1],'r')

for record in actionrecords.readlines():
    iteration, action = record.strip().split(",")
    iterations.append(iteration)
    actions.append(action)

action_names = set(actions)
x = np.array(iterations).astype(float)
actions_a = np.array(actions)

plt.figure()

for name in action_names:
    y = (actions_a == name)
    y = y.cumsum() / x
    #print y, len(y)
    plt.plot(x,y)

plt.xlabel("iteration")
plt.ylabel("action proportion")
plt.title(sys.argv[2])

plt.legend(list(action_names),loc="best")

plt.savefig(sys.argv[3])

