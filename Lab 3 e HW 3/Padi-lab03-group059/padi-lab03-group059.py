# %% [markdown]
# # Learning and Decision Making

# %% [markdown]
# ## Laboratory 3: Partially observable Markov decision problems
# 
# In the end of the lab, you should export the notebook to a Python script (``File >> Download as >> Python (.py)``). Make sure that the resulting script includes all code written in the tasks marked as "**Activity n. N**", together with any replies to specific questions posed. Your file should be named `padi-labKK-groupXXX.py`, where `KK` corresponds to the lab number and the `XXX` corresponds to your group number. Similarly, your homework should consist of a single pdf file named `padi-hwKK-groupXXX.pdf`. You should create a zip file with the lab and homework files and submit it in Fenix **at most 30 minutes after your lab is over**.
# 
# Make sure to strictly respect the specifications in each activity, in terms of the intended inputs, outputs and naming conventions.
# 
# In particular, after completing the activities you should be able to replicate the examples provided (although this, in itself, is no guarantee that the activities are correctly completed).

# %% [markdown]
# ### 1. The POMDP model
# 
# Consider once again POMDP problem from the homework. In this lab, you will interact with a larger version of that same problem, which describes the decision process of a gambler involved in a betting game. The game proceeds in rounds. At each round,
# 
# * The player is dealt a random card from the set {A&spades;, K&spades;, Q&spades;}. The card is left facing down.
# * The player must then either bet about which card he/she was dealt, or quit. There is a cost associated to quitting, but which is inferior to that of losing.
# * After betting/quitting, the card is revealed and the round ends.
# * Before betting, the player may also try to peek which card he/she was dealt (which is cheating). Of course that there is a risk associated with peeking (modeled as a cost). Peeking may or may not succeed.
# 
# The game can be described as a POMDP summarized in the diagram:
# 
# <img src="pomdp.png" width="600px">
# 
# In this lab you will use a POMDP based on the aforementioned domain and investigate how to simulate a partially observable Markov decision problem and track its state. You will also compare different MDP heuristics with the optimal POMDP solution.
# 
# **Throughout the lab, unless if stated otherwise, use $\gamma=0.99$.**
# 
# $$\diamond$$
# 
# In this first activity, you will implement a POMDP model in Python. You will start by loading the POMDP information from a `numpy` binary file, using the `numpy` function `load`. The file contains the list of states, actions, observations, transition probability matrices, observation probability matrices, and cost function.

# %% [markdown]
# ---
# 
# #### Activity 1.        
# 
# Write a function named `load_pomdp` that receives, as input, a string corresponding to the name of the file with the POMDP information, and a real number $\gamma$ between $0$ and $1$. The loaded file contains 6 arrays:
# 
# * An array `X` that contains all the states in the POMDP, represented as strings. In the gambling scenario above, for example, there is a total of 14 states, each describing a stage in the game. Specifically,
#     * `"I"` represents the initial state of the game, before the cards are dealt.
#     * `"2A"`, `"2B"`, and `"2C"` represent the 3 "dealt cards" states. The player only observes that the card has been dealt (corresponding to observation "2"), but does not know which of the three cards, "A", "B", or "C", it has.
#     * `"A"`, `"B"`, `"C"`correspond to the states where the player peeked into the hidden card. For example, `"A"` represents the state where the player was dealt card "A" and peeked into it. These states are reached from `"2A"`, `"2B"`, and `"2C"`, respectively, upon selecting the action "Peek".
#     * States `"3A"`, `"3B"`, and `"3C"` correspond to the states where the player bets. For example, `"3A"` represents the state where the player was dealt card "A" and must now make a bet, which can be "A", "B", or "C".
#     * States `"W"`and `"L"` correspond to winning and losing the game, respectively. State `"Q"` corresponds to the "Quit" state.
#     * State `"F"` represents the final state of the game, right before the game resets.
# * An array `A` that contains all the actions in the POMDP, also represented as strings. In the gambling domain above, for example, each action is represented as a single-letter string among `"a"`, `"b"`, and `"c"`, which represent different actions at different stages of the game. For example, in state `"I"`, since the agent does nothing but await the shuffle, all actions are equivalent and correspond to the action "Wait". However, in the dealt states, the actions `"a"`, `"b"`, and `"c"` correspond, respectively, to the actions "Peek", "Bet" and "Quit". In the betting states, actions `"a"`, `"b"` and `"c"` correspond to betting "A", "B", and "C", respectively.
# * An array `Z` that contains all the observations in the POMDP, also represented as strings. In the gambling domains above, for example, there is a total of 10 observations, corresponding to the observable features of the state: `"I"`, `"2"`, `"A"`, `"B"`, `"C"`, `"3"`, `"W"`, `"Q"`, `"L"`, `"F"`.
# * An array `P` containing `len(A)` subarrays, each with dimension `len(X)` &times; `len(X)` and  corresponding to the transition probability matrix for one action.
# * An array `O` containing `len(A)` subarrays, each with dimension `len(X)` &times; `len(Z)` and  corresponding to the observation probability matrix for one action.
# * An array `c` with dimension `len(X)` &times; `len(A)` containing the cost function for the POMDP.
# 
# Your function should create the POMDP as a tuple `(X, A, Z, (Pa, a = 0, ..., len(A)), (Oa, a = 0, ..., len(A)), c, g)`, where `X` is a tuple containing the states in the POMDP represented as strings (see above), `A` is a tuple containing the actions in the POMDP represented as strings (see above), `Z` is a tuple containing the observations in the POMDP represented as strings (see above), `P` is a tuple with `len(A)` elements, where `P[a]` is an `np.array` corresponding to the transition probability matrix for action `a`, `O` is a tuple with `len(A)` elements, where `O[a]` is an `np.array` corresponding to the observation probability matrix for action `a`, `c` is an `np.array` corresponding to the cost function for the POMDP, and `g` is a float, corresponding to the discount and provided as the argument $\gamma$ of your function. Your function should return the POMDP tuple.
# 
# ---

# %%
import numpy as np

def load_pomdp(filename, gamma):
    with np.load(filename, allow_pickle=True) as data:
        X = tuple(data['X'])
        A = tuple(data['A'])
        Z = tuple(data['Z'])
        P = tuple(data['P'])
        O = tuple(data['O'])
        c = data['c']
        
    return (X, A, Z, P, O, c, gamma)







# %% [markdown]
# We provide below an example of application of the function with the file `pomdp.npz` that you can use as a first "sanity check" for your code. Note that, even fixing the seed, the results you obtain may slightly differ.
# 
# ```python
# import numpy.random as rand
# 
# M = load_pomdp('pomdp.npz', 0.99)
# 
# rand.seed(42)
# 
# # States
# print('= State space (%i states) =' % len(M[0]))
# print('\nStates:')
# for i in range(len(M[0])):
#     print(M[0][i]) 
# 
# # Random state
# x = rand.randint(len(M[0]))
# print('\nRandom state: x =', M[0][x])
# 
# # Last state
# print('\nLast state:', M[0][-1])
# 
# # Actions
# print('= Action space (%i actions) =' % len(M[1]))
# for i in range(len(M[1])):
#     print(M[1][i]) 
# 
# # Random action
# a = rand.randint(len(M[1]))
# print('\nRandom action: a =', M[1][a])
# 
# # Observations
# print('= Observation space (%i observations) =' % len(M[2]))
# print('\nObservations:')
# for i in range(len(M[2])):
#     print(M[2][i]) 
# 
# # Random observation
# z = rand.randint(len(M[2]))
# print('\nRandom observation: z =', M[2][z])
# 
# # Last state
# print('\nLast observation:', M[2][-1])
# 
# # Transition probabilities
# print('\n= Transition probabilities =')
# 
# for i in range(len(M[1])):
#     print('\nTransition probability matrix dimensions (action %s):' % M[1][i], M[3][i].shape)
#     print('Dimensions add up for action "%s"?' % M[1][i], np.isclose(np.sum(M[3][i]), len(M[0])))
#     
# print('\nState-action pair (%s, %s) transitions to state(s)' % (M[0][x], M[1][a]))
# print("s' in", np.array(M[0])[np.where(M[3][a][x, :] > 0)])
# 
# # Observation probabilities
# print('\n= Observation probabilities =')
# 
# for i in range(len(M[1])):
#     print('\nObservation probability matrix dimensions (action %s):' % M[1][i], M[4][i].shape)
#     print('Dimensions add up for action "%s"?' % M[1][i], np.isclose(np.sum(M[4][i]), len(M[0])))
#     
# print('\nState-action pair (%s, %s) yields observation(s)' % (M[0][x], M[1][a]))
# print("z in", np.array(M[2])[np.where(M[4][a][x, :] > 0)])
# 
# # Cost
# print('\n= Costs =')
# 
# print('\nCost for the state-action pair (%s, %s):' % (M[0][x], M[1][a]))
# print('c(s, a) =', M[5][x, a])
# 
# # Discount
# print('\n= Discount =')
# print('\ngamma =', M[6])
# ```
# 
# Output:
# 
# ```
# = State space (14 states) =
# 
# States:
# I
# 2A
# 2B
# 2C
# A
# B
# C
# 3A
# 3B
# 3C
# W
# Q
# L
# F
# 
# Random state: x = C
# 
# Last state: F
# = Action space (3 actions) =
# a
# b
# c
# 
# Random action: a = a
# = Observation space (10 observations) =
# 
# Observations:
# I
# 2
# A
# B
# C
# 3
# W
# Q
# L
# F
# 
# Random observation: z = Q
# 
# Last observation: F
# 
# = Transition probabilities =
# 
# Transition probability matrix dimensions (action a): (14, 14)
# Dimensions add up for action "a"? True
# 
# Transition probability matrix dimensions (action b): (14, 14)
# Dimensions add up for action "b"? True
# 
# Transition probability matrix dimensions (action c): (14, 14)
# Dimensions add up for action "c"? True
# 
# State-action pair (C, a) transitions to state(s)
# s' in ['2C']
# 
# = Observation probabilities =
# 
# Observation probability matrix dimensions (action a): (14, 10)
# Dimensions add up for action "a"? True
# 
# Observation probability matrix dimensions (action b): (14, 10)
# Dimensions add up for action "b"? True
# 
# Observation probability matrix dimensions (action c): (14, 10)
# Dimensions add up for action "c"? True
# 
# State-action pair (C, a) yields observation(s)
# z in ['C']
# 
# = Costs =
# 
# Cost for the state-action pair (C, a):
# c(s, a) = 0.0
# 
# = Discount =
# 
# gamma = 0.99
# ```

# %% [markdown]
# ### 2. Sampling
# 
# You are now going to sample random trajectories of your POMDP and observe the impact it has on the corresponding belief.

# %% [markdown]
# ---
# 
# #### Activity 2.
# 
# Write a function called `gen_trajectory` that generates a random POMDP trajectory using a uniformly random policy. Your function should receive, as input, a POMDP described as a tuple like that from **Activity 1** and two integers, `x0` and `n` and return a tuple with 3 elements, where:
# 
# 1. The first element is a `numpy` array corresponding to a sequence of `n + 1` state indices, $x_0,x_1,\ldots,x_n$, visited by the agent when following a uniform policy (i.e., a policy where actions are selected uniformly at random) from state with index `x0`. In other words, you should select $x_1$ from $x_0$ using a random action; then $x_2$ from $x_1$, etc.
# 2. The second element is a `numpy` array corresponding to the sequence of `n` action indices, $a_0,\ldots,a_{n-1}$, used in the generation of the trajectory in 1.;
# 3. The third element is a `numpy` array corresponding to the sequence of `n` observation indices, $z_1,\ldots,z_n$, experienced by the agent during the trajectory in 1.
# 
# The `numpy` array in 1. should have a shape `(n + 1,)`; the `numpy` arrays from 2. and 3. should have a shape `(n,)`.
# 
# **Note:** Your function should work for **any** POMDP specified as above.
# 
# ---

# %%
def gen_trajectory(pomdp, x0, n):
    X=np.arange(len(pomdp[0]))
    A=np.arange(len(pomdp[1]))
    Z=np.arange(len(pomdp[2]))
    x=np.zeros(n+1,dtype=int)
    a=np.zeros(n,dtype=int)
    z=np.zeros(n,dtype=int)
    
    x[0]=x0
    x_current=x0
    for i in range(0,n):
        
        a_current = rand.choice(A)
        a[i]=a_current
        
        trans_prob=pomdp[3][a_current][x_current]
        x_current= rand.choice(X,p=trans_prob)
        x[i+1]=x_current
       
        z_action = pomdp[4][a_current][x_current]
        z_current = rand.choice(Z,p=z_action)
        z[i] = z_current
    return (x,a,z)


# %% [markdown]
# For example, using the POMDP from **Activity 1** you could obtain the following interaction.
# 
# ```python
# rand.seed(42)
# 
# # Number of steps and initial state
# steps = 10
# x0    = 0 # State I
# 
# # Generate trajectory
# t = gen_trajectory(M, x0, steps)
# 
# # Check shapes
# print('Shape of state trajectory:', t[0].shape)
# print('Shape of state trajectory:', t[1].shape)
# print('Shape of state trajectory:', t[2].shape)
# 
# # Print trajectory
# for i in range(steps):
#     print('\n- Time step %i -' % i)
#     print('State:', M[0][t[0][i]], '(state %i)' % t[0][i])
#     print('Action selected:', M[1][t[1][i]], '(action %i)' % t[1][i])
#     print('Resulting state:', M[0][t[0][i+1]], '(state %i)' % t[0][i+1])
#     print('Observation:', M[2][t[2][i]], '(observation %i)' % t[2][i])
# ```
# 
# Output:
# 
# ```
# Shape of state trajectory: (11,)
# Shape of state trajectory: (10,)
# Shape of state trajectory: (10,)
# 
# - Time step 0 -
# State: I (state 0)
# Action selected: c (action 2)
# Resulting state: 2C (state 3)
# Observation: 2 (observation 1)
# 
# - Time step 1 -
# State: 2C (state 3)
# Action selected: a (action 0)
# Resulting state: C (state 6)
# Observation: C (observation 4)
# 
# - Time step 2 -
# State: C (state 6)
# Action selected: c (action 2)
# Resulting state: 2C (state 3)
# Observation: 2 (observation 1)
# 
# - Time step 3 -
# State: 2C (state 3)
# Action selected: c (action 2)
# Resulting state: Q (state 11)
# Observation: Q (observation 7)
# 
# - Time step 4 -
# State: Q (state 11)
# Action selected: b (action 1)
# Resulting state: F (state 13)
# Observation: F (observation 9)
# 
# - Time step 5 -
# State: F (state 13)
# Action selected: a (action 0)
# Resulting state: I (state 0)
# Observation: I (observation 0)
# 
# - Time step 6 -
# State: I (state 0)
# Action selected: a (action 0)
# Resulting state: 2A (state 1)
# Observation: 2 (observation 1)
# 
# - Time step 7 -
# State: 2A (state 1)
# Action selected: c (action 2)
# Resulting state: Q (state 11)
# Observation: Q (observation 7)
# 
# - Time step 8 -
# State: Q (state 11)
# Action selected: c (action 2)
# Resulting state: F (state 13)
# Observation: F (observation 9)
# 
# - Time step 9 -
# State: F (state 13)
# Action selected: c (action 2)
# Resulting state: I (state 0)
# Observation: I (observation 0)
# ```

# %% [markdown]
# You will now write a function that samples a given number of possible belief points for a POMDP. To do that, you will use the function from **Activity 2**.
# 
# ---
# 
# #### Activity 3.
# 
# Write a function called `sample_beliefs` that receives, as input, a POMDP described as a tuple like that from **Activity 1** and an integer `n`, and return a tuple with `n + 1` elements **or less**, each corresponding to a possible belief state (represented as a $1\times|\mathcal{X}|$ vector). To do so, your function should
# 
# * Generate a trajectory with `n` steps from a random initial state, using the function `gen_trajectory` from **Activity 2**.
# * For the generated trajectory, compute the corresponding sequence of beliefs, assuming that the agent does not know its initial state (i.e., the initial belief is the uniform belief, and should also be considered). 
# 
# Your function should return a tuple with the resulting beliefs, **ignoring duplicate beliefs or beliefs whose distance is smaller than $10^{-3}$.**
# 
# **Suggestion:** You may want to define an auxiliary function `belief_update` that receives a POMDP, a belief, an action and an observation and returns the updated belief.
# 
# **Note:** Your function should work for **any** POMDP specified as above. To compute the distance between vectors, you may find useful `numpy`'s function `linalg.norm`.
# 
# 
# ---

# %%



def belief_update(POMDP, belief, action, observation):
    X, A, Z, P, O, c, gamma = POMDP
    
    
    updated_belief = np.dot(belief ,P[action]* O[action][:, observation])
    updated_belief_norm = np.linalg.norm(updated_belief, 1)
    

    updated_belief /= updated_belief_norm
    updated_belief = np.reshape(updated_belief, (1,-1))
    return updated_belief


def sample_beliefs(pomdp, n):
    l = list()
    n_states = len(pomdp[0])
    X = list(range(n_states))
    
    
    
    x_random= rand.choice(X)
    trajectory = gen_trajectory(pomdp,x_random, n)
    
    current_bel = np.ones(len(X)) / n_states
    
    l.append(np.array(current_bel,ndmin=2))
    
    dup=False
    for i in range(n):
        act = trajectory[1][i]
        obs = trajectory[2][i]
        new_bel = belief_update(pomdp, current_bel, act, obs)
        current_bel = new_bel
        
        for b in l:
            if(np.linalg.norm(b - current_bel) < 10e-3):
                dup=True
        if(not dup):
            l.append(current_bel)
        dup=False
    return l


# %% [markdown]
# For example, using the POMDP from **Activity 1** you could obtain the following interaction.
# 
# ```python
# rand.seed(42)
# 
# # 3 sample beliefs + initial belief
# B = sample_beliefs(M, 3)
# print('%i beliefs sampled:' % len(B))
# for i in range(len(B)):
#     print(np.round(B[i], 3))
#     print('Belief adds to 1?', np.isclose(B[i].sum(), 1.))
# 
# # 100 sample beliefs
# B = sample_beliefs(M, 100)
# print('%i beliefs sampled.' % len(B))
# ```
# 
# Output:
# 
# ```
# 4 beliefs sampled:
# [[0.071 0.071 0.071 0.071 0.071 0.071 0.071 0.071 0.071 0.071 0.071 0.071
#   0.071 0.071]]
# Belief adds to 1? True
# [[0.   0.5  0.25 0.25 0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  ]]
# Belief adds to 1? True
# [[0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]]
# Belief adds to 1? True
# [[0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
# Belief adds to 1? True
# 15 beliefs sampled.
# rand.seed(42)
# 
# # 3 sample beliefs + initial belief
# B = sample_beliefs(M, 3)
# print('%i beliefs sampled:' % len(B))
# for i in range(len(B)):
#     print(np.round(B[i], 3))
#     print('Belief adds to 1?', np.isclose(B[i].sum(), 1.))
# 
# # 100 sample beliefs
# B = sample_beliefs(M, 100)
# print('%i beliefs sampled.' % len(B))
# ```

# %% [markdown]
# ### 3. MDP-based heuristics
# 
# In this section you are going to compare different heuristic approaches for POMDPs discussed in class.

# %% [markdown]
# ---
# 
# #### Activity 4
# 
# Write a function `solve_mdp` that takes as input a POMDP represented as a tuple like that of **Activity 1** and returns a `numpy` array corresponding to the **optimal $Q$-function for the underlying MDP**. Stop the algorithm when the error between iterations is smaller than $10^{-8}$.
# 
# **Note:** Your function should work for **any** POMDP specified as above. Feel free to reuse one of the functions you implemented in Lab 2 (for example, value iteration).
# 
# ---

# %%
def solve_mdp(M):
    X = M[0]
    A = M[1]
    Z = M[2]
    P = M[3]
    O = M[4]
    c = M[5]
    y = M[6]

    J = np.zeros((len(X), 1))
    error = 1
    i = 0

    while error > 10e-8:
        result = list()

        for a in range(0, len(A)):
            ca = c[:,a]
            ca = ca.reshape((len(X),1))
            Qa = ca + y * P[a].dot(J)
            result.append(Qa)
        Jnew = np.min(result, axis=0)

        error = np.linalg.norm(Jnew - J)
        i +=1
        J = Jnew

    final_results = (np.transpose(result)).reshape(len(X), len(A))
    return final_results



# %% [markdown]
# As an example, you can run the following code on the POMDP from **Activity 1**.
# 
# ```python
# Q = solve_mdp(M)
# 
# x = 6 # State C
# print('\nQ-values at state %s:' % M[0][x], np.round(Q[x, :], 3))
# print('Best action at state %s:' % M[0][x], M[1][np.argmin(Q[x, :])])
# 
# x = 3 # State 2C
# print('\nQ-values at state %s:' % M[0][x], np.round(Q[x, :], 3))
# print('Best action at state %s:' % M[0][x], M[1][np.argmin(Q[x, :])])
# 
# x = 12 # L
# print('\nQ-values at state %s:' % M[0][x], np.round(Q[x, :], 3))
# print('Best action at state %s:' % M[0][x], M[1][np.argmin(Q[x, :])])
# ```
# 
# Output:
# 
# ```
# Q-values at state C: [0. 0. 0.]
# Best action at state C: a
# 
# Q-values at state 2C: [0.3   0.    0.742]
# Best action at state 2C: b
# 
# Q-values at state L: [1. 1. 1.]
# Best action at state L: a
# ```

# %% [markdown]
# ---
# 
# #### Activity 5
# 
# You will now test the different MDP heuristics discussed in class. To that purpose, write down a function that, given a belief vector and the solution for the underlying MDP, computes the action prescribed by each of the three MDP heuristics. In particular, you should write down a function named `get_heuristic_action` that receives, as inputs:
# 
# * A belief state represented as a `numpy` array like those of **Activity 3**;
# * The optimal $Q$-function for an MDP (computed, for example, using the function `solve_mdp` from **Activity 4**);
# * A string that can be either `"mls"`, `"av"`, or `"q-mdp"`;
# 
# Your function should return an integer corresponding to the index of the action prescribed by the heuristic indicated by the corresponding string, i.e., the most likely state heuristic for `"mls"`, the action voting heuristic for `"av"`, and the $Q$-MDP heuristic for `"q-mdp"`.
# 
# ---

# %%
def get_heuristic_action(belief, Q, heuristic):
    b = belief[0]

    if heuristic == "av":
        voting = np.zeros((len(Q[1]),))
        for i in range(len(Q[0])):
            index = np.argmin(Q[i])



        return index

    elif heuristic == "q-mdp":
        q_mdp = np.zeros(Q.shape[1])
        for i in range(len(belief)):
            q_mdp += b[1] * Q[i,:]
        return np.argmin(q_mdp)

    elif heuristic == "mls":
        x = np.argmax(b)
        c = Q[x, :]
        return np.argmin(c)
    







# %% [markdown]
# For example, if you run your function in the examples from **Activity 3** using the $Q$-function from **Activity 4**, you can observe the following interaction.
# 
# ```python
# rand.seed(42)
# 
# for b in B[:10]:
#     
#     if np.all(b > 0):
#         print('Belief (approx.) uniform')
#     else:
#         initial = True
# 
#         for i in range(len(M[0])):
#             if b[0, i] > 0:
#                 if initial:
#                     initial = False
#                     print('Belief: [', M[0][i], ': %.3f' % np.round(b[0, i], 3), end='')
#                 else:
#                     print(',', M[0][i], ': %.3f' % np.round(b[0, i], 3), end='')
#         print(']')
# 
#     print('MLS action:', M[1][get_heuristic_action(b, Q, 'mls')], end='; ')
#     print('AV action:', M[1][get_heuristic_action(b, Q, 'av')], end='; ')
#     print('Q-MDP action:', M[1][get_heuristic_action(b, Q, 'q-mdp')])
# 
#     print()
# ```
# 
# Output:
# 
# ```
# Belief (approx.) uniform
# MLS action: b; AV action: b; Q-MDP action: b
# 
# Belief: [ L : 1.000]
# MLS action: b; AV action: a; Q-MDP action: a
# 
# Belief: [ F : 1.000]
# MLS action: a; AV action: c; Q-MDP action: b
# 
# Belief: [ I : 1.000]
# MLS action: c; AV action: a; Q-MDP action: c
# 
# Belief: [ 2A : 1.000]
# MLS action: b; AV action: b; Q-MDP action: b
# 
# Belief: [ Q : 1.000]
# MLS action: a; AV action: a; Q-MDP action: b
# 
# Belief: [ A : 1.000]
# MLS action: b; AV action: a; Q-MDP action: b
# 
# Belief: [ 3A : 1.000]
# MLS action: a; AV action: a; Q-MDP action: a
# 
# Belief: [ 2C : 1.000]
# MLS action: b; AV action: b; Q-MDP action: b
# 
# Belief: [ C : 1.000]
# MLS action: b; AV action: b; Q-MDP action: a
# ```

# %% [markdown]
# You will now implement the last heuristic, the "Fast Informed Bound" (or FIB) heuristic. To that purpose, you will write a function to compute the FIB Q-function.
# 
# ---
# 
# #### Activity 6
# 
# Write a function `solve_fib` that takes as input a POMDP represented as a tuple like that of **Activity 1** and returns a `numpy` array corresponding to the **FIB $Q$-function**, that verifies the recursion
# 
# $$Q_{FIB}(x,a)=c(x,a)+\gamma\sum_{z\in\mathcal{Z}}\min_{a'\in\mathcal{A}}\sum_{x'\in\mathcal{X}}\mathbf{P}(x'\mid x,a)\mathbf{O}(z\mid x',a)Q_{FIB}(x',a').$$
# 
# Stop the algorithm when the error between iterations is smaller than $10^{-1}$. Compare the FIB heuristic with the Q-MDP heuristic. What can you conclude?
# 
# **Note:** Your function should work for **any** POMDP specified as above.
# 
# ---

# %%
def solve_fib(pomdp):
    n_states = len(pomdp[0])
    n_actions = len(pomdp[1])
    Q=np.zeros((n_states,n_actions))
    error=1
    A = pomdp[1]
    Z = pomdp[2]
    P = pomdp[3]
    O = pomdp[4]
    while(error >= 1e-1):
        q_new=np.zeros((n_states,n_actions))
        for a in range(len(A)):
            for z in range(len(Z)):
                q_new[:,a]+=np.min( ( P[a]*O[a][:,z] ).dot(Q),axis=1)
                
        q_new = pomdp[5]+ pomdp[6]*q_new
        error = np.linalg.norm(Q-q_new)
        Q=q_new
    return Q
Qfib = solve_fib(M)

x = 6 # State C
print('\nQ-values at state %s:' % M[0][x], np.round(Qfib[x, :], 3))
print('Best action at state %s:' % M[0][x], M[1][np.argmin(Qfib[x, :])])

x = 3 # State 2C
print('\nQ-values at state %s:' % M[0][x], np.round(Qfib[x, :], 3))
print('Best action at state %s:' % M[0][x], M[1][np.argmin(Qfib[x, :])])

x = 12 # State L
print('\nQ-values at state %s:' % M[0][x], np.round(Qfib[x, :], 3))
print('Best action at state %s:' % M[0][x], M[1][np.argmin(Qfib[x, :])])

print()
import numpy.random as rand
rand.seed(42)

# Comparing the prescribed actions
for b in B[:10]:
    
    if np.all(b > 0):
        print('Belief (approx.) uniform')
    else:
        initial = True

        for i in range(len(M[0])):
            if b[0, i] > 0:
                if initial:
                    initial = False
                    print('Belief: [', M[0][i], ': %.3f' % np.round(b[0, i], 3), end='')
                else:
                    print(',', M[0][i], ': %.3f' % np.round(b[0, i], 3), end='')
        print(']')

    print('MLS action:', M[1][get_heuristic_action(b, Q, 'mls')], end='; ')
    print('AV action:', M[1][get_heuristic_action(b, Q, 'av')], end='; ')
    print('Q-MDP action:', M[1][get_heuristic_action(b, Q, 'q-mdp')], end='; ')
    print('FIB action:', M[1][get_heuristic_action(b, Qfib, 'q-mdp')])

    print()

# %% [markdown]
# Using the function `solve_fib` in the function from `get_heuristic_action` from Activity 5 for the beliefs in the example from **Activity 3**, you can observe the following interaction.
# 
# ```python
# Qfib = solve_fib(M)
# 
# x = 6 # State C
# print('\nQ-values at state %s:' % M[0][x], np.round(Qfib[x, :], 3))
# print('Best action at state %s:' % M[0][x], M[1][np.argmin(Qfib[x, :])])
# 
# x = 3 # State 2C
# print('\nQ-values at state %s:' % M[0][x], np.round(Qfib[x, :], 3))
# print('Best action at state %s:' % M[0][x], M[1][np.argmin(Qfib[x, :])])
# 
# x = 12 # State L
# print('\nQ-values at state %s:' % M[0][x], np.round(Qfib[x, :], 3))
# print('Best action at state %s:' % M[0][x], M[1][np.argmin(Qfib[x, :])])
# 
# print()
# 
# rand.seed(42)
# 
# # Comparing the prescribed actions
# for b in B[:10]:
#     
#     if np.all(b > 0):
#         print('Belief (approx.) uniform')
#     else:
#         initial = True
# 
#         for i in range(len(M[0])):
#             if b[0, i] > 0:
#                 if initial:
#                     initial = False
#                     print('Belief: [', M[0][i], ': %.3f' % np.round(b[0, i], 3), end='')
#                 else:
#                     print(',', M[0][i], ': %.3f' % np.round(b[0, i], 3), end='')
#         print(']')
# 
#     print('MLS action:', M[1][get_heuristic_action(b, Q, 'mls')], end='; ')
#     print('AV action:', M[1][get_heuristic_action(b, Q, 'av')], end='; ')
#     print('Q-MDP action:', M[1][get_heuristic_action(b, Q, 'q-mdp')], end='; ')
#     print('FIB action:', M[1][get_heuristic_action(b, Qfib, 'q-mdp')])
# 
#     print()
# ```
# 
# Output:
# 
# ```
# Q-values at state C: [0. 0. 0.]
# Best action at state C: a
# 
# Q-values at state 2C: [0.3   0.    0.742]
# Best action at state 2C: b
# 
# Q-values at state L: [1. 1. 1.]
# Best action at state L: a
# 
# Belief (approx.) uniform
# MLS action: b; AV action: b; Q-MDP action: b; FIB action: b
# 
# Belief: [ L : 1.000]
# MLS action: a; AV action: a; Q-MDP action: a; FIB action: c
# 
# Belief: [ F : 1.000]
# MLS action: b; AV action: c; Q-MDP action: a; FIB action: c
# 
# Belief: [ I : 1.000]
# MLS action: c; AV action: a; Q-MDP action: a; FIB action: a
# 
# Belief: [ 2A : 1.000]
# MLS action: b; AV action: b; Q-MDP action: b; FIB action: b
# 
# Belief: [ Q : 1.000]
# MLS action: b; AV action: a; Q-MDP action: a; FIB action: b
# 
# Belief: [ A : 1.000]
# MLS action: b; AV action: c; Q-MDP action: a; FIB action: b
# 
# Belief: [ 3A : 1.000]
# MLS action: a; AV action: a; Q-MDP action: a; FIB action: a
# 
# Belief: [ 2C : 1.000]
# MLS action: b; AV action: b; Q-MDP action: b; FIB action: b
# 
# Belief: [ C : 1.000]
# MLS action: a; AV action: a; Q-MDP action: c; FIB action: b
# ```

# %% [markdown]
# <font color="blue">As our results were different from the ones provided we will comment based on those results. Q-MDP and FIB are 2 different methods. The first one tries to maximize the expected Q-value of each action while FIB is designed to approximate the Q-function. There is indeed some states where these two heuristics don't agree and that is due to the way the belief uncertainty integrates the formula of both methods.</font>


