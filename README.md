# Machine Learning Engineer Nanodegree
## Reinforcement Learning
## Project:
Train a Smartcab to Drive

Welcome to the fourth project of the Machine Learning
Engineer Nanodegree! In this notebook, template code has already been provided
for you to aid in your analysis of the *Smartcab* and your implemented learning
algorithm. You will not need to modify the included code beyond what is
requested. There will be questions that you must answer which relate to the
project and the visualizations provided in the notebook. Each section where you
will answer a question is preceded by a **'Question X'** header. Carefully read
each question and provide thorough answers in the following text boxes that
begin with **'Answer:'**. Your project submission will be evaluated based on
your answers to each of the questions and the implementation you provide in
`agent.py`.  

>**Note:** Code and Markdown cells can be executed using the
**Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited
by typically double-clicking the cell to enter edit mode.

-----

## Getting Started
In this project, you will work towards constructing an
optimized Q-Learning driving agent that will navigate a *Smartcab* through its
environment towards a goal. Since the *Smartcab* is expected to drive passengers
from one location to another, the driving agent will be evaluated on two very
important metrics: **Safety** and **Reliability**. A driving agent that gets the
*Smartcab* to its destination while running red lights or narrowly avoiding
accidents would be considered **unsafe**. Similarly, a driving agent that
frequently fails to reach the destination in time would be considered
**unreliable**. Maximizing the driving agent's **safety** and **reliability**
would ensure that *Smartcabs* have a permanent place in the transportation
industry.

**Safety** and **Reliability** are measured using a letter-grade
system as follows:

| Grade 	| Safety 	| Reliability 	|
|:-----:	|:------:
|:-----------:	|
|   A+  	|  Agent commits no traffic violations,<br/>and always
chooses the correct action. | Agent reaches the destination in time<br />for
100% of trips. |
|   A   	|  Agent commits few minor traffic
violations,<br/>such as failing to move on a green light. | Agent reaches the
destination on time<br />for at least 90% of trips. |
|   B   	| Agent commits
frequent minor traffic violations,<br/>such as failing to move on a green light.
| Agent reaches the destination on time<br />for at least 80% of trips. |
|   C
|  Agent commits at least one major traffic violation,<br/> such as driving
through a red light. | Agent reaches the destination on time<br />for at least
70% of trips. |
|   D   	| Agent causes at least one minor accident,<br/> such
as turning left on green with oncoming traffic.       	| Agent reaches the
destination on time<br />for at least 60% of trips. |
|   F   	|  Agent causes
at least one major accident,<br />such as driving through a red light with
cross-traffic.      	| Agent fails to reach the destination on time<br />for at
least 60% of trips. |

To assist evaluating these important metrics, you will
need to load visualization code that will be used later on in the project. Run
the code cell below to import this code which is required for your analysis.

```python
# Import the visualization code
import visuals as vs

# Pretty display for notebooks
%matplotlib inline
```

### Understand the World
Before starting to work on implementing your driving
agent, it's necessary to first understand the world (environment) which the
*Smartcab* and driving agent work in. One of the major components to building a
self-learning agent is understanding the characteristics about the agent, which
includes how the agent operates. To begin, simply run the `agent.py` agent code
exactly how it is -- no need to make any additions whatsoever. Let the resulting
simulation run for some time to see the various working components. Note that in
the visual simulation (if enabled), the **white vehicle** is the *Smartcab*.

### Question 1
In a few sentences, describe what you observe during the
simulation when running the default `agent.py` agent code. Some things you could
consider:
- *Does the Smartcab move at all during the simulation?*
- *What kind
of rewards is the driving agent receiving?*
- *How does the light changing color
affect the rewards?*  

**Hint:** From the `/smartcab/` top-level directory
(where this notebook is located), run the command 
```bash
'python
smartcab/agent.py'
```

**Answer:**

The Smartcab does not move during simulation. So basically it
receives a reward if its decision is "Stay at one place", which corresponds to
'None' value.

Driving agent receives decimal point as a reward. Note, that in
first simulation, driver agent is not moving, but based on the UI, looks like
smartcab is receiving decimal reward for its decision not to move anywhere.

It
receives positive reward, when the light is red - because it can't move,
therefore not moving when the light is red is good in terms of the rewards!
Amazing! Otherwise it receives negative reward.

### Understand the Code
In addition to understanding the world, it is also
necessary to understand the code itself that governs how the world, simulation,
and so on operate. Attempting to create a driving agent would be difficult
without having at least explored the *"hidden"* devices that make everything
work. In the `/smartcab/` top-level directory, there are two folders: `/logs/`
(which will be used later) and `/smartcab/`. Open the `/smartcab/` folder and
explore each Python file included, then answer the following question.

### Question 2
- *In the *`agent.py`* Python file, choose three flags that can
be set and explain how they change the simulation.*
- *In the *`environment.py`*
Python file, what Environment class function is called when an agent performs an
action?*
- *In the *`simulator.py`* Python file, what is the difference between
the *`'render_text()'`* function and the *`'render()'`* function?*
- *In the
*`planner.py`* Python file, will the *`'next_waypoint()`* function consider the
North-South or East-West direction first?*

**Answer:**

Agent.py

1) num_dummies - discrete number of dummy agents in the
environment, default is 100 -  it is basically number of other cars on the road.
If this number is large it will be more difficult to drive when there are a lot
of different cars - dummy agents(bots)

2) grid_size   - discrete number of
intersections (columns, rows), default is (8, 6) 
   It is basically number of
roads from North to south and from West to East. When it is large there are more
posibilities to go for the driving agent.

3) epsilon - continuous value for the
exploration factor, default is 1.
   This should work as a measure weather
explore more(start choosing unknown states) or exploit more(continue running
into states which are familiar)
   
Environment.py

act(self, agent, action)
function is called when an agent performs an action

Simulator.py

render()
function renders GUI display of the simulation
render_text() function renders
non-GUI display of the simulation into termial/command prompt

Planner.py
next_waypoint() function considers East-West directions first

-----
## Implement a Basic Driving Agent

The first step to creating an
optimized Q-Learning driving agent is getting the agent to actually take valid
actions. In this case, a valid action is one of `None`, (do nothing) `'left'`
(turn left), `right'` (turn right), or `'forward'` (go forward). For your first
implementation, navigate to the `'choose_action()'` agent function and make the
driving agent randomly choose one of these actions. Note that you have access to
several class variables that will help you write this functionality, such as
`'self.learning'` and `'self.valid_actions'`. Once implemented, run the agent
file and simulation briefly to confirm that your driving agent is taking a
random action each time step.

### Basic Agent Simulation Results
To obtain results from the initial
simulation, you will need to adjust following flags:
- `'enforce_deadline'` -
Set this to `True` to force the driving agent to capture whether it reaches the
destination in time.
- `'update_delay'` - Set this to a small value (such as
`0.01`) to reduce the time between steps in each trial.
- `'log_metrics'` - Set
this to `True` to log the simluation results as a `.csv` file in `/logs/`.
-
`'n_test'` - Set this to `'10'` to perform 10 testing trials.

Optionally, you
may disable to the visual simulation (which can make the trials go faster) by
setting the `'display'` flag to `False`. Flags that have been set here should be
returned to their default setting when debugging. It is important that you
understand what each flag does and how it affects the simulation!

Once you have
successfully completed the initial simulation (there should have been 20
training trials and 10 testing trials), run the code cell below to visualize the
results. Note that log files are overwritten when identical simulations are run,
so be careful with what log file is being loaded!
Run the agent.py file after
setting the flags from projects/smartcab folder instead of
projects/smartcab/smartcab.

```python
# Load the 'sim_no-learning' log file from the initial simulation results
vs.plot_trials('sim_no-learning.csv')
```

### Question 3
Using the visualization above that was produced from your initial
simulation, provide an analysis and make several observations about the driving
agent. Be sure that you are making at least one observation about each panel
present in the visualization. Some things you could consider:
- *How frequently
is the driving agent making bad decisions? How many of those bad decisions cause
accidents?*
- *Given that the agent is driving randomly, does the rate of
reliability make sense?*
- *What kind of rewards is the agent receiving for its
actions? Do the rewards suggest it has been penalized heavily?*
- *As the number
of trials increases, does the outcome of results change significantly?*
- *Would
this Smartcab be considered safe and/or reliable for its passengers? Why or why
not?*

**Answer:**

1) With probability of approx. 0.46 this driving agent makes bad
decisions. Approximately, with probablility of 0.14 driving agent suffers
accidents, both minor and major. This means that 33% of bad decisions cause
those types of accidents.

2) Rate of reliability does really makes sense when
agent makes on its finite small grid. It is poor but not zero.

3) This driving
agent receives negative reward in most cases, so 10 trials moving average always
lower than -5. This shows that the driving agent is penalized heavily.

4) When
the number of trials changes, the results are not changing significantly.

5)
This smartcab is not safe and not reliable because it frequently causes
accidents and goal is reached only by lucky chance.

-----
## Inform the Driving Agent
The second step to creating an optimized
Q-learning driving agent is defining a set of states that the agent can occupy
in the environment. Depending on the input, sensory data, and additional
variables available to the driving agent, a set of states can be defined for the
agent so that it can eventually *learn* what action it should take when
occupying a state. The condition of `'if state then action'` for each state is
called a **policy**, and is ultimately what the driving agent is expected to
learn. Without defining states, the driving agent would never understand which
action is most optimal -- or even what environmental variables and conditions it
cares about!

### Identify States
Inspecting the `'build_state()'` agent function shows that
the driving agent is given the following data from the environment:
-
`'waypoint'`, which is the direction the *Smartcab* should drive leading to the
destination, relative to the *Smartcab*'s heading.
- `'inputs'`, which is the
sensor data from the *Smartcab*. It includes 
  - `'light'`, the color of the
light.
  - `'left'`, the intended direction of travel for a vehicle to the
*Smartcab*'s left. Returns `None` if no vehicle is present.
  - `'right'`, the
intended direction of travel for a vehicle to the *Smartcab*'s right. Returns
`None` if no vehicle is present.
  - `'oncoming'`, the intended direction of
travel for a vehicle across the intersection from the *Smartcab*. Returns `None`
if no vehicle is present.
- `'deadline'`, which is the number of actions
remaining for the *Smartcab* to reach the destination before running out of
time.

### Question 4
*Which features available to the agent are most relevant for
learning both **safety** and **efficiency**? Why are these features appropriate
for modeling the *Smartcab* in the environment? If you did not choose some
features, why are those features* not *appropriate? Please note that whatever
features you eventually choose for your agent's state, must be argued for here.
That is: your code in agent.py should reflect the features chosen in this
answer.
*

NOTE: You are not allowed to engineer new features for the smartcab.

**Answer:**

In my opinion features needed to make this driving agent safe and
reliable should take into account environment, direction and time necessary to
get to the goal. So waypoint, inputs for oncomming and left trafic would be good
example of state tuple. Waypoint describes the way to the goal, inputs show the
environmental parameters, like traffic light and the position of other driving
agents. If I take all the parameters as state members, the number of all
possible states will be huge and the could not be able to learn from the
simulation.

Waypoint parameter in state is necessary to define the path to the
goal. Inputs['light'] - for determining traffic light on crossroad.
Inputs['oncoming'] - necessary to check oncomming traffic direction and follow
the traffic rules when situation involves oncoming traffic. Inputs['left'] -
used to follow the traffic rule when there is red light on it should be possible
to turn right when no car from the left is moving forward. All other states are
not necessary for this agent to make safe and reliable dicisions.
Inpust['right'] - does not provide any other information which is not reflected
by the parameters above. It simply can be avoided because there are no
exceptions where car from the right is involved in optimal policy. 

Deadline is
not important for safety, but it is important for reliability. However, it is
not included because it is continuous varible, so it will explode the state
space and the agent woun't be able to learn anything.

### Define a State Space
When defining a set of states that the agent can
occupy, it is necessary to consider the *size* of the state space. That is to
say, if you expect the driving agent to learn a **policy** for each state, you
would need to have an optimal action for *every* state the agent can occupy. If
the number of all possible states is very large, it might be the case that the
driving agent never learns what to do in some states, which can lead to
uninformed decisions. For example, consider a case where the following features
are used to define the state of the *Smartcab*:

`('is_raining', 'is_foggy',
'is_red_light', 'turn_left', 'no_traffic', 'previous_turn_left',
'time_of_day')`.

How frequently would the agent occupy a state like `(False,
True, True, True, False, False, '3AM')`? Without a near-infinite amount of time
for training, it's doubtful the agent would ever learn the proper action!

### Question 5
*If a state is defined using the features you've selected from
**Question 4**, what would be the size of the state space? Given what you know
about the environment and how it is simulated, do you think the driving agent
could learn a policy for each possible state within a reasonable number of
training trials?*  
**Hint:** Consider the *combinations* of features to
calculate the total number of states!

**Answer:**

The size of state space is equal to 4. It is an interesting
question, because larger number of parameters in space give not so good
rresults. State now looks like this (waypoint, inputs['light'],  inputs['left'],
inputs['oncoming']). Total number of states equal to 3\*2\*4\*4 = 96. Agent can
learn to drive well from reasonable number of trials because the number of
states is not so big.

### Update the Driving Agent State
For your second implementation, navigate to
the `'build_state()'` agent function. With the justification you've provided in
**Question 4**, you will now set the `'state'` variable to a tuple of all the
features necessary for Q-Learning. Confirm your driving agent is updating its
state by running the agent file and simulation briefly and note whether the
state is displaying. If the visual simulation is used, confirm that the updated
state corresponds with what is seen in the simulation.

**Note:** Remember to
reset simulation flags to their default setting when making this observation!

-----
## Implement a Q-Learning Driving Agent
The third step to creating an
optimized Q-Learning agent is to begin implementing the functionality of
Q-Learning itself. The concept of Q-Learning is fairly straightforward: For
every state the agent visits, create an entry in the Q-table for all state-
action pairs available. Then, when the agent encounters a state and performs an
action, update the Q-value associated with that state-action pair based on the
reward received and the iterative update rule implemented. Of course, additional
benefits come from Q-Learning, such that we can have the agent choose the *best*
action for each state based on the Q-values of each state-action pair possible.
For this project, you will be implementing a *decaying,* $\epsilon$*-greedy*
Q-learning algorithm with *no* discount factor. Follow the implementation
instructions under each **TODO** in the agent functions.

Note that the agent
attribute `self.Q` is a dictionary: This is how the Q-table will be formed. Each
state will be a key of the `self.Q` dictionary, and each value will then be
another dictionary that holds the *action* and *Q-value*. Here is an example:
```
{ 'state-1': { 
    'action-1' : Qvalue-1,
    'action-2' : Qvalue-2,
...
   },
  'state-2': {
    'action-1' : Qvalue-1,
     ...
   },
   ...
}
```
Furthermore, note that you are expected to use a *decaying* $\epsilon$
*(exploration) factor*. Hence, as the number of trials increases, $\epsilon$
should decrease towards 0. This is because the agent is expected to learn from
its behavior and begin acting on its learned behavior. Additionally, The agent
will be tested on what it has learned after $\epsilon$ has passed a certain
threshold (the default threshold is 0.05). For the initial Q-Learning
implementation, you will be implementing a linear decaying function for
$\epsilon$.

### Q-Learning Simulation Results
To obtain results from the initial Q-Learning
implementation, you will need to adjust the following flags and setup:
-
`'enforce_deadline'` - Set this to `True` to force the driving agent to capture
whether it reaches the destination in time.
- `'update_delay'` - Set this to a
small value (such as `0.01`) to reduce the time between steps in each trial.
-
`'log_metrics'` - Set this to `True` to log the simluation results as a `.csv`
file and the Q-table as a `.txt` file in `/logs/`.
- `'n_test'` - Set this to
`'10'` to perform 10 testing trials.
- `'learning'` - Set this to `'True'` to
tell the driving agent to use your Q-Learning implementation.

In addition, use
the following decay function for $\epsilon$:

$$ \epsilon_{t+1} = \epsilon_{t} -
0.05, \hspace{10px}\textrm{for trial number } t$$

If you have difficulty
getting your implementation to work, try setting the `'verbose'` flag to `True`
to help debug. Flags that have been set here should be returned to their default
setting when debugging. It is important that you understand what each flag does
and how it affects the simulation! 

Once you have successfully completed the
initial Q-Learning simulation, run the code cell below to visualize the results.
Note that log files are overwritten when identical simulations are run, so be
careful with what log file is being loaded!

```python
# Load the 'sim_default-learning' file from the default Q-Learning simulation
vs.plot_trials('sim_default-learning.csv')
```

### Question 6
Using the visualization above that was produced from your default
Q-Learning simulation, provide an analysis and make observations about the
driving agent like in **Question 3**. Note that the simulation should have also
produced the Q-table in a text file which can help you make observations about
the agent's learning. Some additional things you could consider:  
- *Are there
any observations that are similar between the basic driving agent and the
default Q-Learning agent?*
- *Approximately how many training trials did the
driving agent require before testing? Does that number make sense given the
epsilon-tolerance?*
- *Is the decaying function you implemented for $\epsilon$
(the exploration factor) accurately represented in the parameters panel?*
- *As
the number of training trials increased, did the number of bad actions decrease?
Did the average reward increase?*
- *How does the safety and reliability rating
compare to the initial driving agent?*

**Answer:**

1) They are pretty much different.

2) My tolerance was 0.05, the
default value. My initial epsilon is 1, and it decreases 0.05 each time. So, the
number of training trials simply becomes (1-0.05)/0.05, which is 19. Plus the
first trial at $\epsilon=1$, there are total 20 trials.

3) Yes. The figure in
the panel accutately represented the parameter panel, where $\epsilon$ is a
straight line with a negative slope of 0.05. The $\alpha$ is constant at 0.5 as
the graph shows.

4) As the number of training trials increased, the number of
bad actions does decrease and the average reward does increase. If we have more
learning trails, they may be even better.

5) While basic agent shows poor
performance in case of both reliability and safety, this q-learning agents shows
better results. For the q-learning driving agent, the reliability rating is "D"
and safety rating is still poor - “F”

-----
## Improve the Q-Learning Driving Agent
The third step to creating an
optimized Q-Learning agent is to perform the optimization! Now that the
Q-Learning algorithm is implemented and the driving agent is successfully
learning, it's necessary to tune settings and adjust learning paramaters so the
driving agent learns both **safety** and **efficiency**. Typically this step
will require a lot of trial and error, as some settings will invariably make the
learning worse. One thing to keep in mind is the act of learning itself and the
time that this takes: In theory, we could allow the agent to learn for an
incredibly long amount of time; however, another goal of Q-Learning is to
*transition from experimenting with unlearned behavior to acting on learned
behavior*. For example, always allowing the agent to perform a random action
during training (if $\epsilon = 1$ and never decays) will certainly make it
*learn*, but never let it *act*. When improving on your Q-Learning
implementation, consider the implications it creates and whether it is
logistically sensible to make a particular adjustment.

### Improved Q-Learning Simulation Results
To obtain results from the initial
Q-Learning implementation, you will need to adjust the following flags and
setup:
- `'enforce_deadline'` - Set this to `True` to force the driving agent to
capture whether it reaches the destination in time.
- `'update_delay'` - Set
this to a small value (such as `0.01`) to reduce the time between steps in each
trial.
- `'log_metrics'` - Set this to `True` to log the simluation results as a
`.csv` file and the Q-table as a `.txt` file in `/logs/`.
- `'learning'` - Set
this to `'True'` to tell the driving agent to use your Q-Learning
implementation.
- `'optimized'` - Set this to `'True'` to tell the driving agent
you are performing an optimized version of the Q-Learning implementation.
Additional flags that can be adjusted as part of optimizing the Q-Learning
agent:
- `'n_test'` - Set this to some positive number (previously 10) to
perform that many testing trials.
- `'alpha'` - Set this to a real number
between 0 - 1 to adjust the learning rate of the Q-Learning algorithm.
-
`'epsilon'` - Set this to a real number between 0 - 1 to adjust the starting
exploration factor of the Q-Learning algorithm.
- `'tolerance'` - set this to
some small value larger than 0 (default was 0.05) to set the epsilon threshold
for testing.

Furthermore, use a decaying function of your choice for $\epsilon$
(the exploration factor). Note that whichever function you use, it **must decay
to **`'tolerance'`** at a reasonable rate**. The Q-Learning agent will not begin
testing until this occurs. Some example decaying functions (for $t$, the number
of trials):

$$ \epsilon = a^t, \textrm{for } 0 < a < 1 \hspace{50px}\epsilon =
\frac{1}{t^2}\hspace{50px}\epsilon = e^{-at}, \textrm{for } 0 < a < 1
\hspace{50px} \epsilon = \cos(at), \textrm{for } 0 < a < 1$$
You may also use a
decaying function for $\alpha$ (the learning rate) if you so choose, however
this is typically less common. If you do so, be sure that it adheres to the
inequality $0 \leq \alpha \leq 1$.

If you have difficulty getting your
implementation to work, try setting the `'verbose'` flag to `True` to help
debug. Flags that have been set here should be returned to their default setting
when debugging. It is important that you understand what each flag does and how
it affects the simulation! 

Once you have successfully completed the improved
Q-Learning simulation, run the code cell below to visualize the results. Note
that log files are overwritten when identical simulations are run, so be careful
with what log file is being loaded!

```python
# Load the 'sim_improved-learning' file from the improved Q-Learning simulation
vs.plot_trials('sim_improved-learning.csv')
```

### Question 7
Using the visualization above that was produced from your
improved Q-Learning simulation, provide a final analysis and make observations
about the improved driving agent like in **Question 6**. Questions you should
answer:  
- *What decaying function was used for epsilon (the exploration
factor)?*
- *Approximately how many training trials were needed for your agent
before begining testing?*
- *What epsilon-tolerance and alpha (learning rate)
did you use? Why did you use them?*
- *How much improvement was made with this
Q-Learner when compared to the default Q-Learner from the previous section?*
-
*Would you say that the Q-Learner results show that your driving agent
successfully learned an appropriate policy?*
- *Are you satisfied with the
safety and reliability ratings of the *Smartcab*?*

**Answer:**

1) I used the following decaying function: self.epsilon = 1 -
(1/(1+math.exp(-k\*self.alpha\*(self.trial_count)))), where k is equal to 0.05.
Initial epsilon is equal to 0.5031

2) It was approximately 550 training trials
3) Tolearance I set to be equal to value 0.001, alpha is equal to 0.25. The
learning rate is the rate at which the Q-table will be updated with new results;
the higher the learning rate, the more weight is attributed to new data. Please
revise this part of your discussion to reflect this. Tolearance represent when
do we stop training trials, i.e. when epsilon is leaa than tolerance this means
that model it trained.
   
   
4) With this model I was able to decrees total
bad actions,  as well as other parameters which correspond to safety. Minor and
major accidents count is near zero

5) I think this model performs really well,
with near human performace. Of course, there can be some improvements but in
general this is very good model.

6) It has A+ in safety and A in reliability
and I'm satisfied with its performance.

### Define an Optimal Policy

Sometimes, the answer to the important question
*"what am I trying to get my agent to learn?"* only has a theoretical answer and
cannot be concretely described. Here, however, you can concretely define what it
is the agent is trying to learn, and that is the U.S. right-of-way traffic laws.
Since these laws are known information, you can further define, for each state
the *Smartcab* is occupying, the optimal action for the driving agent based on
these laws. In that case, we call the set of optimal state-action pairs an
**optimal policy**. Hence, unlike some theoretical answers, it is clear whether
the agent is acting "incorrectly" not only by the reward (penalty) it receives,
but also by pure observation. If the agent drives through a red light, we both
see it receive a negative reward but also know that it is not the correct
behavior. This can be used to your advantage for verifying whether the
**policy** your driving agent has learned is the correct one, or if it is a
**suboptimal policy**.

### Question 8

1. Please summarize what the optimal policy is for the smartcab
in the given environment. What would be the best set of instructions possible
given what we know about the environment? 
   _You can explain with words or a
table, but you should thoroughly discuss the optimal policy._

2. Next,
investigate the `'sim_improved-learning.txt'` text file to see the results of
your improved Q-Learning algorithm. _For each state that has been recorded from
the simulation, is the **policy** (the action with the highest value) correct
for the given state? Are there any states where the policy is different than
what would be expected from an optimal policy?_ 

3. Provide a few examples from
your recorded Q-table which demonstrate that your smartcab learned the optimal
policy. Explain why these entries demonstrate the optimal policy.

4. Try to
find at least one entry where the smartcab did _not_ learn the optimal policy.
Discuss why your cab may have not learned the correct policy for the given
state.

Be sure to document your `state` dictionary below, it should be easy for
the reader to understand what each state represents.

**Answer:** 

1) Below you can see all the possibilities of states for the
driving agent in given environment.

Definition of optimal policy: If the light
is red and if the waypoint is right and there is no left car going forward, then
go right. Otherwise if the light is green, go in the direction of the waypoint.
If waypoint is backward, then go left or right. If there is incoming traffic and
we need to turn left, wait until no incoming car is going forward or (our) left.
2) In the 'sim_improved-learning.txt' text file all actions correspond to
optimal policy

3) Here are some samples of correct behaviour.

('left',
'green', 'forward')
 -- forward : 0.74
 -- right : 0.22
 -- None : -5.55
 --
left : -20.19
 
 Here the cab moves foward, because we need to turn left but
incoming car is moving forward.
 

('right', 'green', 'right')
 -- forward :
0.51
 -- right : 1.76
 -- None : -3.49
 -- left : -15.59
 
 The Smartcab is
turning right and it is good.

('forward', 'green', 'left')
 -- forward : 1.67
-- right : 0.49
 -- None : -5.29
 -- left : 1.18

Again, moving forward is ok in
this case.

('left', 'green', 'right')
 -- forward : 0.79
 -- right : -0.05
 --
None : -4.60
 -- left : -14.89
 
 Here the Smartcab is moving forward because
left turn is blocked by incoming car.

('right', 'red', 'forward')
 -- forward :
-18.34
 -- right : -9.58
 -- None : 0.99
 -- left : -13.63
 
 The agent stops
whe the red light is on.
 
 4) Here you can see incorrect behaviour Q-table.
Agent has high safety rating and here he fails to reach the destination on time.
/-----------------------------------------
| State-action rewards from
Q-Learning
\-----------------------------------------

('left', 'green',
'forward', 'left')
 -- forward : 0.22
 -- right : 0.00
 -- None : -1.92
 -- left
: 2.05

('right', 'green', 'forward', 'right')
 -- forward : 0.00
 -- right :
1.81
 -- None : 0.00
 -- left : -4.97

**('left', 'red', 'right', 'left')
 --
forward : -2.26
 -- right : 0.70
 -- None : 0.00
 -- left : -4.54**

('right',
'red', 'right', None)
 -- forward : -2.74
 -- right : 2.15
 -- None : 0.00
 --
left : -4.33

('right', 'green', 'right', 'forward')
 -- forward : -0.13
 --
right : 1.60
 -- None : -1.31
 -- left : -4.91

**('left', 'red', 'right',
'right')
 -- forward : 0.00
 -- right : 0.00
 -- None : 1.92
 -- left : -9.76**
('right', 'green', None, 'left')
 -- forward : 0.38
 -- right : 1.89
 -- None :
-2.18
 -- left : 0.25

**('left', 'green', 'right', 'right')
 -- forward : 0.29
-- right : -0.19
 -- None : 0.00
 -- left : 0.00**

('forward', 'red', None,
None)
 -- forward : -16.25
 -- right : 0.44
 -- None : 1.62
 -- left : -10.78
('right', 'green', 'left', 'right')
 -- forward : 0.00
 -- right : 1.69
 -- None
: 0.00
 -- left : 0.00

('right', 'red', None, 'forward')
 -- forward : -4.35
-- right : 2.05
 -- None : 0.28
 -- left : -7.62

**('forward', 'red', 'left',
None)
 -- forward : -9.48
 -- right : 0.83
 -- None : 2.21
 -- left : -15.93**
**('left', 'red', 'left', 'forward')
 -- forward : -2.58
 -- right : 0.00
 --
None : 1.92
 -- left : -2.70**

('right', 'red', 'right', 'left')
 -- forward :
-2.47
 -- right : 1.77
 -- None : 0.35
 -- left : 0.00

('left', 'red', None,
None)
 -- forward : -14.30
 -- right : 0.52
 -- None : 1.90
 -- left : -15.23
---------------------------------------------------------------------------------------------
Bold states of the Q-table above indicate smartcab actions that do not follow
the policy. For example, first one, the smartcab when the light is red takes
choice to go right. It can't go left because it will be traffic rules
violations, so here are to choices possible - stay or go right and driving agent
decides to go right. 

Next one is the similar to this but in that case agent
decides to stay on under the red light.

The third one looks a bit strange. Do
not know why agent does not follow the waypoint.

Looks like policy is not
lerned comletely and there are some strange behaviours, but in general my driver
agent performs well to avoid accidents and then misses deadline trying to get to
the destination.

-----
### Optional: Future Rewards - Discount Factor, `'gamma'`
Curiously, as
part of the Q-Learning algorithm, you were asked to **not** use the discount
factor, `'gamma'` in the implementation. Including future rewards in the
algorithm is used to aid in propagating positive rewards backwards from a future
state to the current state. Essentially, if the driving agent is given the
option to make several actions to arrive at different states, including future
rewards will bias the agent towards states that could provide even more rewards.
An example of this would be the driving agent moving towards a goal: With all
actions and rewards equal, moving towards the goal would theoretically yield
better rewards if there is an additional reward for reaching the goal. However,
even though in this project, the driving agent is trying to reach a destination
in the allotted time, including future rewards will not benefit the agent. In
fact, if the agent were given many trials to learn, it could negatively affect
Q-values!

### Optional Question 9
*There are two characteristics about the project that
invalidate the use of future rewards in the Q-Learning algorithm. One
characteristic has to do with the *Smartcab* itself, and the other has to do
with the environment. Can you figure out what they are and why future rewards
won't work for this project?*

**Answer:**

In my opinion future reward is not applicable in this case because
the system is very dynamic. So we can't predict what value future reward will
be. The agent can't see beyond the current intersection. For example it is
difficult to expect some kind of reward not knowing what the light will be and
in what state other cards will be. This is because agent does not have full
information about the environment, it knows only situation where he is located
and future reward cannot be computed here. The agent is not able to select the
best action in advance because simply it does not know state of the environment
at every point in future.

The second point would be that at some point in
future, future reward, which in this case will be correlated with arriving to
destination, could affect safety score. For example agent could try violate
traffic rules just because reward of being at the destination is high comparing
to the rules. Also, each trial the destination changes and future rewards need
to be recalculated each time to contribute successfuly to the model. This
requires full knowledge about the environment which the agent does not have.

> **Note**: Once you have completed all of the code implementations and
successfully answered each question above, you may finalize your work by
exporting the iPython Notebook as an HTML document. You can do this by using the
menu above and navigating to  
**File -> Download as -> HTML (.html)**. Include
the finished document along with this notebook as your submission.
