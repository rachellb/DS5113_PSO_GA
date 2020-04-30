#the intial framework for a particle swarm optimization for Schwefel minimization problem
#author: Charles Nicholson
#for ISE/DSA 5113


#need some python libraries
import copy
import math
from random import Random
import numpy as np

#to setup a random number generator, we will specify a "seed" value
seed = 12345
myPRNG = Random(seed)

#to get a random number between 0 and 1, write call this:             myPRNG.random()
#to get a random number between lwrBnd and upprBnd, write call this:  myPRNG.uniform(lwrBnd,upprBnd)
#to get a random integer between lwrBnd and upprBnd, write call this: myPRNG.randint(lwrBnd,upprBnd)

lowerBound = -500  #bounds for Schwefel Function search space
upperBound = 500   #bounds for Schwefel Function search space

#you may change anything below this line that you wish too -----------------------------------------------------

#note: for the more experienced Python programmers, you might want to consider taking a more object-oriented approach to the PSO implementation, i.e.: a particle class with methods to initialize itself, and update its own velocity and position; a swarm class with a method to iterates through all particles to call update functions, etc.

#number of dimensions of problem
dimensions = 2

#number of particles in swarm
swarmSize = 5


vmax = 0 # need to find out a best vmax

# The Acceleration Constants, currently set to 2
psi1 = 2  
psi2 = 2
    
#Schwefel function to evaluate a real-valued solution x    
# note: the feasible space is an n-dimensional hypercube centered at the origin with side length = 2 * 500
             
def evaluate(x):          
      val = 0
      d = len(x)
      for i in range(d):
            val = val + x[i]*math.sin(math.sqrt(abs(x[i])))
                                        
      val = 418.9829*d - val         
                    
      return val          
          
          

#the swarm will be represented as a list of positions, velocities, values, pbest, and pbest values

pos = [[] for _ in range(swarmSize)]      #position of particles -- will be a list of lists; e.g., for a 2D problem with 3 particles: [[17,4],[-100,2],[87,-1.2]]
vel = [[] for _ in range(swarmSize)]      #velocity of particles -- will be a list of lists similar to the "pos" object 


#note: pos[0] and vel[0] provides the position and velocity of particle 0; pos[1] and vel[1] provides the position and velocity of particle 1; and so on. 


def rankOrder(anyList):
    
    rankOrdered = [0] * len(anyList)
    for i, x in enumerate(sorted(range(len(anyList)), key=lambda y: anyList[y])):  
        rankOrdered[x] = i     

    return rankOrdered

curValue = [] #evaluation value of current position  -- will be a list of real values; curValue[0] provides the evaluation of particle 0 in it's current position
pbest = []    #particles' best historical position -- will be a list of lists: pbest[0] provides the position of particle 0's best historical position
pbestVal = [] #value of pbest position  -- will be a list of real values: pbestBal[0] provides the value of particle 0's pbest location


#initialize the swarm randomly
for i in range(swarmSize):
      for j in range(dimensions):
            pos[i].append(myPRNG.uniform(lowerBound,upperBound))    #assign random value between lower and upper bounds
            vel[i].append(myPRNG.uniform(-1,1))                     #assign random value between -1 and 1   --- maybe these are good bounds?  maybe not...
            
      curValue.append(evaluate(pos[i]))   #evaluate the current position
                                                 
pBest = pos[:]          # initialize pbest to the starting position
pBestVal = curValue[:]  # initialize pbest to the starting position
pBestGlobal = pos[0]    # initialize global best to first position

#Currently missing several elements
#e.g., velocity update function; velocity max limitations; position updates; dealing with infeasible space; identifying the global best; main loop, stopping criterion, etc. 


# updates velocity and position 
def move():
    
    global vel # So I can update vel within the function
    global pos 
    
    # Find out where the best value is currently
    ordering = rankOrder(pBestVal)

    for i in ordering:
        if ordering[i] == 0:
            best = pBest[i] # This will give the x,y coordinates of the best p value
    
    # Random uniform variables between 0,1
    r1 = myPRNG.uniform(0,1)
    r2 = myPRNG.uniform(0,1)
    
    # Equation: v = v + psi1*r1(Pi - Xi) + psi2*r2(Pg-Xi)
    
    e1 = psi1*r1*(np.subtract(best,pos))        #psi1*r1(Pi - Xi)
    e2 = psi2*r2*(np.subtract(pBestGlobal,pos)) #psi2*r2(Pg-Xi)
    
    
    e3 = np.add(e1,e2)                          #psi1*r1(Pi - Xi) + psi2*r2(Pg-Xi)
    
    vel = np.add(vel,e3)
    
    for i in vel:           # Making sure velocity never goes above or below |vmax|
        if vel[i] > vmax:
            vel[i] = vmax
        if vel[i] < -vmax:
            vel[i] = -vmax
   
    pos = np.add(pos, vel)
       
# General code loop ---------------------------------------

something = 0 # Stopping criterion, will fill in later

while something:                                                              
    
    # Now I need to update the velocity and positions
    move()
    
    # I have to evaluate the positions and update pBest and pBestVal if they're better
    for i in pos:
        if evaluate(pos[i]) < pBestVal[i]: # Minimization problem
            pBest[i] = list(pos[i])              # Copy this position into pBest
            pBestVal[i] = list(evaluate(pos[i])) # Copy the value into pBestVal
            
    
    # Now the same for the global best
    for i in pBestVal:
        if pBestVal[i] <  evaluate(pBestGlobal):
            pBestGlobal = list(pBest[i])
    
    # Maybe stopping criterion can be if 90% of particles < 1 standard deviation away from mean, stop?




