#the intial framework for a particle swarm optimization for Schwefel minimization problem
#author: Charles Nicholson
#for ISE/DSA 5113


#need some python libraries
import copy
import math
from random import Random
import numpy as np
import matplotlib.pyplot as plt


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

vmax = 100

# The Acceleration Constants, currently set to 2
phi1 = 1  # The cognitive (particle dependent) portion
phi2 = 3  # The social (swarm dependent) portion
    
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
pBestLocal = list(pos) # Initializing to best being initial positions
#Currently missing several elements
#e.g., velocity update function; velocity max limitations; position updates; dealing with infeasible space; identifying the global best; main loop, stopping criterion, etc. 

def ring():
    
    global pBestLocal
    
    # The modulus allows for wrapping (so the first item is neighbors with last)
    # Currently uses 3 if statements to check which among neighbors is best. 
    
    for i in range(len(pBestVal)):
       if pBestVal[(i-1)%len(pBestVal)] < evaluate(pBestLocal[i]):
           pBestLocal[i] = list(pos[(i-1)%len(pBestVal)])
       if pBestVal[i] < evaluate(pBestLocal[i]):
           pBestLocal[i] = list(pos[i])
       if pBestVal[(i+1)%len(pBestVal)] < evaluate(pBestLocal[i]):
           pBestLocal[i] = list(pos[(i+1)%len(pos)])
       
    

        


# updates velocity and position 
def move():
    
    global vel # So I can update velocity within the function
    global pos # For updating position
    

    # Random uniform variables between 0,1
    r1 = myPRNG.uniform(0,1)
    r2 = myPRNG.uniform(0,1)
    
    # Equation: v = v + phi1*r1(Pi - Xi) + phi2*r2(Pg-Xi)
    
    
    # The equation is broken up into 3 parts for ease of coding
    
    #phi1*r1(Pi - Xi)
    e1 = phi1*r1*(np.subtract(pBest,pos))        
    #phi2*r2(Pg-Xi) # For global best structure
    e2 = phi2*r2*(np.subtract(pBestGlobal,pos)) 
    #ring()
    
    #phi2*r2(Pg-Xi) # For local neighborhood structure
    #e2 = phi2*r2*(np.subtract(pBestLocal,pos)) 
    #phi1*r1(Pi - Xi) + phi2*r2(Pg-Xi)
    e3 = np.add(e1,e2)                          
    
    # Updating the velocity
    vel = np.add(vel,e3)
    
    
    # Making sure velocity never goes above or below |vmax|
    for i in range(len(vel)):           
        for j in range(len(vel[i])):
            if vel[i][j] > vmax:
                vel[i][j] = vmax
            if vel[i][j] < -vmax:
                vel[i][j] = -vmax
   
    #Updating the position
    pos = np.add(pos, vel)
    
    # Need to deal with infeasibility. What about Torus? Then wouldn't need to mess with velocity
    # Should wrap around to the other side of the screen
    for i in range(len(pos)):
        for j in range(len(pos[i])):
            if pos[i][j] > upperBound:
                pos[i][j] = lowerBound + (pos[i][j]-upperBound)
            if pos[i][j] < lowerBound:
                pos[i][j] = upperBound + (pos[i][j]+upperBound)
            
    
       
# General code loop ---------------------------------------

# Step 1 Initialize the swarm (done higher up)

iterations = 0
while iterations < 3000:                                                              
    
    
    # Step 2 Evaluate fitness of each particle
    # Step 3(a) Evaluate individual best and update
    for i in range(len(pos)):
        if evaluate(pos[i]) < pBestVal[i]:       # Minimization problem
            pBest[i] = list(pos[i])              # Copy this position into pBest
            pBestVal[i] = evaluate(pos[i]) # Copy the value into pBestVal
            
    
    
    # Step 3(b) Update global best
    for i in range(len(pBestVal)):
        if pBestVal[i] <  evaluate(pBestGlobal):
            pBestGlobal = list(pBest[i])

    # Functions for plotting initial position and location after first move
    
    if iterations <= 1 and dimensions == 2:  
        x = []
        y = []
        for i in range(len(pos)):
            x.append(pos[i][0])
            y.append(pos[i][1])
        
        if iterations == 0: 
            plt.title("Initial location")
        else:
            plt.title("After first move")
        plt.axhline(linewidth=1, color='grey')
        plt.axvline(linewidth=1, color='grey')
        plt.ylabel("y")
        plt.xlabel("x")
        plt.scatter(x,y)
        plt.show()
    
    
    # Step 4 Update velocity and position of each particle
    move()
    
    
    # Stopping criterion is number of iterations
    iterations = iterations + 1
        
    

print("Best Value is: ")
print(evaluate(pBestGlobal))

