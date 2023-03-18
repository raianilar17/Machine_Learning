# Range-Simulation-of-E-Bikes

### OBJECTIVES
To see how far a vehicle can go before the battery is flat 

The simulation of the range of a vehicle involves the continuous running of driving cycles
or schedules until there is no more energy left.

## Initial Setup

1. An array of velocity values V must have been created, corresponding to the driving
cycle.
2. The value of N must have been found.
3. set up the vehicle parameters such as the mass, the battery size and type,
and so on. The electrical power taken by the accessories P ac should be set at this point.
4. set up arrays for the data to be stored just for one cycle; this
data can be lost at the end of each cycle. This is also the charge removed, depth of
discharge, and distance travelled, but we might also save other data, such as information
about torque, or motor power, or battery current, as it is sometimes useful to be able to
plot this data for just one cycle.

Having set the system up, the vehicle is put through one driving cycle, using the
velocities given to calculate the acceleration, and thus the tractive effort, and thus the
motor power, torque and speed. This is used to find the motor efficiency, which is used
to find the electrical power going into the motor. Combined with the accessory power,
this is used to find the battery current. This is then used to recalculate the battery state
of charge. This calculation is repeated in one second steps until the end of the cycle.

One second steps are the most convenient, as most driving cycles are defined in terms of one second intervals. Also, many
of the formulas become much simpler. However, it is quite easy to adapt any of the programs given here for different time
steps, and shorter steps are sometimes used.

The end of cycle data arrays are then updated, and if the battery still has enough charge,
the process is repeated for another cycle.

If we wish to find the range to exactly 90% discharged, then the DD < 0.9 ;

Range Calculations Formula

  ## Range = D(N)*0.8/DoD(N)

# Future Work

apply this algorithm on different electric vechicles...

# Work in Progress...
