# BOIDS - Bird Flocking Simulation Program#

## Preface
I became interested in this simulation after learning about flocking bird behaviors and emergent properties in my
COGS100 Class - "Cyborgs, Now and in the Future".

## The Big Question, How Do Birds Flock?

Bird flocking appears highly complex and coordinated, yet it emerges from very simple local rules followed by each individual bird. The BOIDS model demonstrates how complex and collective behaviors can arise without central control, producing seemingly infinite flocking patterns from a small set of rules.

This project models flocking behavior using three simple programs applied to every bird (agent) in the system:
1. Separation - Each bird maintains a reasonable distance from other birds to prevent collisions.

2. Alignment - Birds attempt to align their movement with the average direction of nearby birds.

3. Cohesion - Each bird tries to stay close to the group by moving toward the average position of nearby birds.

## Emergence and Complexity

This model demonstrates how complexity emerges from simple components. None of the individual rules contain complex flocking behavior on their own. However, when combined and applied locally across many birds, coordinated flocking patterns emerge at the system level.

This is a defining characteristic of emergence. The final behavior cannot be explicitly programmed in detail. Instead, it must be discovered by identifying simple rules that interact to produce complex outcomes.
