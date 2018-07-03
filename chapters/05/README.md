# Notes

## Example 5.3: Blackjack with Monte Carlo ES

In the problem description, it is stated that:

> As the initial policy we use the policy evaluated in the previous
> blackjack example, that which sticks only on 20 or 21.

I found this to be misleading. I intepret an "initial policy" as the
policy for a state that has not been visited. That logic, however,
does not produce the results in the book.

Instead, what is expressed in the Monte Carlo ES algorithm is what
should be followed: here, the initial policy for each state is
"arbitrary," which can be interpreted as random. Using random for
unexplored states, and previous knowledge for explored, produces the
policies displayed in the book (pi^* Figure 5.5).

However, using random selection as initial policies, and the previous
(less-than 20) policy for subsequent unexplored states also
works. That is the implementation chosen here.

# Secondary Resources:

* [Playing Blackjack with Monte Carlo Methods](http://outlace.com/rlpart2.html)