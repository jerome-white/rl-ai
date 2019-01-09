# Notes

## Example 6.2

Applying TD(0) as described in the book is not the way this problem is
done; [this reference](https://math.stackexchange.com/q/1884168)
alludes to the difference.

## Example 6.5

Although the book says alpha is 0.1, the figure can only be reproduced
using a value of 0.5:

```bash
$> python example-6.5.py --alpha 0.5
```

## Example 6.6

For best results, the experiment needs to be repeated several times;
fifty to 100, for example (in part to be consistent with
[ShangtongZhang](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/35b2bb7d500edc789920c8e1ff5cd4b0edd2e113/chapter06/cliff_walking.py#L172))

```bash
$> python example-6.5.py --alpha 0.5 --repeat 100
```

(And, again, it appears alpha should be 0.5, not 0.1.)

## Example 6.7

Reproducing the results from this problem is extremely difficult; and
the only reference implementations seem to be those provided at the
books website
([Lisp](http://incompleteideas.net/book/code/queuing.lisp),
[C](http://incompleteideas.net/book/code/queuing.c)). Challenges:

1. There is an implicit assumption that when zero servers are
   available, customers cannot be accepted. Because the policy (Q)
   starts with zeros, this means the learned rewards for acceptence
   under no available servers will always be higher than for
   rejection. Thus, Figure 6.17b should be at zero instead of -15. To
   be strict about this in the implementation those values should be
   set to negative infinity:

   ```python
   Q.q[0,:,1] = -np.inf
   ```

2. It is unclear what the correct model parameters should be. It looks
   like alpha should be 0.01, and helps if high-priority probability
   is 0.25:

   ```bash
   $> python example-6.7.py --alpha 0.01 --high-priority 0.25
   ```

   The alpha value is consistent with the authors C implementation,
   while the priority value is consistent with the Lisp
   implementation. The Lisp implementation actually doesn't weight
   priorities at all; which is what a priority value of 0.25 actually
   means in this implementation.

   For other hyper parameters, the reference implementations use
   different values --- both from the book, and from each other.

3. What constitutes a high-priority customer is unclear. Whether it is
   just the highest priority, or its the top-*n*. This implementation
   assumes the former.
