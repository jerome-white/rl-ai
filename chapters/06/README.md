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
books website. Challenges:

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
   like alpha should be 0.01:

   ```bash
   $> python example-6.7.py --alpha 0.01
   ```

   This is consistent with the authors reference [C
   implementation](http://incompleteideas.net/book/code/queuing.c),
   and gets the figures closest to those in the book. The reference
   implementations use different values --- both from the book, and
   from each oether -- for the server-free probability and the
   high-priority probability as well.

3. What constitutes a high-priority customer is unclear. Whether it is
   just the highest priority, or its the top-*n*. This implementation
   assumes the former.
