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
