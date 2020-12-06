# ProblemSolving-LeetCode

## Cracking the code interview:

1. 01.01. [Is Unique LCCI](https://leetcode-cn.com/problems/is-unique-lcci/)

### Set:
```python
def isUnique(self, astr):
    s = set(astr)
    return len(s) == len(astr)
```

### Brutal: two for loop or check `in`

```python
def isUnique(self, astr):
    for i in range(len(astr)):
        if astr[i] in astr[i+1:]:
            return False
    return True
```

check `astr[i] in astr[i+1:]` is also using nested for under the hood.

### Bool array

```python
def isUnique(self, astr):
    arr = [0] * 26
    for i in range(len(astr)):
        index = ord(astr[i]) - ord('a')
        if arr[index] == 1:
            return False
        else:
            arr[index] += 1
    return True
```

### Bit Operation:

Whenever there is bool array, we can consider using bit to save space.

We can use an integer 0, which has 32 bits, we use its right 26 bits as markers. First we need to know `1 << 4` we get ...0001000, `1 << 6` we get ...000100000, so first we calculate the binary of each character by `c - 'a'`, then shift it, do & with current. if result is not 0, then there is duplication, other wise we update markers by `|` with current makers.

```python
def isUnique(self, astr: str) -> bool:
    mark = 0
    for char in astr:
      move_bit = ord(char) - ord('a')
      if (mark & (1 << move_bit)) != 0:
        return False
      else:
        mark |= (1 << move_bit)
    return True
```

