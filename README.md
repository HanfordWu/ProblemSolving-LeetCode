# ProblemSolving-LeetCode

## Cracking the code interview:

### [01.01. Is Unique LCCI](https://leetcode-cn.com/problems/is-unique-lcci/)

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

### [01.02. Check Permutation LCCI](https://leetcode-cn.com/problems/check-permutation-lcci/)

### Sort then compare

```python
def CheckPermutation(self, s1, s2):
    return ''.join(sorted(s1)) == ''.join(sorted(s2))
```

### Character Array

```python
def CheckPermutation(self, s1, s2):
    if len(s1) != len(s2):
        return False

    arr = [0] * 52
    for i in range(len(s1)):
        arr[ord(s1[i]) - ord('a')] += 1
    for i in range(len(s2)):
        arr[ord(s2[i]) - ord('a')] -= 1
    return all(v == 0 for v in arr)
```

### [01.03. String to URL LCCI](https://leetcode-cn.com/problems/string-to-url-lcci/)

```python
def replaceSpaces(self, S: str, length: int) -> str:
        return S[:length].replace(' ','%20')
```

### Java String API:

```java
public String replaceSpaces(String S, int length) {    
    char[] ch = new char[length * 3];
    int index = 0;
    for (int i = 0; i < length; i++) {
        char c = S.charAt(i);
        if (c == ' ') {
            ch[index++] = '%';
            ch[index++] = '2';
            ch[index++] = '0';
        } else {
            ch[index] = c;
            index++;
        }
    }
    return new String(ch, 0, index);
}
```
### Java StringBuilder:
```java
public String replaceSpaces(String S, int length) {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < length; i++) {
        char ch = S.charAt(i);
        if (ch == ' ') {
            sb.append("%20");
            continue;
        }
        sb.append(ch);
    }
    return sb.toString();
}
```

### [01.04. Palindrome Permutation LCCI](https://leetcode-cn.com/problems/palindrome-permutation-lcci/)

count the number of each character, the numbers can only have 1 odd number at most.

```python
def canPermutePalindrome(self, s):
    count = [i for i in Counter(s).values() if i%2 != 0]
    return len(count) <= 1
    # return sum(1 for k, v in collections.Counter(s).items() if v % 2 != 0) <= 1
```

In Java, there is no convenient way to count the number of each character, so we can only use hashmap or character array to count.

```java
public boolean canPermutePalindrome(String s) {
    char[] chars = s.toCharArray();
    int[] target = new int[128];
    for (char c : chars) {
        target[c] += 1;
    }
    int count = 0;
    for (int i : target) {
        if(i % 2 != 0) {
            count++;
        }
        if(count > 1) {
            return false;
        }
    }
    return true;
}
```

>When we are checking if something appear even times, we can use set: if set contains current element, then remove it from set, if set doesn't contain current element, put it into set, finally check if the set is empty.

```java
public boolean canPermutePalindrome(String s) {
    Set<Character> set = new HashSet<>();
    for(int i = 0; i < s.length(); i++){
        if(set.contains(s.charAt(i))){
            set.remove(s.charAt(i));
        }else{
            set.add(s.charAt(i));
        }
    }
    return set.size() < 2;
}
```

### [01.05. One Away LCCI](https://leetcode-cn.com/problems/one-away-lcci/)

### Find the first difference, then compare the rest of the string.

`leetcode` and `leettode`, the first difference is `c` and `t`, we compare `ode` and `ode`.

`leetcode` and `leetode`, we keep the first string longer, the first difference is `c` and `o`, we compare `ode` with `ode`.

```python
def oneEditAway(self, first, second):
    if abs(len(first) - len(second)) >1:
        return False
    if len(first) < len(second): //guarantee first is longer than second
        return self.oneEditAway(second, first)
    for i in range(len(second)):
        if first[i] != second[i]:
            if len(first) == len(second):
                return first[i+1:] == second[i+1:] //leetcode and leettode
            return first[i+1:] == second[i:] //leetcode and leetode
    return True
```

### Two pointers compare from two ends, if different, they stop, then compare the length of the rest of string

The first pointer stop at the first difference, second pointer stops at second difference, if they are the same difference(rest string length is 1), then return true. Otherwise return false.


`leetcode` and `leettode`, first pointer stop at t, second pointer stop at o, two pointer difference < 1, return true

`leetcode` and `leetode`, first pointer stop at t, second pointer stop at o, two pointer difference < 1, return true.

```c++
public:
    bool oneEditAway(string first, string second) {
        if(first==second){
            return true;
        }
        const int len1=first.size();
        const int len2=second.size();
        if(abs(len1-len2)>1){
            return false;
        }
        int i=0,j=len1-1,k=len2-1;
        while(i<len1 && i<len2 && first[i]==second[i]){ // i从左至右扫描
            ++i;
        }
        while(j>=0 && k>=0 && first[j]==second[k]){ // j、k从右至左扫描
            --j;
            --k;
        }
        return j-i<1 && k-i<1;
    }
```

### [01.06. Compress String LCCI](https://leetcode-cn.com/problems/compress-string-lcci/)

Nothing but count

```python
def compressString(self, S: str) -> str:
    if not S:
        return ""
    ch = S[0]
    ans = ''
    cnt = 0
    for c in S:
        if c == ch:
            cnt += 1
        else:
            ans += ch + str(cnt)
            ch = c
            cnt = 1
    ans += ch + str(cnt)
    return ans if len(ans) < len(S) else S
```

### [01.07. Rotate Matrix LCCI](https://leetcode-cn.com/problems/rotate-matrix-lcci/)

n is even:

![alt](https://pic.leetcode-cn.com/194630bf90343475a07278a0840d93ad891206acd50be1b81e75eb357d1e2c07-rotate.gif)

n is odd:

![alt](https://pic.leetcode-cn.com/a2a3d0691e9979fee19e5f69f12b8b5205fd1b955d27661d36168f51aa0ba796-image.png)

So whatever n is even or odd, we only iterate the left corner part, rotate every element of that part with other three parts.

```python
def rotate(self, matrix):
    n = len(matrix)
    for i in range(n//2):
        for j in range((n+1)//2):
            matrix[i][j], matrix[n - j - 1][i], matrix[n - i - 1][n - j - 1], matrix[j][n - i - 1] \
                = matrix[n - j - 1][i], matrix[n - i - 1][n - j - 1], matrix[j][n - i - 1], matrix[i][j]
```

Or we can vertically revers, then diagonal reverse:

```python
def rotate(self, matrix: List[List[int]]) -> None:
    n = len(matrix)
    # 水平翻转
    for i in range(n // 2):
        for j in range(n):
            matrix[i][j], matrix[n - i - 1][j] = matrix[n - i - 1][j], matrix[i][j]
    # 主对角线翻转
    for i in range(n):
        for j in range(i):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
```


### [01.08. Zero Matrix LCCI](https://leetcode-cn.com/problems/zero-matrix-lcci/)

First, we cannot change the elements in place when we are iterating the matrix, because after we modify the elements, the matrix got changed.

We can use two array to store the row and column number that has 0 in the first iteration, then iterate the matrix again, set corresponding row and column 0.

```python
def setZeroes(self, matrix: List[List[int]]) -> None:
    row=set()
    column=set()
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j]==0:
                row.add(i)
                column.add(j)
    for i in row:
        for j in range(len(matrix[0])):
            matrix[i][j]=0
    for j in column:
        for i in range(len(matrix)):
            matrix[i][j]=0
    return
```

But if we cannot use extra array to store the row and column number, we must find another place to store the row and column.

The answer is the first row and column elements. We use them to store the information of corresponding row and column.

But because we are modifying first column and row, so we need to store whether the first row and column has 0.

```python
def setZeroes(self, matrix):
    first_row = False
    first_col = False
    #check if first row has 0
    for i in matrix[0]:
        if i == 0:
            first_row = True
    #check if first column has 0
    for i in range(len(matrix)):
        if matrix[i][0] == 0:
            first_col = True
    #if that element is 0, set that first row and first column as 0
    for i in range(1, len(matrix)):
        for j in range(1, len(matrix[0])):
            if matrix[i][j] == 0:
                matrix[0][j] = 0
                matrix[i][0] = 0
    #if that first row or first column is 0, set that row or column as 0
    for i in range(1, len(matrix)):
        for j in range(1, len(matrix[0])):
            if matrix[0][j] == 0 or matrix[i][0]==0:
                matrix[i][j] =0
    #if first col has 0, set all elements of first column as 0
    if first_col:
        for i in range(len(matrix)):
            matrix[i][0] = 0
    #if first row has 0, set all elements of first row as 0
    if first_row:
        for i in range(len(matrix[0])):
            matrix[0][i] = 0
```

### [01.09. String Rotation LCCI](https://leetcode-cn.com/problems/string-rotation-lcci/)

First, consider if they are the same length.

If the same length, repeat any one of them, the new string must contain the other one.

```python
def isFlipedString(self, s1, s2):
    if len(s1) != len(s2):
        return False
    s = s2 + s2
    return s1 in s
```


### [02.01. Remove Duplicate Node LCCI](https://leetcode-cn.com/problems/remove-duplicate-node-lcci/)

Note that head is the head of a linked list, so we have to iterate the list by head = head.next

The idea is put all occurred value into a set, if already contain the value, skip the node.

```python
def removeDuplicateNodes(self, head):
    """
    :type head: ListNode
    :rtype: ListNode
    """
    if not head:
        return head
    occurred = {head.val}
    pos = head
    # 枚举前驱节点
    while pos.next:
        # 当前待删除节点
        cur = pos.next
        if cur.val not in occurred:
            occurred.add(cur.val)
            pos = pos.next
        else:
            pos.next = pos.next.next
    return head
```

### [02.02. Kth Node From End of List LCCI](https://leetcode-cn.com/problems/kth-node-from-end-of-list-lcci/)


### Two pointer with k distance

```python
def kthToLast(self, head, k):
    if not head:
        return head
    fast = head
    slow = head

    while fast.next and k > 1:
        fast = fast.next
        k-=1

    while fast.next:
        fast = fast.next
        slow = slow.next
    return slow.val
```

### Push to stack, and pop

```java
public int kthToLast(ListNode head, int k) {
    Stack<ListNode> stack = new Stack<>();
    //链表节点压栈
    while (head != null) {
        stack.push(head);
        head = head.next;
    }
    //在出栈串成新的链表
    ListNode firstNode = stack.pop();
    while (--k > 0) {
        ListNode temp = stack.pop();
        temp.next = firstNode;
        firstNode = temp;
    }
    return firstNode.val;
}
```

### Recursion

Whenever there is singly linked list, we should think about recursion.

Recursion is referring Go and Back. Go is because function call, back is because function return.

So we want  kth to last element, we can have a global variable start from 0 at the last node, then back to increase till it is equal to k.

```python
def kthToLast(self, head, k):
    if head is None:
        return 0
    n = self.kthToLast(head.next, k)

    self.count += 1

    return head.val if self.count == k else n
```

The recursion return process means, if count == k, I return head.val, otherwise, I return what I get from may next recursion. The whole process only two function return it selves value, the last one(0), and the count == k one(head.val). And the head.val replaced the 0.


### [02.03. Delete Middle Node LCCI](https://leetcode-cn.com/problems/delete-middle-node-lcci/)

Manipulate singly linked list, sometimes we may don't change the list, but only change the value.

```python
def deleteNode(self, node):
    node.val = node.next.val
    node.next = node.next.next
```

