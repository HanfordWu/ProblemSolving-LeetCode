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

### [02.04. Partition List LCCI](https://leetcode-cn.com/problems/partition-list-lcci/)

This is the idea of quick sort, we partition the list according to a pivotal.

The idea of quick sort partition is that, two pointer p1 and p2, p1 is reserved for all smaller values than pivotal, p2 is go ahead to find smaller values, if no found, p2 continue moving on, if found, p2 swap value with p1, then p1 and p2 both moving next. Until p2 reach the end.

```python
def partition(self, head, x):
    if not head: return head

    p, q = head, head
    while q:
        if q.val < x:
            q.val, p.val = p.val, q.val
            p = p.next
        q = q.next
    return head
```
Above code is doing Node swapping by swapping the value.

Another straightforward solution is we create two Linked list, the first one link the smaller nodes, the second link the bigger nodes. and finally, we put the bigger list after the smaller list.

### [02.05. Sum Lists LCCI](https://leetcode-cn.com/problems/sum-lists-lcci/)
<p><strong>Example</strong></p>

<pre><strong>Input</strong>(7 -&gt; 1 -&gt; 6) + (5 -&gt; 9 -&gt; 2)，即617 + 295
<strong>Output</strong>2 -&gt; 1 -&gt; 9，即912
</pre>

```python
def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
    head = ListNode(0)
    node = head
    carry = 0
    while l1 or l2:
        if l1 == None:
            node.next = l2
            l1 = ListNode(0)
        if l2 == None:
            node.next = l1
            l2 = ListNode(0)
        sum_up = carry + l1.val + l2.val
        node.next = ListNode(sum_up % 10)
        carry = sum_up // 10 
        node = node.next
        l1 = l1.next 
        l2 = l2.next 
    if carry:
        node.next = ListNode(carry)
    return head.next
```

Note that if one of list is longer, we use 0 to make up.


Advanced:
Suppose the digits are stored in forward order. Repeat the above problem：
<pre><strong>Input:</strong>(6 -&gt; 1 -&gt; 7) + (2 -&gt; 9 -&gt; 5)，即617 + 295
<strong>Output:</strong>9 -&gt; 1 -&gt; 2，即912
</pre>

Here we can use recursion, but recursion is difficult to understand, so I am using another straightforward way, we convert each list to a number, sum them, the convert back to a list.

```python
def addTwoNumbers(self, l1, l2):
    # two array to store the fetched integer
    list1 = []
    list2 = []
    # populate the list with numbers
    self.add(list1, l1)
    self.add(list2, l2)

    # calculate the value of the number
    num1 = self.getNumberFromList(list1)
    num2 = self.getNumberFromList(list2)

    # calculate the sum
    s = num1 + num2

    # convert the sum to a linked list
    head = ListNode(-1)
    result = ListNode(s % 10)
    head.next = result
    r = s // 10
    while r != 0:
        temp = head.next
        head.next = ListNode(r % 10)
        head.next.next = temp
        r = r // 10
    return head.next

def getNumberFromList(self, list) -> int:
    weight = 1
    num1 = 0
    for i in list1[::-1]:
        num1 += weight * i
        weight *= 10
    return num1

def add(self, list, linkedList):
    if not linkedList:
        return
    self.add(list, linkedList.next)
    list.append(linkedList.val)
    return
```

### [02.06. Palindrome Linked List LCCI](https://leetcode-cn.com/problems/palindrome-linked-list-lcci/)

Take advantage of other data structure. 

For palindrome, reverse it, we can get the same as original list.

#### Use stack:

```python
def isPalindrome(self, head):
    stack = []
    temHead = head
    while temHead:
        stack.append(temHead.val)
        temHead = temHead.next

    while head:
        if head.val != stack.pop():
            return False
        head = head.next
    return True
```

#### Reverse the singly linked list:

Using stack has a O(n) space complexity, we can do better.

First, using two pointers to find the middle of the linked list.
Second, reverse the later half.
Third, compare the first part with second part, check if they are the same.

```python
def isPalindrome(self, head):
    faster = head
    slower = head
    # find the middle
    while faster and faster.next:
        faster = faster.next.next
        slower = slower.next
    # If number is odd, we take the next of the middle
    if faster:
        slower = slower.next

    faster = head
    # reverse the second part
    pre = None

    while slower:
        nex = slower.next
        slower.next = pre
        pre = slower
        slower = nex

    # compare the second part with the first part
    while pre:
        if faster.val != pre.val:
            return False
        pre = pre.next
        faster = faster.next
    return True
```


### [02.07. Intersection of Two Linked Lists LCCI](https://leetcode-cn.com/problems/intersection-of-two-linked-lists-lcci/)

#### Using a HashSet
We are checking if the set already has the object, we can use a HashSet, if hash method is not overrode, the default is comparing "==", the address of the object, which is exactly the identity of the object.

```python
def getIntersectionNode(self, headA, headB):
    se = set()
    while headA:
        se.add(headA)
        headA = headA.next

    while headB:
        if headB in se:
            return headB
        headB = headB.next
    return None
```

#### We can also find the intersection in place

Two pointers A and B, A iterate from headA, after reaching the end, switch to headB. B iterate from headB, after reaching the end, switch to headA. A and B will meet at the intersection. If headA and headB don't intersect, A and B will meet at null.

```python
def getIntersectionNode(self, headA, headB):
    pointer1 = headA
    pointer2 = headB

    while pointer1 != pointer2:
        pointer1 = headB if pointer1 is None else pointer1.next
        pointer2 = headA if pointer2 is None else pointer2.next

    return pointer1
```

### [02.08. Linked List Cycle LCCI](https://leetcode-cn.com/problems/linked-list-cycle-lcci/)

#### Using a set to check if the node has appeared

```python
def detectCycle(self, head):
    my_set = set()
    cursor = head
    while cursor:
        if cursor in my_set:
            return cursor
        my_set.add(cursor)
        cursor = cursor.next
    return None
```

#### Two pointers, faster and slower

If there is cycle, they will meet, then move faster to head, move them one step once, till they meet again, that the intersection.

```python
def detectCycle(self, head):
    faster = head
    slower = head

    while faster and faster.next:
        faster = faster.next.next
        slower = slower.next
        if faster == slower:
            faster = head
            while faster != slower:
                faster = faster.next
                slower = slower.next
            return faster
    return None
```

### [03.01. Three in One LCCI](https://leetcode-cn.com/problems/three-in-one-lcci/)

If implement two stacks with one array, we can grow two stacks from two ends. So that we can make full use of the stack.

Now we are asked to implement three stacks with one array, and it tells us the size of the stack, so we can create an array with three times of the stack, and grow them individually.

But I am going to use another way, three stacks can be stored into three some spaces, stack0 occupy 0,3,6,9..., stack1 occupy 1,4,7..., stack2 occupy 2,5,8..., **for convenience, we can use position 0,1,2 to store the corresponding next position of the stacks. the real elements start from 3,4,5.**

```python
def __init__(self, stackSize):
    """
    :type stackSize: int
    """
    self.arr = [None] * (stackSize*3+3)
    self.size = stackSize
    self.arr[0] = 3
    self.arr[1] = 4
    self.arr[2] = 5

def push(self, stackNum, value):
    """
    :type stackNum: int
    :type value: int
    :rtype: None
    """
    if self.arr[stackNum] // 3 >   self.size:
        return
    self.arr[self.arr[stackNum]] = value
    self.arr[stackNum] += 3

def pop(self, stackNum):
    """
    :type stackNum: int
    :rtype: int
    """
    if self.arr[stackNum] < 6:
        return -1
    self.arr[stackNum] -= 3
    res = self.arr[self.arr[stackNum]]
    self.arr[self.arr[stackNum]] = None
    return res


def peek(self, stackNum):
    """
    :type stackNum: int
    :rtype: int
    """
    if self.arr[stackNum] < 6:
        return -1
    res = self.arr[self.arr[stackNum]-3]
    return res

def isEmpty(self, stackNum):
    """
    :type stackNum: int
    :rtype: bool
    """
    return self.arr[stackNum] < 6
```

### [03.02. Min Stack LCCI](https://leetcode-cn.com/problems/min-stack-lcci/)

#### Having another stack to keep track of the minimal element

```python
def __init__(self):
    """
    initialize your data structure here.
    """
    self.stack = []
    self.min_stack = []

def push(self, x):
    """
    :type x: int
    :rtype: None
    """
    if self.min_stack and self.min_stack[-1] < x:
        self.min_stack.append(self.min_stack[-1])
        self.stack.append(x)
        return
    self.stack.append(x)
    self.min_stack.append(x)

def pop(self):
    """
    :rtype: None
    """
    if len(self.min_stack) == 0:
        return
    self.min_stack.pop()
    return self.stack.pop()

def top(self):
    """
    :rtype: int
    """
    if len(self.min_stack) == 0:
        return
    return self.stack[-1]

def getMin(self):
    """
    :rtype: int
    """
    return self.min_stack[-1]
```
The idea is keeping track of the extreme values:

![alt](https://ch3302files.storage.live.com/y4mSDfx0su6DUE8JSHGcqSM-j0q1yzOnzMI64lEXgCjbI7jaubQLBwU8JkwTxj-RJd4M805w1A0WzUYNJkk9367g9CChm3PY8tZNrkr3x4kBGnNCb05YmmxJhhg0FskyBMuMz2zOIkCr1Me7H1u5PX4vGV2whEyd7lMCcFuU5Vd_Srj3sSR5dRZkYyT_ycWaZSk?width=419&height=299&cropmode=none)

### [03.03. Stack of Plates LCCI](https://leetcode-cn.com/problems/stack-of-plates-lcci/)

#### Two dimensions array
The tricky part is some special boundary case.

```python
def __init__(self, cap):
    """
    :type cap: int
    """
    self.cap = cap
    self.set = []

def push(self, val):
    """
    :type val: int
    :rtype: None
    """
    # 如果初始容量小于0 直接return
    if self.cap <= 0:
        return

    if len(self.set) == 0 or len(self.set[-1]) >= self.cap:
        # 当栈满了，或没有栈了，则新建一个栈
        self.set.append([val])
    else:
        self.set[-1].append(val)

def pop(self):
    """
    :rtype: int
    """
    if len(self.set) == 0:
        return -1
    res = self.set[-1].pop()
    if len(self.set[-1]) == 0:
        # 如果pop后栈为空，则删除该栈
        self.set.pop()

    return res

def popAt(self, index):
    """
    :type index: int
    :rtype: int
    """
    if len(self.set) <= index:
        return -1
    res = self.set[index].pop()
    if len(self.set[index]) == 0:
        # 如果pop后栈为空，则删除该栈
        self.set.remove(self.set[index])
    return res
```

### [面试题 03.04. Implement Queue using Stacks LCCI](https://leetcode-cn.com/problems/implement-queue-using-stacks-lcci/)

#### Two stacks, A is for push, B is for pop

push is doing A.push()

pop is doing B.pop()

Whenever B is empty, push all A to B. Don't have to switch upside down for every push and pop.

```python
def __init__(self):
    """
    Initialize your data structure here.
    """

    self.pushs = []
    self.pops = []

def push(self, x):
    """
    Push element x to the back of queue.
    :type x: int
    :rtype: None
    """

    self.pushs.append(x)

def pop(self):
    """
    Removes the element from in front of queue and returns that element.
    :rtype: int
    """
    if len(self.pops) == 0:
        for i in range(len(self.pushs)):
            self.pops.append(self.pushs.pop())
    return self.pops.pop()


def peek(self):
    """
    Get the front element.
    :rtype: int
    """
    if len(self.pops) == 0:
        for i in range(len(self.pushs)):
            self.pops.append(self.pushs.pop())
    temp = self.pops.pop()
    self.pops.append(temp)
    return temp


def empty(self):
    """
    Returns whether the queue is empty.
    :rtype: bool
    """
    if len(self.pushs)==0 and len(self.pops)==0:
        return True
    else:
        return False
```

