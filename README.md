# 算法模板

![来刷题了](https://img.fuiboom.com/img/title.png)

算法模板，最科学的刷题方式，最快速的刷题路径，一个月从入门到 offer，你值得拥有 🐶~

算法模板顾名思义就是刷题的套路模板，掌握了刷题模板之后，刷题也变得好玩起来了~

> 此项目是自己找工作时，从 0 开始刷 LeetCode 的心得记录，通过各种刷题文章、专栏、视频等总结了一套自己的刷题模板。
>
> 这个模板主要是介绍了一些通用的刷题模板，以及一些常见问题，如到底要刷多少题，按什么顺序来刷题，如何提高刷题效率等。

## 在线文档

在线文档 Gitbook：[算法模板 🔥](https://greyireland.gitbook.io/algorithm-pattern/)

## 核心内容

### 入门篇 🐶

- [go 语言入门](./introduction/golang.md)
- [算法快速入门](./introduction/quickstart.md)

### 数据结构篇 🐰

- [二叉树](./data_structure/binary_tree.md)
- [链表](./data_structure/linked_list.md)
- [栈和队列](./data_structure/stack_queue.md)
- [二进制](./data_structure/binary_op.md)
- [图]
 > 拓扑排序：[207. 课程表(图的拓扑排序)](https://leetcode-cn.com/problems/course-schedule/solution/207-ke-cheng-biao-tu-de-tuo-bu-pai-xu-by-zgu9/)  
 判断一个有向图是否无环
 ```java
  public boolean canFinish(int numCourses, int[][] prerequisites) {

        //拓扑排序：原理：使用邻节表结构，对一个图，将入度为0的点入队，
        // 不断去除入度为0的顶点（去掉因果的'因'），出队时将邻接子节点的入度减1，在重复上过程将入度为0入队出队
        //若全部都能去除，则图无环【去不掉则有环，环内不存在入度为0的顶点。】
        //O(n+m)：n个节点，m条邻边

        /*自己理解：使用队列来存入度为0的点，是为了抓住绳子的头部，顺藤摸瓜，避免每次都要全部遍历一遍找入度为0的点，再减子入度*/
        
        //邻接表抽象结构：[入度][数据]-->邻接链表
        int[] indegrees = new int[numCourses];
        List<List<Integer>> adjacency = new ArrayList<>();
        for (int i = 0; i < numCourses; i++) {
            adjacency.add(new ArrayList<>());
        }
        //构图的邻接表
        for (int[]cp :  prerequisites){
            //添加邻接链表
            adjacency.get(cp[1]).add(cp[0]);//因为课程[0,1]是先修1在0，后面那个是父 
            //为邻接子节点入度+1
            indegrees[cp[0]]++;
        }
        
        //将图中入度为0的所有节点先入队【全部的根节点】
        Deque<Integer> deque = new LinkedBlockingDeque<>();
        for (int i = 0; i < indegrees.length; i++) {
            if(indegrees[i] == 0){
                deque.add(i);
            }
        }

        while (!deque.isEmpty()){
            int pre = deque.poll();
            numCourses--;//去掉一个课程，看最后是否能去除完【无环】
            //将将邻接子节点入度 减1，并把其中的入度为0的加入队列
            for (int cur : adjacency.get(pre)) {
                indegrees[cur]--;
                if(indegrees[cur] == 0){
                    deque.add(cur);
                }
            }
        }
        return numCourses == 0;
    }
 ```
 > 图的遍历：[797. 所有可能的路径](https://leetcode-cn.com/problems/course-schedule/solution/207-ke-cheng-biao-tu-de-tuo-bu-pai-xu-by-zgu9/)  
 ```java
 /*题目：
 给一个有 n 个结点的有向无环图，找到所有从 0 到 n-1 的路径并输出（不要求按顺序）
二维数组的第 i 个数组中的单元都表示有向图中 i 号结点所能到达的下一些结点（译者注：有向图是有方向的，即规定了 a→b 你就不能从 b→a ）空就是没有下一个结点了。
 */
 
 /*思路：“图的遍历和回溯算法一样
注：使用回溯模板，图的根结点没有加上，需要额外添加
 */
    public List<List<Integer>> allPathsSourceTarget(int[][] graph) {
        listAll = new LinkedList<List<Integer>>();
        LinkedList<Integer> list = new LinkedList<Integer>();
        list.add(0);//添加根节点
        traverse(graph,new boolean[graph.length], 0,list);
        return listAll;
    }
    List<List<Integer>> listAll;
    public void traverse(int[][] graph,boolean[] visited, int i,List<Integer> list){
        //basecase

        if(i == graph.length - 1){
            listAll.add(new LinkedList<Integer>(list));
            return;
        }
       //选择
        for(int j = 0; j < graph[i].length; j++){
            if( visited[graph[i][j]] == true)//已经被访问过
                continue;
            //访问
            visited[graph[i][j]] = true;
            list.add(graph[i][j]);
            traverse(graph,visited, graph[i][j],list);
            //撤销
            visited[graph[i][j]] = false;
            list.remove(list.size() - 1);
        }
    }
 ```
>邻接表与邻接矩阵  
邻接表很直观，我把每个节点x的邻居都存到一个列表里，然后把x和这个列表关联起来，这样就可以通过一个节点x找到它的所有相邻节点。   
邻接矩阵则是一个二维布尔数组，我们权且成为matrix，如果节点x和y是相连的，那么就把matrix[x][y]设为true。如果想找节点x的邻居，去扫一圈matrix[x][..]就行了。  
![图片](https://user-images.githubusercontent.com/73264826/123726952-6c2c5880-d8c3-11eb-8145-ff4cbbc8f77e.png)



    
### 基础算法篇 🐮

- [二分搜索](./basic_algorithm/binary_search.md)
- [排序算法](./basic_algorithm/sort.md)
- [动态规划](./basic_algorithm/dp.md)

### 算法思维 🦁

- [递归思维](./advanced_algorithm/recursion.md)
- [滑动窗口思想](./advanced_algorithm/slide_window.md)
- [二叉搜索树](./advanced_algorithm/binary_search_tree.md)
- [回溯法](./advanced_algorithm/backtrack.md)

### 第三章、必会算法技巧
#### 1.前缀和数组
> 1 前缀和数组

```java
  //前缀和：可以随意求一个区间[i，j]的累加和 或者累加或
  加--减
  或--异或
    static int subarraySum(int[] nums, int k) {
        int n = nums.length;
        // 构造前缀和
        int[] sum = new int[n + 1];
        sum[0] = 0;
        for (int i = 0; i < n; i++)
            sum[i + 1] = sum[i] | nums[i];//或 相当于加
        int ans = 0;
        // 穷举所有子数组
        for (int i = 1; i <= n; i++)
            for (int j = 0; j < i; j++){
                if ((sum[i] ^ sum[j]) <= k)//异或相当于减
                    ans++;
            }
        return ans;
    }
```
- [前缀和技巧：解决子数组问题 ](https://mp.weixin.qq.com/s?__biz=MzAxODQxMDM0Mw==&mid=2247484488&idx=1&sn=848f76e86fce722e70e265d0c6f84dc3&chksm=9bd7fa40aca07356a6f16db72f5a56529044b1bdb2dcce2de4efe59e0338f0c313de682aef29&scene=21#wechat_redirect)

#### 2.LRU
```java
 class LRUCache {
        /*思路：由hash + 双端链表组成  
        由hash快速定位到链表中哪个节点
        将该节点删除并移到首部，当超出容限，删除链尾和hash中的键值
        主要子函数：
            moveToHead：包含：removeNode，addToHead
            removeTail
         */
        class DLinkedNode {
            int key;
            int value;
            DLinkedNode prev;
            DLinkedNode next;
            public DLinkedNode() {}
            public DLinkedNode(int _key, int _value) {key = _key; value = _value;}
        }
        private Map<Integer, DLinkedNode> cache = new HashMap<Integer, DLinkedNode>();
        private int size;
        private int capacity;
        private DLinkedNode head, tail;
        
        public LRUCache(int capacity) {
            this.size = 0;
            this.capacity = capacity;
            // 使用伪头部和伪尾部节点
            head = new DLinkedNode();
            tail = new DLinkedNode();
            //哑巴节点首尾连起来，初始时，中间没有任何节点
            head.next = tail;
            tail.prev = head;
        }
        public int get(int key) {
            DLinkedNode node = cache.get(key);
            if(node == null)
                return -1;
            // 如果 key 存在，先通过哈希表定位，再移到头部
            moveToHead(node);
            return node.value;
        }
        public void put(int key, int value) {
            DLinkedNode node = cache.get(key);
            if (node == null) {
                // 如果 key 不存在，创建一个新的节点
                DLinkedNode newNode = new DLinkedNode(key, value);
                // 添加进哈希表
                cache.put(key, newNode);
                // 添加至双向链表的头部
                addToHead(newNode);
                ++size;
                if (size > capacity) {
                    // 如果超出容量，删除双向链表的尾部节点
                    DLinkedNode tail = removeTail();
                    // 删除哈希表中对应的项
                    cache.remove(tail.key);
                    --size;
                }
            }
            else {
                // 如果 key 存在，先通过哈希表定位，再修改 value，并移到头部
                node.value = value;
                moveToHead(node);
            }
        }
        private void addToHead(DLinkedNode node) {
            DLinkedNode oldHead = head.next;
            //插入新的头
            head.next = node;
            node.prev = head;
            node.next = oldHead;
            oldHead.prev = node;
        }

        private void removeNode(DLinkedNode node) {
            node.prev.next = node.next;
            node.next.prev = node.prev;
        }

        private void moveToHead(DLinkedNode node) {
            removeNode(node);
            addToHead(node);
        }

        private DLinkedNode removeTail() {
            DLinkedNode removeNode = tail.prev;
            removeNode(removeNode);
            return removeNode;
        }
    }
    - [面试题 16.25. LRU 缓存 ](https://leetcode-cn.com/problems/lru-cache-lcci/)
```



## 心得体会

文章大部分是对题目的思路介绍，和一些问题的解析，有了思路还是需要自己手动写写的，所以每篇文章最后都有对应的练习题

刷完这些练习题，基本对数据结构和算法有自己的认识体会，基本大部分面试题都能写得出来，国内的 BAT、TMD 应该都不是问题

从 4 月份找工作开始，从 0 开始刷 LeetCode，中间大概花了一个半月(6 周)左右时间刷完 240 题。

![一个半月刷完240题](https://img.fuiboom.com/img/leetcode_time.png)

![刷题记录](https://img.fuiboom.com/img/leetcode_record.png)

开始刷题时，确实是无从下手，因为从序号开始刷，刷到几道题就遇到 hard 的题型，会卡住很久，后面去评论区看别人怎么刷题，也去 Google 搜索最好的刷题方式，发现按题型刷题会舒服很多，基本一个类型的题目，一天能做很多，慢慢刷题也不再枯燥，做起来也很有意思，最后也收到不错的 offer（最后去了宇宙系）。

回到最开始的问题，面试到底要刷多少题，其实这个取决于你想进什么样公司，你定的目标如果是国内一线大厂，个人感觉大概 200 至 300 题基本就满足大部分面试需要了。第二个问题是按什么顺序刷及如何提高效率，这个也是本 repo 的目的，给你指定了一个刷题的顺序，以及刷题的模板，有了方向和技巧后，就去动手吧~ 希望刷完之后，你也能自己总结一套属于自己的刷题模板，有所收获，有所成长~

## 推荐的刷题路径

按此 repo 目录刷一遍，如果中间有题目卡住了先跳过，然后刷题一遍 LeetCode 探索基础卡片，最后快要面试时刷题一遍剑指 offer。

为什么这么要这么刷，因为 repo 里面的题目是按类型归类，都是一些常见的高频题，很有代表性，大部分都是可以用模板加一点变形做出来，刷完后对大部分题目有基本的认识。然后刷一遍探索卡片，巩固一下一些基础知识点，总结这些知识点。最后剑指 offer 是大部分公司的出题源头，刷完面试中基本会遇到现题或者变形题，基本刷完这三部分，大部分国内公司的面试题应该就没什么问题了~

1、 [algorithm-pattern 练习题](https://greyireland.gitbook.io/algorithm-pattern/)

![练习题](https://img.fuiboom.com/img/repo_practice.png)

2、 [LeetCode 卡片](https://leetcode-cn.com/explore/)

![探索卡片](https://img.fuiboom.com/img/leetcode_explore.png)

3、 [剑指 offer](https://leetcode-cn.com/problemset/lcof/)

![剑指offer](https://img.fuiboom.com/img/leetcode_jzoffer.png)

刷题时间可以合理分配，如果打算准备面试了，建议前面两部分 一个半月 （6 周）时间刷完，最后剑指 offer 半个月刷完，边刷可以边投简历进行面试，遇到不会的不用着急，往模板上套就对了，如果面试管给你提示，那就好好做，不要错过这大好机会~

> 注意点：如果为了找工作刷题，遇到 hard 的题如果有思路就做，没思路先跳过，先把基础打好，再来刷 hard 可能效果会更好~

## 面试资源

分享一些计算机的经典书籍，大部分对面试应该都有帮助，强烈推荐 🌝

[我看过的 100 本书](https://github.com/greyireland/awesome-programming-books-1)

## 更新计划

持续更新中，觉得还可以的话点个 **star** 收藏呀 ⭐️~

【 Github 】[https://github.com/greyireland/algorithm-pattern](https://github.com/greyireland/algorithm-pattern) ⭐️

## 完成打卡

完成计划之后，可以提交 Pull requests，在下面添加自己的项目仓库，完成自己的算法模板打卡呀~

| 完成 | 用户                                              | 项目地址                                                            |
| ---- | ------------------------------------------------- | ------------------------------------------------------------------- |
| ✅   | [easyui](https://github.com/easyui/) | [algorithm-pattern-swift(Swift 实现)](https://github.com/easyui/algorithm-pattern-swift),[在线文档 Gitbook](https://zyj.gitbook.io/algorithm-pattern-swift/) |
| ✅   | [wardseptember](https://github.com/wardseptember) | [notes(Java 实现)](https://github.com/wardseptember/notes)          |
| ✅   | [dashidhy](https://github.com/dashidhy) | [algorithm-pattern-python(Python 实现)](https://github.com/dashidhy/algorithm-pattern-python) |
| ✅   | [binzi56](https://github.com/binzi56) | [algorithm-pattern-c(c++ 实现)](https://github.com/binzi56/algorithm-pattern-c) |
