# 回溯法

## 行：
### >1.总结：
做回溯题，画树，兄弟节点(回溯每一层，当不满足时就剪枝 continue))每一个分支表示一个解，可以根据分支结束来写回溯的结束条件。
### >2. 回溯模板
```java
	result = []
	def backtrack(路径, 选择列表):
	    if 满足结束条件:
		result.add(路径)
		return

	    for 选择 in 选择列表:
		做选择
		backtrack(路径, 选择列表)
		撤销选择
```
### >3.子集、组合、排列的模板

#### 子集
![图片](https://user-images.githubusercontent.com/73264826/121987265-bd6c2080-cdca-11eb-9d7f-8b917c105d33.png)
```java
/**题目
问题很简单，输入一个不包含重复数字的数组，要求算法输出这些数字的所有子集。
*/
/**思路
话树，且使用前序遍历，保存树的所有节点为子集，(要用一个 start 排除已经选择过的数字。)
*/

//子集 (要用一个 start 排除已经选择过的数字。)
void backtrack(vector<int>& nums, int start, vector<int>& track) {
    res.push_back(track);
    for (int i = start; i < nums.size(); i++) {
        // 做选择
        track.push_back(nums[i]);
        // 回溯
        backtrack(nums, i + 1, track);
        // 撤销选择
        track.pop_back();
    }
}

```

#### 组合
![图片](https://user-images.githubusercontent.com/73264826/121987754-ada10c00-cdcb-11eb-9034-488f0c67cb59.png)

```java
/**题目：输入两个数字 n, k，算法输出 [1..n] 中 k 个数字的所有组合。
*/
/**思路：
画树，树的叶子节点为解
k 限制了树的高度，n 限制了树的宽度(要用一个 start 排除已经选择过的数字。)
*/

//组合：k 限制了树的高度，n 限制了树的宽度(要用一个 start 排除已经选择过的数字。)
void backtrack(int n, int k, int start, vector<int>& track) {
    // 到达树的底部
    if (k == track.size()) {
        res.push_back(track);
        return;
    }
    // 注意 i 从 start 开始递增
    for (int i = start; i <= n; i++) {
        // 做选择
        track.push_back(i);
        backtrack(n, k, i + 1, track);
        // 撤销选择
        track.pop_back();
    }
}
```
#### 排列
![图片](https://user-images.githubusercontent.com/73264826/121988146-66674b00-cdcc-11eb-97ad-a91c6212b2fe.png)

```java
/**题目：输入一个不包含重复数字的数组 nums，返回这些数字的全部排列。
*/
/**思路：一般做题穷举最常见的模板，选择是随机的，使用vistied[i]保存选择的，在一堆苹果里面每个随机选择拿一个，标记vistied，再拿剩下的，直到拿完位置。
叶子节点就是解
*/

//排列 (contains【也是vistied】 方法排除已经选择的数字)
void backtrack(int[] nums, LinkedList<Integer> track) {
    // 触发结束条件
    if (track.size() == nums.length) {
        res.add(new LinkedList(track));
        return;
    }
    
    for (int i = 0; i < nums.length; i++) {
        // 排除不合法的选择
        if (track.contains(nums[i]))
            continue;
        // 做选择
        track.add(nums[i]);
        // 进入下一层决策树
        backtrack(nums, track);
        // 取消选择
        track.removeLast();
    }
}
记住这几种树的形状，就足以应对大部分回溯算法问题了，无非就是 start【子集和组合，选择是有先后顺序】 或者 contains【排列，选择是随机选，没有先后】 剪枝，也没啥别的技巧了。
```
##### 20210615 中兴提前批笔试题，【回溯 排列做】
问题：选择列表【int i = 0,每种技能的伤害】， 每种技能当妖怪的血量小于多少，会伤害双倍【不要被额外的条件吓住，就是在每次选择技能时，再判断该技能是否能造成双倍而已，只是再多了两种分支判断选择而已】， 使用vist[i]保存当前随机选择的i，再往下递归继续选剩下的技能，直到妖怪的血量为0    
思路：其实就是画一个树，先选哪个技能，其子节点再往下选剩余技能，叶子节点【打死妖怪】就是选择的可行解    


### > 4.回溯与递归、位运算解题比较：
回溯的做选择和撤销选择，其实和题解中的用递归或位运算从1~n,每个元素加入或者不加入两种情况思路一样  
### > 5.遇到重复子集问题
(需要先排序，重复元素的一小段此时的顺序就是唯一的排列，避免重复)，为了消去重复的子集，需要先将选择列表进行排序，再在每一轮start选子集的时候，遇到相同和前一个的就跳过，  
[重复子集](https://leetcode-cn.com/problems/subsets-ii/)  

### > 6. [22.括号生成](https://leetcode-cn.com/problems/generate-parentheses/)  
穷举所有可能，剪枝  
```java
/*
回溯法：
    选择：当前字符选择左，还是右
    剪枝：左括号<maxLen，右括号<左括号 才为当前有效字符串
*/
 List<String> result = new LinkedList<>();
    int maxLen = 0;
    public List<String> generateParenthesis(int n) {
        maxLen = n;
        recall(0 , 0, 0, new StringBuilder());
        return result;
    }
    private void recall(int index , int leftCount, int rightCount, StringBuilder path){
        //basecase
        if(index == maxLen * 2){
            if(leftCount == rightCount)
                //加入这条路径
                result.add(path.toString());
            else
                return;
        }
        //剪枝:左括号<maxLen，右括号<左括号 才为当前有效字符串
        //选择:添加左
        if(leftCount < maxLen){
             path.append("(");
            recall(index + 1 , leftCount + 1, rightCount,  path);
            path.deleteCharAt(path.length() - 1);//回溯
        }
        //选择:添加右【利用必须leftCount > rightCount剪枝】
        if(leftCount > rightCount){
            path.append(")");
            recall(index + 1 , leftCount , rightCount + 1,  path);
            path.deleteCharAt(path.length() - 1);//回溯
        }
    }
```


## 背景

回溯法（backtrack）常用于遍历列表所有子集，是 DFS 深度搜索一种，一般用于全排列，穷尽所有可能，遍历的过程实际上是一个决策树的遍历过程。时间复杂度一般 O(N!)，它不像动态规划存在重叠子问题可以优化，回溯算法就是纯暴力穷举，复杂度一般都很高。

## 模板

```go
result = []
func backtrack(选择列表,路径):
    if 满足结束条件:
        result.add(路径)
        return
    for 选择 in 选择列表:
        做选择
        backtrack(选择列表,路径)
        撤销选择
```

核心就是从选择列表里做一个选择，然后一直递归往下搜索答案，如果遇到路径不通，就返回来撤销这次选择。

## 示例

### [subsets](https://leetcode-cn.com/problems/subsets/)

> 给定一组不含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）。

遍历过程

![image.png](https://img.fuiboom.com/img/backtrack.png)

```go
func subsets(nums []int) [][]int {
	// 保存最终结果
	result := make([][]int, 0)
	// 保存中间结果
	list := make([]int, 0)
	backtrack(nums, 0, list, &result)
	return result
}

// nums 给定的集合
// pos 下次添加到集合中的元素位置索引
// list 临时结果集合(每次需要复制保存)
// result 最终结果
func backtrack(nums []int, pos int, list []int, result *[][]int) {
	// 把临时结果复制出来保存到最终结果
	ans := make([]int, len(list))
	copy(ans, list)
	*result = append(*result, ans)
	// 选择、处理结果、再撤销选择
	for i := pos; i < len(nums); i++ {
		list = append(list, nums[i])
		backtrack(nums, i+1, list, result)
		list = list[0 : len(list)-1]
	}
}
```

### [subsets-ii](https://leetcode-cn.com/problems/subsets-ii/)

> 给定一个可能包含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）。说明：解集不能包含重复的子集。
#### 行：这类包含重复元素序列，都需要进行排序，相同的值挨着的那个序列就是唯一的排列，选择时从左往右选未被选择的点，这样不会重复，原本的排列顺序就是其解
```go
import (
	"sort"
)

func subsetsWithDup(nums []int) [][]int {
	// 保存最终结果
	result := make([][]int, 0)
	// 保存中间结果
	list := make([]int, 0)
	// 先排序
	sort.Ints(nums)
	backtrack(nums, 0, list, &result)
	return result
}

// nums 给定的集合
// pos 下次添加到集合中的元素位置索引
// list 临时结果集合(每次需要复制保存)
// result 最终结果
func backtrack(nums []int, pos int, list []int, result *[][]int) {
	// 把临时结果复制出来保存到最终结果
	ans := make([]int, len(list))
	copy(ans, list)
	*result = append(*result, ans)
	// 选择时需要剪枝、处理、撤销选择
	for i := pos; i < len(nums); i++ {
        // 排序之后，如果再遇到重复元素，则不选择此元素
		if i != pos && nums[i] == nums[i-1] {
			continue
		}
		list = append(list, nums[i])
		backtrack(nums, i+1, list, result)
		list = list[0 : len(list)-1]
	}
}
```

### [permutations](https://leetcode-cn.com/problems/permutations/)

> 给定一个   没有重复   数字的序列，返回其所有可能的全排列。

思路：需要记录已经选择过的元素，满足条件的结果才进行返回

```go
func permute(nums []int) [][]int {
    result := make([][]int, 0)
    list := make([]int, 0)
    // 标记这个元素是否已经添加到结果集
    visited := make([]bool, len(nums))
    backtrack(nums, visited, list, &result)
    return result
}

// nums 输入集合
// visited 当前递归标记过的元素
// list 临时结果集(路径)
// result 最终结果
func backtrack(nums []int, visited []bool, list []int, result *[][]int) {
    // 返回条件：临时结果和输入集合长度一致 才是全排列
    if len(list) == len(nums) {
        ans := make([]int, len(list))
        copy(ans, list)
        *result = append(*result, ans)
        return
    }
    for i := 0; i < len(nums); i++ {
        // 已经添加过的元素，直接跳过
        if visited[i] {
            continue
        }
        // 添加元素
        list = append(list, nums[i])
        visited[i] = true
        backtrack(nums, visited, list, result)
        // 移除元素
        visited[i] = false
        list = list[0 : len(list)-1]
    }
}
```

### [permutations-ii](https://leetcode-cn.com/problems/permutations-ii/)

> 给定一个可包含重复数字的序列，返回所有不重复的全排列。
#### 行：这类包含重复元素序列，都需要进行排序，相同的值挨着的那个序列就是唯一的排列，选择时从左往右选未被选择的点，这样不会重复，原本的排列顺序就是其解
```go
import (
	"sort"
)

func permuteUnique(nums []int) [][]int {
	result := make([][]int, 0)
	list := make([]int, 0)
	// 标记这个元素是否已经添加到结果集
	visited := make([]bool, len(nums))
	sort.Ints(nums)
	backtrack(nums, visited, list, &result)
	return result
}

// nums 输入集合
// visited 当前递归标记过的元素
// list 临时结果集
// result 最终结果
func backtrack(nums []int, visited []bool, list []int, result *[][]int) {
	// 临时结果和输入集合长度一致 才是全排列
	if len(list) == len(nums) {
		subResult := make([]int, len(list))
		copy(subResult, list)
		*result = append(*result, subResult)
	}
	for i := 0; i < len(nums); i++ {
		// 已经添加过的元素，直接跳过
		if visited[i] {
			continue
		}
        // 上一个元素和当前相同，并且没有访问过就跳过
		if i != 0 && nums[i] == nums[i-1] && !visited[i-1] {
			continue
		}
		list = append(list, nums[i])
		visited[i] = true
		backtrack(nums, visited, list, result)
		visited[i] = false
		list = list[0 : len(list)-1]
	}
}
```

## 练习

- [ ] [subsets](https://leetcode-cn.com/problems/subsets/)
- [ ] [subsets-ii](https://leetcode-cn.com/problems/subsets-ii/)
- [ ] [permutations](https://leetcode-cn.com/problems/permutations/)
- [ ] [permutations-ii](https://leetcode-cn.com/problems/permutations-ii/)

挑战题目 
- [ ] [combination-sum](https://leetcode-cn.com/problems/combination-sum/)
> 行
> 做题思路：回溯（这道题和 纸币买东西题类似）  
> 先画出树的结构，每个节点下所有的节点再作为子节点。  
> 剪枝：由于这个树是全排列，在路径之和大于target时就剪掉  
> 之后发现路径会出现重复的子序列（由于是全排列），因此设置一个start，使得树的每个节点的子节点必须大于等于父节点，这样不会出现重复的，小的永远在前，这样重复序列的排列只有一种情况。 
- [ ] [letter-combinations-of-a-phone-number](https://leetcode-cn.com/problems/letter-combinations-of-a-phone-number/)
- [ ] [palindrome-partitioning](https://leetcode-cn.com/problems/palindrome-partitioning/)
> 行 看的题解 做回溯题，画树，兄弟节点(回溯每一层，当不满足时就剪枝 continue))每一个分支表示一个解，可以根据分支结束来写回溯的结束条件  
- [ ] [restore-ip-addresses](https://leetcode-cn.com/problems/restore-ip-addresses/)  
> 行   
>   思路：耶，自己写出来了。  
>    方法：先画树，树的每一层兄弟节点就是回溯的一次for  
>          根据画树时的剪枝来进行for里面的兄弟节点剪枝，根据每一条分支的结尾来写结束语句  
>    注意点；1.substring的end索引需要多移一位  
>           2.注意索引start == s.length() 而不是<  
>           3.StringBuilder 拼接用append() 而不是+= ; "a"+"b"纯粹字符串拼接其实也是先创建了一个StringBuilder对象，在进行append();  

- [ ] [permutations](https://leetcode-cn.com/problems/permutations/)
