# 二叉搜索树
## 自我总结：
> 1)中序遍历--》升序  
> 2) 删除二叉搜索树的某个点，分三种情况，主要思路还是二叉搜索树的升序规律，找它的前驱或者后继来填  
```java
    看的题解：
    找前驱：先走当前节点的左节点，然后一直右
    找后继：先走当前节点的右节点，然后一直左

    算法思路：
        (1)如果该节点是叶子节点，则直接删除它：root = null。
        (2)如果该节点不是叶子节点且有右节点，则用它的后继节点的值替代 root.val = successor.val，然后删除后继节点。
        (3)如果该节点不是叶子节点且只有左节点，则用它的前驱节点的值替代 root.val = predecessor.val，然后删除前驱节点。
```  
> 3) 高效计算数据流的中位数 尽量使用二叉搜索树的【左大右小特性】，这样速度是logn级别的，而使用【中序遍历为升序】的特性，速度才为o(n)    
> 4) BST 转化累加树 :从大到小降序打印 BST 节点的值，如果维护一个外部累加变量sum，然后把sum赋值给 BST 中的每一个节点，不就将 BST 转化成累加树了吗?    
[把二叉搜索树转换为累加树](https://leetcode-cn.com/problems/convert-bst-to-greater-tree/)  

> 5) 验证二叉搜索树 : 左子树.max < cur < 右子树,min , 因此利用前序遍历，逐步缩小范围往下判断每个结点是否符合，当不符合则向上回溯false， 利用前序遍历 比 后序遍历的 好处，当发现错误时，可以及时停止向下了，减少不必要的递归。  
[验证二叉搜索树](https://leetcode-cn.com/problems/validate-binary-search-tree/)  

> 6) 平衡二叉树：一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过 1 （思路：使用分治法从底往上比较左右子树深度得到本子树的深度，判断此时左右子树是否满足不平衡，进行标志记录。）  
 
 > 7) 在BST中搜索一个数  (行：利用二叉搜索树剪枝)  
   [700. 二叉搜索树中的搜索](https://leetcode-cn.com/problems/search-in-a-binary-search-tree/)  
 ```java
 boolean isInBST(TreeNode root, int target) {
    if (root == null) return false;
    if (root.val == target)
        return true;
    if (root.val < target) 
        return isInBST(root.right, target);
    if (root.val > target)
        return isInBST(root.left, target);
    // root 该做的事做完了，顺带把框架也完成了，妙
}
 ```


## 定义

- 每个节点中的值必须大于（或等于）存储在其左侧子树中的任何值。
- 每个节点中的值必须小于（或等于）存储在其右子树中的任何值。

## 应用

[validate-binary-search-tree](https://leetcode-cn.com/problems/validate-binary-search-tree/)

> 验证二叉搜索树

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func isValidBST(root *TreeNode) bool {
    return dfs(root).valid
}
type ResultType struct{
    max int
    min int
    valid bool
}
func dfs(root *TreeNode)(result ResultType){
    if root==nil{
        result.max=-1<<63
        result.min=1<<63-1
        result.valid=true
        return
    }

    left:=dfs(root.Left)
    right:=dfs(root.Right)

    // 1、满足左边最大值<root<右边最小值 && 左右两边valid
    if root.Val>left.max && root.Val<right.min && left.valid && right.valid {
        result.valid=true
    }
    // 2、更新当前节点的最大最小值
    result.max=Max(Max(left.max,right.max),root.Val)
    result.min=Min(Min(left.min,right.min),root.Val)
    return
}
func Max(a,b int)int{
    if a>b{
        return a
    }
    return b
}
func Min(a,b int)int{
    if a>b{
        return b
    }
    return a
}

```

[insert-into-a-binary-search-tree](https://leetcode-cn.com/problems/insert-into-a-binary-search-tree/)

> 给定二叉搜索树（BST）的根节点和要插入树中的值，将值插入二叉搜索树。 返回插入后二叉搜索树的根节点。 保证原始二叉搜索树中不存在新值。

```go
func insertIntoBST(root *TreeNode, val int) *TreeNode {
    if root==nil{
        return &TreeNode{Val:val}
    }
    if root.Val<val{
        root.Right=insertIntoBST(root.Right,val)
    }else{
        root.Left=insertIntoBST(root.Left,val)
    }
    return root
}
```

[delete-node-in-a-bst](https://leetcode-cn.com/problems/delete-node-in-a-bst/)

> 给定一个二叉搜索树的根节点 root 和一个值 key，删除二叉搜索树中的  key  对应的节点，并保证二叉搜索树的性质不变。返回二叉搜索树（有可能被更新）的根节点的引用。

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func deleteNode(root *TreeNode, key int) *TreeNode {
    // 删除节点分为三种情况：
    // 1、只有左节点 替换为右
    // 2、只有右节点 替换为左
    // 3、有左右子节点 左子节点连接到右边最左节点即可
    if root ==nil{
        return root
    }
    if root.Val<key{
        root.Right=deleteNode(root.Right,key)
    }else if root.Val>key{
        root.Left=deleteNode(root.Left,key)
    }else if root.Val==key{
        if root.Left==nil{
            return root.Right
        }else if root.Right==nil{
            return root.Left
        }else{
            cur:=root.Right
            // 一直向左找到最后一个左节点即可
            for cur.Left!=nil{
                cur=cur.Left
            }
            cur.Left=root.Left
            return root.Right
        }
    }
    return root
}
```

[balanced-binary-tree](https://leetcode-cn.com/problems/balanced-binary-tree/)

> 给定一个二叉树，判断它是否是高度平衡的二叉树。

```go
type ResultType struct{
    height int
    valid bool
}
func isBalanced(root *TreeNode) bool {
    return dfs(root).valid
}
func dfs(root *TreeNode)(result ResultType){
    if root==nil{
        result.valid=true
        result.height=0
        return
    }
    left:=dfs(root.Left)
    right:=dfs(root.Right)
    // 满足所有特点：二叉搜索树&&平衡
    if left.valid&&right.valid&&abs(left.height,right.height)<=1{
        result.valid=true
    }
    result.height=Max(left.height,right.height)+1
    return
}
func abs(a,b int)int{
    if a>b{
        return a-b
    }
    return b-a
}
func Max(a,b int)int{
    if a>b{
        return a
    }
    return b
}

```

## 练习

- [ ] [validate-binary-search-tree](https://leetcode-cn.com/problems/validate-binary-search-tree/)
- [ ] [insert-into-a-binary-search-tree](https://leetcode-cn.com/problems/insert-into-a-binary-search-tree/)
- [ ] [delete-node-in-a-bst](https://leetcode-cn.com/problems/delete-node-in-a-bst/)
- [ ] [balanced-binary-tree](https://leetcode-cn.com/problems/balanced-binary-tree/)
