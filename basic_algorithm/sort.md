# 排序

## 常考排序

### 快速排序  
> 快速排序就是个二叉树的前序遍历，归并排序就是个二叉树的后续遍历```
```java
//快排框架：前序遍历
void sort(int[] nums, int lo, int hi) {
    /****** 前序遍历位置 ******/
    // 通过交换元素构建分界点 p
    int p = partition(nums, lo, hi);
    /************************/

    sort(nums, lo, p - 1);
    sort(nums, p + 1, hi);
}
//归并框架：后序遍历
void sort(int[] nums, int lo, int hi) {
    int mid = (lo + hi) / 2;
    sort(nums, lo, mid);
    sort(nums, mid + 1, hi);

    /****** 后序遍历位置 ******/
    // 合并两个排好序的子数组
    merge(nums, lo, mid, hi);
    /************************/
}
```

```java
行
/* 快速排序主函数 */
void sort(int[] nums) {
    // 一般要在这用洗牌算法将 nums 数组打乱，
    // 以保证较高的效率，我们暂时省略这个细节
    sort(nums, 0, nums.length - 1);
}
void quickSort(int[] nums, int low, int high){

	if(low >= high)return;
	int pivot = partition(nums,low,high);
	quickSort(nums,low,pivot - 1);
	quickSort(nums, pivot + 1, high);
}
int partition(int[] nums, int low, int high){//分区
	int pivotKey = nums[low];
	while(low < high){
	    while (low < high && nums[high] >= pivotKey)
		high--;
	    swap(nums,low,high);//行：每次移动暂停切换时，都换交换一次，将pivotKey换到另一端，继而移动下一端

	    while (low < high && nums[low] <= pivotKey)
		low++;
	    swap(nums,low,high);
	}
	return low;
}
void swap(int[] nums, int low, int high){
	int temp = nums[low];
	nums[low] = nums[high];
	nums[high] = temp;
}
```
```java
行：快速选择算法(快排+二分查找结合)
题目要求的是「第 k 个最大元素」  
总结一下，快速选择算法就是快速排序的简化版，复用了 partition 函数，快速定位第 k 大的元素。相当于对数组部分排序而不需要完全排序，从而提高算法效率，将平均时间复杂度降到 O(N)。
int findKthLargest(int[] nums, int k) {
    int lo = 0, hi = nums.length - 1;
    // 索引转化
    k = nums.length - k;
    while (lo <= hi) {
        // 在 nums[lo..hi] 中选一个分界点
        int p = partition(nums, lo, hi);
        if (p < k) {
            // 第 k 大的元素在 nums[p+1..hi] 中
            lo = p + 1;
        } else if (p > k) {
            // 第 k 大的元素在 nums[lo..p-1] 中
            hi = p - 1;
        } else {
            // 找到第 k 大元素
            return nums[p];
        }
    }
    return -1;
}
```

```go
func QuickSort(nums []int) []int {
    // 思路：把一个数组分为左右两段，左段小于右段
    quickSort(nums, 0, len(nums)-1)
    return nums

}
// 原地交换，所以传入交换索引
func quickSort(nums []int, start, end int) {
    if start < end {
        // 分治法：divide
        pivot := partition(nums, start, end)
        quickSort(nums, 0, pivot-1)
        quickSort(nums, pivot+1, end)
    }
}
// 分区
func partition(nums []int, start, end int) int {
    // 选取最后一个元素作为基准pivot
    p := nums[end]
    i := start
    // 最后一个值就是基准所以不用比较
    for j := start; j < end; j++ {
        if nums[j] < p {
            swap(nums, i, j)
            i++
        }
    }
    // 把基准值换到中间
    swap(nums, i, end)
    return i
}
// 交换两个元素
func swap(nums []int, i, j int) {
    t := nums[i]
    nums[i] = nums[j]
    nums[j] = t
}
```

### 归并排序

```go
func MergeSort(nums []int) []int {
    return mergeSort(nums)
}
func mergeSort(nums []int) []int {
    if len(nums) <= 1 {
        return nums
    }
    // 分治法：divide 分为两段
    mid := len(nums) / 2
    left := mergeSort(nums[:mid])
    right := mergeSort(nums[mid:])
    // 合并两段数据
    result := merge(left, right)
    return result
}
func merge(left, right []int) (result []int) {
    // 两边数组合并游标
    l := 0
    r := 0
    // 注意不能越界
    for l < len(left) && r < len(right) {
        // 谁小合并谁
        if left[l] > right[r] {
            result = append(result, right[r])
            r++
        } else {
            result = append(result, left[l])
            l++
        }
    }
    // 剩余部分合并
    result = append(result, left[l:]...)
    result = append(result, right[r:]...)
    return
}
```

### 堆排序

#### 行：二叉堆 推排序思想

>    构建初始堆，将待排序列构成一个大顶堆，升序大顶堆  
>    将堆顶元素与堆尾元素交换，并断开(从待排序列中移除)堆尾元素。  
>    重新构建堆。  
>    重复2~3，直到待排序列中只剩下一个元素(堆顶元素)。  

```java 
//行：二叉堆(最大堆)的性质，每个结点比其子节点都大，因为为完全二叉树很容易找到父、左子、右子节点。用数组来存这个完全二叉树，arr[0]不用(如：2的父：2/2=1，索引1必须为根)
// 父节点的索引
int parent(int root) {
    return root / 2;
}
// 左孩子的索引
int left(int root) {
    return root * 2;
}
// 右孩子的索引
int right(int root) {
    return root * 2 + 1;
}
```
```java
//行：上浮函数
private void swim(int k) {
    // 如果浮到堆顶，就不能再上浮了
    while (k > 1 && less(parent(k), k)) {
        // 如果第 k 个元素比上层大
        // 将 k 换上去
        exch(parent(k), k);
        k = parent(k);
    }
}
```
```java
//下沉函数
private void sink(int k) {
    // 如果沉到堆底，就沉不下去了
    while (left(k) <= N) {
        // 先假设左边节点较大
        int older = left(k);
        // 如果右边节点存在，比一下大小
        if (right(k) <= N && less(older, right(k)))
            older = right(k);
        // 结点 k 比俩孩子都大，就不必下沉了
        if (less(older, k)) break;
        // 否则，不符合最大堆的结构，下沉 k 结点
        exch(k, older);
        k = older;
    }
}
```

用数组表示的完美二叉树 complete binary tree

> 完美二叉树 VS 其他二叉树

![image.png](https://img.fuiboom.com/img/tree_type.png)

[动画展示](https://www.bilibili.com/video/av18980178/)

![image.png](https://img.fuiboom.com/img/heap.png)

核心代码

```go
package main

func HeapSort(a []int) []int {
    // 1、无序数组a
	// 2、将无序数组a构建为一个大根堆【行：因为sink需要节点有左右子节点，因此构建时需要从完全二叉树的一个父节点开始下沉归位】
	for i := len(a)/2 - 1; i >= 0; i-- {
		sink(a, i, len(a))
	}
	// 3、交换a[0]和a[len(a)-1]
	// 4、然后把前面这段数组继续下沉保持堆结构，如此循环即可
	for i := len(a) - 1; i >= 1; i-- {
		// 从后往前填充值
		swap(a, 0, i)
		// 前面的长度也减一
		sink(a, 0, i)
	}
	return a
}
func sink(a []int, i int, length int) {
	for {
		// 左节点索引(从0开始，所以左节点为i*2+1)
		l := i*2 + 1
		// 有节点索引
		r := i*2 + 2
		// idx保存根、左、右三者之间较大值的索引
		idx := i
		// 存在左节点，左节点值较大，则取左节点
		if l < length && a[l] > a[idx] {
			idx = l
		}
		// 存在有节点，且值较大，取右节点
		if r < length && a[r] > a[idx] {
			idx = r
		}
		// 如果根节点较大，则不用下沉
		if idx == i {
			break
		}
		// 如果根节点较小，则交换值，并继续下沉
		swap(a, i, idx)
		// 继续下沉idx节点
		i = idx
	}
}
func swap(a []int, i, j int) {
	a[i], a[j] = a[j], a[i]
}

```

## 参考

[十大经典排序](https://www.cnblogs.com/onepixel/p/7674659.html)

[二叉堆](https://labuladong.gitbook.io/algo/shu-ju-jie-gou-xi-lie/er-cha-dui-xiang-jie-shi-xian-you-xian-ji-dui-lie)

## 练习

- [ ] 手写快排、归并、堆排序
