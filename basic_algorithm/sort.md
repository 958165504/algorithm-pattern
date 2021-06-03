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

```

[快排](https://leetcode-cn.com/problems/ba-shu-zu-pai-cheng-zui-xiao-de-shu-lcof/solution/jian-zhi-offer-45-ba-shu-zu-pai-cheng-zu-4q3r/)
```java
行
private void quickSort(String[] strs, int left , int right){
        if(left >= right)
            return;
        int i = left;
        int j = right;
        String tmp = strs[left];//初始枢轴值
        while(i < j){
            //从后往前 大:j--
            while(i < j && (j >= tmp) j--;//注：需要添加相等，否则全部一样ij不动 出不去大循环
            //交换
            strs[i] = strs[j];
            //从前往后: 小:i++
            while(i < j && (i <= tmp) i++;
            //交换
            strs[j] = strs[i];
        }
        //将轴值放到中间
        strs[i] = tmp;
        quickSort(strs, left , i - 1);
        quickSort(strs, i + 1 , right);
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

### 归并排序
> 数组版 归并  
```java
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

    // 归并排序
    private static void merge_sort(int[] arr, int l, int r) {
        // 递归结束条件
        if (l >= r) return;
        // 以下都为逻辑部分
        int mid = l + ((r - l) >> 1);
        merge_sort(arr, l, mid);
        merge_sort(arr, mid + 1, r);
	
        int[] tmp = new int[r - l + 1]; // 临时数组, 用于临时存储 [l,r]区间内排好序的数据
        int i = l, j = mid + 1, k = 0;  // 两个指针
        // 进行归并
        while (i <= mid && j <= r) {
            if (arr[i] <= arr[j]) 
                tmp[k++] = arr[i++];
            else
                tmp[k++] = arr[j++];
        }

        while (i <= mid) tmp[k++] = arr[i++];
        while (j <= r) tmp[k++] = arr[j++];

        // 进行赋值
        for (i = l, j = 0; i <= r; i++, j++)
            arr[i] = tmp[j];
    }
```

### 堆排序
[215. 数组中的第K个最大元素](https://leetcode-cn.com/problems/kth-largest-element-in-an-array/)  

取前k个最小值，构建k大小的大顶堆，然后将所有值一次与堆顶元素比较，更小则取代【我的位置，给我出来，相当于堆是一个暂时的缓存】，直到最后剩k个最小的元素在堆中  
取前k个最大值，构建k大小的小顶堆，然后将所有值一次与堆顶元素比较，更大则取代【我的位置，给我出来，相当于堆是一个暂时的缓存】，直到最后剩k个最大的元素在堆中  

#### 行：二叉堆 推排序思想
>    构建初始堆，将待排序列构成一个大顶堆，升序大顶堆  
>    将堆顶元素与堆尾元素交换，并断开(从待排序列中移除)堆尾元素。  
>    重新构建堆。  
>    重复2~3，直到待排序列中只剩下一个元素(堆顶元素)。  

```java
  /**
     * 堆排序
     */
    public static void headSort(int[] nums) {
        /**
         * 大顶堆：arr[i] >= arr[2i+1] && arr[i] >= arr[2i+2]  
         * 小顶堆：arr[i] <= arr[2i+1] && arr[i] <= arr[2i+2]  
         */
	 //构造初始堆,从第一个非叶子节点开始调整,左右孩子节点中较大的交换到父节点中
        for (int i = (nums.length) / 2 - 1; i >= 0; i--) {
            heapAdjust(nums, nums.length, i);
        }
        //排序，将最大的节点放在堆尾，然后从根节点重新调整
        for (int i = nums.length - 1; i >= 1; i--) {
            int temp = nums[0];
            nums[0] = nums[i];
            nums[i] = temp;
            heapAdjust(nums, i, 0);
        }
    }
    public static void heapAdjust(int[] nums, int len, int i){
        int k = i;
        int temp = nums[i];
        int index = 2 * k + 1;
        while(index < len){
            //找左右子节点较大的
            if(index + 1 < len){
                if(nums[index] < nums[index + 1])
                    index = index + 1;
            }
            //判断父子大小，交换父节点与子节点
            if(temp < nums[index]){//注意这儿是比较temp 不是nums[k]
                nums[k] = nums[index];
                k = index;
                index = 2 * k + 1;
            }else {
                break;
            }
        }
        nums[k] = temp;
    }
```

## 参考

[十大经典排序](https://www.cnblogs.com/onepixel/p/7674659.html)

[二叉堆](https://labuladong.gitbook.io/algo/shu-ju-jie-gou-xi-lie/er-cha-dui-xiang-jie-shi-xian-you-xian-ji-dui-lie)

## 练习

- [ ] 手写快排、归并、堆排序
