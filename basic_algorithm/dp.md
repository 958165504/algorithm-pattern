# 动态规划
动态规划和记忆化递归区别：动态规划需要自己规定路径移动顺序【我觉得操蛋的地方】，记忆化只是列举每种计划层层递归，是正常的逻辑，不用考虑顺序。但动归可以进一步优化空间【所以必须要学，【表情】![图片](https://user-images.githubusercontent.com/73264826/120952174-19a4c400-c77d-11eb-9f03-1590b4ce5fcd.png)】  
> 【动态规划做股票买卖的感悟20210626】  
	状态：动态规划就是先找出所有状态，在确定dp定义  
	选择：穷举所有状态，来根据选择来计算不同的状态，直至穷举完  

# 一、自我总结
### > 1) 步骤：明确「状态」 -> 定义 dp 数组/函数的含义 -> 明确「选择取最值」-> 明确 base case  
```java
# 凑零钱问题
a.先确定「状态」，也就是原问题和子问题中变化的变量。由于硬币数量无限，所以唯一的状态就是目标金额amount。
b.然后确定dp函数的定义：函数 dp(n)表示，当前的目标金额是n，至少需要dp(n)个硬币凑出该金额。
c.然后确定「选择」并择优，也就是对于每个状态，可以做出什么选择改变当前状态。具体到这个问题，无论当的目标金额是多少，选择就是从面额列表coins中选择一个硬币，然后目标金额就会减少：
# 伪码框架
def coinChange(coins: List[int], amount: int):
    # 定义：要凑出金额 n，至少要 dp(n) 个硬币
    def dp(n):
        # 做选择，需要硬币最少的那个结果就是答案
        for coin in coins:
            res = min(res, 1 + dp(n - coin))
        return res
    # 我们要求目标金额是 amount
    return dp(amount)

d.最后明确 base case，显然目标金额为 0 时，所需硬币数量为 0；当目标金额小于 0 时，无解，返回 -1：
```
### > 2) 最优子结构性质作为动态规划问题的必要条件，一定是让你求最值的，以后碰到那种恶心人的—**最值题，思路往动态规划想就对了，这就是套路**。  动态规划不就是**从最简单的 base case 往后推导吗**，可以想象成一个链式反应，不断以小博大。但只有符合最优子结构的问题，才有发生这种链式反应的性质。找最优子结构的过程，其实就是证明状态转移方程正确性的过程，**方程符合最优子结构就可以写暴力解了**，写出暴力解就可以看出有没有重叠子问题了，有则优化，无则 OK。这也是套路，经常刷题的朋友应该能体会。  

### > 3) dp 数组的遍历方向 :抓住两点： 1、遍历的过程中，所需的状态必须是已经计算出来的。2、遍历的终点必须是存储结果的那个位置。  
[labuladong 动态规划答疑篇 ](https://mp.weixin.qq.com/s?__biz=MzAxODQxMDM0Mw==&mid=2247484832&idx=1&sn=44ad2505ac5c276bf36eea1c503b78c3&chksm=9bd7fba8aca072be32f66e6c39d76ef4e91bdbf4ef993014d4fee82896687ad61da4f4fc4eda&scene=21#wechat_redirect)  

# 二、子序列类型问题
> 总结:两子序列问题：  

子序列的动态规划：都是以ij结尾，当i==j，直接向前，当不相等，看删除i还是j【由于序列可以不连续，因此考虑删i或j】  
如：1）编辑距离：考虑word1转换未word2,当i!=j，考虑删除i,增加i，替换i等操作。  
    2）最长公共子序列：i!=j时，考虑删除ih或者j  

### > 1)编辑距离： 解决两个字符串的动态规划问题 ，一般都是用**两个指针i,j分别指向两个字符串的最后，然后一步步往前走**，缩小问题的规模。
[labuladong 经动态规划：编辑距离 ](https://mp.weixin.qq.com/s?__biz=MzAxODQxMDM0Mw==&mid=2247484731&idx=3&sn=aa642cbf670feee73e20428775dff0b5&chksm=9bd7fb33aca0722568ab71ead8d23e3a9422515800f0587ff7c6ef93ad45b91b9e9920d8728e&scene=21#wechat_redirect)  
[72. 编辑距离](https://leetcode-cn.com/problems/edit-distance/)  
```java
/**题目
> 题目：给你两个单词 word1 和 word2，请你计算出将 word1 转换成 word2 所使用的最少操作数 。  
> 你可以对一个单词进行如下三种操作：  
>   插入一个字符  
>   删除一个字符  
>   替换一个字符  
*/

/**思路
 *  选择
     *      if(s[i] == s[j]) dp[i,j] = dp[i - 1, j -1] ,直接跳过
     *      if(s[i] != s[j])
     *          dp[i , j] = dp[i, j - 1] + 1  //插入
     *          dp[i , j] = dp[i - 1, j] + 1  //删除
     *          dp[i , j] = dp[i - 1, j - 1] + 1  //改
*/

//二刷：20210607
    public int minDistance(String word1, String word2) {
        int[][] dp = new int[word1.length() + 1][word2.length() + 1];
        //basecase
        //第一列
        for (int i = 1; i < dp.length; i++) {
            dp[i][0] = dp[i - 1][0] + 1;
        }
        //第一行
        for (int i = 1; i < dp[0].length; i++) {
            dp[0][i] = dp[0][i - 1] + 1;
        }
        for (int i = 1; i <= word1.length(); i++) {
            for (int j = 1; j <= word2.length(); j++) {
                if(word1.charAt(i - 1) == word2.charAt(j - 1))
                    dp[i][j] = dp[i - 1][j - 1];
                else {
                    //增加
                    int tmp1= dp[i][j - 1] + 1;
                    //删除
                    int tmp2 = dp[i - 1][j] + 1;
                    //修改
                    int tmp3 = dp[i - 1][j - 1] + 1;
                    dp[i][j] = Math.min(Math.min(tmp1,tmp2),tmp3);
                }
            }
        }
        return dp[dp.length - 1][dp[0].length - 1];
    }
```


### > 2）最长递增子序列 ： 
[300. 最长递增子序列](https://leetcode-cn.com/problems/longest-increasing-subsequence/)    
已知dp[1....4]， 求dp[5]。 nums[5] = 3，既然是递增子序列，我们只要找到前面那些结尾比 3 小的子序列，然后把 3 接到最后，就可以形成一个新的递增子序列，而且这个新的子序列长度加一。需要将nums[5]回溯 从头开始找比其小的dp值，重新组成最长序列的最大长度。  
```java
/**题目
给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。
子序列是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，[3,6,2,7] 是数组 [0,3,1,6,2,2,7] 的子序列。
*/

/**思路
二刷：20210608
状态i
dp[i]:以i为结尾的子序列长度最大值
选择：回溯i， for j=0;j<i;j++  当i>j dp[i] = max(dp[i], dp[j]+1),得出该区间最大长度
用记忆化搜索也行，递归相等于去掉最外层的那个fori
 */
    public int lengthOfLIS(int[] nums) {
        int[] dp = new int[nums.length];
        //basecase
        Arrays.fill(dp,1);
        int maxLen = dp[0];
        //选择
        for(int i = 0; i < nums.length; i++){
            for(int j = 0; j < i; j++){
                if(nums[i] > nums[j]){
                     dp[i] = Math.max(dp[i],dp[j] + 1);
                }
            }
            maxLen = Math.max(dp[i],maxLen);//注：需要回溯完整个j,在得出最大值
        }
        return maxLen;
    }
```
### > 3)最长递增子序列之信封嵌套问题 
[354. 俄罗斯套娃信封问题](https://leetcode-cn.com/problems/russian-doll-envelopes/)    
```java
/**题目
给你一个二维整数数组 envelopes ，其中 envelopes[i] = [wi, hi] ，表示第 i 个信封的宽度和高度。
当另一个信封的宽度和高度都比这个信封大的时候，这个信封就可以放进另一个信封里，如同俄罗斯套娃一样。
请计算 最多能有多少个 信封能组成一组“俄罗斯套娃”信封（即可以把一个信封放到另一个信封里面）。
*/
/**思路：
  先排序使得先根据宽w升序，h再降序，在根据h序列找最长递增序列
对h降序原因：对于宽度w相同的数对，要对其高度h进行降序排序。因为两个宽度相同的信封不能相互包含的，而逆序排序保证在w相同的数对中最多只选取一个计入 LIS，因此选最大的h放前面。  
*/

 public int maxEnvelopes(int[][] envelopes) {
        //先排序使得先根据宽w升序，h再降序，在根据h序列找最长递增序列
        Arrays.sort(envelopes, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[0] == o2[0] ? o2[1] - o1[1] : o1[0] - o2[0];
            }
        });

        //动态规划找h的最大递增序列
        int[] dp = new int[envelopes.length];
        //basecase
        Arrays.fill(dp,1);
        int maxRes = 0;
        for (int i = 0; i < dp.length; i++) {
            for (int j = 0; j < i; j++) {
                if(envelopes[i][1] > envelopes[j][1]){
                    dp[i] = Math.max(dp[i],dp[j] + 1);
                }
            }
            maxRes = Math.max(dp[i],maxRes);
        }
        return maxRes;
    }
```
### >4) 最长公共子序列
```java
/*题目:给定两个字符串 text1 和 text2，返回这两个字符串的最长 公共子序列 的长度。如果不存在 公共子序列 ，返回 0 。
例如，"ace" 是 "abcde" 的子序列，但 "aec" 不是 "abcde" 的子序列。
*/
/*思路：
1.dp[i][j]定义：以ij为结尾的两字符串的最长公共子序列  
2.basecase:第一行第一列全为0，初始化数组已经为0，不用赋初值了
3.选择
        /**
        1.当i==j,一起前进
        2.当i!=j,考虑删除i或者j,取两种情况所得最长公共的最大值
         */
	 
总结两子序列问题：子序列的动态规划：都是以ij结尾，当i==j，直接向前，当不相等，看删除i还是j【由于序列可以不连续，因此考虑删i或j】
如：编辑距离：考虑word1转换未word2,当i!=j，考虑删除i,增加i，替换i等操作。
   最长公共子序列：i!=j时，考虑删除ih或者j
*/



  public int  longestCommonSubsequence(String text1, String text2) {
        //定义以ij为结尾的两字符串的最长公共子序列  
        int[][] dp = new int[text1.length() + 1][text2.length() + 1];
        //basecase:第一行第一列全为0，初始化数组已经为0，不用赋初值了

        //选择
        /**
        1.当i==j,一起前进
        2.当i!=j,考虑删除i或者j,取两种情况所得最长公共的最大值
         */
        for(int i = 1; i <= text1.length(); i++){
            for(int j = 1; j <= text2.length(); j++){
                if(text1.charAt(i - 1) == text2.charAt(j - 1)){
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                }else{
                    int temp1 =  dp[i][j - 1];//删i
                    int temp2 =  dp[i - 1][j];//删j
                    dp[i][j] = Math.max(temp1,temp2);
                }
            }
        }
        return dp[text1.length()][text2.length()];
    }
```


### > 4) 子序列解题模板：最长回文子序列 【子序列问题是，对ij比较，相等则一起移动，不相等则看删除i还j】
```java
1、第一种思路模板是一个一维的 dp 数组：
int n = array.length;
int[] dp = new int[n];
for (int i = 1; i < n; i++) {
    for (int j = 0; j < i; j++) {
        dp[i] = 最值(dp[i], dp[j] + ...)
    }
}
举个我们写过的例子 最长递增子序列，在这个思路中 dp 数组的定义是：在子数组array[0..i]中，以array[i]结尾的目标子序列（最长递增子序列）的长度是dp[i]。

2、第二种思路模板是一个二维的 dp 数组：
int n = arr.length;
int[][] dp = new dp[n][n];
for (int i = 0; i < n; i++) {
    for (int j = 1; j < n; j++) {
        if (arr[i] == arr[j]) 
            dp[i][j] = dp[i][j] + ...
        else
            dp[i][j] = 最值(...)
    }
}
这种思路运用相对更多一些，尤其是涉及两个字符串/数组的子序列。本思路中 dp 数组含义又分为「只涉及一个字符串」和「涉及两个字符串」两种情况。  
```
[516. 最长回文子序列（动态规划）](https://leetcode-cn.com/problems/longest-palindromic-subsequence/solution/516-zui-chang-hui-wen-zi-xu-lie-dong-tai-sily/)  
注：找到状态转移和 base case 之后，一定要观察 DP table，看看怎么遍历才能保证通过已计算出来的结果解决新的问题【比如本题的 从底往上计算扫描】  

```java
 public int longestPalindromeSubseq(String s) {
 
 	/**题目：给定一个字符串 s ，找到其中最长的回文子序列，并返回该序列的长度。可以假设 s 的最大长度为 1000*/
	
        /*
        思路：
        通过ij，分割成所有子区间[i,j]
        然后判断子区间的最大回文数
            i==j,dp[i][j] = dp[i+1][j-1]+2
            i!=j,考虑删除i还是j，取最大值。dp[i][j]=max(dp[i][j-1],dp[i+1][j])
        最后返回dp[0][n-1],[0,n-1]最大区间的最大回文数
         */
        int[][]dp = new int[s.length()][s.length()];
        int n = s.length();
        //basecase
        for(int i = 0; i < n; i++){
            dp[i][i] = 1;
        }
        for(int i = n - 1; i >= 0; i--){
            for(int j = i + 1;j <= n - 1; j ++){
                if(s.charAt(i) == s.charAt(j)){
                    dp[i][j] = dp[i + 1][j - 1] + 2;
                }else{
                    //删i还是j
                    dp[i][j] = Math.max(dp[i + 1][j],dp[i][j - 1]);
                }
            }
        }
        return dp[0][n - 1];
    }
```

# 三、子串问题
>子串的问题，都是连续的，因此基本都是以i为结尾，只用看dp[i-1]推导出dp[i],而子序列不是连续的，因此需要遍历0-i-1来推导出dp[i]  

### > 1) 53. 最大子序和：  
[53. 最大子序和](https://leetcode-cn.com/problems/maximum-subarray//)  
```java
/**题目：给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
*/

/**思路：
思路：这种连续子串的动态规划，一般都是以i为结尾，只用看前面i-1的情况，子序列问题，就需要看0-i-1 来得出i
*/
    public int maxSubArray(int[] nums) {
        int[] dp = new int[nums.length];
        for(int i = 0; i < nums.length; i++){
            dp[i] = nums[i];
        }
        int res = dp[0];
        for(int i = 1; i < nums.length; i++){
            dp[i] = dp[i - 1] > 0 ? dp[i - 1] + dp[i]:dp[i];
            res = Math.max(res, dp[i]);
        }
        return res;
    }
```

### > 7) 回文子串解题思路：利用判断【i，j】区间是否为回文动态规划解法为基础来变形
> 总结：关于回文的，都可以基于判断【i，j】区间是否为回文动态规划解法为基础，来进行穷举所有区间判断，来变相解题  
dp[i][j] = (array[i] == array[j])&&( j - i < 3 || dp[i - 1][j - 1]);//动态规划状态
    当i == j, 就去掉两端转移看i+1,j-1是否为回文
    当j-i<3,只有一个 或两个 三个 且两端相等，那一定为回文  
    
[647. 回文子串（利用判断【i，j】区间是否为回文动态规划解法为基础）](https://leetcode-cn.com/problems/palindromic-substrings/solution/647-hui-wen-zi-chuan-li-yong-pan-duan-ij-aofk/)  
```java
/**题目：
给定一个字符串，你的任务是计算这个字符串中有多少个回文子串。
具有不同开始位置或结束位置的子串，即使是由相同的字符组成，也会被视作不同的子串。
*/

/**思路
首先根据ij划分所有子区间
若i==j，则dp[i][j] = dp[i+1][j-1]?true:false;//看下一层子集是否为回文
若i!=j, 直接为false.由于是子串不是子序列，因此不用考虑删i还是删j
 */

    public int countSubstrings(String s) {
        boolean[][] dp = new boolean[s.length()][s.length()];
        //basecase
        int res = 0;
        for(int i = 0; i < s.length(); i++){
            dp[i][i] = true;
        }
        for(int i = s.length() - 1; i >=0 ; i--){
            for(int j = i; j < s.length(); j++){
                if(s.charAt(i) == s.charAt(j)){
                    if(j - i <= 2){
                        dp[i][j] = true;
                    }else
                        dp[i][j] = dp[i + 1][j - 1]? true : false;
                }else{
                    dp[i][j] = false;
                }
                if(dp[i][j])
                    res++;
            }
        }
        return res;
    }
```
# 三、背包类型

### > 1) 0-1背包问题【选还是不选】
```java
/*题目：
给你一个可装载重量为W的背包和N个物品，每个物品有重量和价值两个属性。其中第i个物品的重量为wt[i]，价值为val[i]，现在让你用这个背包装物品，最多能装的价值是多少？
*/
/*思路：感觉这个动归的状态转移方程和递归从上往下的选择公式是一样的*/
int knapsack(int W, int N, vector<int>& wt, vector<int>& val) {
    // vector 全填入 0，base case 已初始化
    vector<vector<int>> dp(N + 1, vector<int>(W + 1, 0));
    for (int i = 1; i <= N; i++) {
        for (int w = 1; w <= W; w++) {
            if (w - wt[i-1] < 0) {
                // 当前背包容量装不下，只能选择不装入背包
                dp[i][w] = dp[i - 1][w];
            } else {
                // 装入或者不装入背包，择优
                dp[i][w] = max(dp[i - 1][w - wt[i-1]] + val[i-1], 
                               dp[i - 1][w]);
            }
        }
    }
    return dp[N][W];
}
```
#### 分割等和子集 转为为01背包问题
> 每个商品i选还是不选，能否刚好装满背包sum
[416. 分割等和子集](https://leetcode-cn.com/problems/partition-equal-subset-sum/)  
```java
题目：给你一个 只包含正整数 的 非空 数组 nums 。请你判断是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。
思路：
    官方解答： 转为01背包问题：每个商品选i还是不选，看最中能否刚好装满背包W 
    
     public boolean canPartition(int[] nums) {
        int len = nums.length;
        // 题目已经说非空数组，可以不做非空判断
        int sum = 0;
        for (int num : nums) {
            sum += num;
        }
        // 特判：如果是奇数，就不符合要求
        if ((sum & 1) == 1) {
            return false;
        }

        int target = sum / 2;
        // 创建二维状态数组，行：物品索引，列：容量（包括 0）
        boolean[][] dp = new boolean[len][target + 1];

        // 先填表格第 0 行，第 1 个数只能让容积为它自己的背包恰好装满
        if (nums[0] <= target) {
            dp[0][nums[0]] = true;
        }
        // 再填表格后面几行
        for (int i = 1; i < len; i++) {
            for (int j = 0; j <= target; j++) {
                // 直接从上一行先把结果抄下来，然后再修正
                dp[i][j] = dp[i - 1][j];

                if (nums[i] == j) {
                    dp[i][j] = true;
                    continue;
                }
                //选还是不选
                if (nums[i] < j) {
                    dp[i][j] = dp[i - 1][j] || dp[i - 1][j - nums[i]];
                }
            }
        }
        return dp[len - 1][target];
    }
```
### > 2)完全背包问题
[518. 零钱兑换 II](https://leetcode-cn.com/problems/partition-equal-subset-sum/)  
```java
/*题目：给你一个整数数组 coins 表示不同面额的硬币，另给一个整数 amount 表示总金额。
请你计算并返回可以凑成总金额的硬币组合数。如果任何硬币组合都无法凑出总金额，返回 0 。
假设每一种面额的硬币有无限个。 
*/
/*思路：
前i个物品装满j的方法数 = 第i物品 选 + 不选的方法数
比如：你想用面值为 2 的硬币【当前i】凑出金额 5【当前j】，那么如果你知道了凑出金额 3 的方法【dp[i][j - conis[i]]】，再加上一枚面额为 2 的硬币，不就可以凑出 5 了嘛
dp[i][j] = dp[i][j] = dp[i -1][j]  
		+ dp[i][j - coins[i - 1]];//注：这儿是可重复性的物品，因此选择i,i仍然不动，不为i-1,这是完全背包与01背包的区别

*/
    public int change(int amount, int[] coins) {
        int[][] dp = new int[coins.length + 1][amount + 1];//前i个货币能装满amount的数量
        //basecase
        for(int i = 0; i <= coins.length; i++){
            dp[i][0] = 1;
        }
        for(int i = 1; i <= coins.length; i++){
            for(int j = 1; j <= amount; j++){
                if(j - coins[i - 1] < 0){
                    dp[i][j] = dp[i - 1][j];//不选
                }else{
                    //选和不选两种方法相加的数量就是总的方法数
                    dp[i][j] = dp[i -1][j] 
                    + dp[i][j - coins[i - 1]];//注：这儿是可重复性的物品，因此选择i,i仍然不动，不为i-1,这是完全背包与01背包的区别
                }
            }
        }
        return  dp[coins.length][amount];
    }
```
[322. 零钱兑换](https://leetcode-cn.com/problems/coin-change/)  
```java
/*题目：给定不同面额的硬币 coins 和一个总金额 amount。编写一个函数来计算可以凑成总金额所需的最少的硬币个数。如果没有任何一种硬币组合能组成总金额，返回 -1。
你可以认为每种硬币的数量是无限的。
*/
/*思路
状态：变量 amount
    选择：遍历coins
    dp[n] :n元时最少的张数凑齐
    base：amout为0时，只需要0张纸币
*/
public int coinChange(int[] coins, int amount) {
        int[] dp = new int[amount + 1];
        Arrays.fill(dp,amount + 1);//先赋初值，为最大值，全由1组成且+1不能的个数，之后为了比较装入最小值
        dp[0] = 0;//base
        //遍历状态1的所有取值
        for (int i = 1; i <= amount; i++) {
            for (int j = 0; j < coins.length; j++) {
                if(i - coins[j] < 0)
                    continue;
                //选择最小的：选择还是不选择i
                dp[i] = Math.min(dp[i], dp[i - coins[j]] + 1);
            }
        }
        return dp[amount] == amount + 1? -1 : dp[amount];
    }
```
```java
/*题目：给定正整数 n，找到若干个完全平方数（比如 1, 4, 9, 16, ...）使得它们的和等于 n。你需要让组成和的完全平方数的个数最少。
给你一个整数 n ，返回和为 n 的完全平方数的 最少数量 。。*/
/*
思路：转换为零钱兑换1问题，从选择列表钱种类中【无限个】，凑齐n的最小个数
*/
    //二刷：20210624【完全背包问题】零钱兑换1
    public int numSquares(int n) {
        /*
        dp[n] = min( dp[n],dp[n - list(i)] + 1)//组成n的最小个数，选还是不选i两种方案最小值
        状态n
        选择：每个i选还是不选两种方案的最小值
         */
        //构造选择列表
        int maxSquareNum = (int) (Math.sqrt(n)) + 1;//考虑到根号为小数情况
        int[] list = new int[maxSquareNum + 1];
        for(int i = 1; i <= maxSquareNum; i++){
            list[i - 1] = i * i;
        }
        int [] dp = new int[n + 1];
         //basecase 
         Arrays.fill(dp, n + 1);//全由1组成且+1，不能的个数
         dp[0] = 0;//背包为0
         for(int i = 1; i <= n; i++){
             //在每个背包容量下，选择最小的一个选择列表i
             for(int j = 0; j < list.length; j++){
                 if(i - list[j] >= 0){
                     dp[i] = Math.min(dp[i],dp[i - list[j]] + 1);//i选还是不选
                 }else{
                     //不选
                     dp[i] = dp[i];
                     break;//后面的选择列表直接不看了，肯定相减<0
                 }
             }
         }
         return dp[n];
    }
```

# 四、贪心类型
### > 9) 贪心算法
> 总结） 本题贪心算法与动态规划区别  
> 动归：动归是递归区间的所以子节点并找出步数最小值，而贪心，不用每个子节点都递归，而是找其中最具有潜力的节点即可。  
> 【动态是穷举，而贪心是莽夫每次拿最好的出来，求局部最优根本不考虑之后是否还有更好的解，是否是全局最优】  

[435. 无重叠区间](https://leetcode-cn.com/problems/non-overlapping-intervals/)  
本题体验动态规划的思想是：要选择最早的end的区间，因为要尽量腾出更多的空间来装其他的区间，不断的局部最优求得全局最优【自我感觉：贪心算法就是莽夫算法，每次都用最好的，没有绕弯子。比如斗地主，贪心算法每次都是出最大的王炸牌，没有博弈，是莽夫】
```java
/*题目：给定一个区间的集合，找到需要移除区间的最小数量，使剩余区间互不重叠。
注意:
    可以认为区间的终点总是大于它的起点。
    区间 [1,2] 和 [2,3] 的边界相互“接触”，但没有相互重叠。
*/

/*思路：
        * 贪心算法：贪心算法之区间调度问题 ：本题转换为求最多不重叠区间问题。  移除最小数量 = len - 最多不重叠区间
        求最多不重叠区间的思想：先根据end从小到大排序，每次选取最小end的区间，从lists中删除与end重叠的区间【end0 > start2】（因为尽量要让更多的区间不重合，因此要选择最早的end的区间，因为要尽量腾出更多的空间来装其他的区间，然后删除与它相交的；一直这样不断找新的end，删除与其相交的，直到区间集合lists为空）
       
       
       * 求最多不重叠区间：先排序,从低往上，取end1 <= start2
        *                   是：则收下这个区间，取end2为末尾依次递进
        *                   不是：则比较下一个区间 end1 <= start3
        * 因为是排了序的，相当于求：尽量使得区间填满时间线【使得时间线最长】
*/
    public int eraseOverlapIntervals(int[][] intervals) {
        /*
        * 贪心算法：贪心算法之区间调度问题 ：本题转换为求最多不重叠区间问题。  移除最小数量 = len - 最多不重叠区间
        求最多不重叠区间的思想：先根据end从小到大排序，每次选取最小end的区间，从lists中删除与end重叠的区间【end0 > start2】（因为尽量要让更多的区间不重合，因此要选择最早的end的区间，因为要尽量腾出更多的空间来装其他的区间，然后删除与它相交的；一直这样不断找新的end，删除与其相交的，直到区间集合lists为空）
        * 求最多不重叠区间：先排序,从低往上，取end1 <= start2
        *                   是：则收下这个区间，取end2为末尾依次递进
        *                   不是：则比较下一个区间 end1 <= start3
        * 因为是排了序的，相当于求：尽量使得区间填满时间线【使得时间线最长】
        * */

         if(intervals.length <= 0)
            return 0;

        /*先求最多不重叠区间，然后用len- 最多不重叠区间 就是移除区间的最小数*/
        //以各区间的end 升序排序
        Arrays.sort(intervals, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[1] - o2[1];
            }
        });

        int noIntersectCont = 1;//不相交数至少为1
        int end = intervals[0][1];
        for (int i = 1; i < intervals.length; i++) {
            if(end <= intervals[i][0]){//不重叠的区间，更新下一个区间的end
                noIntersectCont++;
                //更新末尾end
                end = intervals[i][1];
            }
        }
        return intervals.length - noIntersectCont;//需要去掉最少的空间
    }
```
[452. 用最少数量的箭引爆气球 | 也是最多不重叠区间数问题【区间调度】】](https://leetcode-cn.com/problems/minimum-number-of-arrows-to-burst-balloons/)  
```java
/*题目：一支弓箭可以沿着 x 轴从不同点完全垂直地射出。在坐标 x 处射出一支箭，若有一个气球的直径的开始和结束坐标为 xstart，xend， 且满足  xstart ≤ x ≤ xend，则该气球会被引爆。可以射出的弓箭的数量没有限制。 弓箭一旦被射出之后，可以无限地前进。我们想找到使得所有气球全部被引爆，所需的弓箭的最小数量。
给你一个数组 points ，其中 points [i] = [xstart,xend] ，返回引爆所有气球所必须射出的最小弓箭数。
*/
/*思路：

        * 贪心算法： 本题和区间调度问题一模一样，可转换为求最多的不重叠区间，即最少的箭矢
        * 先排序,从低往上，取end1 < start2【和区间调度算法相比，<= 变为< ，就算按着气球 也能引爆】
        *                   是：则收下这个区间，取end2为末尾依次递进
        *                   不是：则比较下一个区间 end1 < start3
        * 
*/

    public int findMinArrowShots(int[][] points) {
        /*
        * 贪心算法： 本题和区间调度问题一模一样，可转换为求最多的不重叠区间，即最少的箭矢
        * 先排序,从低往上，取end1 < start2【和区间调度算法相比，<= 变为< ，就算按着气球 也能引爆】
        *                   是：则收下这个区间，取end2为末尾依次递进
        *                   不是：则比较下一个区间 end1 < start3
        * 
        * */
        if(points.length <= 0)
            return 0;
        Arrays.sort(points, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                ////return o1[1] - o2[1];//以后排序不要用减法，因为-2147483645 - 2147483646 造成int溢出 变成正数
                return o1[1] < o2[1]? -1 : 1;
            }
        });
        int noIntersectCont = 1;//不相交数至少为1
        int end = points[0][1];
        for (int i = 1; i < points.length; i++) {
            if(end < points[i][0]){//当不重叠的区间，更新下一个区间的end，去除与end相交的区间，直至区间列表为空
                noIntersectCont++;
                //更新末尾end
                end = points[i][1];
            }
        }
        return noIntersectCont;//需要多少只箭矢，就是多少个不重叠的区间
    }
```
    
[55. 跳跃游戏](https://leetcode-cn.com/problems/jump-game/)  
```java
/*题目：给定一个非负整数数组 nums ，你最初位于数组的 第一个下标 。
数组中的每个元素代表你在该位置可以跳跃的最大长度。
判断你是否能够到达最后一个下标。
*/
//二刷：20210624 贪心算法
/*思路：攻城掠地：根据每个炮兵的炮程向前一步一步推进，并实时更远最远的边疆，看是否能到达末尾，若炮兵位置都超出最远边界了，则不能达到末尾 */
     public boolean canJump(int[] nums) {
        int longest = nums[0];//维护一个最长的距离变量
        //遍历每个点，更新最远距离，若最远距离能到达末尾，则可以跳
        for(int i = 0; i < nums.length; i++){
            if(i > longest){//若i都追上longest，则不能到达末尾
                return false;
            }
            if(i + nums[i] > longest){
                longest = i + nums[i];
            }
            if(longest >= nums.length - 1)
                return true;
        }
        return false;
     }
```

```java
/*题目：给定一个非负整数数组，你最初位于数组的第一个位置。
数组中的每个元素代表你在该位置可以跳跃的最大长度。
你的目标是使用最少的跳跃次数到达数组的最后一个位置。
假设你总是可以到达数组的最后一个位置。
*/
/*每次在射程范围内选取下一个更有潜力的炮兵*/
 //二刷：20210624 贪心算法
     public int jump(int[] nums) {
         if(nums.length == 1)
            return 0;
        int farest = nums[0];//维护一个最远距离
        int index = 0;//当前位置
        int step = 1;//步数
        while(farest < nums.length - 1){
            //选择下一个跳跃点
            int oldFarest = farest;
            for(int i = index; i < nums.length && i <= oldFarest; i++){
                if(i + nums[i] > farest){
                    farest = i + nums[i];
                    index = i;//下一个跳到该点
                }
            }
            step++;
        }
        return step;
     }
     
     或者
int jump(vector<int>& nums) {
    int n = nums.size();
    int end = 0, farthest = 0;
    int jumps = 0;
    for (int i = 0; i < n - 1; i++) {
        farthest = max(nums[i] + i, farthest);
        if (end == i) {//已完成本次区间去找最有潜力的炮兵，开始进入下一区间，寻找更有潜力炮兵
            jumps++;
            end = farthest;
        }
    }
    return jumps;
}


本题贪心算法与动态规划区别
动归：动归是递归区间的所以子节点并找出步数最小值，而贪心，不用每个子节点都递归，而是找其中最具有潜力的节点即可。
【动态是穷举，而贪心是莽夫每次拿最好的出来，求局部最优根本不考虑之后是否还有更好的解，是否是全局最优】
vector<int> memo;
// 主函数
int jump(vector<int>& nums) {
    int n = nums.size();
    // 备忘录都初始化为 n，相当于 INT_MAX
    // 因为从 0 调到 n - 1 最多 n - 1 步
    memo = vector<int>(n, n);
    return dp(nums, 0);
}

int dp(vector<int>& nums, int p) {
    int n = nums.size();
    // base case
    if (p >= n - 1) {
        return 0;
    }
    // 子问题已经计算过
    if (memo[p] != n) {
        return memo[p];
    }
    int steps = nums[p];
    // 你可以选择跳 1 步，2 步...
    for (int i = 1; i <= steps; i++) {
        // 穷举每一个选择
        // 计算每一个子问题的结果
        int subProblem = dp(nums, p + i);
        // 取其中最小的作为最终结果
        memo[p] = min(memo[p], subProblem + 1);
    }
    return memo[p];
}
```


# 五、其他经典类型
### > 10) 打家劫舍
[198. 打家劫舍](https://leetcode-cn.com/problems/house-robber/)  

```java

    //*方法一：动态规划从上往下：打家劫舍1：动态规划，先写出递归的，容易想到*/
	//1.状态：就是index索引，到哪儿了
    	////2.选择：当前点抢还是不抢
    	////两种选择的最大值，每种选择都局部最优，推向全局最优

    //方法2：从下往上
    /*2.打家劫舍1：动态规划 从下往上解法*/
    //【自己还是反映不过来从底往上的思路，都是看了递归 才写出来的
    // 从低往上的动态规划：看做哪选择，需要和前面的哪个dp[]递推而来
    // 】
    //basecase对于我也比较困难 找到


// /*方法一：动态规划从上往下：打家劫舍1：动态规划，先写出递归的，容易想到*/
    // int [] memeory ;
    // public int rob(int[] nums) {

    //     //备忘录 赋初值
    //     memeory = new int[nums.length];
    //     Arrays.fill(memeory,-1);
    //     return robTraverse(nums,0);
    // }
    // public int robTraverse(int[] nums,int index) {
    //     //1.状态：就是index索引，到哪儿了
    //     //2.选择：当前点抢还是不抢
    //     //两种选择的最大值，每种选择都局部最优，推向全局最优

    //     //basecase
    //     if(index >= nums.length)
    //         return 0;
    //     //选择之前，看下备忘录有现成的没,消除重复子问题
    //     if(memeory[index] != -1)
    //         return memeory[index];

    //     //选择：抢不抢
    //     int money1 = robTraverse(nums,index + 2) + nums[index];//枪
    //     int money2 = robTraverse(nums,index + 1);//不枪

    //     return memeory[index] = Math.max(money1,money2) ;//返回两种选择的最大值，每种选择都局部最优，推向全局最优
    // }
    //方法2：从下往上
    /*2.打家劫舍1：动态规划 从下往上解法*/
    //【自己还是反映不过来从底往上的思路，都是看了递归 才写出来的
    // 从低往上的动态规划：看做哪选择，需要和前面的哪个dp[]递推而来
    // 】
    //basecase对于我也比较困难 找到
    public int rob(int[] nums) {
        if(nums == null)
            return 0;
        if(nums.length <= 1)
            return nums[0];

        //1.dp[i] 定义：状态索引i时，可抢到的最大金额
        int[] dp =new int[nums.length];
        /*边界条件为
        1.dp[0]=nums[0]只有一间房屋 则偷窃该房屋
        2.dp[1]=max(nums[0],nums[1]) 只有两间房屋，选择其中金额较高的房屋进行偷窃
        */
        dp[0] = nums[0];
        dp[1] = Math.max(nums[0],nums[1]);

        //2.选择：
        for (int i = 2; i < dp.length; i++) {
                //选择 ：抢还是不抢
                dp[i] = Math.max(dp[i - 2] + nums[i], dp[i - 1]); //抢 | 不抢
                //【自己还是反映不过来从底往上的思路，都是看了递归 才写出来的
                // 从低往上的动态规划：看做哪选择，需要和前面的哪个dp[]递推而来
                // 】
        }
        return dp[dp.length - 1];
    }

```

> 【动态规划做股票买卖的感悟20210626】  
	状态：动态规划就是先找出所有状态，在确定dp定义  
	选择：穷举所有状态，来根据选择来计算不同的状态，直至穷举完  
### 股票买卖问题[动态规划+备忘录很好理解]
> 参考东哥的算法[股票问题的一种通用解法 ](https://mp.weixin.qq.com/s?__biz=MzAxODQxMDM0Mw==&mid=2247484509&idx=1&sn=21ace57f19d996d46e82bd7d806a2e3c&source=41#wechat_redirect)  
>模板
```java
//动态规划+备忘录：是在买卖股票的最佳时机I的基础上【只能一次交易】，不断的穷举卖出，在递归下一轮分片交易[sell+1,end]
    /* 核心动态规划 选择代码，穷举
    //选择：在以本次start买入的情况下，列举之后所有作为卖出时间点，再往下递归的最大值
    for (int sell = start + 1; sell <= prices.length - 1; sell++) {
            if(prices[sell] < curmin){
                curmin = prices[sell];
            }
            maxProfit = Math.max(maxProfit,prices[sell] - curmin + maxProfitTraverse(prices, sell + 1));
            //在sell索引轮卖掉，进入下一次交易选择区间[sell+1,end];
        }
    */
```

[121. 买卖股票的最佳时机](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/)  
```java
/*题目：
给定一个数组 prices ，它的第 i 个元素 prices[i] 表示一支给定股票第 i 天的价格。
你只能选择 某一天 买入这只股票，并选择在 未来的某一个不同的日子 卖出该股票。设计一个算法来计算你所能获取的最大利润。
返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 0 。
*/

/*思路： 二刷：20210625 动态规划
状态：动态规划就是先找出所有状态，在确定dp定义
选择：穷举所有状态，来根据选择来计算不同的状态，直至穷举完

本题状态：第i天，当前是否持股和昨天是否持股有关系，为此我们需要把 是否持股 设计到状态数组中。
定义dp[i][0] 今天不持股的最大收益
选择：dp[i][0] = max(dp[i - 1][1] + price[i],dp[i - 1][0])//今天不持股和昨天持股或不持股的关系，选取收益最大值
dp[i][1] = max(-price[i],dp[i - 1][1])同理
*/

    public int maxProfit(int[] prices) {
        int[][] dp = new int[prices.length][2];//dp[i][0];第i天不持有的收益，dp[i][1]第i天持有的收益
        //basecase
        dp[0][0] = 0;//第1天不买入，收益为0
        dp[0][1] = -prices[0];//第一天买入，收益为prices[0]
        for(int i = 1; i < prices.length; i++){
            dp[i][0] = Math.max(dp[i - 1][0],dp[i - 1][1] +  + prices[i]);
            dp[i][1] = Math.max(/*dp[i - 1][0]*/ - prices[i],dp[i - 1][1]);//注：这儿交易只能一次，因此昨天不持股，今天持股，收益为-prices[i]，而dp[i - 1][0] - prices[i]表示可以多次交易情况
        }
        return dp[prices.length - 1][0];
    }
```


[122. 买卖股票的最佳时机 II（动态规划+备忘录）](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/)  
```java
//动态规划+备忘录：是在买卖股票的最佳时机I的基础上【只能一次交易】，不断的穷举卖出，在递归下一轮分片[sell+1,end]
    /* 核心动态规划 选择代码，穷举
    //选择：在以本次start买入的情况下，列举之后所有作为卖出时间点，再往下递归的最大值
    for (int sell = start + 1; sell <= prices.length - 1; sell++) {
            if(prices[sell] < curmin){
                curmin = prices[sell];
            }
            maxProfit = Math.max(maxProfit,prices[sell] - curmin + maxProfitTraverse(prices, sell + 1));
            //在sell索引轮卖掉，进入下一次交易选择区间[sell+1,end];
        }
    */
     public int maxProfit(int[] prices) {
        memo = new int[prices.length];
        Arrays.fill(memo,-1);
        return maxProfitTraverse(prices,0);
    }

    int[] memo;
    public int maxProfitTraverse(int[] prices, int start) {
        //basecase
        if(start >= prices.length - 1)
            return 0;
        if(memo[start] != -1)
            return memo[start];
        int maxProfit = 0;//本次递归交易的最大收益
        int curmin = prices[start];//当前交易最低点
        //选择：在以本次start买入的情况下，列举之后所有作为卖出时间点，再往下递归的最大值
        for (int sell = start + 1; sell <= prices.length - 1; sell++) {
           
            if(prices[sell] < curmin){
                curmin = prices[sell];
            }
            maxProfit = Math.max(maxProfit,prices[sell] - curmin + maxProfitTraverse(prices, sell + 1));
            //在sell索引轮卖掉，进入下一次交易选择区间[sell+1,end];
        }
        return memo[start] = maxProfit;
    }
```
```java
    //二刷：20210625 动态规划
    public int maxProfit(int[] prices) {
        int[][] dp = new int[prices.length][2];
        //basecase
        dp[0][0] = 0;
        dp[0][1] = -prices[0];
        for(int i = 1; i < prices.length; i++){
            dp[i][0] = Math.max(dp[i - 1][0],dp[i - 1][1] + prices[i]);
            dp[i][1] = Math.max(dp[i - 1][0] - prices[i],dp[i - 1][1]);
        }
        return dp[prices.length - 1][0];
    }
```
[123. 买卖股票的最佳时机 III](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iii/)  
```java
 public int maxProfit(int[] prices) {
/*题目：给定一个数组，它的第 i 个元素是一支给定的股票在第 i 天的价格。
设计一个算法来计算你所能获取的最大利润。你最多可以完成 两笔 交易。
注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
 */

        int[][][] dp = new int[prices.length][3][2];//[天][交易次数][持是否有] 规定买入时交易+1
        //basecase 交易0次没有意义，就不初始化了
        dp[0][1][1] = -prices[0];
        dp[0][1][0] = 0;//同一天买入卖出
        dp[0][2][1] = Integer.MIN_VALUE; //-prices[0];;//
        dp[0][2][0] = 0;//同一天买入卖出两次
        
        for(int i = 1; i < prices.length; i++){
            //顺序 必须是先持有再卖出 先1后0,
            dp[i][1][1] = Math.max(dp[i - 1][1][1],dp[i - 1][0][0] - prices[i]);
            dp[i][1][0] = Math.max(dp[i - 1][1][0], dp[i - 1][1][1] + prices[i]);
            dp[i][2][1] = Math.max(dp[i - 1][2][1], dp[i - 1][1][0] - prices[i]);
            dp[i][2][0] = Math.max(dp[i - 1][2][0], dp[i - 1][2][1] + prices[i]);
        }
        return Math.max(dp[prices.length - 1][2][0],dp[prices.length - 1][1][0]);
    }

```

[309. 最佳买卖股票时机含冷冻期](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/)  
```java
    /*
    思路：用labuladong的递归模板一改，还是递归的动归简单
        //选择：基础还是简单的只有一次交易动归：//以当前start为买入点，维护本次交易的买入最小值，为了动归计算本次卖出点收益最大值dp[i]=max(dp[i-1],prices[i] - cruMin)，在不断向下递归
        for (int sell = start + 1; sell <= prices.length - 1; sell++) {
            //维护最小值
            if(prices[sell] < curMin)
                curMin = prices[sell];
            curMaxProfit = Math.max(curMaxProfit, (prices[sell] - curMin) + dp(prices,sell + 2));//冷冻期+2
        }
    */
    public int maxProfit(int[] prices) {
        memo = new int[prices.length];
        Arrays.fill(memo, -1);
        int maxProfit = 0;
        maxProfit = dp(prices, 0);
        return maxProfit;
    }
    int[] memo;
    private int dp(int[] prices, int start){//以start为买入点，收获的最大值
        //basecase
        if(start >= prices.length)//最后一天才买入股票，卖不出去了，收益为0
            return 0;
        //备忘录
        if(memo[start] != -1)
            return memo[start];
        int curMin = prices[start];//以当前start为买入点，维护本次交易的买入最小值，
                                    //为了动归计算本次卖出点收益最大值dp[i]=max(dp[i-1],prices[i] - cruMin)
        int curMaxProfit = 0;
        //选择
        for (int sell = start + 1; sell <= prices.length - 1; sell++) {
            //维护最小值
            if(prices[sell] < curMin)
                curMin = prices[sell];
            curMaxProfit = Math.max(curMaxProfit, (prices[sell] - curMin) + dp(prices,sell + 2));//这儿跳2冷冻期
        }  
        return memo[start] = curMaxProfit;
    }
```
```java
/*题目：
给定一个整数数组，其中第 i 个元素代表了第 i 天的股票价格 。​
设计一个算法计算出最大利润。在满足以下约束条件下，你可以尽可能地完成更多的交易（多次买卖一支股票）:
    你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
    卖出股票后，你无法在第二天买入股票 (即冷冻期为 1 天)。
*/
/*思路：二刷：20210626：动态规划 
状态：[天数][持有][冷冻]
选择：   dp[i][0][0] = Math.max(dp[i - 1][0][1],dp[i - 1][0][0]);
        dp[i][1][0] = Math.max(dp[i - 1][1][0],dp[i - 1][0][0] - prices[i]);//今天啥也不干，昨天不冷冻买入，昨天冷冻买入 //注：这儿昨天冷冻买入情况不知道为啥不能加
        dp[i][0][1] = dp[i - 1][1][0] + prices[i];//今天卖出导致冷冻或者
*/
    public int maxProfit(int[] prices) {
        int[][][] dp = new int[prices.length][2][2];//[天数][持有][冷冻]
        //basecase
        dp[0][0][0] = 0;
        dp[0][1][0] = -prices[0];//第1天买入
        dp[0][0][1] = 0;//第一天买入又卖出，所以为冷冻期

        //选择
        for(int i = 1; i < prices.length; i++){
            dp[i][0][0] = Math.max(dp[i - 1][0][1],dp[i - 1][0][0]);
            dp[i][1][0] = Math.max(dp[i - 1][1][0],dp[i - 1][0][0] - prices[i]);//今天啥也不干，昨天不冷冻买入，昨天冷冻买入 //注：这儿昨天冷冻买入情况不知道为啥不能加
            dp[i][0][1] = dp[i - 1][1][0] + prices[i];//今天卖出导致冷冻或者
        }
        return Math.max(dp[prices.length - 1][0][0],dp[prices.length - 1][0][1]);
    }

```


[188. 买卖股票的最佳时机 IV](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iv/)  
    注意：这儿的状态有两个 start , k，因此备忘录int[][] memo是二维的  
```java
 /*思路：在122. 买卖股票的最佳时机 II（动态规划+备忘录）基础上，增加了交易次数k
    注意：这儿的状态有两个 start , k，因此备忘录int[][] memo是二维的
    
    //动态规划+备忘录：是在买卖股票的最佳时机I的基础上【只能一次交易】，不断的穷举卖出，在递归下一轮分片[sell+1,end]

        核心动态规划 选择代码，穷举
        for (int sell = start + 1; sell <= prices.length - 1; sell++) {
            if(prices[sell] < curmin){
                curmin = prices[sell];
            }
            maxProfit = Math.max(maxProfit,prices[sell] - curmin + maxProfitTraverse(prices, sell + 1));
            //在sell索引轮卖掉，进入下一次交易选择区间[sell+1,end];
        }    
    */
    int[][] memo;//注：动态规划的状态有两个，所以备忘录为二维
    public int maxProfit(int k, int[] prices) {
        memo = new int[prices.length][k+1];
        for (int i = 0; i < memo.length; i++) {
            Arrays.fill(memo[i], -1);
        }
        int maxProfit = 0;
        maxProfit = dp(prices, 0,k);
        return maxProfit;
    }
    private int dp(int[] prices, int start ,int k){//以start为买入点，收获的最大值
        //basecase
        if(start >= prices.length || k <= 0)//最后一天才买入股票，卖不出去了或者交易次数没有了 收益为0
            return 0;
        //备忘录
        if(memo[start][k] != -1)
            return memo[start][k];
        int curMin = prices[start];//以当前start为买入点，维护本次交易的买入最小值，
                               //为了动归计算本次卖出点收益最大值dp[i]=max(dp[i-1],prices[i] - cruMin)
        int curMaxProfit = 0;
        //选择
        for (int sell = start + 1; sell <= prices.length - 1; sell++) {
            //维护最小值
            if(prices[sell] < curMin)
                curMin = prices[sell];
            curMaxProfit = Math.max(curMaxProfit, (prices[sell] - curMin) + dp(prices,sell + 1,k - 1));
        }
        return memo[start][k] = curMaxProfit;
    } 
```
[714. 买卖股票的最佳时机含手续费](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/)  
```java
/*题目：
给定一个整数数组 prices，其中第 i 个元素代表了第 i 天的股票价格 ；非负整数 fee 代表了交易股票的手续费用。
你可以无限次地完成交易，但是你每笔交易都需要付手续费。如果你已经购买了一个股票，在卖出它之前你就不能再继续购买股票了。
返回获得利润的最大值。
注意：这里的一笔交易指买入持有并卖出股票的整个过程，每笔交易你只需要为支付一次手续费。*/

/*思路：
状态:[天数][是否持有]
选择：穷举计算所有状态，直至结束
     dp[i][0] = Math.max(dp[i - 1][0],dp[i - 1][1] + prices[i] - fee);
     dp[i][1] = Math.max(dp[i - 1][0] - prices[i],dp[i - 1][1]);
*/


    public int maxProfit(int[] prices, int fee) {
        int[][] dp = new int[prices.length][2];//[天数][是否持有]
        //basecase
        dp[0][0] = 0;
        dp[0][1] = - prices[0];
        //选择
        for(int i = 1; i < prices.length; i++){
            dp[i][0] = Math.max(dp[i - 1][0],dp[i - 1][1] + prices[i] - fee);
            dp[i][1] = Math.max(dp[i - 1][0] - prices[i],dp[i - 1][1]);
        }
        return dp[prices.length - 1][0];
    }
```




## 背景

先从一道题目开始~

如题  [triangle](https://leetcode-cn.com/problems/triangle/)

> 给定一个三角形，找出自顶向下的最小路径和。每一步只能移动到下一行中相邻的结点上。

例如，给定三角形：

```text
[
     [2],
    [3,4],
   [6,5,7],
  [4,1,8,3]
]
```

自顶向下的最小路径和为  11（即，2 + 3 + 5 + 1 = 11）。

使用 DFS（遍历 或者 分治法）

遍历

![image.png](https://img.fuiboom.com/img/dp_triangle.png)

分治法

![image.png](https://img.fuiboom.com/img/dp_dc.png)

优化 DFS，缓存已经被计算的值（称为：记忆化搜索 本质上：动态规划）

![image.png](https://img.fuiboom.com/img/dp_memory_search.png)

动态规划就是把大问题变成小问题，并解决了小问题重复计算的方法称为动态规划

动态规划和 DFS 区别

- 二叉树 子问题是没有交集，所以大部分二叉树都用递归或者分治法，即 DFS，就可以解决
- 像 triangle 这种是有重复走的情况，**子问题是有交集**，所以可以用动态规划来解决

动态规划，自底向上

```go
func minimumTotal(triangle [][]int) int {
	if len(triangle) == 0 || len(triangle[0]) == 0 {
		return 0
	}
	// 1、状态定义：f[i][j] 表示从i,j出发，到达最后一层的最短路径
	var l = len(triangle)
	var f = make([][]int, l)
	// 2、初始化
	for i := 0; i < l; i++ {
		for j := 0; j < len(triangle[i]); j++ {
			if f[i] == nil {
				f[i] = make([]int, len(triangle[i]))
			}
			f[i][j] = triangle[i][j]
		}
	}
	// 3、递推求解
	for i := len(triangle) - 2; i >= 0; i-- {
		for j := 0; j < len(triangle[i]); j++ {
			f[i][j] = min(f[i+1][j], f[i+1][j+1]) + triangle[i][j]
		}
	}
	// 4、答案
	return f[0][0]
}
func min(a, b int) int {
	if a > b {
		return b
	}
	return a
}

```

动态规划，自顶向下

```go
// 测试用例：
// [
// [2],
// [3,4],
// [6,5,7],
// [4,1,8,3]
// ]
func minimumTotal(triangle [][]int) int {
    if len(triangle) == 0 || len(triangle[0]) == 0 {
        return 0
    }
    // 1、状态定义：f[i][j] 表示从0,0出发，到达i,j的最短路径
    var l = len(triangle)
    var f = make([][]int, l)
    // 2、初始化
    for i := 0; i < l; i++ {
        for j := 0; j < len(triangle[i]); j++ {
            if f[i] == nil {
                f[i] = make([]int, len(triangle[i]))
            }
            f[i][j] = triangle[i][j]
        }
    }
    // 递推求解
    for i := 1; i < l; i++ {
        for j := 0; j < len(triangle[i]); j++ {
            // 这里分为两种情况：
            // 1、上一层没有左边值
            // 2、上一层没有右边值
            if j-1 < 0 {
                f[i][j] = f[i-1][j] + triangle[i][j]
            } else if j >= len(f[i-1]) {
                f[i][j] = f[i-1][j-1] + triangle[i][j]
            } else {
                f[i][j] = min(f[i-1][j], f[i-1][j-1]) + triangle[i][j]
            }
        }
    }
    result := f[l-1][0]
    for i := 1; i < len(f[l-1]); i++ {
        result = min(result, f[l-1][i])
    }
    return result
}
func min(a, b int) int {
    if a > b {
        return b
    }
    return a
}
```

## 递归和动规关系

递归是一种程序的实现方式：函数的自我调用

```go
Function(x) {
	...
	Funciton(x-1);
	...
}
```

动态规划：是一种解决问 题的思想，大规模问题的结果，是由小规模问 题的结果运算得来的。动态规划可用递归来实现(Memorization Search)

## 使用场景

满足两个条件

- 满足以下条件之一
  - 求最大/最小值（Maximum/Minimum ）
  - 求是否可行（Yes/No ）
  - 求可行个数（Count(\*) ）
- 满足不能排序或者交换（Can not sort / swap ）

如题：[longest-consecutive-sequence](https://leetcode-cn.com/problems/longest-consecutive-sequence/)  位置可以交换，所以不用动态规划

## 四点要素

1. **状态 State**
   - 灵感，创造力，存储小规模问题的结果
2. 方程 Function
   - 状态之间的联系，怎么通过小的状态，来算大的状态
3. 初始化 Intialization
   - 最极限的小状态是什么, 起点
4. 答案 Answer
   - 最大的那个状态是什么，终点

## 常见四种类型

1. Matrix DP (10%)
1. Sequence (40%)
1. Two Sequences DP (40%)
1. Backpack (10%)

> 注意点
>
> - 贪心算法大多题目靠背答案，所以如果能用动态规划就尽量用动规，不用贪心算法

## 1、矩阵类型（10%）

### [minimum-path-sum](https://leetcode-cn.com/problems/minimum-path-sum/)

> 给定一个包含非负整数的  *m* x *n*  网格，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。

思路：动态规划
1、state: f[x][y]从起点走到 x,y 的最短路径
2、function: f[x][y] = min(f[x-1][y], f[x][y-1]) + A[x][y]
3、intialize: f[0][0] = A[0][0]、f[i][0] = sum(0,0 -> i,0)、 f[0][i] = sum(0,0 -> 0,i)
4、answer: f[n-1][m-1]

```go
func minPathSum(grid [][]int) int {
    // 思路：动态规划
    // f[i][j] 表示i,j到0,0的和最小
    if len(grid) == 0 || len(grid[0]) == 0 {
        return 0
    }
    // 复用原来的矩阵列表
    // 初始化：f[i][0]、f[0][j]
    for i := 1; i < len(grid); i++ {
        grid[i][0] = grid[i][0] + grid[i-1][0]
    }
    for j := 1; j < len(grid[0]); j++ {
        grid[0][j] = grid[0][j] + grid[0][j-1]
    }
    for i := 1; i < len(grid); i++ {
        for j := 1; j < len(grid[i]); j++ {
            grid[i][j] = min(grid[i][j-1], grid[i-1][j]) + grid[i][j]
        }
    }
    return grid[len(grid)-1][len(grid[0])-1]
}
func min(a, b int) int {
    if a > b {
        return b
    }
    return a
}
```

### [unique-paths](https://leetcode-cn.com/problems/unique-paths/)

> 一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为“Start” ）。
> 机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。
> 问总共有多少条不同的路径？

```go
func uniquePaths(m int, n int) int {
	// f[i][j] 表示i,j到0,0路径数
	f := make([][]int, m)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if f[i] == nil {
				f[i] = make([]int, n)
			}
			f[i][j] = 1
		}
	}
	for i := 1; i < m; i++ {
		for j := 1; j < n; j++ {
			f[i][j] = f[i-1][j] + f[i][j-1]
		}
	}
	return f[m-1][n-1]
}
```

### [unique-paths-ii](https://leetcode-cn.com/problems/unique-paths-ii/)

> 一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为“Start” ）。
> 机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。
> 问总共有多少条不同的路径？
> 现在考虑网格中有障碍物。那么从左上角到右下角将会有多少条不同的路径？

```go
func uniquePathsWithObstacles(obstacleGrid [][]int) int {
	// f[i][j] = f[i-1][j] + f[i][j-1] 并检查障碍物
	if obstacleGrid[0][0] == 1 {
		return 0
	}
	m := len(obstacleGrid)
	n := len(obstacleGrid[0])
	f := make([][]int, m)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if f[i] == nil {
				f[i] = make([]int, n)
			}
			f[i][j] = 1
		}
	}
	for i := 1; i < m; i++ {
		if obstacleGrid[i][0] == 1 || f[i-1][0] == 0 {
			f[i][0] = 0
		}
	}
	for j := 1; j < n; j++ {
		if obstacleGrid[0][j] == 1 || f[0][j-1] == 0 {
			f[0][j] = 0
		}
	}
	for i := 1; i < m; i++ {
		for j := 1; j < n; j++ {
			if obstacleGrid[i][j] == 1 {
				f[i][j] = 0
			} else {
				f[i][j] = f[i-1][j] + f[i][j-1]
			}
		}
	}
	return f[m-1][n-1]
}
```

## 2、序列类型（40%）

### [climbing-stairs](https://leetcode-cn.com/problems/climbing-stairs/)

> 假设你正在爬楼梯。需要  *n*  阶你才能到达楼顶。

```go
func climbStairs(n int) int {
    // f[i] = f[i-1] + f[i-2]
    if n == 1 || n == 0 {
        return n
    }
    f := make([]int, n+1)
    f[1] = 1
    f[2] = 2
    for i := 3; i <= n; i++ {
        f[i] = f[i-1] + f[i-2]
    }
    return f[n]
}
```

### [jump-game](https://leetcode-cn.com/problems/jump-game/)

> 给定一个非负整数数组，你最初位于数组的第一个位置。
> 数组中的每个元素代表你在该位置可以跳跃的最大长度。
> 判断你是否能够到达最后一个位置。

```go
func canJump(nums []int) bool {
    // 思路：看最后一跳
    // 状态：f[i] 表示是否能从0跳到i
    // 推导：f[i] = OR(f[j],j<i&&j能跳到i) 判断之前所有的点最后一跳是否能跳到当前点
    // 初始化：f[0] = 0
    // 结果： f[n-1]
    if len(nums) == 0 {
        return true
    }
    f := make([]bool, len(nums))
    f[0] = true
    for i := 1; i < len(nums); i++ {
        for j := 0; j < i; j++ {
            if f[j] == true && nums[j]+j >= i {
                f[i] = true
            }
        }
    }
    return f[len(nums)-1]
}
```

### [jump-game-ii](https://leetcode-cn.com/problems/jump-game-ii/)

> 给定一个非负整数数组，你最初位于数组的第一个位置。
> 数组中的每个元素代表你在该位置可以跳跃的最大长度。
> 你的目标是使用最少的跳跃次数到达数组的最后一个位置。

```go
// v1动态规划（其他语言超时参考v2）
func jump(nums []int) int {
    // 状态：f[i] 表示从起点到当前位置最小次数
    // 推导：f[i] = f[j],a[j]+j >=i,min(f[j]+1)
    // 初始化：f[0] = 0
    // 结果：f[n-1]
    f := make([]int, len(nums))
    f[0] = 0
    for i := 1; i < len(nums); i++ {
        // f[i] 最大值为i
        f[i] = i
        // 遍历之前结果取一个最小值+1
        for j := 0; j < i; j++ {
            if nums[j]+j >= i {
                f[i] = min(f[j]+1,f[i])
            }
        }
    }
    return f[len(nums)-1]
}
func min(a, b int) int {
    if a > b {
        return b
    }
    return a
}
```

```go
// v2 动态规划+贪心优化
func jump(nums []int) int {
    n:=len(nums)
    f := make([]int, n)
    f[0] = 0
    for i := 1; i < n; i++ {
        // 取第一个能跳到当前位置的点即可
        // 因为跳跃次数的结果集是单调递增的，所以贪心思路是正确的
        idx:=0
        for idx<n&&idx+nums[idx]<i{
            idx++
        }
        f[i]=f[idx]+1
    }
    return f[n-1]
}

```

### [palindrome-partitioning-ii](https://leetcode-cn.com/problems/palindrome-partitioning-ii/)

> 给定一个字符串 _s_，将 _s_ 分割成一些子串，使每个子串都是回文串。
> 返回符合要求的最少分割次数。

```go
func minCut(s string) int {
	// state: f[i] "前i"个字符组成的子字符串需要最少几次cut(个数-1为索引)
	// function: f[i] = MIN{f[j]+1}, j < i && [j+1 ~ i]这一段是一个回文串
	// intialize: f[i] = i - 1 (f[0] = -1)
	// answer: f[s.length()]
	if len(s) == 0 || len(s) == 1 {
		return 0
	}
	f := make([]int, len(s)+1)
	f[0] = -1
	f[1] = 0
	for i := 1; i <= len(s); i++ {
		f[i] = i - 1
		for j := 0; j < i; j++ {
			if isPalindrome(s, j, i-1) {
				f[i] = min(f[i], f[j]+1)
			}
		}
	}
	return f[len(s)]
}
func min(a, b int) int {
	if a > b {
		return b
	}
	return a
}
func isPalindrome(s string, i, j int) bool {
	for i < j {
		if s[i] != s[j] {
			return false
		}
		i++
		j--
	}
	return true
}
```

注意点

- 判断回文字符串时，可以提前用动态规划算好，减少时间复杂度

### [longest-increasing-subsequence](https://leetcode-cn.com/problems/longest-increasing-subsequence/)

> 给定一个无序的整数数组，找到其中最长上升子序列的长度。

```go
func lengthOfLIS(nums []int) int {
    // f[i] 表示从0开始到i结尾的最长序列长度
    // f[i] = max(f[j])+1 ,a[j]<a[i]
    // f[0...n-1] = 1
    // max(f[0]...f[n-1])
    if len(nums) == 0 || len(nums) == 1 {
        return len(nums)
    }
    f := make([]int, len(nums))
    f[0] = 1
    for i := 1; i < len(nums); i++ {
        f[i] = 1
        for j := 0; j < i; j++ {
            if nums[j] < nums[i] {
                f[i] = max(f[i], f[j]+1)
            }
        }
    }
    result := f[0]
    for i := 1; i < len(nums); i++ {
        result = max(result, f[i])
    }
    return result

}
func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

### [word-break](https://leetcode-cn.com/problems/word-break/)

> 给定一个**非空**字符串  *s*  和一个包含**非空**单词列表的字典  *wordDict*，判定  *s*  是否可以被空格拆分为一个或多个在字典中出现的单词。

```go
func wordBreak(s string, wordDict []string) bool {
	// f[i] 表示前i个字符是否可以被切分
	// f[i] = f[j] && s[j+1~i] in wordDict
	// f[0] = true
	// return f[len]

	if len(s) == 0 {
		return true
	}
	f := make([]bool, len(s)+1)
	f[0] = true
	max,dict := maxLen(wordDict)
	for i := 1; i <= len(s); i++ {
		l := 0
		if i - max > 0 {
			l = i - max
		}
		for j := l; j < i; j++ {
			if f[j] && inDict(s[j:i],dict) {
				f[i] = true
                break
			}
		}
	}
	return f[len(s)]
}



func maxLen(wordDict []string) (int,map[string]bool) {
    dict := make(map[string]bool)
	max := 0
	for _, v := range wordDict {
		dict[v] = true
		if len(v) > max {
			max = len(v)
		}
	}
	return max,dict
}

func inDict(s string,dict map[string]bool) bool {
	_, ok := dict[s]
	return ok
}

```

小结

常见处理方式是给 0 位置占位，这样处理问题时一视同仁，初始化则在原来基础上 length+1，返回结果 f[n]

- 状态可以为前 i 个
- 初始化 length+1
- 取值 index=i-1
- 返回值：f[n]或者 f[m][n]

## Two Sequences DP（40%）

### [longest-common-subsequence](https://leetcode-cn.com/problems/longest-common-subsequence/)

> 给定两个字符串  text1 和  text2，返回这两个字符串的最长公共子序列。
> 一个字符串的   子序列   是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。
> 例如，"ace" 是 "abcde" 的子序列，但 "aec" 不是 "abcde" 的子序列。两个字符串的「公共子序列」是这两个字符串所共同拥有的子序列。

```go
func longestCommonSubsequence(a string, b string) int {
    // dp[i][j] a前i个和b前j个字符最长公共子序列
    // dp[m+1][n+1]
    //   ' a d c e
    // ' 0 0 0 0 0
    // a 0 1 1 1 1
    // c 0 1 1 2 1
    //
    dp:=make([][]int,len(a)+1)
    for i:=0;i<=len(a);i++ {
        dp[i]=make([]int,len(b)+1)
    }
    for i:=1;i<=len(a);i++ {
        for j:=1;j<=len(b);j++ {
            // 相等取左上元素+1，否则取左或上的较大值
            if a[i-1]==b[j-1] {
                dp[i][j]=dp[i-1][j-1]+1
            } else {
                dp[i][j]=max(dp[i-1][j],dp[i][j-1])
            }
        }
    }
    return dp[len(a)][len(b)]
}
func max(a,b int)int {
    if a>b{
        return a
    }
    return b
}
```

注意点

- go 切片初始化

```go
dp:=make([][]int,len(a)+1)
for i:=0;i<=len(a);i++ {
    dp[i]=make([]int,len(b)+1)
}
```

- 从 1 开始遍历到最大长度
- 索引需要减一

### [edit-distance](https://leetcode-cn.com/problems/edit-distance/)

> 给你两个单词  word1 和  word2，请你计算出将  word1  转换成  word2 所使用的最少操作数  
> 你可以对一个单词进行如下三种操作：
> 插入一个字符
> 删除一个字符
> 替换一个字符

思路：和上题很类似，相等则不需要操作，否则取删除、插入、替换最小操作次数的值+1

```go
func minDistance(word1 string, word2 string) int {
    // dp[i][j] 表示a字符串的前i个字符编辑为b字符串的前j个字符最少需要多少次操作
    // dp[i][j] = OR(dp[i-1][j-1]，a[i]==b[j],min(dp[i-1][j],dp[i][j-1],dp[i-1][j-1])+1)
    dp:=make([][]int,len(word1)+1)
    for i:=0;i<len(dp);i++{
        dp[i]=make([]int,len(word2)+1)
    }
    for i:=0;i<len(dp);i++{
        dp[i][0]=i
    }
    for j:=0;j<len(dp[0]);j++{
        dp[0][j]=j
    }
    for i:=1;i<=len(word1);i++{
        for j:=1;j<=len(word2);j++{
            // 相等则不需要操作
            if word1[i-1]==word2[j-1] {
                dp[i][j]=dp[i-1][j-1]
            }else{ // 否则取删除、插入、替换最小操作次数的值+1
                dp[i][j]=min(min(dp[i-1][j],dp[i][j-1]),dp[i-1][j-1])+1
            }
        }
    }
    return dp[len(word1)][len(word2)]
}
func min(a,b int)int{
    if a>b{
        return b
    }
    return a
}
```

说明

> 另外一种做法：MAXLEN(a,b)-LCS(a,b)

## 零钱和背包（10%）

### [coin-change](https://leetcode-cn.com/problems/coin-change/)

> 给定不同面额的硬币 coins 和一个总金额 amount。编写一个函数来计算可以凑成总金额所需的最少的硬币个数。如果没有任何一种硬币组合能组成总金额，返回  -1。

思路：和其他 DP 不太一样，i 表示钱或者容量

```go
func coinChange(coins []int, amount int) int {
    // 状态 dp[i]表示金额为i时，组成的最小硬币个数
    // 推导 dp[i]  = min(dp[i-1], dp[i-2], dp[i-5])+1, 前提 i-coins[j] > 0
    // 初始化为最大值 dp[i]=amount+1
    // 返回值 dp[n] or dp[n]>amount =>-1
    dp:=make([]int,amount+1)
    for i:=0;i<=amount;i++{
        dp[i]=amount+1
    }
    dp[0]=0
    for i:=1;i<=amount;i++{
        for j:=0;j<len(coins);j++{
            if  i-coins[j]>=0  {
                dp[i]=min(dp[i],dp[i-coins[j]]+1)
            }
        }
    }
    if dp[amount] > amount {
        return -1
    }
    return dp[amount]

}
func min(a,b int)int{
    if a>b{
        return b
    }
    return a
}
```

注意

> dp[i-a[j]] 决策 a[j]是否参与

### [backpack](https://www.lintcode.com/problem/backpack/description)

> 在 n 个物品中挑选若干物品装入背包，最多能装多满？假设背包的大小为 m，每个物品的大小为 A[i]

```go
func backPack (m int, A []int) int {
    // write your code here
    // f[i][j] 前i个物品，是否能装j
    // f[i][j] =f[i-1][j] f[i-1][j-a[i] j>a[i]
    // f[0][0]=true f[...][0]=true
    // f[n][X]
    f:=make([][]bool,len(A)+1)
    for i:=0;i<=len(A);i++{
        f[i]=make([]bool,m+1)
    }
    f[0][0]=true
    for i:=1;i<=len(A);i++{
        for j:=0;j<=m;j++{
            f[i][j]=f[i-1][j]
            if j-A[i-1]>=0 && f[i-1][j-A[i-1]]{
                f[i][j]=true
            }
        }
    }
    for i:=m;i>=0;i--{
        if f[len(A)][i] {
            return i
        }
    }
    return 0
}

```

### [backpack-ii](https://www.lintcode.com/problem/backpack-ii/description)

> 有 `n` 个物品和一个大小为 `m` 的背包. 给定数组 `A` 表示每个物品的大小和数组 `V` 表示每个物品的价值.
> 问最多能装入背包的总价值是多大?

思路：f[i][j] 前 i 个物品，装入 j 背包 最大价值

```go
func backPackII (m int, A []int, V []int) int {
    // write your code here
    // f[i][j] 前i个物品，装入j背包 最大价值
    // f[i][j] =max(f[i-1][j] ,f[i-1][j-A[i]]+V[i]) 是否加入A[i]物品
    // f[0][0]=0 f[0][...]=0 f[...][0]=0
    f:=make([][]int,len(A)+1)
    for i:=0;i<len(A)+1;i++{
        f[i]=make([]int,m+1)
    }
    for i:=1;i<=len(A);i++{
        for j:=0;j<=m;j++{
            f[i][j]=f[i-1][j]
            if j-A[i-1] >= 0{
                f[i][j]=max(f[i-1][j],f[i-1][j-A[i-1]]+V[i-1])
            }
        }
    }
    return f[len(A)][m]
}
func max(a,b int)int{
    if a>b{
        return a
    }
    return b
}
```

## 练习

Matrix DP (10%)

- [ ] [triangle](https://leetcode-cn.com/problems/triangle/)
- [ ] [minimum-path-sum](https://leetcode-cn.com/problems/minimum-path-sum/)
- [ ] [unique-paths](https://leetcode-cn.com/problems/unique-paths/)
- [ ] [unique-paths-ii](https://leetcode-cn.com/problems/unique-paths-ii/)

Sequence (40%)

- [ ] [climbing-stairs](https://leetcode-cn.com/problems/climbing-stairs/)
- [ ] [jump-game](https://leetcode-cn.com/problems/jump-game/)
- [ ] [jump-game-ii](https://leetcode-cn.com/problems/jump-game-ii/)
- [ ] [palindrome-partitioning-ii](https://leetcode-cn.com/problems/palindrome-partitioning-ii/)
- [ ] [longest-increasing-subsequence](https://leetcode-cn.com/problems/longest-increasing-subsequence/)
- [ ] [word-break](https://leetcode-cn.com/problems/word-break/)

Two Sequences DP (40%)

- [ ] [longest-common-subsequence](https://leetcode-cn.com/problems/longest-common-subsequence/)
- [ ] [edit-distance](https://leetcode-cn.com/problems/edit-distance/)

Backpack & Coin Change (10%)

- [ ] [coin-change](https://leetcode-cn.com/problems/coin-change/)
- [ ] [backpack](https://www.lintcode.com/problem/backpack/description)
- [ ] [backpack-ii](https://www.lintcode.com/problem/backpack-ii/description)


## 自我总结
1.思维框架：
		`明确 base case -> 明确「状态」-> 明确「选择」 -> 定义 dp 数组/函数的含义。`
	
	# 初始化 base case
	dp[0][0][...] = base
	# 进行状态转移
	for 状态1 in 状态1的所有取值：
   	 	for 状态2 in 状态2的所有取值：
      	  for ...
          	  dp[状态1][状态2][...] = 求最值(选择1，选择2...)
	
2.
动态规划是穷举所有例子，递归、备忘录（从上往下）、dpTable(从下往上)
我对做走方格题可以，做序列题感觉吃力，不知道在怎么找状态转移方程
	






