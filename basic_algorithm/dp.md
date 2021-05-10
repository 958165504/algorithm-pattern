# 动态规划

# 自我总结
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

### > 4)编辑距离： 解决两个字符串的动态规划问题 ，一般都是用**两个指针i,j分别指向两个字符串的最后，然后一步步往前走**，缩小问题的规模。
[labuladong 经动态规划：编辑距离 ](https://mp.weixin.qq.com/s?__biz=MzAxODQxMDM0Mw==&mid=2247484731&idx=3&sn=aa642cbf670feee73e20428775dff0b5&chksm=9bd7fb33aca0722568ab71ead8d23e3a9422515800f0587ff7c6ef93ad45b91b9e9920d8728e&scene=21#wechat_redirect)  
[72. 编辑距离](https://leetcode-cn.com/problems/edit-distance/)  

### > 5）最长递增子序列 ： 已知dp[1....4]， 求dp[5]。 nums[5] = 3，既然是递增子序列，我们只要找到前面那些结尾比 3 小的子序列，然后把 3 接到最后，就可以形成一个新的递增子序列，而且这个新的子序列长度加一。需要将nums[5]回溯 从头开始找比其小的dp值，重新组成最长序列的最大长度。  
![image](https://mmbiz.qpic.cn/mmbiz_png/map09icNxZ4kgXtfMiaNRfjKJK5DiaHNAiaEckTjx0BjeFdSIXalPct8LfFicaGnZyaRCK0H0HYNF6nAfZHblloRu4w/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)  

### > 6)最长递增子序列之信封嵌套问题 
```java
先对宽度w进行升序排序，如果遇到w相同的情况，则按照高度h降序排序。之后把所有的h作为一个数组，在这个数组上计算 LIS 的长度就是答案。  
对h降序原因：对于宽度w相同的数对，要对其高度h进行降序排序。因为两个宽度相同的信封不能相互包含的，而逆序排序保证在w相同的数对中最多只选取一个计入 LIS。  
```
### > 7) 子序列解题模板：最长回文子序列 【子序列问题是，对ij比较，相等则一起移动，不相等则看删除i还j】
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
注：找到状态转移和 base case 之后，一定要观察 DP table，看看怎么遍历才能保证通过已计算出来的结果解决新的问题【比如本题的 从底往上计算扫描  
```java
    public int longestPalindromeSubseq(String s) {
        int n = s.length();
        int [][] dp = new int[n][n];
        /*
        * 动态规划
        * dp[i][j]定义，索引i j之间的最长回文子序列的最大长度
        * 选择：当字符i , j 相等 dp[i][j] = dp[i+1][j-1] + 2;
        *                  不相等 dp[i][j] = max(dp[i][j-1], dp[i+1][j])//当ij不等，分别舍弃j，i加入之前的子串，看求这两种情况的最大值
        * basecase: 当下标i==j，一个字符串时,回文长度为1
        */

        //basecase
        for (int i = 0; i < n; i++) {
            dp[i][i] = 1;
        }
        for (int i = n - 1; i >= 0; i--) {
            for (int j = i + 1; j <= n - 1 ; j++) {
                if(s.charAt(i) == s.charAt(j)){
                    dp[i][j] = dp[i+1][j-1] + 2;
                }else {
                    dp[i][j] = Math.max(dp[i][j-1], dp[i+1][j]);
                }
            }
        }
        return dp[0][n - 1];
    }
```


### > 7) 回文子串解题思路：利用判断【i，j】区间是否为回文动态规划解法为基础来变形
> 总结：关于回文的，都可以基于判断【i，j】区间是否为回文动态规划解法为基础，来进行穷举所有区间判断，来变相解题  
dp[i][j] = (array[i] == array[j])&&( j - i < 3 || dp[i - 1][j - 1]);//动态规划状态
    当i == j, 就去掉两端转移看i+1,j-1是否为回文
    当j-i<3,只有一个 或两个 三个 且两端相等，那一定为回文  
    
[647. 回文子串（利用判断【i，j】区间是否为回文动态规划解法为基础）](https://leetcode-cn.com/problems/palindromic-substrings/solution/647-hui-wen-zi-chuan-li-yong-pan-duan-ij-aofk/)  
```java
/*
思路：利用判断【i，j】区间是否为回文动态规划解法基础上，穷举出全部区间【i，j】，在用【i，j】是否为回文的动归解法判断，最后对每个回文区间累加计数，得到全部的回文子串数，O（n^2）

拓展 :
	1）5. 最长回文子串：根据判断【i，j】区间是否为回文动态规划解法，还可以用在求 ：穷举所有区间[ij]，来动归判断是否为回文，若是，然后不断更新j-i最大值，来求得最长回文串）
        2）求最长子序列也是差不都的动归状态转移方程  
				选择：当字符i , j 相等 dp[i][j] = dp[i+1][j-1] + 2;
                                不相等 dp[i][j] = max(dp[i][j-1], dp[i+1][j])//当ij不等，分别舍弃j，i加入之前的子串，看求这两种情况的最大值

*/
 public int countSubstrings(String s) {
        int polindromeCount = 0;//回文个数
        char[] array = s.toCharArray();
        boolean [][] dp = new boolean[array.length][array.length];//[i,j]区间是否为回文串
        for (int j = 0; j < dp.length; j++) {
            for (int i = 0; i <= j; i++) {
                dp[i][j] = (array[i] == array[j])&&( j - i < 3 || dp[i + 1][j - 1]);//动态规划状态转移：
                if (dp[i][j])
                    polindromeCount++;
            }
        }
        return polindromeCount;
    }
```

### > 8) 0-1背包问题
```java
dp[i][w]的定义如下：对于前i个物品，当前背包的容量为w，这种情况下可以装的最大价值是dp[i][w]。  
int dp[N+1][W+1]
dp[0][..] = 0
dp[..][0] = 0

for i in [1..N]:
    for w in [1..W]:
        dp[i][w] = max(
            把物品 i 装进背包,
            不把物品 i 装进背包
        )
return dp[N][W]
```
[416. 分割等和子集](https://leetcode-cn.com/problems/partition-equal-subset-sum/)  

### > 9) 贪心算法
```java
/*
* 贪心算法：贪心算法之区间调度问题
* 先排序,从低往上，取end1 <= start2
* 			是：则收下这个区间，取end2为末尾依次递进
* 			不是：则比较下一个区间 end1 <= start3
* 因为是排了序的，相当于求：尽量使得区间填满时间线【使得时间线最长】
* */
```
[435. 无重叠区间](https://leetcode-cn.com/problems/non-overlapping-intervals/)  

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

### >11 乘积最大子数组（动态规划破解双重for）
[152. 乘积最大子数组](https://leetcode-cn.com/problems/maximum-product-subarray/solution/152-cheng-ji-zui-da-zi-shu-zu-dong-tai-g-6nhn/)  
//看的官方的解答
    /*遇到暴力解需要两个for时，考虑用动态规划的话，只考虑一个端点再动，这样一次遍历完成O(n)
    定义dp[i]：表示以i为结尾的子数组乘积max，dp[i] = max{dp[i-1]*ai,ai},
    已知前面i-1为结尾子数组，到了i就考虑把i接到末尾，还是单独成一派

    由于存在正负数，需要分正负两种情况比较最大值
    如：a={5,6,−3,4,−3}，最大值序列{5,30,−3,4,−3}，但是最大值!=30,而是全部相乘
    /
     */
```java
	public int maxProduct(int[] nums) {
            int length = nums.length;
            int[] maxF = new int[length];
            int[] minF = new int[length];
            //赋初值，最坏的情况，最大值应该为自己一个数
            System.arraycopy(nums, 0, maxF, 0, length);
            System.arraycopy(nums, 0, minF, 0, length);
            for (int i = 1; i < length; ++i) {
                maxF[i] = Math.max(maxF[i - 1] * nums[i], Math.max(nums[i], minF[i - 1] * nums[i]));
                minF[i] = Math.min(minF[i - 1] * nums[i], Math.min(nums[i], maxF[i - 1] * nums[i]));
            }
            int ans = maxF[0];
            for (int i = 1; i < length; ++i) {
                ans = Math.max(ans, maxF[i]);
            }
            return ans;
        }
```

### 凑零钱类型问题
[279. 完全平方数](https://leetcode-cn.com/problems/perfect-squares/)  
```java
 /*
    这道题和纸币买卖题一样，有一个选择列表，求金额的最少纸币数
    方法一：递归暴力解：先算出选择列表choseList，然后想买卖问题一样，每次遍历试下所有的纸币，在每一轮选择循环中，选取最少的纸币数，但会溢出  可加上备忘录消除重复解‘
    方法二：动态规划从下往上：由于动归是从底往上，i之前算出了，因此原理和递归+备忘录一样
    for(i:n)
        for(k:选择列表))
            dp[i] = min(dp[i], dp[i - k]+1)
    */

    public int numSquares(int n) {
        //动态规划
        //生成选择列表
        int maxSquareNum = (int) (Math.sqrt(n)) + 1;//考虑到根号为小数情况
        int[] choseList = new int[maxSquareNum + 1];
        for (int i = 1; i <= maxSquareNum; i++) {
            choseList[i] = i * i;
        }
        //动态规划进行选择
        int[] dp = new int[n + 1];
        Arrays.fill(dp,Integer.MAX_VALUE);
        dp[0] = 0;//basecase
        for (int i = 1; i <= n; i++) {
            //选择列表
            for (int j = 1; j < choseList.length; j++) {
                if(i - choseList[j] < 0)
                    break;
                dp[i] = Math.min(dp[i - choseList[j]] + 1, dp[i]);
            }
        }
        return dp[n];
    }
```

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
	






