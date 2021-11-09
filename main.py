# This is a sample Python script.

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def two_sum(nums, target):
    num_dict = {}
    for i, num in enumerate(nums):
        check = target - num
        if check in num_dict:
            return num_dict[check], i
        num_dict[num] = i
    return None


def test_two_sum():
    indexes = two_sum([2, 7, 11, 15], 9)
    print(indexes)
    indexes = two_sum([3, 2, 4], 6)
    print(indexes)


def length_longest_substring_without_repeating(string):
    char_set = set()
    left_index = 0
    result = 0
    for i, c in enumerate(string):
        while c in char_set:
            char_set.remove(string[left_index])
            left_index += 1
        char_set.add(c)
        result = max(result, i - left_index + 1)
    return result


def test_length_longest_substring_without_repeating():
    print(length_longest_substring_without_repeating("abcabcbb"))
    print(length_longest_substring_without_repeating("bbbbb"))
    print(length_longest_substring_without_repeating("pwwkew"))
    print(length_longest_substring_without_repeating(""))


def best_time_stock(prices):
    pointer_left = 0
    result = 0
    for pointer_right, p in enumerate(prices[1:]):
        if p < prices[pointer_left]:
            pointer_left = pointer_right
        else:
            result = max(result, p - prices[pointer_left])

    return result


def test_best_time_stock():
    print(best_time_stock([7, 1, 5, 6, 4]))


def contains_duplicate(nums):
    hash_set = set()
    for n in nums:
        if n in hash_set:
            return True
        hash_set.add(n)
    return False


def test_contains_duplicate():
    print(contains_duplicate([1, 2, 3, 1]))


def product_of_array_except_self(nums):
    output = [1] * len(nums)

    p_fix = 1
    for i, num in enumerate(nums):
        output[i] = p_fix
        p_fix *= num

    p_fix = 1
    for i in range(len(nums) - 1, -1, -1):
        output[i] *= p_fix
        p_fix *= nums[i]

    return output


def test_product_of_array_except_self():
    print(product_of_array_except_self([1, 2, 3, 4]))


def maximum_subarray(nums):
    current_sum = 0
    max_sum = nums[0]
    for num in nums:
        if current_sum < 0:
            current_sum = 0
        current_sum += num
        max_sum = max(max_sum, current_sum)

    return max_sum


def test_maximum_subarray():
    print(maximum_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]))


def max_product(nums):
    result = max(nums)
    current_max_prod = 1
    current_min_prod = 1
    for num in nums:
        if num == 0:
            current_max_prod = 1
            current_min_prod = 1
            continue
        tmp_max = current_max_prod * num
        tmp_min = current_min_prod * num
        current_max_prod = max(num, tmp_min, tmp_max)
        current_min_prod = min(num, tmp_min, tmp_max)
        result = max(result, current_max_prod)
    return result


def test_max_product():
    print(max_product([2, 3, -2, 4]))
    print(max_product([-1, -2, -3]))


def min_sorted_array(nums):
    left_pointer = 0
    right_pointer = len(nums) - 1
    result = nums[0]
    while left_pointer <= right_pointer:
        if nums[left_pointer] < nums[right_pointer]:
            result = min(result, nums[left_pointer])
            break

        middle_pointer = (left_pointer + right_pointer) // 2
        result = min(result, nums[middle_pointer])
        if nums[middle_pointer] >= nums[left_pointer]:
            left_pointer = middle_pointer + 1
        else:
            right_pointer = middle_pointer - 1
    return result


def test_min_sorted_array():
    print(min_sorted_array([3, 4, 5, 1, 2]))


def search_sorted_array(nums, n):
    result = -1
    left_pointer = 0
    right_pointer = len(nums) - 1
    while left_pointer <= right_pointer:
        middle_pointer = (left_pointer + right_pointer) // 2
        if nums[middle_pointer] == n:
            return middle_pointer
        if nums[left_pointer] <= n < nums[middle_pointer]:
            right_pointer = middle_pointer - 1
        else:
            left_pointer = middle_pointer + 1
    return result


def test_search_sorted_array():
    print(search_sorted_array([4, 5, 6, 7, 0, 1, 2], 0))
    print(search_sorted_array([0, 1, 2, 3, 4, 5, 6, 7], 0))


def three_sum(nums):
    nums.sort()
    sum_list = []
    for i, num in enumerate(nums):
        if i > 0 and num == nums[i - 1]:
            continue
        left_pointer = i + 1
        right_pointer = len(nums) - 1
        while left_pointer < right_pointer:
            sum_num = num + nums[left_pointer] + nums[right_pointer]
            if sum_num > 0:
                right_pointer -= 1
            elif sum_num < 0:
                left_pointer += 1
            else:
                sum_list.append([num, nums[left_pointer], nums[right_pointer]])
                left_pointer += 1
                while nums[left_pointer] == nums[left_pointer - 1] and left_pointer < right_pointer:
                    left_pointer += 1
    return sum_list


def test_three_sum():
    print(three_sum([-1, 0, 1, 2, -1, 4]))
    print(three_sum([-3, 3, 4, -3, 1, 2]))


def water_container(heights):
    max_area = 0
    left_pointer = 0
    right_pointer = len(heights) - 1
    while left_pointer < right_pointer:
        area = (right_pointer - left_pointer) * min(heights[left_pointer], heights[right_pointer])
        max_area = max(max_area, area)
        if heights[left_pointer] < heights[right_pointer]:
            left_pointer += 1
        else:
            right_pointer -= 1
    return max_area


def test_water_container():
    print(water_container([1, 8, 6, 2, 5, 4, 8, 3, 7]))


def number_of_1_bits(number):
    bits = 0
    while number:
        number &= (number - 1)
        bits += 1
    return bits


def test_number_of_1_bits():
    print(number_of_1_bits(11))


def counting_bits(number):
    list_bits = []
    for i in range(number + 1):
        bits = 0
        n = i
        while n:
            n &= (n - 1)
            bits += 1
        list_bits.append(bits)
    return list_bits


def test_counting_bits():
    print(counting_bits(2))


def missing_number(nums):
    result = len(nums)
    for i in range(len(nums)):
        result += (i - nums[i])
    return result


def test_missing_number():
    print(missing_number([3, 0, 1]))


def reverse_bits(n):
    result = 0
    for i in range(32):
        bit = (n >> i) & 1
        result |= (bit << (31 - i))
    return result


def test_reverse_bits():
    print(reverse_bits(4))


def climbing_stairs(n):
    one = 1
    two = 1
    for i in range(n - 1):
        tmp = one
        one += two
        two = tmp
    return one


def test_climbing_stairs():
    print(climbing_stairs(3))
    print(climbing_stairs(5))


def coins_change(coins, amount):
    error = amount + 1
    change_list = [[error]] * error
    change_list[0] = []

    for i in range(error):
        if change_list[i] == [error]:
            continue
        for c in coins:
            if c + i > amount:
                continue
            combination = change_list[i] + [c]
            if change_list[i + c] == [error] or len(change_list[i + c]) > len(combination):
                change_list[i + c] = combination
    return len(change_list[-1])


def test_coins_change():
    print(coins_change([1, 2, 5], 11))


def longest_increasing_subsequence(nums):
    lengths = [1] * len(nums)
    for i in range(len(nums) - 1, -1, -1):
        for j in range(1 + i, len(nums)):
            if nums[i] < nums[j]:
                lengths[i] = max(1, lengths[j] + 1)
    return max(lengths)


def test_longest_increasing_subsequence():
    print(longest_increasing_subsequence([1, 2, 4, 3]))


def longest_common_sequence(str1, str2):
    dp = [[0 for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]
    for i in range(len(str1) - 1, -1, -1):
        for j in range(len(str2) - 1, -1, -1):
            if str1[i] == str2[j]:
                dp[i][j] = 1 + dp[i + 1][j + 1]
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
    return dp[0][0]


def word_break(s, words):
    dp = [False] * (len(s) + 1)
    dp[len(s)] = True

    for i in range(len(s) - 1, -1, -1):
        for w in words:
            if (i + len(w)) <= len(s) and s[i : i + len(w)] == w:
                dp[i] = dp[i + len(w)]
            if dp[i]:
                break
    return dp[0]


def combination_sum(candidates, target):
    result = []

    def dfs(i, current, total):
        if total == target:
            result.append(current.copy())
            return
        if total > target or i >= len(candidates):
            return

        current.append(candidates[i])
        dfs(i, current, total + candidates[i])

        current.pop()
        dfs(i + 1, current, total)

    dfs(0, [], 0)
    return result


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test_longest_increasing_subsequence()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
